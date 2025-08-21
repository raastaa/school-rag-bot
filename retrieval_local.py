# retrieval_local.py
from __future__ import annotations
from typing import List, Dict, Tuple
import os
import html
import re

from gigachat_client import GigaChatEmbedder, chat
from store_qdrant import search as qsearch, get_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pypdf import PdfReader, PdfWriter

# Параметры
MAX_ITEMS      = 3
SNIPPET_LIMIT  = 800
NEIGH_BEFORE   = 0          # только центр
NEIGH_AFTER    = 1          # и ОДИН чанк после
THRESHOLD      = float(os.getenv("RELEVANCE_THRESHOLD", "0.82"))  # порог релевантности (>=)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "school_docs")

def _escape(t: str) -> str:
    # экранируем для parse_mode="HTML"
    return html.escape(t or "", quote=False)

def _block_header(pl: Dict) -> str:
    src   = _escape(pl.get("source") or "Источник")
    p_from = pl.get("page_from")
    head = f"Информация найдена в файле <b>{src}</b>"
    if p_from:
        head += f" (стр. {p_from})"
    return head

def _concat_limit(parts: List[str], limit: int) -> str:
    text = "\n\n".join(p.strip() for p in parts if p)
    if len(text) <= limit:
        return _escape(text)
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return _escape(cut) + "…"

def _sort_key(pl: Dict) -> tuple:
    seq = pl.get("seq")
    pf  = pl.get("page_from") or 0
    pid = pl.get("id") or ""
    seq_key = seq if isinstance(seq, int) else -1
    return (seq_key, pf, str(pid))


def _slice_pdf_pages(src_path: str, start: int, end: int) -> str:
    """Сохраняет диапазон страниц [start, end] из PDF в новый файл."""
    reader = PdfReader(src_path)
    total = len(reader.pages)
    if total == 0:
        return src_path
    s = max(1, start)
    e = min(total, end)
    writer = PdfWriter()
    for i in range(s - 1, e):
        writer.add_page(reader.pages[i])
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_dir = os.path.join("outputs", "snippets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_{s}_{e}.pdf")
    with open(out_path, "wb") as f:
        writer.write(f)
    return out_path

def _collect_doc_points(path: str, hard_cap: int = 5000) -> List[Dict]:
    """Читаем все точки по данному файлу (payload.path == path) через embedded-клиент."""
    cli = get_client()
    out: List[Dict] = []
    next_page = None
    fetched = 0
    while True:
        points, next_page = cli.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            with_payload=True,
            offset=next_page,
            scroll_filter=Filter(must=[FieldCondition(key="path", match=MatchValue(value=path))]),
        )
        if not points:
            break
        for p in points:
            if p.payload:
                out.append(p.payload)
                fetched += 1
                if fetched >= hard_cap:
                    return out
        if next_page is None:
            break
    return out

def _neighbors_preview(all_payloads: List[Dict], center_id: str) -> str:
    """Берём центр и 1 следующий чанк (окно NEIGH_BEFORE=0, NEIGH_AFTER=1)."""
    if not all_payloads:
        return ""
    all_sorted = sorted(all_payloads, key=_sort_key)
    idx = next((i for i, pl in enumerate(all_sorted) if pl.get("id") == center_id), None)
    if idx is None:
        center = next((pl for pl in all_sorted if pl.get("id") == center_id), None)
        txt = (center.get("text") if center else "") or ""
        return _concat_limit([txt], SNIPPET_LIMIT)
    lo = max(0, idx - NEIGH_BEFORE)
    hi = min(len(all_sorted), idx + NEIGH_AFTER + 1)
    parts = [pl.get("text") or "" for pl in all_sorted[lo:hi]]
    return _concat_limit(parts, SNIPPET_LIMIT)

async def _summarize_text(text: str) -> str:
    """Получаем краткое резюме и алгоритм действий через GigaChat."""
    if not text:
        return ""
    # ограничим размер текста, чтобы не превышать лимиты модели
    cut = text[:12000]
    prompt = (
        "На основе приведённого текста составь краткую выжимку (2-3 абзаца) "
        "и чёткий пошаговый алгоритм действий. Ничего не придумывай вне этого текста.\n\n"
        f"Текст:\n{cut}\n\n"
        "Ответ сформируй строго без каких-либо вступлений:\n"
        "<абзацы выжимки без заголовка>\n\nАлгоритм действий:\n1. ..."
    )
    print("[_summarize_text] Source text:")
    print(cut)
    print("[_summarize_text] Prompt sent to GigaChat:")
    print(prompt)
    result = await chat(prompt)
    print("[_summarize_text] Response:")
    print(result)
    return result


async def _score_answer(question: str, text: str) -> float:
    """Оценивает релевантность фрагмента вопросу через GigaChat (0..1)."""
    if not text:
        return 0.0
    prompt = (
        f"Вопрос:\n{question}\n\n"
        f"Фрагмент:\n{text}\n\n"
        "Оцени насколько фрагмент отвечает на вопрос по шкале от 0 до 1. "
        "Ответь только числом." 
    )
    resp = await chat(prompt)
    if not resp:
        return 0.0
    val = resp.strip().replace(",", ".")
    try:
        score = float(val)
    except ValueError:
        m = re.search(r"0?\.\d+|1(?:\.0+)?", val)
        score = float(m.group()) if m else 0.0
    return max(0.0, min(1.0, score))

def extract_scored(hits: list) -> list[tuple[dict, float | None]]:
    """Удобно вытащить (payload, score) для логирования."""
    out = []
    for h in hits or []:
        out.append((getattr(h, "payload", None) or {}, getattr(h, "score", None)))
    return out

async def retrieve_local(
    question: str,
    top_k: int = MAX_ITEMS,
    prefer_spravochnik: bool = True
) -> Tuple[str, str, str, List[Dict], List[str], Dict[str, list]]:
    """
    Возвращает:
      header (HTML), summary_html, algorithm_text,
      cites (для ссылок/метаданных), files (пути к файлам),
      diag({'passed','rejected','gigachat'}).
    """
    print(f"[retrieve_local] question: {question}")
    emb = GigaChatEmbedder()
    print("[retrieve_local] embedding question")
    qvec = (await emb.embed([question]))[0]
    print(f"[retrieve_local] embedding vector size: {len(qvec)}")

    if not prefer_spravochnik:
        print("[retrieve_local] searching without spravochnik filter")
        hits = qsearch(qvec, top_k=top_k)
    else:
        k1 = max(2, top_k // 2)
        k2 = top_k - k1
        print(f"[retrieve_local] searching with spravochnik preference k1={k1}, k2={k2}")
        h1 = qsearch(qvec, top_k=k1, source_filter="spravochnik")
        h2 = qsearch(qvec, top_k=top_k)
        seen = set(p.payload.get("id") for p in h1 if p.payload)
        h2 = [p for p in h2 if p.payload and p.payload.get("id") not in seen]
        hits = h1 + h2[:k2]
    print(f"[retrieve_local] hits retrieved: {len(hits)}")

    if not hits:
        return "Ничего релевантного не найдено в локальном справочнике.", "", "", [], [], {"passed": [], "rejected": []}

    # обрежем по количеству и применим порог
    hits = hits[:top_k]
    passed, rejected = [], []
    for h in hits:
        sc = getattr(h, "score", None)
        if sc is None or sc >= THRESHOLD:
            passed.append(h)
        else:
            rejected.append(h)
    print(f"[retrieve_local] passed={len(passed)}, rejected={len(rejected)}")

    if not passed:
        # совсем пусто после порога
        return "Ничего релевантного не найдено в локальном справочнике.", "", "", [], [], {
            "passed": [], "rejected": extract_scored(rejected)
        }

    doc_cache: Dict[str, List[Dict]] = {}
    scored: List[tuple] = []
    for h in passed:
        pl = h.payload or {}
        path = pl.get("path")
        center_id = pl.get("id") or str(h.id)
        if path not in doc_cache:
            doc_cache[path] = _collect_doc_points(path) if path else []
        doc_payloads = doc_cache[path]
        preview = _neighbors_preview(doc_payloads, center_id)
        gc_score = await _score_answer(question, preview)
        scored.append((h, gc_score, preview, doc_payloads))

    if not scored:
        return "Ничего релевантного не найдено в локальном справочнике.", "", "", [], [], {
            "passed": extract_scored(passed),
            "rejected": extract_scored(rejected),
        }

    scored.sort(key=lambda x: x[1], reverse=True)
    best_h, _, best_preview, best_doc_payloads = scored[0]
    pl = best_h.payload or {}
    path = pl.get("path")

    header = _block_header(pl)
    score = getattr(best_h, "score", None)
    score_txt = ""
    if score is not None:
        score_pct = int(round(score * 100))
        score_txt = f" — коэфф. совпадения {score_pct}%"

    summary_html = ""
    algorithm_text = ""
    if path:
        print(f"[retrieve_local] summarizing for file: {path}")
        if pl.get("page_from"):
            pf = pl.get("page_from") or 1
            pt = pl.get("page_to") or pf
            start_p = pf - 2
            end_p = pt + 2
            snippet_chunks: List[str] = []
            for pld in best_doc_payloads:
                p_page = pld.get("page_from") or 0
                if start_p <= p_page <= end_p:
                    snippet_chunks.append(pld.get("text") or "")
            summary_text = "\n".join(snippet_chunks)
        else:
            ordered = sorted(best_doc_payloads, key=_sort_key)
            summary_text = "\n".join(p.get("text") or "" for p in ordered)
        summary_raw = await _summarize_text(summary_text)
        if summary_raw:
            sr = summary_raw.strip()
            lines_sr = sr.splitlines()
            cleaned: List[str] = []
            started = False
            for line in lines_sr:
                low = line.lower().lstrip()
                if not started:
                    if "краткая выжимка" in low:
                        started = True
                        continue
                    if low.startswith("запрос:") or low.startswith("источник:") or low.startswith("общая информация:"):
                        continue
                cleaned.append(line)
                started = True
            sr = "\n".join(cleaned).strip()
            marker = "Алгоритм действий:"
            if marker in sr:
                before, alg = sr.split(marker, 1)
                summary_html = _escape(before.strip())
                algorithm_text = alg.strip()
            else:
                summary_html = _escape(sr)
    else:
        # If the document path is missing, still attempt to summarize the preview
        # so that local_summary and local_algorithm are not empty
        summary_raw = await _summarize_text(best_preview)
        if summary_raw:
            sr = summary_raw.strip()
            marker = "Алгоритм действий:"
            if marker in sr:
                before, alg = sr.split(marker, 1)
                summary_html = _escape(before.strip())
                algorithm_text = alg.strip()
            else:
                summary_html = _escape(sr)

    header_block = f"{header}{score_txt}"

    cites = [{
        "source": pl.get("source"),
        "page_from": pl.get("page_from"),
        "path": path,
        "text": best_preview,
    }]

    files_out: List[str] = []
    if path:
        if pl.get("source_group") == "spravochnik" and pl.get("page_from"):
            pf = pl.get("page_from") or 1
            pt = pl.get("page_to") or pf
            snippet = _slice_pdf_pages(path, pf - 2, pt + 2)
            files_out.append(snippet)
        else:
            files_out.append(path)

    diag = {
        "passed": extract_scored(passed),
        "rejected": extract_scored(rejected),
        "gigachat": [(getattr(h, "payload", None) or {}, sc) for h, sc, _, _ in scored],
    }
    return header_block.strip(), summary_html, algorithm_text, cites, files_out, diag
