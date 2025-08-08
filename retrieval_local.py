# retrieval_local.py
from __future__ import annotations
from typing import List, Dict, Tuple
import os
import html

from gigachat_client import GigaChatEmbedder, chat
from store_qdrant import search as qsearch, get_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Параметры
MAX_ITEMS      = 3
SNIPPET_LIMIT  = 800
NEIGH_BEFORE   = 0          # только центр
NEIGH_AFTER    = 1          # и ОДИН чанк после
THRESHOLD      = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))  # порог релевантности (>=)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "school_docs")

def _escape(t: str) -> str:
    # экранируем для parse_mode="HTML"
    return html.escape(t or "", quote=False)

def _block_header(pl: Dict) -> str:
    src   = _escape(pl.get("source") or "Источник")
    p_from = pl.get("page_from")
    group = _escape(pl.get("source_group") or "local")
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
        "Ответ сформируй в виде:\n"
        "Краткая выжимка:\n<абзацы>\n\nАлгоритм действий:\n1. ..."
    )
    return await chat(prompt)

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
) -> Tuple[str, List[Dict], List[str], Dict[str, list]]:
    """
    Возвращает:
      msg_text (HTML), cites (для ссылок/метаданных), files (пути к файлам), diag({'passed','rejected'}).
    """
    emb = GigaChatEmbedder()
    qvec = (await emb.embed([question]))[0]

    if not prefer_spravochnik:
        hits = qsearch(qvec, top_k=top_k)
    else:
        k1 = max(2, top_k // 2)
        k2 = top_k - k1
        h1 = qsearch(qvec, top_k=k1, source_filter="spravochnik")
        h2 = qsearch(qvec, top_k=top_k)
        seen = set(p.payload.get("id") for p in h1 if p.payload)
        h2 = [p for p in h2 if p.payload and p.payload.get("id") not in seen]
        hits = h1 + h2[:k2]

    if not hits:
        return "Ничего релевантного не найдено в локальном справочнике.", [], [], {"passed": [], "rejected": []}

    # обрежем по количеству и применим порог
    hits = hits[:top_k]
    passed, rejected = [], []
    for h in hits:
        sc = getattr(h, "score", None)
        if sc is None or sc >= THRESHOLD:
            passed.append(h)
        else:
            rejected.append(h)

    if not passed:
        # совсем пусто после порога
        return "Ничего релевантного не найдено в локальном справочнике.", [], [], {
            "passed": [], "rejected": extract_scored(rejected)
        }

    lines: List[str] = ["Найдено в локальной базе:\n"]
    cites: List[Dict] = []
    files_set = set()

    for h in passed:
        pl = h.payload or {}
        path = pl.get("path")
        center_id = pl.get("id") or str(h.id)
        header = _block_header(pl)
        score = getattr(h, "score", None)
        score_txt = f" (коэфф. {score:.3f})" if score is not None else ""
        preview = _neighbors_preview(_collect_doc_points(path) if path else [], center_id)
        block = f"• {header}{score_txt}\n{preview}" if preview else f"• {header}{score_txt}"

        summary = ""
        if path and pl.get("source_group") != "spravochnik":
            all_payloads = _collect_doc_points(path)
            ordered = sorted(all_payloads, key=_sort_key)
            full_text = "\n".join(p.get("text") or "" for p in ordered)
            summary_raw = await _summarize_text(full_text)
            if summary_raw:
                sr = summary_raw.strip()
                sr = sr.replace("Краткая выжимка:\n", "", 1).replace("Краткая выжимка:", "", 1).strip()
                marker = "Алгоритм действий:"
                if marker in sr:
                    before, alg = sr.split(marker, 1)
                    before_html = _escape(before.strip())
                    alg_html = _escape(marker + "\n" + alg.strip())
                    prefix = f"{before_html}\n" if before_html else ""
                    summary = f"{prefix}<b>{alg_html}</b>"
                else:
                    summary = _escape(sr)
        if summary:
            block += f"\n{summary}"
        lines.append(block + "\n")
        cites.append({
            "source": pl.get("source"),
            "page_from": pl.get("page_from"),
            "path": path,
            "text": preview
        })
        if path:
            files_set.add(path)

    msg = "\n".join(lines).strip()
    diag = {
        "passed": extract_scored(passed),
        "rejected": extract_scored(rejected),
    }
    return msg, cites, list(files_set), diag
