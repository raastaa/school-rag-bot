# retrieval_local.py
from __future__ import annotations
from typing import List, Dict, Tuple
import os
import html
import json
import re
import time
import numpy as np
from rank_bm25 import BM25Okapi
import logging

from gigachat_client import GigaChatEmbedder, chat, generate_query_hyde
from store_qdrant import search as qsearch, get_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from db_local import get_feedback_by_source
from models import DocChunk
from config import (
    TOP_K_INITIAL,
    TOP_K_MAX,
    MIN_UNIQUE_SOURCES,
    MMR_LAMBDA,
    RERANK_ALPHA,
    HYDE_N,
    NEIGHBOR_RADIUS,
    PERCENTILE_CUT,
    MIN_RESULTS_FLOOR,
    LOCAL_SEARCH_CACHE_TTL_SEC,
    DIRECTOR_TOP_K_INITIAL,
    DIRECTOR_NEIGHBOR_RADIUS,
    DIRECTOR_PERCENTILE_CUT,
    DIRECTOR_MIN_RESULTS_FLOOR,
    DIRECTOR_MMR_LAMBDA,
    DIRECTOR_RERANK_ALPHA,
    DIRECTOR_HYDE_N,
    DIRECTOR_MIN_UNIQUE_SECTIONS,
)

# Параметры
MAX_ITEMS = TOP_K_INITIAL
SNIPPET_LIMIT = 2000
NEIGH_BEFORE = NEIGHBOR_RADIUS
NEIGH_AFTER = NEIGHBOR_RADIUS
THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.82"))  # базовый порог
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "school_docs")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() not in {"0", "false"}
# Максимальная длина текста резюме
SUMMARY_LIMIT = 1000

logger = logging.getLogger(__name__)

DIRECTOR_KEYWORDS_RE = re.compile(
    r"(приказ|приказы|приказываю|распоряжение|бланк|регистрация|книга приказов|основание|контроль за исполнением|дата|номер|заголовок|место издания|подпись)",
    re.IGNORECASE,
)


# --- feedback stats cache ---
_FEEDBACK_CACHE: Dict[str, Tuple[float, int]] = {}
_FEEDBACK_CACHE_TS = 0.0
_FEEDBACK_CACHE_TTL = float(os.getenv("FEEDBACK_CACHE_TTL", "3600"))


def _feedback_stats() -> Dict[str, Tuple[float, int]]:
    """Возвращает кэшированную статистику рейтингов по источникам."""
    global _FEEDBACK_CACHE, _FEEDBACK_CACHE_TS
    if time.time() - _FEEDBACK_CACHE_TS > _FEEDBACK_CACHE_TTL:
        _FEEDBACK_CACHE = get_feedback_by_source()
        _FEEDBACK_CACHE_TS = time.time()
    return _FEEDBACK_CACHE


def _apply_feedback(score: float, source: str | None) -> float:
    if source is None:
        return score
    stats = _feedback_stats()
    data = stats.get(source)
    if not data:
        return score
    avg = data[0] if isinstance(data, tuple) else data
    return score * (1.0 + avg)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def select_chunks_mmr(
    qv: np.ndarray, docs: List[DocChunk], k: int = 5, lam: float = MMR_LAMBDA
) -> List[DocChunk]:
    selected: List[DocChunk] = []
    rest = docs[:]
    while rest and len(selected) < k:
        best, best_val = None, -1e9
        for d in rest:
            rel = _cosine(qv, np.array(d.vector))
            div = (
                0.0
                if not selected
                else max(
                    _cosine(np.array(d.vector), np.array(s.vector)) for s in selected
                )
            )
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best, best_val = d, val
        if best is None:
            break
        selected.append(best)
        rest.remove(best)
    return selected


def rerank_combined(
    query: str, docs: List[DocChunk], alpha: float = RERANK_ALPHA
) -> List[DocChunk]:
    if not docs:
        return []
    tokenized_corpus = [d.text.split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm_scores = bm25.get_scores(query.split())
    bm_scores = (bm_scores - bm_scores.min()) / (
        bm_scores.max() - bm_scores.min() + 1e-9
    )
    vec_scores = np.array([d.score for d in docs])
    vec_scores = (vec_scores - vec_scores.min()) / (
        vec_scores.max() - vec_scores.min() + 1e-9
    )
    final = alpha * vec_scores + (1 - alpha) * bm_scores
    ranked = sorted(zip(docs, final), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked]


_SEARCH_CACHE: Dict[str, Tuple[float, List[DocChunk]]] = {}


def _cache_key(queries: List[str], top_k: int) -> str:
    return json.dumps({"q": queries, "k": top_k}, sort_keys=True)


async def fused_search(
    queries: List[str], top_k: int = TOP_K_INITIAL
) -> List[DocChunk]:
    key = _cache_key(queries, top_k)
    now = time.time()
    cached = _SEARCH_CACHE.get(key)
    if cached and now - cached[0] < LOCAL_SEARCH_CACHE_TTL_SEC:
        logger.info("fused_search cache hit for %s", key)
        return cached[1]
    logger.info("Starting fused search for %d queries, top_k=%d", len(queries), top_k)
    emb = GigaChatEmbedder()
    logger.info("Embedding queries")
    vecs = await emb.embed(queries)
    docs: Dict[str, DocChunk] = {}
    for q, vec in zip(queries, vecs):
        logger.info("Searching Qdrant for query: %s", q)
        hits = qsearch(vec, top_k=top_k * 2)
        for h in hits:
            pl = h.payload or {}
            if "id" not in pl:
                pl["id"] = str(h.id)
            if pl["id"] not in docs:
                docs[pl["id"]] = DocChunk(
                    id=pl["id"],
                    doc_id=pl.get("doc_id"),
                    path=pl.get("path"),
                    page_from=pl.get("page_from"),
                    page_to=pl.get("page_to"),
                    text=pl.get("text", ""),
                    vector=h.vector or [],
                    score=h.score or 0.0,
                    section=pl.get("section"),
                    payload=pl,
                )
    out = list(docs.values())
    logger.info("Collected %d documents before feedback adjustment", len(out))
    for d in out:
        src = d.payload.get("source") if d.payload else None
        d.score = _apply_feedback(d.score, src)
    if RERANK_ENABLED:
        logger.info("Reranking %d documents", len(out))
        out = rerank_combined(queries[0], out)
    logger.info("Fused search returning %d documents", len(out))
    _SEARCH_CACHE[key] = (now, out)
    return out


def _escape(t: str) -> str:
    # экранируем для parse_mode="HTML"
    return html.escape(t or "", quote=False)


def _highlight_terms(text: str, query: str | None) -> str:
    """Экранирует текст и выделяет ключевые слова запроса тегом <b>."""
    safe = _escape(text)
    if not query:
        return safe
    words = [w for w in re.split(r"\s+", query) if w]
    if not words:
        return safe
    escaped_words = [re.escape(_escape(w)) for w in words]
    pattern = re.compile("(" + "|".join(escaped_words) + ")", re.IGNORECASE)
    return pattern.sub(lambda m: f"<b>{m.group(0)}</b>", safe)


def _block_header(pl: Dict) -> str:
    src = _escape(pl.get("source") or "Источник")
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
    pf = pl.get("page_from") or 0
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
            scroll_filter=Filter(
                must=[FieldCondition(key="path", match=MatchValue(value=path))]
            ),
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
    idx = next(
        (i for i, pl in enumerate(all_sorted) if pl.get("id") == center_id), None
    )
    if idx is None:
        center = next((pl for pl in all_sorted if pl.get("id") == center_id), None)
        txt = (center.get("text") if center else "") or ""
        return _concat_limit([txt], SNIPPET_LIMIT)
    lo = max(0, idx - NEIGH_BEFORE)
    hi = min(len(all_sorted), idx + NEIGH_AFTER + 1)
    parts = [pl.get("text") or "" for pl in all_sorted[lo:hi]]
    return _concat_limit(parts, SNIPPET_LIMIT)


def _expand_paragraph(all_payloads: List[Dict], center_id: str) -> str:
    """Берём центр и расширяем до границ абзаца или главы.

    На практике используем простое эвристическое правило: границей
    считаем перевод строки. Если искомый чанк не найден, возвращаем его
    собственный текст.
    """
    if not all_payloads:
        return ""
    ordered = sorted(all_payloads, key=_sort_key)
    texts = [pl.get("text") or "" for pl in ordered]
    idx = next((i for i, pl in enumerate(ordered) if pl.get("id") == center_id), None)
    if idx is None:
        return ordered[0].get("text") or ""
    # соберём полный текст и позиции начала каждого чанка
    offsets: List[int] = []
    cur = 0
    for t in texts:
        offsets.append(cur)
        cur += len(t) + 1  # учтём перевод строки между чанками
    full = "\n".join(texts)
    start_pos = offsets[idx]
    end_pos = start_pos + len(texts[idx])
    para_start = full.rfind("\n", 0, start_pos)
    para_start = 0 if para_start == -1 else para_start + 1
    para_end = full.find("\n", end_pos)
    para_end = len(full) if para_end == -1 else para_end
    return full[para_start:para_end].strip()


async def _summarize_text(text: str) -> str:
    """Получаем короткое введение и алгоритм действий через GigaChat."""
    if not text:
        return ""
    # ограничим размер текста, чтобы не превышать лимиты модели
    cut = text[:12000]
    prompt = (
        "На основе приведённого текста составь короткое информационное введение "
        "и пошаговый алгоритм действий для директора школы. Ничего не придумывай "
        "и не добавляй сведений вне этого текста. Если сделать введение или алгоритм "
        "невозможно, сообщи об этом.\n\n"
        f"Текст:\n{cut}\n\n"
        "Ответ оформи строго в формате:\n"
        "Введение:\n<введение или сообщение об отсутствии>\n\n"
        "Алгоритм действий:\n1. ..."
    )
    return await chat(prompt)


async def summarize(texts: List[str]) -> str:
    """Возвращает краткое резюме списка текстов с помощью LLM."""
    if not texts:
        return ""
    joined = "\n\n".join(t for t in texts if t)
    if not joined:
        return ""
    joined = joined[:12000]
    prompt = (
        "Сделай краткое резюме по следующим фрагментам. Не добавляй сведений вне "
        "этих фрагментов.\n\n" + joined
    )
    summary = await chat(prompt)
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    cut = summary[:SUMMARY_LIMIT]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _expand_chapter(all_payloads: List[Dict], center_id: str) -> str:
    """Расширяем текст до границ условной главы.

    Границей главы считаем двойной перевод строки. Если определить
    границы не удаётся, возвращаем текст параграфа.
    """
    if not all_payloads:
        return ""
    ordered = sorted(all_payloads, key=_sort_key)
    texts = [pl.get("text") or "" for pl in ordered]
    idx = next((i for i, pl in enumerate(ordered) if pl.get("id") == center_id), None)
    if idx is None:
        return ordered[0].get("text") or ""
    offsets: List[int] = []
    cur = 0
    for t in texts:
        offsets.append(cur)
        cur += len(t) + 1
    full = "\n".join(texts)
    start_pos = offsets[idx]
    end_pos = start_pos + len(texts[idx])
    chap_start = full.rfind("\n\n", 0, start_pos)
    chap_start = 0 if chap_start == -1 else chap_start + 2
    chap_end = full.find("\n\n", end_pos)
    chap_end = len(full) if chap_end == -1 else chap_end
    return full[chap_start:chap_end].strip()


def extract_scored(hits: list) -> list[tuple[dict, float | None]]:
    """Удобно вытащить (payload, score) для логирования."""
    out = []
    for h in hits or []:
        out.append((getattr(h, "payload", None) or {}, getattr(h, "score", None)))
    return out


def preview_from_payload(pl: Dict, query: str | None = None) -> str:
    """Возвращает короткий фрагмент текста (до двух предложений) для превью."""
    path = pl.get("path")
    center_id = pl.get("id")
    doc_payloads = _collect_doc_points(path) if path else []
    txt = _neighbors_preview(doc_payloads, str(center_id))
    sentences = re.split(r"(?<=[.!?])\s+", txt.strip())
    preview = " ".join(sentences[:2]).strip()
    return _highlight_terms(preview, query)


async def format_answer_from_payload(pl: Dict) -> Tuple[str, List[Dict], List[str]]:
    """Формирует ответ по данному payload (без поиска)."""
    lines: List[str] = []
    cites: List[Dict] = []
    files_set = set()

    path = pl.get("path")
    center_id = pl.get("id")
    doc_payloads = _collect_doc_points(path) if path else []

    summary = ""
    if path:
        summary_text = ""
        if pl.get("source_group") == "spravochnik":
            # Для справочника берём текст целой главы, посвящённой запросу.
            summary_text = _expand_chapter(doc_payloads, str(center_id))
        else:
            ordered = sorted(doc_payloads, key=_sort_key)
            summary_text = "\n".join(p.get("text") or "" for p in ordered)
        summary_raw = await _summarize_text(summary_text)
        if summary_raw:
            sr = summary_raw.strip()
            lines_sr = sr.splitlines()
            cleaned: List[str] = []
            started = False
            for line in lines_sr:
                low = line.lower().lstrip()
                if not started and (
                    low.startswith("запрос:")
                    or low.startswith("источник:")
                    or low.startswith("общая информация:")
                ):
                    continue
                cleaned.append(line)
                started = True
            sr = "\n".join(cleaned).strip()
            sr = re.sub(r"(?i)введение:\s*", "", sr).strip()
            marker_old = "Алгоритм действий:"
            marker_new = "Примерный план действий:"
            if marker_old in sr:
                before, alg = sr.split(marker_old, 1)
                before_html = _escape(before.strip())
                alg_html = _escape(marker_new + "\n" + alg.strip())
                prefix = f"{before_html}\n\n" if before_html else ""
                summary = f"{prefix}<b>{alg_html}</b>"
            else:
                summary = _escape(sr)

    if summary:
        lines.append(summary)

    cites.append(
        {
            "source": pl.get("source"),
            "page_from": pl.get("page_from"),
            "path": path,
            "text": _neighbors_preview(doc_payloads, str(center_id)),
        }
    )
    if path:
        files_set.add(path)

    msg = "\n".join(lines).strip()
    if not msg:
        msg = "Информация не найдена в локальной базе."
    return msg, cites, list(files_set)


async def retrieve_local(
    question: str, top_k: int = MAX_ITEMS, prefer_spravochnik: bool = True
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

    for h in hits:
        sc = getattr(h, "score", 0.0) or 0.0
        src = (h.payload or {}).get("source")
        h.score = _apply_feedback(sc, src)

    hits = sorted(hits, key=lambda x: getattr(x, "score", 0.0), reverse=True)

    if not hits:
        return (
            "Ничего релевантного не найдено в локальном справочнике.",
            [],
            [],
            {"passed": [], "rejected": []},
        )

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
        return (
            "Ничего релевантного не найдено в локальном справочнике.",
            [],
            [],
            {"passed": [], "rejected": extract_scored(rejected)},
        )

    h = passed[0]
    pl = h.payload or {}
    if "id" not in pl:
        pl["id"] = str(h.id)
    msg, cites, files = await format_answer_from_payload(pl)
    diag = {
        "passed": extract_scored(passed),
        "rejected": extract_scored(rejected),
    }
    return msg, cites, files, diag


async def retrieve_local_hits(
    question: str,
    top_k: int = MAX_ITEMS,
    prefer_spravochnik: bool = True,
    mode: str | None = None,
) -> Tuple[List[Dict], str, Dict[str, list]]:
    """Возвращает список payload'ов релевантных чанков без форматирования."""
    _ = mode  # параметр для совместимости интерфейса
    logger.info(
        "retrieve_local_hits: question='%s', top_k=%d, prefer_spravochnik=%s",
        question,
        top_k,
        prefer_spravochnik,
    )
    logger.info("Generating HYDE queries")
    hyde_q = await generate_query_hyde(question, n=HYDE_N)
    logger.info("HYDE queries: %s", hyde_q)
    logger.info("Running fused search")
    docs = await fused_search([question] + hyde_q, top_k=TOP_K_MAX)
    emb = GigaChatEmbedder()
    logger.info("Embedding original question for MMR")
    qvec = np.array((await emb.embed([question]))[0])
    logger.info("Selecting top %d chunks with MMR", top_k)
    docs_sel = select_chunks_mmr(qvec, docs, k=top_k, lam=MMR_LAMBDA)

    # ensure minimal unique sources
    seen_ids = {d.doc_id for d in docs_sel if d.doc_id}
    if len(seen_ids) < MIN_UNIQUE_SOURCES:
        for d in docs:
            if d.doc_id not in seen_ids:
                docs_sel.append(d)
                seen_ids.add(d.doc_id)
            if len(seen_ids) >= MIN_UNIQUE_SOURCES or len(docs_sel) >= TOP_K_MAX:
                break

    # remove near-duplicates
    unique: List[DocChunk] = []
    for d in docs_sel:
        dv = np.array(d.vector)
        if any(_cosine(dv, np.array(u.vector)) > 0.95 for u in unique):
            continue
        unique.append(d)
    docs_sel = unique

    # dynamic percentile threshold with floor
    if docs_sel:
        scores = np.array([d.score for d in docs_sel])
        perc = float(np.percentile(scores, PERCENTILE_CUT * 100))
        dyn_thr = max(THRESHOLD, perc)
        docs_sel = [d for d in docs_sel if d.score >= dyn_thr]
        if len(docs_sel) < MIN_RESULTS_FLOOR:
            docs_sel = sorted(unique, key=lambda d: d.score, reverse=True)[
                :MIN_RESULTS_FLOOR
            ]
    else:
        dyn_thr = THRESHOLD

    passed_payloads, passed_diag, rejected_diag = [], [], []
    for d in docs_sel:
        sc = d.score
        pl: Dict = d.payload or {}
        passed_payloads.append(pl)
        passed_diag.append((pl, sc))
    for d in unique:
        if d not in docs_sel:
            rejected_diag.append((d.payload or {}, d.score))

    logger.info(
        "Documents passed threshold %.2f: %d/%d",
        dyn_thr,
        len(passed_payloads),
        len(unique),
    )
    logger.info(
        "diag params: top_k_used=%d unique_sources=%d mmr_lambda=%.2f percentile_cut=%.2f neighbor_radius=%d",
        len(passed_payloads),
        len({pl.get("doc_id") for pl in passed_payloads if pl.get("doc_id")}),
        MMR_LAMBDA,
        PERCENTILE_CUT,
        NEIGHBOR_RADIUS,
    )
    logger.info("Summarizing documents")
    summary = await summarize([pl.get("text", "") for pl in passed_payloads])
    diag = {"passed": passed_diag, "rejected": rejected_diag}
    logger.info("retrieve_local_hits finished")
    return passed_payloads, summary, diag


async def retrieve_director_strict(
    query: str,
    k_initial: int = DIRECTOR_TOP_K_INITIAL,
    neighbor_radius: int = DIRECTOR_NEIGHBOR_RADIUS,
) -> List[DocChunk]:
    """Specialised retrieval limited to the director's handbook."""
    _ = neighbor_radius  # reserved for future windowing
    logger.info("retrieve_director_strict: %s", query)
    hyde_q = await generate_query_hyde(query, n=DIRECTOR_HYDE_N)
    emb = GigaChatEmbedder()
    vecs = await emb.embed([query] + hyde_q)
    docs: Dict[str, DocChunk] = {}
    for vec in vecs:
        hits = qsearch(vec, top_k=k_initial * 2, doc_tag="director_handbook")
        for h in hits:
            pl = h.payload or {}
            if pl.get("doc_tag") != "director_handbook":
                continue
            if "id" not in pl:
                pl["id"] = str(h.id)
            if pl["id"] not in docs:
                docs[pl["id"]] = DocChunk(
                    id=pl["id"],
                    doc_id=pl.get("doc_id"),
                    path=pl.get("path"),
                    page_from=pl.get("page_from"),
                    page_to=pl.get("page_to"),
                    text=pl.get("text", ""),
                    vector=h.vector or [],
                    score=h.score or 0.0,
                    section=pl.get("section"),
                    payload=pl,
                )
            else:
                if h.score and h.score > docs[pl["id"]].score:
                    docs[pl["id"]].score = h.score
    all_docs = list(docs.values())
    if not all_docs:
        logger.info("director_handbook: no hits, fallback to default")
        return []
    if RERANK_ENABLED:
        all_docs = rerank_combined(query, all_docs, alpha=DIRECTOR_RERANK_ALPHA)
    qvec = np.array(vecs[0])
    sel = select_chunks_mmr(qvec, all_docs, k=k_initial, lam=DIRECTOR_MMR_LAMBDA)
    seen = {
        d.payload.get("heading_path")
        for d in sel
        if d.payload.get("heading_path")
    }
    if len(seen) < DIRECTOR_MIN_UNIQUE_SECTIONS:
        for d in all_docs:
            sec = d.payload.get("heading_path")
            if sec not in seen:
                sel.append(d)
                seen.add(sec)
            if len(seen) >= DIRECTOR_MIN_UNIQUE_SECTIONS:
                break
    unique: List[DocChunk] = []
    for d in sel:
        dv = np.array(d.vector)
        if any(_cosine(dv, np.array(u.vector)) > 0.95 for u in unique):
            continue
        unique.append(d)
    sel = unique
    if sel:
        scores = np.array([d.score for d in sel])
        perc = float(np.percentile(scores, DIRECTOR_PERCENTILE_CUT * 100))
        sel = [d for d in sel if d.score >= perc]
        if len(sel) < DIRECTOR_MIN_RESULTS_FLOOR:
            sel = sorted(unique, key=lambda d: d.score, reverse=True)[
                :DIRECTOR_MIN_RESULTS_FLOOR
            ]
    filtered = [d for d in sel if DIRECTOR_KEYWORDS_RE.search(d.text)]
    if len(filtered) >= DIRECTOR_MIN_RESULTS_FLOOR:
        sel = filtered
    logger.info(
        "director_handbook retrieval: k_initial=%d unique_sections=%d percent_cut=%.2f mmr_lambda=%.2f",
        k_initial,
        len({d.payload.get('heading_path') for d in sel if d.payload.get('heading_path')}),
        DIRECTOR_PERCENTILE_CUT,
        DIRECTOR_MMR_LAMBDA,
    )
    return sel
