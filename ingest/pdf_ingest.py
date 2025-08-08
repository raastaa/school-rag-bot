# ingest/pdf_ingest.py
"""
Индексатор PDF:
- читает PDF постранично;
- делит текст на чанки с мягкой сегментацией по предложениям и жёстким
  ограничением по токенам (<= EMB_MAX, по умолчанию 514);
- получает эмбеддинги через официальный SDK gigachat;
- пишет векторы и метаданные в локальную файловую БД Qdrant (embedded).

Зависимости:
    pypdf, qdrant-client, gigachat, tiktoken, python-dotenv

Окружение (.env):
    QDRANT_PATH=./qdrant_storage
    QDRANT_COLLECTION=school_docs

    EMBEDDING_MAX_TOKENS=514
    EMBEDDING_TARGET_TOKENS=480
    EMBEDDING_OVERLAP_TOKENS=60
"""

from __future__ import annotations

import os
import uuid
from typing import Optional, List, Tuple, Dict, Any

from pypdf import PdfReader
from dotenv import load_dotenv

from text_utils import (
    split_into_chunks,      # гарантирует чанки <= EMB_MAX
    split_text_hard,        # жёсткое деление по токенам
    count_tokens,
    EMB_MAX,
)

from gigachat_client import GigaChatEmbedder, detect_dim
from store_qdrant import ensure_collection, upsert_chunks

# Для перехвата 413 из официального SDK:
try:
    from gigachat.exceptions import ResponseError  # type: ignore
except Exception:  # на случай другой версии SDK
    class ResponseError(Exception):  # заглушка
        def __init__(self, *args, status_code: int | None = None, **kwargs):
            super().__init__(*args)
            self.status_code = status_code

load_dotenv()

# Сколько строк отправляем в одном запросе к /embeddings
BATCH = int(os.getenv("EMBEDDING_BATCH", "64"))


# ----------------------------- helpers ----------------------------- #

def read_pdf_pages(file_path: str) -> List[Tuple[int, str]]:
    """
    Возвращает список (page_number, text) начиная с 1.
    Не падает на ошибках извлечения текста, подставляя "".
    """
    reader = PdfReader(file_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append((i, page.extract_text() or ""))
        except Exception:
            pages.append((i, ""))
    return pages


def guess_source_group(file_name: str) -> str:
    """
    Простая эвристика для группировки источников.
    Нужна для приоритезации поиска (spravochnik -> zabedu -> web, и т.п.)
    """
    name = (file_name or "").lower()
    if "spravochnik" in name or "справочник" in name or "sprav" in name:
        return "spravochnik"
    if "zabedu" in name:
        return "zabedu"
    return "upload"


# ----------------------------- main api ---------------------------- #

async def ingest_pdf(
    file_path: str,
    title: Optional[str] = None,
    source_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Индексация одного PDF-файла в Qdrant.
    Возвращает краткий отчёт: {'document': ..., 'pages': N, 'chunks': M}.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    # 1) Определяем размерность эмбеддингов и убеждаемся, что коллекция создана
    dim = await detect_dim()                   # например, 1024
    ensure_collection(dim)                     # создаст коллекцию, если нет

    # 2) Читаем PDF и режем на чанки (<= EMB_MAX токенов каждый)
    pages = read_pdf_pages(file_path)
    chunks = split_into_chunks(pages)          # target/overlap берутся из .env (text_utils)

    # 3) Готовим payload’ы и тексты для векторизации
    base_name = os.path.basename(file_path)
    src = source_label or base_name
    group = guess_source_group(src)
    doc_title = title or base_name
    abs_path = os.path.abspath(file_path)

    payloads: List[Dict[str, Any]] = []
    texts: List[str] = []

    for text_c, p_from, p_to in chunks:
        # Доп. страховка (на вход уже должны приходить <= EMB_MAX)
        if count_tokens(text_c) > EMB_MAX:
            for piece in split_text_hard(text_c, EMB_MAX):
                payloads.append({
                    "id": str(uuid.uuid4()),
                    "source": src,
                    "source_group": group,
                    "title": doc_title,
                    "page_from": p_from,
                    "page_to": p_to,
                    "path": abs_path,
                    "type": "pdf",
                    "token_count": count_tokens(piece),
                    # для превью в hits; полный текст для LLM
                    "text": piece[:1000],
                })
                texts.append(piece)
        else:
            payloads.append({
                "id": str(uuid.uuid4()),
                "source": src,
                "source_group": group,
                "title": doc_title,
                "page_from": p_from,
                "page_to": p_to,
                "path": abs_path,
                "type": "pdf",
                "token_count": count_tokens(text_c),
                "text": text_c[:1000],
            })
            texts.append(text_c)

    # 4) Эмбеддинги и запись в Qdrant с обработкой 413
    embedder = GigaChatEmbedder()
    total = 0
    i = 0

    while i < len(texts):
        batch_texts = texts[i:i + BATCH]
        batch_payloads = payloads[i:i + BATCH]
        try:
            vecs = await embedder.embed(batch_texts)
            upsert_chunks(vecs, batch_payloads)
            total += len(batch_texts)
            i += BATCH
        except ResponseError as e:
            # Если SDK отдал 413 — делим «провинившиеся» элементы и пробуем снова.
            status = getattr(e, "status_code", None)
            if status == 413:
                new_texts: List[str] = []
                new_payloads: List[Dict[str, Any]] = []
                changed = False
                for t, pl in zip(batch_texts, batch_payloads):
                    if count_tokens(t) > EMB_MAX:
                        changed = True
                        parts = split_text_hard(t, EMB_MAX)
                        for piece in parts:
                            np = dict(pl)
                            np["id"] = str(uuid.uuid4())
                            np["text"] = piece[:1000]
                            np["token_count"] = count_tokens(piece)
                            new_texts.append(piece)
                            new_payloads.append(np)
                    else:
                        new_texts.append(t)
                        new_payloads.append(pl)
                if not changed:
                    # На всякий случай: если сервер считает иначе, разрежем все элементы батча пополам по токенам
                    really_new_texts: List[str] = []
                    really_new_payloads: List[Dict[str, Any]] = []
                    for t, pl in zip(new_texts, new_payloads):
                        # простая двоичная резка
                        toks = count_tokens(t)
                        if toks > EMB_MAX:
                            # если вдруг остался длиннее — разделим пополам жёстко
                            for piece in split_text_hard(t, EMB_MAX):
                                np = dict(pl)
                                np["id"] = str(uuid.uuid4())
                                np["text"] = piece[:1000]
                                np["token_count"] = count_tokens(piece)
                                really_new_texts.append(piece)
                                really_new_payloads.append(np)
                        else:
                            really_new_texts.append(t)
                            really_new_payloads.append(pl)
                    new_texts, new_payloads = really_new_texts, really_new_payloads

                # Подменяем текущий батч и повторяем (i не двигаем)
                texts[i:i + BATCH] = new_texts
                payloads[i:i + BATCH] = new_payloads
                continue
            # Любая другая ошибка — пробрасываем наверх
            raise

    return {
        "document": doc_title,
        "pages": len(pages),
        "chunks": total,
        "source_group": group,
        "file": abs_path,
    }
