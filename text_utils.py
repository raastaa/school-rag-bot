from typing import List, Tuple
import re
import tiktoken

from config import settings

EMB_MAX = settings.EMBEDDING_MAX_TOKENS
EMB_TARGET = settings.EMBEDDING_TARGET_TOKENS
EMB_OVERLAP = settings.EMBEDDING_OVERLAP_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")
_HYPHENS = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]" 


def normalize_pdf_text(raw: str) -> str:
    if not raw:
        return ""

    t = raw

    # 0) вычистить спец-символ мягкого переноса и BOM
    t = t.replace("\u00AD", "")   # soft hyphen
    t = t.replace("\ufeff", "")

    # 1) склейка слов, разорванных переносом строки + дефисом:
    #    образо-\nвание  → образование
    t = re.sub(rf"(\w){_HYPHENS}\s*\n\s*(\w)", r"\1\2", t, flags=re.UNICODE)

    # 2) склейка слов, разорванных дефисом + большим количеством пробелов (иногда PDF даёт пробел вместо \n)
    #    образо-   вание → образование  (только если похоже на перенос)
    t = re.sub(rf"(\w){_HYPHENS}\s{{1,5}}(\w)", r"\1\2", t, flags=re.UNICODE)

    # 3) удалить лидирующие «—»/«-» в начале строк как артефакт переносов:
    #    \n- слово  → \nслово
    t = re.sub(rf"(?m)^\s*{_HYPHENS}\s+", "", t)

    # 4) схлопнуть лишние пробелы, но сохраним перенос абзацев как одиночный \n
    #    сначала нормализуем многострочные пробелы:
    t = re.sub(r"[ \t]+\n", "\n", t)     # пробелы перед \n
    t = re.sub(r"\n[ \t]+", "\n", t)     # пробелы после \n
    #    двойные/тройные \n → один \n
    t = re.sub(r"\n{2,}", "\n", t)
    #    внутри строк — схлопываем в один пробел
    t = re.sub(r"[ \t]{2,}", " ", t)

    # 5) финально — обрезка краёв
    return t.strip()

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def _split_sentences(text: str) -> List[str]:
    # простая сегментация по предложениям; переносы строк уже нормализованы
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [p for p in parts if p]

def _truncate_tokens(tokens: List[int], max_tokens: int) -> List[int]:
    return tokens[:max_tokens] if len(tokens) > max_tokens else tokens

def split_text_hard(text: str, max_tokens: int) -> List[str]:
    toks = _enc.encode(re.sub(r"\s+", " ", text).strip())
    out, i = [], 0
    while i < len(toks):
        j = min(i + max_tokens, len(toks))
        out.append(_enc.decode(toks[i:j]))
        i = j
    return out

def split_into_chunks(pages: List[Tuple[int, str]],
                      max_tokens: int = EMB_TARGET,
                      overlap: int = EMB_OVERLAP) -> List[Tuple[str, int, int]]:
    """
    Возвращает чанки (text, page_from, page_to) и ГАРАНТИРУЕТ:
    - каждый чанк ≤ EMB_MAX токенов;
    - корректные номера страниц с учётом переходов между страницами.
    """
    chunks: List[Tuple[str, int, int]] = []
    buf_tokens: List[int] = []
    buf_pages: List[int] = []  # собираем ВСЕ страницы, через которые прошёл чанк

    def flush():
        nonlocal buf_tokens, buf_pages
        if not buf_tokens:
            return
        p_from = min(buf_pages) if buf_pages else None
        p_to   = max(buf_pages) if buf_pages else None

        if len(buf_tokens) > EMB_MAX:
            # жёстко режем и помечаем тот же диапазон страниц для всех частей
            for piece in split_text_hard(_enc.decode(buf_tokens), EMB_MAX):
                chunks.append((piece, p_from, p_to))
            buf_tokens, buf_pages = [], []
            return

        text = _enc.decode(buf_tokens)
        chunks.append((text, p_from, p_to))

        # overlap по токенам + последний номер страницы сохраняем
        if overlap > 0 and len(buf_tokens) > overlap:
            keep = buf_tokens[-overlap:]
            last_page = p_to
            buf_tokens = keep
            buf_pages  = [last_page] if last_page is not None else []
        else:
            buf_tokens, buf_pages = [], []

    for pnum, raw in pages:
        t = normalize_pdf_text(raw)
        if not t:
            continue

        for sent in _split_sentences(t):
            toks = _enc.encode(sent)

            # если одно предложение само по себе > EMB_MAX — разрежем и добавим
            if len(toks) > EMB_MAX:
                for piece in split_text_hard(sent, EMB_MAX):
                    ptoks = _enc.encode(piece)
                    if not buf_tokens:
                        buf_pages = [pnum]
                    # добавляем страницу ЭТОГО фрагмента
                    if not buf_pages or buf_pages[-1] != pnum:
                        buf_pages.append(pnum)
                    buf_tokens.extend(ptoks)
                    if len(buf_tokens) >= max_tokens:
                        flush()
                continue

            if not buf_tokens:
                buf_pages = [pnum]
            # обязательно фиксируем переход страницы
            if not buf_pages or buf_pages[-1] != pnum:
                buf_pages.append(pnum)

            # если добавление предложения переполнит «целевой» размер — сбрасываем
            if len(buf_tokens) + len(toks) > max_tokens:
                flush()
                if not buf_tokens:
                    buf_pages = [pnum]

            buf_tokens.extend(toks)

            # страховка: не перелезть лимит GigaChat
            if len(buf_tokens) >= EMB_MAX:
                flush()

    flush()

    # финальная страховка
    safe: List[Tuple[str, int, int]] = []
    for text, pf, pt in chunks:
        if count_tokens(text) > EMB_MAX:
            for piece in split_text_hard(text, EMB_MAX):
                safe.append((piece, pf, pt))
        else:
            safe.append((text, pf, pt))
    return safe
