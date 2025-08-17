# gigachat_client.py
"""
Адаптер под официальный SDK `gigachat`:
- GigaChatEmbedder.embed(texts) -> List[List[float]]
- detect_dim() -> int

Требуемые переменные окружения (.env):
  GIGACHAT_CREDENTIALS=...             # строка Authorization Key
  GIGACHAT_SCOPE=GIGACHAT_API_PERS     # или GIGACHAT_API_B2B/CORP
  GIGACHAT_EMBEDDINGS_MODEL=Embeddings # либо EmbeddingsGigaR
  GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1  (опционально)
  GIGACHAT_VERIFY_SSL=true|false
  GIGACHAT_CA_BUNDLE=/path/to/chain.pem  (опционально)
"""

import os
import asyncio
from typing import List, Sequence, Any, Literal
from dotenv import load_dotenv

# официальный SDK
from gigachat import GigaChat
from gigachat.exceptions import ResponseError

load_dotenv()


class RateLimitError(RuntimeError):
    """Превышен лимит запросов к GigaChat."""


CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
EMBEDDINGS_MODEL = os.getenv("GIGACHAT_EMBEDDINGS_MODEL", "Embeddings")
BASE_URL = os.getenv("GIGACHAT_BASE_URL", "").rstrip("/") or None

# TLS
VERIFY_SSL_ENV = os.getenv("GIGACHAT_VERIFY_SSL", "true").strip().lower()
CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE", "").strip() or None
# verify_ssl_certs: bool; если нужен кастомный корневой, используем переменные окружения certifi/SSL,
# но сам SDK принимает только bool, поэтому оставляем системный truststore + возможность отключить проверку.
VERIFY_SSL_CERTS: bool = VERIFY_SSL_ENV in ("1", "true", "yes")


def _new_client() -> GigaChat:
    # По документации SDK: credentials (str), scope (optional),
    # model (optional), base_url (optional), verify_ssl_certs (optional).
    # Никаких api_url/auth_url/verify тут нет. :contentReference[oaicite:2]{index=2}
    kwargs: dict[str, Any] = {
        "credentials": CREDENTIALS,
        "scope": SCOPE,
        "verify_ssl_certs": VERIFY_SSL_CERTS,
    }
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    # Если требуется кастомный CA, положите его в системное хранилище
    # или задайте переменные окружения REQUESTS_CA_BUNDLE/SSL_CERT_FILE.
    # SDK не принимает путь к CA напрямую (только bool).
    return GigaChat(**kwargs)


class GigaChatEmbedder:
    def __init__(self):
        if not CREDENTIALS:
            raise RuntimeError("Не задан GIGACHAT_CREDENTIALS в .env")
        self._client = _new_client()

    def _embed_sync(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        # SDK-метод embeddings принимает texts=..., model=...
        try:
            res = self._client.embeddings(texts=list(texts), model=EMBEDDINGS_MODEL)
        except ResponseError as e:  # noqa: PERF203
            if len(e.args) > 1 and e.args[1] == 429:
                raise RateLimitError("Too Many Requests") from e
            raise
        # Ответ SDK может быть объектом с .data (элементы с .embedding)
        data = getattr(res, "data", None) or (
            res.get("data") if isinstance(res, dict) else None
        )
        if not data:
            raise RuntimeError("Неожиданный формат ответа от GigaChat.embeddings()")
        out: List[List[float]] = []
        for item in data:
            vec = getattr(item, "embedding", None)
            if vec is None and isinstance(item, dict):
                vec = item.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError("Элемент ответа не содержит 'embedding'")
            out.append(vec)
        return out

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)


def _probe_dim_sync() -> int:
    cli = GigaChatEmbedder()
    vecs = cli._embed_sync(["dim_probe"])
    if not vecs or not vecs[0]:
        raise RuntimeError("Не удалось получить эмбеддинг для определения размерности.")
    return len(vecs[0])


async def detect_dim() -> int:
    return await asyncio.to_thread(_probe_dim_sync)


_LLM_SEM = asyncio.Semaphore(int(os.getenv("LLM_CONCURRENCY", "3")))


async def chat(prompt: str, timeout: int = 30) -> str:
    """Простой вызов chat-комплитов GigaChat с ретраями."""
    if not CREDENTIALS:
        return ""
    async with _LLM_SEM:
        for attempt in range(3):
            cli = _new_client()
            try:
                resp = await asyncio.wait_for(cli.achat(prompt), timeout=timeout)
                choices = getattr(resp, "choices", None) or []
                if not choices:
                    return ""
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", "") if msg else ""
                return content.strip()
            except ResponseError as e:  # noqa: PERF203
                if len(e.args) > 1 and e.args[1] == 429:
                    raise RateLimitError("Too Many Requests") from e
                if attempt == 2:
                    return ""
                await asyncio.sleep(2**attempt)
            except Exception:
                if attempt == 2:
                    return ""
                await asyncio.sleep(2**attempt)
            finally:
                try:
                    await cli.aclose()
                except Exception:
                    pass

        return ""


async def self_check_sufficiency(
    query: str, snippets: list[str]
) -> Literal["sufficient", "insufficient"]:
    prompt = (
        "Ты ассистент валидации. Дано: вопрос пользователя и 1-3 фрагмента из базы. "
        "Определи, достаточно ли фрагментов, чтобы ответить точно. "
        "Ответь одним словом: sufficient или insufficient.\n\n"
        f"Вопрос: {query}\nФрагменты:\n- " + "\n- ".join(snippets[:3])
    )
    resp = await chat(prompt, timeout=15)
    r = resp.strip().lower()
    return "insufficient" if "insufficient" in r else "sufficient"


async def generate_query_hyde(query: str, n: int = 2) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        prompt = (
            "Сформулируй краткий перефраз вопроса для улучшения поиска.\n"
            f"Вопрос: {query}"
        )
        txt = await chat(prompt, timeout=15)
        if txt:
            out.append(txt.strip())
    return out


__all__ = [
    "GigaChatEmbedder",
    "detect_dim",
    "chat",
    "self_check_sufficiency",
    "generate_query_hyde",
    "RateLimitError",
]
