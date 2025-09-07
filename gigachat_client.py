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

import asyncio
from typing import List, Sequence, Any, Optional
import logging

from config import settings

# официальный SDK
from gigachat import GigaChat

logger = logging.getLogger(__name__)

CREDENTIALS = settings.GIGACHAT_CREDENTIALS
SCOPE = settings.GIGACHAT_SCOPE
EMBEDDINGS_MODEL = settings.GIGACHAT_EMBEDDINGS_MODEL
BASE_URL = settings.GIGACHAT_BASE_URL.rstrip("/") or None if settings.GIGACHAT_BASE_URL else None

VERIFY_SSL_CERTS: bool = settings.GIGACHAT_VERIFY_SSL.lower() in ("1", "true", "yes")

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
        res = self._client.embeddings(texts=list(texts), model=EMBEDDINGS_MODEL)
        # Ответ SDK может быть объектом с .data (элементы с .embedding)
        data = getattr(res, "data", None) or (res.get("data") if isinstance(res, dict) else None)
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

async def chat(prompt: str) -> str:
    """Простой вызов chat-комплитов GigaChat.

    Возвращает текст первого сообщения или пустую строку при ошибке.
    """
    if not CREDENTIALS:
        return ""
    cli = _new_client()
    try:
        resp = await cli.achat(prompt)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", "") if msg else ""
        return content.strip()
    except Exception as e:
        logger.exception("GigaChat chat error: %s", e)
        return ""
    finally:
        try:
            await cli.aclose()
        except Exception:
            pass

__all__ = ["GigaChatEmbedder", "detect_dim", "chat"]
