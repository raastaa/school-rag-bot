# web_search.py
import time
import httpx
from typing import List, Dict, Tuple
import logging

from config import settings

API_KEY = settings.GOOGLE_API_KEY
CX_SMP  = settings.GOOGLE_CSE_ID_SMP  # PSE для сайта (site-only)
CX_WEB  = settings.GOOGLE_CSE_ID_WEB  # PSE для всего веба

logger = logging.getLogger(__name__)

_CACHE: dict[Tuple[str, str, int], Tuple[float, List[Dict]]] = {}
CACHE_TTL = settings.WEB_SEARCH_CACHE_TTL

class WebSearchError(Exception): pass

async def _google_cse(query: str, cx: str, num: int) -> List[Dict]:
    if not (API_KEY and cx):
        raise WebSearchError("GOOGLE_API_KEY или cx не заданы")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY, "cx": cx, "q": query,
        "num": min(num, 10), "lr": "lang_ru", "hl": "ru", "safe": "off",
    }
    key = (query, cx, num)
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < CACHE_TTL:
        logger.debug("web search cache hit for %s", key)
        return _CACHE[key][1]
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise WebSearchError(f"CSE {e.response.status_code}: {e.response.text[:300]}") from e
        js = r.json()
    items = js.get("items") or []
    result = [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]
    _CACHE[key] = (now, result)
    return result

async def search_google_site(query: str, num: int = 5) -> List[Dict]:
    """Сначала PSE для сайта; если пусто — fallback: веб-cx + site:."""
    # 1) основной запрос в site-cx
    items = await _google_cse(query, CX_SMP, num)
    if items:
        return items
    # 2) fallback: используем веб-cx и принудительный оператор site:
    q2 = f"site:smp.edu.ru/kniga-direktora20 {query}"
    return await _google_cse(q2, CX_WEB, num)

async def search_google_web(query: str, num: int = 5) -> List[Dict]:
    """Общий веб."""
    return await _google_cse(query, CX_WEB, num)
