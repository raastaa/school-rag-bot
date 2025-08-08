# web_search.py
import os, httpx
from typing import List, Dict

API_KEY = os.getenv("GOOGLE_API_KEY")
CX_SMP  = os.getenv("GOOGLE_CSE_ID_SMP")  # PSE для сайта (site-only)
CX_WEB  = os.getenv("GOOGLE_CSE_ID_WEB")  # PSE для всего веба

class WebSearchError(Exception): pass

async def _google_cse(query: str, cx: str, num: int) -> List[Dict]:
    if not (API_KEY and cx):
        raise WebSearchError("GOOGLE_API_KEY или cx не заданы")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY, "cx": cx, "q": query,
        "num": min(num, 10), "lr": "lang_ru", "hl": "ru", "safe": "off",
    }
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise WebSearchError(f"CSE {e.response.status_code}: {e.response.text[:300]}") from e
        js = r.json()
    items = js.get("items") or []
    return [{"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")} for it in items]

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
