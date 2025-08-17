# retrieve_web_live.py
from typing import Tuple, List, Dict
from web_search import search_google_web

async def retrieve_web_live(question: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
    results = await search_google_web(question, num=max_results)
    if not results:
        return "В интернете релевантных результатов не найдено.", []
    lines = []
    for i, it in enumerate(results, 1):
        t = it.get("title") or it["link"]
        s = (it.get("snippet") or "").strip()
        lines.append(f"{i}. <a href='{it['link']}'>{t}</a>\n{s}")
    return "\n\n".join(lines), results
