# retrieve_site_live.py
from typing import Tuple, List, Dict
from web_search import search_google_site


async def retrieve_site_live(
    question: str, max_results: int = 5, mode: str | None = None
) -> Tuple[str, List[Dict]]:
    """
    Возвращает человекочитаемый текст и список результатов:
    [{title, link, snippet}]
    """
    _ = mode  # параметр для совместимости интерфейса
    results = await search_google_site(question, num=max_results)
    if not results:
        return "На сайте smp.edu.ru релевантных результатов не найдено.", []

    lines = []
    for i, it in enumerate(results, 1):
        t = it.get("title") or it["link"]
        s = (it.get("snippet") or "").strip()
        lines.append(f"{i}. <a href='{it['link']}'>{t}</a>\n{s}")
    return "\n\n".join(lines), results
