import re
from typing import Literal

_KEYWORDS = [
    "приказ",
    "распоряжение",
    "постановление",
    "регистрация приказов",
    "бланк приказа",
    "структура приказа",
    "приказываю",
    "основание приказа",
    "контроль за исполнением",
]


def route_source(query: str) -> Literal["director_handbook", "default"]:
    """Classify a user query and decide which knowledge source to use.

    If the query clearly refers to director handbook topics ("приказ" etc.)
    return ``"director_handbook"`` otherwise ``"default"``.
    The check is keyword based (no LLM calls).
    """

    q = (query or "").lower()
    if "справочник директора" in q:
        return "director_handbook"
    for kw in _KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q):
            return "director_handbook"
    return "default"


__all__ = ["route_source"]
