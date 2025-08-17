import pytest
from text_utils import normalize_pdf_text, split_text_hard


def test_normalize_pdf_text():
    raw = "образо-\nвание  "
    assert normalize_pdf_text(raw) == "образование"


def test_split_text_hard():
    text = " ".join(str(i) for i in range(50))
    chunks = split_text_hard(text, 10)
    assert all(len(c.split()) <= 10 for c in chunks)
