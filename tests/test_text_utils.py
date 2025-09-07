import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("BOT_TOKEN", "test")

from text_utils import normalize_pdf_text, count_tokens

def test_normalize_pdf_text():
    raw = "обра\u00ADзо-\nвание  \n\n"
    assert normalize_pdf_text(raw) == "образование"

def test_count_tokens():
    text = "hello world"
    assert isinstance(count_tokens(text), int)
