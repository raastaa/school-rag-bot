import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("BOT_TOKEN", "test")

gigachat_stub = types.ModuleType("gigachat")
gigachat_stub.GigaChat = object
sys.modules.setdefault("gigachat", gigachat_stub)

sq_stub = types.ModuleType("store_qdrant")
def _dummy(*args, **kwargs):
    return None
sq_stub.ensure_collection = _dummy
sq_stub.upsert_chunks = _dummy
sys.modules.setdefault("store_qdrant", sq_stub)

from ingest.pdf_ingest import guess_source_group
from text_utils import split_into_chunks, count_tokens, EMB_MAX


def test_guess_source_group():
    assert guess_source_group("spravochnik_file.pdf") == "spravochnik"
    assert guess_source_group("zabedu_info.pdf") == "zabedu"
    assert guess_source_group("other.pdf") == "upload"


def test_chunk_splitting():
    pages = [(1, "слово " * 300)]
    chunks = split_into_chunks(pages, max_tokens=50, overlap=0)
    assert len(chunks) > 1
    assert all(count_tokens(text) <= EMB_MAX for text, _, _ in chunks)
