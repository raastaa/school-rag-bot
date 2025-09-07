import importlib
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("BOT_TOKEN", "test")

sq_stub = types.ModuleType("store_qdrant")
sq_stub.search = lambda *a, **k: []
sq_stub.get_client = lambda: None
sys.modules["store_qdrant"] = sq_stub

qc_stub = types.ModuleType("qdrant_client")
qc_stub.__path__ = []
models_stub = types.ModuleType("qdrant_client.models")
for name in ["Filter", "FieldCondition", "MatchValue"]:
    setattr(models_stub, name, object)
sys.modules["qdrant_client"] = qc_stub
sys.modules["qdrant_client.models"] = models_stub


def test_threshold_env(monkeypatch):
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.5")
    import config
    importlib.reload(config)
    rl = importlib.import_module('retrieval_local')
    importlib.reload(rl)
    assert rl.THRESHOLD == 0.5


def test_sort_key():
    from retrieval_local import _sort_key
    payloads = [
        {"seq": 5, "page_from": 2, "id": "b"},
        {"seq": 1, "page_from": 3, "id": "a"},
    ]
    sorted_payloads = sorted(payloads, key=_sort_key)
    assert [p["seq"] for p in sorted_payloads] == [1, 5]
