import asyncio
import types

import retrieval_local


class DummyEmbed:
    async def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


async def fake_hyde(q, n):
    return []


def make_point(payload, score):
    return types.SimpleNamespace(payload=payload, score=score, vector=[0.1, 0.2])


def test_director_retrieve(monkeypatch):
    monkeypatch.setattr(retrieval_local, "GigaChatEmbedder", lambda: DummyEmbed())
    monkeypatch.setattr(retrieval_local, "generate_query_hyde", fake_hyde)
    monkeypatch.setattr(retrieval_local, "DIRECTOR_MIN_RESULTS_FLOOR", 1)

    def fake_qsearch(vec, top_k=5, doc_tag=None):
        assert doc_tag == "director_handbook"
        return [
            make_point(
                {
                    "id": "1",
                    "text": "Приказ должен иметь дату и номер",
                    "doc_tag": "director_handbook",
                    "heading_path": "A",
                },
                0.9,
            ),
            make_point(
                {
                    "id": "2",
                    "text": "Случайный текст",
                    "doc_tag": "director_handbook",
                    "heading_path": "B",
                },
                0.8,
            ),
        ]

    monkeypatch.setattr(retrieval_local, "qsearch", fake_qsearch)

    res = asyncio.run(
        retrieval_local.retrieve_director_strict("как правильно составить приказ")
    )
    assert res
    assert all("приказ" in r.text.lower() for r in res)
