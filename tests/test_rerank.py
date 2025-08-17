from retrieval_local import Doc, rerank_combined


def test_rerank_prefers_bm25_when_alpha_zero():
    docs = [
        Doc(id="1", text="alpha beta", vector=[0, 0], score=0.1, payload={}),
        Doc(id="2", text="gamma", vector=[0, 0], score=0.1, payload={}),
        Doc(id="3", text="delta", vector=[0, 0], score=0.1, payload={}),
    ]
    ranked = rerank_combined("gamma", docs, alpha=0.0)
    assert ranked[0].id == "2"
