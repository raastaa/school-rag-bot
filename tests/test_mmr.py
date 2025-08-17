import numpy as np
from retrieval_local import Doc, select_chunks_mmr


def test_mmr_diversity():
    qv = np.array([1.0, 0.0])
    docs = [
        Doc(id="1", text="a", vector=[1, 0], score=0.9, payload={}),
        Doc(id="2", text="b", vector=[0.99, 0.01], score=0.8, payload={}),
        Doc(id="3", text="c", vector=[0, 1], score=0.7, payload={}),
    ]
    selected = select_chunks_mmr(qv, docs, k=2, lam=0.3)
    ids = {d.id for d in selected}
    assert ids == {"1", "3"}
