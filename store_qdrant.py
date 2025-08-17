# store_qdrant.py
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from dotenv import load_dotenv

load_dotenv()
# store_qdrant.py (вверху)
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION", "school_docs"
)  # <-- убедитесь, что эта строка есть на модуле уровне


_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _client = QdrantClient(path=QDRANT_PATH)  # embedded
    return _client


def ensure_collection(dim: int, distance: Distance = Distance.COSINE):
    cli = get_client()
    if QDRANT_COLLECTION not in [c.name for c in cli.get_collections().collections]:
        cli.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=distance),
        )


def upsert_chunks(vectors: List[List[float]], payloads: List[Dict[str, Any]]):
    cli = get_client()
    points = [
        PointStruct(id=payloads[i].get("id"), vector=vectors[i], payload=payloads[i])
        for i in range(len(vectors))
    ]
    cli.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search(
    query_vector: List[float], top_k: int = 5, source_filter: Optional[str] = None
):
    cli = get_client()
    flt = None
    if source_filter:
        flt = Filter(
            must=[
                FieldCondition(
                    key="source_group", match=MatchValue(value=source_filter)
                )
            ]
        )
    res = cli.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=flt,
        with_vectors=True,
    )
    return res  # list[ScoredPoint] -> .payload / .score
