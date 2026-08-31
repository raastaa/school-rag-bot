from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingProvider(Protocol):
    name: str

    def encode_passages(self, texts: list[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbeddingProvider:
    """Deterministic offline embeddings for reproducible tests and demos."""

    name = "char-ngram-hashing-2048"

    def __init__(self, n_features: int = 2048):
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.astype(np.float32).toarray().tolist()

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


class SentenceTransformerEmbeddingProvider:
    name: str

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Для нейросетевых эмбеддингов установите зависимости из requirements-ml.txt"
            ) from exc
        self.name = model_name
        self.uses_e5_prefix = "e5" in model_name.lower()
        self.model = SentenceTransformer(model_name)

    def _prepare(self, texts: list[str], kind: str) -> list[str]:
        if not self.uses_e5_prefix:
            return texts
        return [f"{kind}: {text}" for text in texts]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode(
            self._prepare(texts, "passage"), normalize_embeddings=True
        )
        return values.astype(np.float32).tolist()

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        values = self.model.encode(
            self._prepare(texts, "query"), normalize_embeddings=True
        )
        return values.astype(np.float32).tolist()


def create_embedding_provider(provider: str, model_name: str) -> EmbeddingProvider:
    if provider.lower() in {"e5", "sentence-transformers", "sentence_transformers"}:
        return SentenceTransformerEmbeddingProvider(model_name)
    return HashingEmbeddingProvider()
