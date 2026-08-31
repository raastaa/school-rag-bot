from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    uploads_dir: Path
    chroma_dir: Path
    registry_path: Path
    collection_name: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    chunk_size: int
    chunk_overlap: int
    top_k_candidates: int
    max_context_chunks: int
    min_relevance: float
    max_file_size_mb: int


def get_settings(data_dir: str | Path | None = None) -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    resolved_data = Path(data_dir) if data_dir else Path(os.getenv("DATA_DIR", base_dir / "data"))
    return Settings(
        base_dir=base_dir,
        data_dir=resolved_data,
        uploads_dir=resolved_data / "uploads",
        chroma_dir=resolved_data / "chroma",
        registry_path=resolved_data / "registry.sqlite",
        collection_name=os.getenv("CHROMA_COLLECTION", "school_director_documents"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        llm_provider=os.getenv("LLM_PROVIDER", "extractive"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
        top_k_candidates=int(os.getenv("TOP_K_CANDIDATES", "8")),
        max_context_chunks=int(os.getenv("MAX_CONTEXT_CHUNKS", "4")),
        min_relevance=float(os.getenv("MIN_RELEVANCE", "0.05")),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "30")),
    )
