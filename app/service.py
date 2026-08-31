from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from .config import Settings
from .document_processing import SUPPORTED_EXTENSIONS, extract_pages, split_pages
from .embeddings import create_embedding_provider
from .llm import LLMClient
from .registry import DocumentRegistry


class DuplicateDocumentError(ValueError):
    pass


class DocumentNotFoundError(LookupError):
    pass


class KnowledgeBaseService:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.registry = DocumentRegistry(settings.registry_path)
        self.embedding = create_embedding_provider(settings.embedding_provider, settings.embedding_model)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.llm = LLMClient(
            settings.llm_provider,
            settings.llm_model,
            settings.llm_base_url,
            settings.llm_api_key,
        )

    def upload_bytes(self, filename: str, data: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Поддерживаются форматы: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        if not data:
            raise ValueError("Файл пуст")
        if len(data) > self.settings.max_file_size_mb * 1024 * 1024:
            raise OverflowError(f"Максимальный размер файла — {self.settings.max_file_size_mb} МБ")

        file_hash = hashlib.sha256(data).hexdigest()
        if self.registry.by_hash(file_hash):
            raise DuplicateDocumentError("Документ уже загружен")

        document_id = str(uuid.uuid4())
        uploaded_at = datetime.now(timezone.utc).isoformat()
        safe_name = Path(filename).name.replace("\x00", "")
        stored_path = self.settings.uploads_dir / f"{document_id}_{safe_name}"
        stored_path.write_bytes(data)

        try:
            pages = extract_pages(stored_path)
            if not pages or sum(len(page.text) for page in pages) < 50:
                raise ValueError("Не удалось извлечь текст. Возможно, документ требует OCR")
            chunks = split_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
            if not chunks:
                raise ValueError("После разбиения документа не получено текстовых фрагментов")

            ids = [f"{document_id}_p{item.page}_c{item.chunk}" for item in chunks]
            documents = [item.text for item in chunks]
            metadatas = [
                {
                    "document_id": document_id,
                    "filename": safe_name,
                    "page": item.page,
                    "chunk": item.chunk,
                    "uploaded_at": uploaded_at,
                }
                for item in chunks
            ]
            embeddings = self.embedding.encode_passages(documents)
            batch_size = 200
            for start in range(0, len(chunks), batch_size):
                stop = start + batch_size
                self.collection.add(
                    ids=ids[start:stop],
                    documents=documents[start:stop],
                    metadatas=metadatas[start:stop],
                    embeddings=embeddings[start:stop],
                )
            return self.registry.add(
                document_id=document_id,
                filename=safe_name,
                stored_path=str(stored_path),
                file_hash=file_hash,
                pages=len(pages),
                chunks=len(chunks),
                uploaded_at=uploaded_at,
            )
        except Exception:
            stored_path.unlink(missing_ok=True)
            self.collection.delete(where={"document_id": document_id})
            raise

    def upload_path(self, path: Path) -> dict:
        return self.upload_bytes(path.name, path.read_bytes())

    def list_documents(self) -> list[dict]:
        return self.registry.list()

    def delete_document(self, document_id: str) -> None:
        document = self.registry.get(document_id)
        if not document:
            raise DocumentNotFoundError("Документ не найден")
        self.collection.delete(where={"document_id": document_id})
        Path(document["stored_path"]).unlink(missing_ok=True)
        self.registry.delete(document_id)

    def search(self, question: str, top_k: int | None = None) -> list[dict]:
        if not self.registry.list():
            return []
        count = min(top_k or self.settings.top_k_candidates, self.collection.count())
        query_embedding = self.embedding.encode_queries([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, count),
            include=["documents", "metadatas", "distances"],
        )
        found: list[dict] = []
        seen: set[tuple] = set()
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            key = (metadata["document_id"], metadata["page"], text[:120])
            score = max(0.0, 1.0 - float(distance))
            if key in seen or score < self.settings.min_relevance:
                continue
            seen.add(key)
            found.append({**metadata, "text": text, "score": round(score, 4)})
        return found

    def ask(self, question: str, top_k: int = 4) -> dict:
        contexts = self.search(question, max(top_k, self.settings.top_k_candidates))[:top_k]
        if not contexts:
            return {
                "answer": "В загруженной базе недостаточно информации для точного ответа.",
                "sources": [],
                "answer_status": "insufficient_context",
                "llm_provider": self.settings.llm_provider,
            }
        llm_result = self.llm.answer(question, contexts)
        return {
            "answer": llm_result.text,
            "sources": contexts,
            "answer_status": "grounded",
            "llm_provider": llm_result.provider,
        }

    def reset(self) -> None:
        for document in self.registry.list():
            Path(document["stored_path"]).unlink(missing_ok=True)
        if self.settings.data_dir.exists():
            shutil.rmtree(self.settings.data_dir)
