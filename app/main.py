from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .config import get_settings
from .models import AnswerResponse, DeleteResponse, DocumentResponse, QuestionRequest, UploadResponse
from .service import DocumentNotFoundError, DuplicateDocumentError, KnowledgeBaseService


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    settings = get_settings(data_dir)
    service = KnowledgeBaseService(settings)
    app = FastAPI(
        title="ИИ-помощник директора школы",
        version="1.0.0",
        description="RAG-сервис по нормативной и методической базе образовательной организации",
    )
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((settings.base_dir / "templates" / "index.html").read_text(encoding="utf-8"))

    @app.post("/upload", response_model=UploadResponse, status_code=201)
    async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
        try:
            data = await file.read()
            document = service.upload_bytes(file.filename or "document", data)
            return UploadResponse(
                document_id=document["id"],
                filename=document["filename"],
                pages=document["pages"],
                chunks=document["chunks"],
            )
        except DuplicateDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Ошибка индексации: {exc}") from exc

    @app.post("/ask", response_model=AnswerResponse)
    async def ask_question(request: QuestionRequest) -> AnswerResponse:
        try:
            return AnswerResponse(**service.ask(request.question, request.top_k))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/documents", response_model=list[DocumentResponse])
    async def list_documents() -> list[DocumentResponse]:
        return [DocumentResponse(**item) for item in service.list_documents()]

    @app.delete("/documents/{document_id}", response_model=DeleteResponse)
    async def delete_document(document_id: str) -> DeleteResponse:
        try:
            service.delete_document(document_id)
            return DeleteResponse(document_id=document_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "documents": len(service.list_documents()),
            "chunks": service.collection.count(),
            "embedding": service.embedding.name,
            "llm": settings.llm_provider,
        }

    return app


app = create_app()

