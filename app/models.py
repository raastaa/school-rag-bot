from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class SourceItem(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk: int
    score: float
    text: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    answer_status: str
    llm_provider: str


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_hash: str
    pages: int
    chunks: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str = "indexed"


class DeleteResponse(BaseModel):
    document_id: str
    status: str = "deleted"

