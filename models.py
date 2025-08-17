# models.py
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import Column, Integer, BigInteger, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


@dataclass
class DocChunk:
    """Универсальная структура для чанка документа."""

    id: str
    doc_id: Optional[str] = None
    path: Optional[str] = None
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    text: str = ""
    vector: List[float] = field(default_factory=list)
    score: float = 0.0
    section: Optional[str] = None
    payload: Optional[dict] = None


# Базовый документ (файл/страница/URL)
class Document(Base):  # type: ignore[misc, valid-type]
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(Text, nullable=False)  # путь/URL/источник
    title = Column(Text)
    page_from = Column(Integer)
    page_to = Column(Integer)
    meta = Column(JSONB)

    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


# Текстовые фрагменты + вектор
# !! ВРЕМЕННО Vector(1536). Проверим фактический размер на шаге 2.
class Chunk(Base):  # type: ignore[misc, valid-type]
    __tablename__ = "chunks"
    id = Column(BigInteger, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024))  # при необходимости мигрируем
    token_count = Column(Integer)

    document = relationship("Document", back_populates="chunks")


# Индексы создадим через SQL в init_db.py (ivfflat)
