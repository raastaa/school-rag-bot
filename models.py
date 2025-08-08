# models.py
from sqlalchemy import (
    Column, Integer, BigInteger, Text, JSON, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

# Базовый документ (файл/страница/URL)
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(Text, nullable=False)     # путь/URL/источник
    title = Column(Text)
    page_from = Column(Integer)
    page_to = Column(Integer)
    meta = Column(JSONB)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

# Текстовые фрагменты + вектор
# !! ВРЕМЕННО Vector(1536). Проверим фактический размер на шаге 2.
class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(BigInteger, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024))          # при необходимости мигрируем
    token_count = Column(Integer)

    document = relationship("Document", back_populates="chunks")

# Индексы создадим через SQL в init_db.py (ivfflat)
