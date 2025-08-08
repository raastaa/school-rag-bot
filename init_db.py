# init_db.py
from sqlalchemy import text
from db import get_engine
from models import Base

DDL_INDEXES = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT,
  page_from INT,
  page_to INT,
  meta JSONB
);

CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id INT REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  embedding vector(1024),
  token_count INT,
  UNIQUE(document_id, content)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
"""

if __name__ == "__main__":
    engine = get_engine()
    with engine.begin() as conn:
        Base.metadata.create_all(bind=conn)
        conn.execute(text(DDL_INDEXES))
    print("DB initialized.")
