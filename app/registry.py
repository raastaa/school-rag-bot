from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class DocumentRegistry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL
                )
                """
            )

    def add(self, *, document_id: str, filename: str, stored_path: str, file_hash: str, pages: int, chunks: int, uploaded_at: str | None = None) -> dict:
        uploaded_at = uploaded_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, filename, stored_path, file_hash, pages, chunks, uploaded_at),
            )
        return self.get(document_id)

    def by_hash(self, file_hash: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        return dict(row) if row else None

    def get(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [dict(row) for row in rows]

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
