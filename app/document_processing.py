from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class TextChunk:
    page: int
    chunk: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    if suffix == ".pdf":
        pdf = fitz.open(path)
        pages = [PageText(i + 1, normalize_text(page.get_text("text"))) for i, page in enumerate(pdf)]
        pdf.close()
        return [page for page in pages if page.text]

    if suffix == ".docx":
        doc = Document(path)
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [PageText(1, normalize_text(text))]

    if suffix == ".csv":
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        rows = csv.reader(io.StringIO(raw))
        text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        return [PageText(1, normalize_text(text))]

    return [PageText(1, normalize_text(path.read_text(encoding="utf-8-sig", errors="replace")))]


def _paragraphs(text: str) -> list[str]:
    raw = [item.strip() for item in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[А-ЯA-Z0-9])", text)]
    return [item for item in raw if item]


def split_pages(pages: list[PageText], chunk_size: int = 1000, overlap: int = 180) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in pages:
        units = _paragraphs(page.text)
        current = ""
        page_chunks: list[str] = []
        for unit in units:
            if len(unit) > chunk_size:
                if current:
                    page_chunks.append(current.strip())
                    current = ""
                step = max(1, chunk_size - overlap)
                page_chunks.extend(unit[start : start + chunk_size].strip() for start in range(0, len(unit), step))
                continue
            candidate = f"{current} {unit}".strip()
            if current and len(candidate) > chunk_size:
                page_chunks.append(current.strip())
                tail = current[-overlap:] if overlap else ""
                current = f"{tail} {unit}".strip()
            else:
                current = candidate
        if current:
            page_chunks.append(current.strip())
        chunks.extend(TextChunk(page.page, index, text) for index, text in enumerate(page_chunks, 1) if len(text) >= 40)
    return chunks

