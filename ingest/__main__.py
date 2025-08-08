import asyncio, sys
from ingest.pdf_ingest import ingest_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest path/to/file.pdf [title]")
        sys.exit(1)
    path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    res = asyncio.run(ingest_pdf(path, title=title))
    print("Indexed:", res)
