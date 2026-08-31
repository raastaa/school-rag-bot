from pathlib import Path

from app.config import get_settings
from app.service import DuplicateDocumentError, KnowledgeBaseService


def main() -> None:
    project = Path(__file__).resolve().parent
    service = KnowledgeBaseService(get_settings())
    for path in sorted((project / "data" / "seed").glob("*")):
        if not path.is_file():
            continue
        try:
            item = service.upload_path(path)
            print(f"Indexed: {item['filename']} — {item['chunks']} chunks")
        except DuplicateDocumentError:
            print(f"Skipped duplicate: {path.name}")


if __name__ == "__main__":
    main()

