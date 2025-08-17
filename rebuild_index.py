import argparse
import asyncio
import os
import shutil
from typing import List

from store_qdrant import get_client, QDRANT_COLLECTION, QDRANT_PATH
from ingest.ingest_generic import ingest_path
from db_local import init_db, DB_PATH


async def rebuild(paths: List[str], clear_storage: bool, reset_db: bool) -> None:
    """Recreate search index and optionally local DB."""
    cli = get_client()
    try:
        cli.delete_collection(QDRANT_COLLECTION)
        print(f"Deleted collection {QDRANT_COLLECTION}")
    except Exception as e:
        print(f"Failed to delete collection {QDRANT_COLLECTION}: {e}")

    if clear_storage and os.path.isdir(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH)
        print(f"Removed storage at {QDRANT_PATH}")

    if reset_db:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        print(f"Reinitialized {DB_PATH}")

    for p in paths:
        if os.path.isdir(p):
            print(f"Ingesting {p}...")
            await ingest_path(p)
        else:
            print(f"Skipping {p}: not a directory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Qdrant index and local DB")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["teach", "uploads"],
        help="Directories to ingest (default: teach and uploads)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of existing data",
    )
    parser.add_argument(
        "--keep-storage",
        action="store_true",
        help="Do not delete local Qdrant storage",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Do not recreate app.db",
    )
    args = parser.parse_args()

    if not args.yes:
        parser.error("Use --yes to confirm deletion")

    asyncio.run(
        rebuild(
            args.paths,
            clear_storage=not args.keep_storage,
            reset_db=not args.keep_db,
        )
    )


if __name__ == "__main__":
    main()
