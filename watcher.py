import os
import asyncio
import hashlib
from typing import List

from db_local import upsert_file_index
from ingest.ingest_generic import ingest_file


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for n in filenames:
            yield os.path.join(dirpath, n)


async def start_watcher(paths: List[str], interval_sec: int = 60):
    while True:
        for p in paths:
            if not os.path.isdir(p):
                continue
            for file in iter_files(p):
                try:
                    size = os.path.getsize(file)
                    mtime = os.path.getmtime(file)
                    sha = sha256_of_file(file)
                    upsert_file_index(file, size, mtime, sha)
                    await ingest_file(file, title=os.path.basename(file))
                except Exception:
                    continue
        await asyncio.sleep(interval_sec)
