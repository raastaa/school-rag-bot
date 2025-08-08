# ingest/ingest_zip.py
from __future__ import annotations
import os, zipfile, argparse, asyncio, shutil
from ingest_generic import ingest_path

def safe_extract(zip_path: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for m in z.infolist():
            # защита от path traversal
            extracted = os.path.normpath(os.path.join(target_dir, m.filename))
            if not extracted.startswith(os.path.abspath(target_dir) + os.sep) and extracted != os.path.abspath(target_dir):
                continue
            z.extract(m, target_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Путь к ZIP-архиву")
    ap.add_argument("--group", default="strategy", help="source_group для Qdrant (по умолчанию strategy)")
    ap.add_argument("--dst", default="uploads/strategy_session", help="Куда распаковывать")
    args = ap.parse_args()

    # чистый каталог под этот архив
    if os.path.exists(args.dst):
        shutil.rmtree(args.dst, ignore_errors=True)
    safe_extract(args.zip, args.dst)
    print(f"Extracted to: {args.dst}")

    res = asyncio.run(ingest_path(args.dst, source_group=args.group))
    print("Done:", res)
