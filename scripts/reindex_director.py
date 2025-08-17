import os
import argparse
import asyncio
from statistics import mean

from ingest.ingest_generic import ingest_file
from store_qdrant import get_client, QDRANT_COLLECTION
from qdrant_client.models import Filter, FieldCondition, MatchValue


def _ext_ok(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in {
        ".pdf",
        ".docx",
        ".html",
        ".htm",
        ".txt",
        ".pptx",
        ".xlsx",
        ".xlsm",
    }


async def reindex(path: str, drop: bool = False):
    cli = get_client()
    if drop:
        cli.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="doc_tag", match=MatchValue(value="director_handbook"))]
            ),
        )
    total_docs = 0
    total_chunks = 0
    sizes: list[int] = []
    for dirpath, _, filenames in os.walk(path):
        rel = os.path.relpath(dirpath, path)
        heading_path = None if rel in ("", ".") else " > ".join(rel.split(os.sep))
        for name in filenames:
            if not _ext_ok(name):
                continue
            fp = os.path.join(dirpath, name)
            res = await ingest_file(
                fp,
                title=name,
                source_group="spravochnik",
                doc_tag="director_handbook",
                heading_path=heading_path,
            )
            total_docs += 1
            total_chunks += res.get("chunks", 0)
            if res.get("chunks", 0):
                avg = res.get("token_total", 0) / max(res.get("chunks"), 1)
                sizes.append(int(avg))
            print(f"[OK] {name}: chunks={res.get('chunks')}")
    avg_size = int(mean(sizes)) if sizes else 0
    print(
        f"Documents indexed: {total_docs}, chunks: {total_chunks}, average chunk tokens: {avg_size}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    asyncio.run(reindex(args.path, drop=args.drop))
