# School RAG Bot

Utility script `rebuild_index.py` allows rebuilding the local search index.
It removes the existing Qdrant collection, optionally clears the local
`qdrant_storage` directory and recreates `app.db`, then re-ingests files.

## Usage

```bash
python rebuild_index.py [paths ...] --yes [--keep-storage] [--keep-db]
```

- `paths` – directories to index (defaults: `teach` and `uploads`).
- `--yes` – confirm destructive actions (required).
- `--keep-storage` – keep existing Qdrant files.
- `--keep-db` – do not recreate `app.db`.

