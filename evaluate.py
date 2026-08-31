from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from app.config import get_settings
from app.service import KnowledgeBaseService


QUESTIONS = [
    {"question": "Что включает единая система оценки качества образования?", "expected": "Znanie"},
    {"question": "Как директору использовать результаты оценки качества образования?", "expected": "Znanie"},
    {"question": "Какие требования задают федеральные государственные образовательные стандарты?", "expected": "Znanie"},
    {"question": "Как в школе организуется внеурочная деятельность?", "expected": "Znanie"},
    {"question": "Какие материалы относятся к Всероссийской олимпиаде школьников?", "expected": "Znanie"},
    {"question": "Для чего используется портал «Единое содержание общего образования»?", "expected": "Znanie"},
    {"question": "Как организовать здоровьесберегающую среду в школе?", "expected": "Zdorove"},
    {"question": "Какие меры помогают сохранять здоровье обучающихся?", "expected": "Zdorove"},
    {"question": "Какие рекомендации даны по организации питания обучающихся?", "expected": "Zdorove"},
    {"question": "Как составлять учебное расписание с учетом здоровья детей?", "expected": "Zdorove"},
    {"question": "Как проводить мониторинг физического здоровья обучающихся?", "expected": "Zdorove"},
    {"question": "Какие меры используются для профилактики травматизма в школе?", "expected": "Zdorove"},
]


def main() -> None:
    project = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="school_rag_eval_") as temp_dir:
        service = KnowledgeBaseService(get_settings(Path(temp_dir) / "data"))
        for path in sorted((project / "data" / "seed").glob("*.pdf")):
            service.upload_path(path)

        rows = []
        reciprocal_ranks = []
        hits = 0
        for item in QUESTIONS:
            started = time.perf_counter()
            found = service.search(item["question"], top_k=3)
            elapsed = time.perf_counter() - started
            rank = next((i for i, result in enumerate(found, 1) if item["expected"] in result["filename"]), None)
            hits += int(rank is not None)
            reciprocal_ranks.append(1 / rank if rank else 0)
            rows.append({
                "question": item["question"],
                "expected": item["expected"],
                "top_document": found[0]["filename"] if found else None,
                "rank": rank,
                "latency_ms": round(elapsed * 1000, 1),
            })

        metrics = {
            "documents": len(service.list_documents()),
            "chunks": service.collection.count(),
            "questions": len(QUESTIONS),
            "hit_at_3": round(hits / len(QUESTIONS), 3),
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1),
            "embedding": service.embedding.name,
            "details": rows,
        }
        output = project / "evaluation_results.json"
        output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
