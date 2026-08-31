from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.document_processing import extract_pages, split_pages


PROJECT = Path(__file__).resolve().parent
SEED_DIR = PROJECT / "data" / "seed"
RESULT_JSON = PROJECT / "embedding_benchmark_results.json"
RESULT_CSV = PROJECT / "embedding_benchmark_results.csv"

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
]

# Экспертно размеченные страницы: релевантным считается чанк из указанного
# документа и одной из допустимых страниц. Разметка проверена вручную по тексту.
QUERIES = [
    {
        "question": "Что включает план внеурочной деятельности школы?",
        "document": "01_Kniga_direktora_Znanie.pdf",
        "pages": [25, 49],
    },
    {
        "question": "Как проводится школьный этап Всероссийской олимпиады школьников?",
        "document": "01_Kniga_direktora_Znanie.pdf",
        "pages": [53, 54],
    },
    {
        "question": "Что такое индивидуальный учебный план обучающегося?",
        "document": "01_Kniga_direktora_Znanie.pdf",
        "pages": [18, 24],
    },
    {
        "question": "Как составлять календарный учебный график по триместрам?",
        "document": "01_Kniga_direktora_Znanie.pdf",
        "pages": [25, 26],
    },
    {
        "question": "Из каких разделов состоит федеральная образовательная программа?",
        "document": "01_Kniga_direktora_Znanie.pdf",
        "pages": [10, 11, 12],
    },
    {
        "question": "Какие рекомендации даны по составлению школьного расписания?",
        "document": "02_Kniga_direktora_Zdorove.pdf",
        "pages": [39, 40],
    },
    {
        "question": "Как организовать качественное горячее питание учеников начальной школы?",
        "document": "02_Kniga_direktora_Zdorove.pdf",
        "pages": [35, 36, 37],
    },
    {
        "question": "Как школе выстроить профилактику травматизма обучающихся?",
        "document": "02_Kniga_direktora_Zdorove.pdf",
        "pages": [47, 48],
    },
    {
        "question": "Какие материалы предназначены для обучения работников первой помощи?",
        "document": "02_Kniga_direktora_Zdorove.pdf",
        "pages": [50],
    },
    {
        "question": "Что делать при наружном кровотечении у пострадавшего?",
        "document": "02_Kniga_direktora_Zdorove.pdf",
        "pages": [51, 52, 53],
    },
]


def build_corpus() -> list[dict]:
    corpus: list[dict] = []
    for path in sorted(SEED_DIR.glob("*.pdf")):
        for chunk in split_pages(extract_pages(path), chunk_size=1000, overlap=180):
            corpus.append(
                {
                    "id": f"{path.name}:p{chunk.page}:c{chunk.chunk}",
                    "filename": path.name,
                    "page": chunk.page,
                    "chunk": chunk.chunk,
                    "text": chunk.text,
                }
            )
    return corpus


def is_relevant(item: dict, query: dict) -> bool:
    return item["filename"] == query["document"] and item["page"] in query["pages"]


def benchmark_model(model_name: str, corpus: list[dict]) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    load_s = time.perf_counter() - started
    is_e5 = "e5" in model_name.lower()

    passages = [item["text"] for item in corpus]
    if is_e5:
        passages = [f"passage: {text}" for text in passages]

    started = time.perf_counter()
    corpus_embeddings = model.encode(
        passages,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    corpus_encode_s = time.perf_counter() - started

    details: list[dict] = []
    query_times: list[float] = []
    for query in QUERIES:
        text = f"query: {query['question']}" if is_e5 else query["question"]
        started = time.perf_counter()
        query_embedding = model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0].astype(np.float32)
        scores = corpus_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:3]
        query_ms = (time.perf_counter() - started) * 1000
        query_times.append(query_ms)
        top = [corpus[int(index)] | {"score": float(scores[int(index)])} for index in top_indices]
        rank = next((rank for rank, item in enumerate(top, 1) if is_relevant(item, query)), None)
        details.append(
            {
                "model": model_name,
                "question": query["question"],
                "expected_document": query["document"],
                "expected_pages": query["pages"],
                "rank": rank,
                "top_3": [
                    {
                        "filename": item["filename"],
                        "page": item["page"],
                        "chunk": item["chunk"],
                        "score": round(item["score"], 4),
                    }
                    for item in top
                ],
            }
        )

    ranks = [item["rank"] for item in details]
    summary = {
        "model": model_name,
        "dimensions": int(corpus_embeddings.shape[1]),
        "chunks": len(corpus),
        "queries": len(QUERIES),
        "hit_at_3": round(sum(rank is not None for rank in ranks) / len(ranks), 4),
        "mrr": round(sum(1 / rank if rank else 0 for rank in ranks) / len(ranks), 4),
        "load_s": round(load_s, 3),
        "corpus_encode_s": round(corpus_encode_s, 3),
        "chunks_per_s": round(len(corpus) / corpus_encode_s, 2),
        "mean_query_ms": round(float(np.mean(query_times)), 2),
        "p95_query_ms": round(float(np.percentile(query_times, 95)), 2),
    }
    del model, corpus_embeddings
    gc.collect()
    return summary, details


def main() -> None:
    corpus = build_corpus()
    summaries: list[dict] = []
    details: list[dict] = []
    for model_name in MODELS:
        print(f"Benchmark: {model_name}", flush=True)
        summary, model_details = benchmark_model(model_name, corpus)
        summaries.append(summary)
        details.extend(model_details)
        print(summary, flush=True)

    winner = sorted(
        summaries,
        key=lambda item: (-item["hit_at_3"], -item["mrr"], item["mean_query_ms"]),
    )[0]["model"]
    payload = {
        "chunk_size": 1000,
        "chunk_overlap": 180,
        "corpus_documents": len(list(SEED_DIR.glob("*.pdf"))),
        "corpus_chunks": len(corpus),
        "evaluation_queries": len(QUERIES),
        "selection_rule": "max Hit@3, затем max MRR, затем min mean_query_ms",
        "selected_model": winner,
        "summary": summaries,
        "details": details,
    }
    RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(RESULT_CSV, index=False)
    print(f"Selected: {winner}")
    print(RESULT_JSON)


if __name__ == "__main__":
    main()
