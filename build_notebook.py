from __future__ import annotations

import ast
import base64
import contextlib
import io
import os
import traceback
from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "notebooks" / "School_Director_RAG_Taras_Metelsky.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# ИИ-помощник директора школы

**Вариант 1:** RAG-система для анализа документов  
**Выполнил:** Метельский Тарас Леонидович

Цель — разработать FastAPI-сервис, который индексирует школьную нормативную и методическую базу, отвечает только по найденному контексту и возвращает документ, страницу и чанк каждого источника.
"""),
    md("""
## 1. Датасет и обоснование выбора

Использованы два официальных тома «Настольной книги директора школы»: «Знание» и «Здоровье». Корпус содержит русскоязычные нормативные выдержки и методические рекомендации, объединяет разные управленческие темы и позволяет проверять найденный ответ по странице. Это практичнее случайных новостей: результат можно использовать как прототип помощника руководителя школы.
"""),
    code("""
import json
import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fastapi.testclient import TestClient

os.environ["EMBEDDING_PROVIDER"] = "hashing"
from app.config import get_settings
from app.llm import SYSTEM_PROMPT
from app.main import create_app
from app.service import KnowledgeBaseService
from embedding_benchmark import MODELS, QUERIES
os.environ["EMBEDDING_PROVIDER"] = "sentence-transformers"
os.environ["EMBEDDING_MODEL"] = "intfloat/multilingual-e5-small"

PROJECT = Path.cwd()
SEED_DIR = PROJECT / "data" / "seed"
seed_files = sorted(SEED_DIR.glob("*.pdf"))
display(pd.DataFrame({
    "Документ": [path.name for path in seed_files],
    "Размер, МБ": [round(path.stat().st_size / 1024**2, 2) for path in seed_files],
}))
"""),
    md("""
## 2. Чанкинг

PDF обрабатывается постранично. Абзацы объединяются до **1000 символов**, overlap — **180 символов**. Такой размер обычно вмещает законченный пункт или несколько связанных предложений, но остаётся достаточно узким для точного поиска. Перекрытие сохраняет смысл на границах фрагментов. В метаданные записываются имя файла, дата загрузки, `document_id`, страница и номер чанка.
"""),
    code("""
from app.document_processing import extract_pages, split_pages

chunk_rows = []
for path in seed_files:
    pages = extract_pages(path)
    chunks = split_pages(pages, chunk_size=1000, overlap=180)
    chunk_rows.append({"Документ": path.name, "Страниц": len(pages), "Чанков": len(chunks)})
chunk_df = pd.DataFrame(chunk_rows)
display(chunk_df)
print("Всего чанков:", int(chunk_df["Чанков"].sum()))
"""),
    md("""
## 3. Сравнение трёх моделей эмбеддингов

Все модели проверены на одних и тех же 547 чанках и 10 вручную размеченных запросах. Релевантность задана парой «документ + допустимая страница».

- `Hit@3` — доля вопросов, для которых релевантный чанк найден в первой тройке;
- `MRR` — среднее обратное значение ранга первого релевантного чанка;
- дополнительно измерены время кодирования корпуса и задержка запроса.

Полный воспроизводимый эксперимент находится в `embedding_benchmark.py`; результаты сохранены в JSON и CSV.
"""),
    code("""
benchmark = json.loads((PROJECT / "embedding_benchmark_results.json").read_text(encoding="utf-8"))
benchmark_df = pd.DataFrame(benchmark["summary"])[[
    "model", "dimensions", "hit_at_3", "mrr", "corpus_encode_s", "chunks_per_s", "mean_query_ms"
]].rename(columns={
    "model": "Модель", "dimensions": "Размерность", "hit_at_3": "Hit@3",
    "mrr": "MRR", "corpus_encode_s": "Кодирование корпуса, с",
    "chunks_per_s": "Чанков/с", "mean_query_ms": "Запрос, мс",
})
display(benchmark_df)
benchmark_df.set_index("Модель")[["Hit@3", "MRR"]].plot(
    kind="bar", figsize=(10, 4), color=["#2563eb", "#0f9d76"]
)
plt.ylim(0, 1)
plt.ylabel("Значение метрики")
plt.title("Качество top-3 поиска на 10 запросах")
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.show()
"""),
    code("""
selected = benchmark["selected_model"]
selected_details = [row for row in benchmark["details"] if row["model"] == selected]
display(pd.DataFrame([{
    "Вопрос": row["question"],
    "Ожидаемые страницы": ", ".join(map(str, row["expected_pages"])),
    "Ранг": row["rank"] if row["rank"] is not None else "не найдено",
    "Top-1": f"{row['top_3'][0]['filename']}, стр. {row['top_3'][0]['page']}",
} for row in selected_details]))
print("Выбранная модель:", selected)
print("Правило выбора:", benchmark["selection_rule"])
"""),
    md("""
## 4. Выбор модели

`all-MiniLM-L6-v2` ориентирована прежде всего на английский язык и на данном русском корпусе показала `Hit@3 = 0.00`. Многоязычная MiniLM достигла `0.70`, а `multilingual-e5-small` — `0.90` при `MRR = 0.85`. Поэтому в RAG-системе используется **`intfloat/multilingual-e5-small`**. Для E5 запросы получают префикс `query:`, документы — `passage:`.

Метрики относятся только к небольшой размеченной выборке и не доказывают юридическую корректность ответов.
"""),
    md("""
## 5. Индексация в ChromaDB выбранной моделью

ChromaDB хранит E5-векторы и метаданные. При загрузке рассчитывается SHA-256, поэтому дубликат отклоняется. Удаление документа очищает реестр, файл и все его векторы.
"""),
    code("""
work_dir = Path(tempfile.mkdtemp(prefix="school_director_notebook_"))
settings = get_settings(work_dir / "data")
service = KnowledgeBaseService(settings)

indexed = []
for path in seed_files:
    started = time.perf_counter()
    item = service.upload_path(path)
    indexed.append({
        "Документ": item["filename"], "Страниц": item["pages"],
        "Чанков": item["chunks"], "Дата загрузки": item["uploaded_at"],
        "Время, с": round(time.perf_counter() - started, 2),
    })
display(pd.DataFrame(indexed))
print("Векторов в ChromaDB:", service.collection.count())
print("Embedding model:", service.embedding.name)
"""),
    md("""
## 6. Поиск релевантных чанков

Вопрос кодируется выбранной E5-моделью. ChromaDB выполняет cosine search, после чего результаты очищаются от дублей и фильтруются по релевантности.
"""),
    code("""
question = "Какие рекомендации даны по организации питания обучающихся?"
results = service.search(question, top_k=5)
display(pd.DataFrame([{
    "Документ": item["filename"], "Страница": item["page"], "Чанк": item["chunk"],
    "Релевантность": item["score"], "Фрагмент": item["text"][:180] + "…",
} for item in results]))

plt.figure(figsize=(9, 3.6))
plt.barh([f"стр. {r['page']} / чанк {r['chunk']}" for r in results][::-1],
         [r["score"] for r in results][::-1], color="#2563eb")
plt.xlabel("Cosine similarity")
plt.title("Пять наиболее релевантных фрагментов")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
"""),
    md("""
## 7. Генерация ответа и источники

LLM подключается через OpenAI-совместимый API или Ollama. Промпт запрещает использовать сведения вне контекста, придумывать статьи и даты, требует различать нормы и рекомендации и честно сообщать о недостатке данных. Без API-ключа работает безопасный extractive-режим.
"""),
    code("print(SYSTEM_PROMPT)"),
    code("""
answer = service.ask(question, top_k=4)
print("Статус:", answer["answer_status"])
print("Провайдер:", answer["llm_provider"])
print("Ответ:", answer["answer"], sep="\\n")
display(pd.DataFrame([{
    "Источник": source["filename"], "Страница": source["page"],
    "Чанк": source["chunk"], "Score": source["score"],
} for source in answer["sources"]]))
"""),
    md("""
## 8. FastAPI и пользовательский интерфейс

Реализованы `GET /`, `POST /upload`, `POST /ask`, `GET /documents`, `DELETE /documents/{id}`. Pydantic проверяет вопрос и `top_k`; предусмотрены ответы 400, 404, 409, 413, 422, 500 и 503. HTML-страница содержит загрузку, список документов, вопрос, ответ и источники.

Endpoint-цикл ниже проверяется на быстром deterministic backend; E5 уже отдельно проверена и использована при индексации основного корпуса выше.
"""),
    code("""
os.environ["EMBEDDING_PROVIDER"] = "hashing"
client = TestClient(create_app(work_dir / "api_data"))
sample_text = (
    "Охрана здоровья обучающихся включает организацию питания, обеспечение безопасности "
    "и определение оптимальной учебной нагрузки. Руководитель образовательной организации "
    "организует контроль условий обучения."
)
checks = []
r = client.get("/"); checks.append(("GET /", r.status_code, "HTML"))
r = client.post("/upload", files={"file": ("demo_rules.txt", sample_text.encode(), "text/plain")})
document_id = r.json()["document_id"]; checks.append(("POST /upload", r.status_code, "indexed"))
r = client.post("/ask", json={"question": "Кто организует контроль условий обучения?", "top_k": 3})
checks.append(("POST /ask", r.status_code, r.json()["answer_status"]))
r = client.get("/documents"); checks.append(("GET /documents", r.status_code, f"{len(r.json())} документ"))
r = client.delete(f"/documents/{document_id}"); checks.append(("DELETE /documents/{id}", r.status_code, r.json()["status"]))
display(pd.DataFrame(checks, columns=["Эндпоинт", "HTTP", "Результат"]))
os.environ["EMBEDDING_PROVIDER"] = "sentence-transformers"
"""),
    md("""
## 9. Итог

Создан работающий RAG-сервис по школьной документальной базе. Три модели эмбеддингов сравнены на одинаковом корпусе; выбрана `multilingual-e5-small`. Документы индексируются в ChromaDB, LLM отвечает по найденным чанкам, а API возвращает проверяемые источники. Все обязательные endpoint-сценарии реализованы.

Ограничения: десять запросов — небольшая оценочная выборка; методическая книга не заменяет актуальную редакцию нормативного акта; сканированные PDF требуют OCR; ответы для практического решения должны проверяться ответственным специалистом.
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def mime_bundle(value):
    if hasattr(value, "to_html"):
        return {"text/html": value.to_html(), "text/plain": repr(value)}
    return {"text/plain": repr(value)}


namespace = {"__name__": "__main__"}
displayed = []


def capture_display(value):
    displayed.append(nbf.v4.new_output("display_data", data=mime_bundle(value), metadata={}))


namespace["display"] = capture_display
os.chdir(PROJECT)
execution_count = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    execution_count += 1
    cell.execution_count = execution_count
    displayed = []
    namespace["display"] = capture_display
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        tree = ast.parse(cell.source)
        last_expression = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = tree.body[:-1] if last_expression else tree.body
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if body:
                exec(compile(ast.Module(body=body, type_ignores=[]), "<notebook>", "exec"), namespace)
            value = eval(compile(ast.Expression(last_expression.value), "<notebook>", "eval"), namespace) if last_expression else None
        outputs = []
        if stdout.getvalue():
            outputs.append(nbf.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
        if stderr.getvalue():
            outputs.append(nbf.v4.new_output("stream", name="stderr", text=stderr.getvalue()))
        outputs.extend(displayed)
        if value is not None:
            outputs.append(nbf.v4.new_output("execute_result", execution_count=execution_count, data=mime_bundle(value), metadata={}))
        plt = namespace.get("plt")
        if plt is not None:
            for figure_number in plt.get_fignums():
                buffer = io.BytesIO()
                plt.figure(figure_number).savefig(buffer, format="png", dpi=140, bbox_inches="tight")
                outputs.append(nbf.v4.new_output(
                    "display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}, metadata={}
                ))
                plt.close(figure_number)
        cell.outputs = outputs
    except Exception as exc:
        cell.outputs = [nbf.v4.new_output(
            "error", ename=type(exc).__name__, evalue=str(exc), traceback=traceback.format_exc().splitlines()
        )]
        raise

nbf.write(notebook, OUTPUT)
print(OUTPUT)
