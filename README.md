# ИИ-помощник директора школы

RAG-сервис по нормативной и методической базе образовательной организации. Проект принимает PDF, DOCX, TXT, MD и CSV, разбивает документы на чанки, сохраняет векторы в ChromaDB и отвечает на вопросы с указанием документа, страницы и фрагмента.

## Реализованные требования

- ChromaDB с постоянным хранением;
- сравнение трёх моделей эмбеддингов на 10 размеченных запросах;
- выбранная нейросетевая модель `intfloat/multilingual-e5-small` используется сервисом по умолчанию;
- воспроизводимый локальный embedding-backend для запуска без загрузки тяжёлой модели;
- интеграция OpenAI-совместимого API и Ollama;
- строгий промпт против выдумывания нормативных требований;
- источники в каждом подтверждённом ответе;
- FastAPI, Pydantic, обработка ошибок;
- `GET /`, `POST /upload`, `POST /ask`, `GET /documents`, `DELETE /documents/{id}`;
- минимальный HTML/CSS/JavaScript-интерфейс;
- защита от дубликатов по SHA-256;
- тесты полного API-цикла;
- оценка поиска Hit@3 и MRR.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_seed_data.py
python seed_database.py
uvicorn app.main:app --reload
```

Открыть: `http://127.0.0.1:8000`  
Swagger: `http://127.0.0.1:8000/docs`

При первом запуске автоматически загружается `intfloat/multilingual-e5-small`.

На Windows активация окружения выполняется командой `.venv\\Scripts\\activate`.

## Исследование эмбеддингов

```bash
python embedding_benchmark.py
```

На одинаковом корпусе из 547 чанков сравниваются:

- `sentence-transformers/all-MiniLM-L6-v2`;
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- `intfloat/multilingual-e5-small`.

Оцениваются `Hit@3`, `MRR`, время кодирования корпуса и средняя задержка запроса. Результаты сохраняются в `embedding_benchmark_results.json` и `.csv`. По правилу «максимальный Hit@3, затем MRR, затем минимальная задержка» выбрана `multilingual-e5-small` (`Hit@3 = 0.90`, `MRR = 0.85`).

Для быстрых автоматических тестов без загрузки модели используется отдельный детерминированный hashing-backend. Он не подменяет исследование и рабочую конфигурацию.

## LLM

OpenAI-совместимый API:

```bash
export LLM_PROVIDER=openai
export LLM_API_KEY=...
export LLM_MODEL=gpt-4.1-mini
```

Ollama:

```bash
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=qwen2.5:7b
```

Без ключа работает безопасный extractive-режим: он возвращает выжимку из найденного фрагмента и не добавляет внешние факты.

## Проверка

```bash
python -m pytest -q
python evaluate.py
```

## Состав репозитория

- `app/` — FastAPI, ChromaDB, обработка документов, эмбеддинги и LLM;
- `templates/index.html` — пользовательский интерфейс;
- `embedding_benchmark.py` — сравнение трёх моделей;
- `download_seed_data.py` — загрузка двух официальных PDF с сайта ЕДСОО;
- `notebooks/School_Director_RAG_Taras_Metelsky.ipynb` — выполненная исследовательская часть;
- `tests/` — автоматические тесты API;
- `data/seed/` — демонстрационные документы;
- `Dockerfile` — контейнерный запуск.

Два официальных тома «Настольной книги директора» включены в `data/seed` для воспроизводимой демонстрации. Перед практическим использованием необходимо загрузить действующие редакции нормативных актов и хранить для каждого документа дату редакции и период действия.
