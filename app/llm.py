from __future__ import annotations

import json
from dataclasses import dataclass

import httpx


SYSTEM_PROMPT = """Ты — информационно-справочный помощник руководителя российской образовательной организации.
Отвечай только на основании предоставленного контекста из загруженной нормативной и методической базы.

Правила:
1. Не добавляй сведения, которых нет в контексте.
2. Не придумывай названия документов, даты, номера статей и обязанности.
3. Если данных недостаточно, прямо сообщи об этом.
4. Отделяй обязательные требования от методических рекомендаций.
5. При противоречии источников укажи на него и не выбирай норму самостоятельно.
6. Ответ должен быть кратким, понятным и проверяемым.
7. Ссылайся на источники в формате [Источник N].
"""


@dataclass
class LLMResult:
    text: str
    provider: str


class LLMClient:
    def __init__(self, provider: str, model: str, base_url: str, api_key: str):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def answer(self, question: str, contexts: list[dict]) -> LLMResult:
        context_text = "\n\n".join(
            f"[Источник {i}] {item['filename']}, стр. {item['page']}, фрагмент {item['chunk']}\n{item['text']}"
            for i, item in enumerate(contexts, 1)
        )
        prompt = f"Вопрос пользователя:\n{question}\n\nКонтекст:\n{context_text}"

        if self.provider in {"openai", "openai-compatible"}:
            if not self.api_key:
                raise RuntimeError("Не задан LLM_API_KEY")
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=60,
            )
            response.raise_for_status()
            return LLMResult(response.json()["choices"][0]["message"]["content"].strip(), self.provider)

        if self.provider == "ollama":
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            return LLMResult(response.json()["message"]["content"].strip(), "ollama")

        # Reproducible no-key mode for tests: it never invents facts and cites the retrieved passage.
        best = contexts[0]
        sentences = [part.strip() for part in best["text"].replace("\n", " ").split(".") if len(part.strip()) > 35]
        excerpt = ". ".join(sentences[:3]).strip()
        if excerpt and not excerpt.endswith("."):
            excerpt += "."
        return LLMResult(
            f"По загруженной базе: {excerpt} [Источник 1]",
            "extractive-fallback",
        )

