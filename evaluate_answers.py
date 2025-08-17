import argparse
import json
import sys
from pathlib import Path

# requests не имеет типовых определений
import requests  # type: ignore[import-untyped]


def evaluate(test_file: Path, api_url: str) -> float:
    try:
        data = json.loads(test_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Не удалось прочитать файл {test_file}: {e}", file=sys.stderr)
        return 0.0

    if not isinstance(data, list):
        print("Ожидается список тестов.", file=sys.stderr)
        return 0.0

    total = len(data)
    success = 0

    for idx, item in enumerate(data, 1):
        question = item.get("question", "")
        keywords = item.get("expected_keywords", [])

        try:
            resp = requests.post(api_url, json={"question": question}, timeout=30)
            resp.raise_for_status()
            answer = resp.json().get("answer", "")
        except Exception as e:
            print(f"[{idx}] Не удалось получить ответ: {e}")
            continue

        answer_lower = answer.lower()
        missing = [kw for kw in keywords if kw.lower() not in answer_lower]
        if missing:
            print(f"[{idx}] В ответе отсутствуют ключевые слова: {', '.join(missing)}")
        else:
            success += 1
            print(f"[{idx}] ОК")

    ratio = success / total if total else 0.0
    print(f"Успешных ответов: {success}/{total} ({ratio:.0%})")
    return ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка качества ответов бота")
    parser.add_argument(
        "test_file",
        type=Path,
        help="JSON-файл с вопросами и ключевыми словами",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/ask",
        help="URL эндпоинта, к которому отправляются вопросы",
    )
    args = parser.parse_args()
    evaluate(args.test_file, args.api_url)
