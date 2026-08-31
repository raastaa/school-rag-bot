from pathlib import Path

import httpx


SEED_DIR = Path(__file__).resolve().parent / "data" / "seed"
FILES = {
    "01_Kniga_direktora_Znanie.pdf": "https://edsoo.ru/wp-content/uploads/2023/12/301-1432-01-znanie.pdf",
    "02_Kniga_direktora_Zdorove.pdf": "https://edsoo.ru/wp-content/uploads/2023/12/41-0708-01-zdorove.pdf",
}


def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for filename, url in FILES.items():
            target = SEED_DIR / filename
            if target.exists() and target.stat().st_size > 100_000:
                print(f"Уже загружен: {filename}")
                continue
            print(f"Загрузка: {filename}")
            response = client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)
            print(f"Сохранён: {target} ({target.stat().st_size / 1024**2:.1f} МБ)")


if __name__ == "__main__":
    main()
