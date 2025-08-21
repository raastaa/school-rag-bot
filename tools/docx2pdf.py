#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Конвертирует все .docx из указанной папки (по умолчанию: ./teach) в PDF.
Требуется установленный LibreOffice (soffice).

Примеры:
  python tools/docx2pdf.py              # конвертирует ./teach/*.docx
  python tools/docx2pdf.py --src ./teach --out ./teach/pdf
  python tools/docx2pdf.py --recursive  # ищет .docx во вложенных папках
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

def find_soffice() -> Path | None:
    # ищем soffice в PATH
    p = shutil.which("soffice")
    return Path(p) if p else None

def convert_batch(soffice_path: Path, files: list[Path], outdir: Path) -> int:
    """
    Вызывает LibreOffice один раз на партию файлов.
    Возвращает код возврата процесса (0 — ок).
    """
    if not files:
        return 0
    # LibreOffice понимает множественные аргументы: soffice --headless --convert-to pdf --outdir OUT file1 file2 ...
    cmd = [str(soffice_path), "--headless", "--convert-to", "pdf", "--outdir", str(outdir)]
    cmd += [str(f) for f in files]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write("LibreOffice error:\n")
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
    else:
        sys.stdout.write(proc.stdout)
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description="DOCX → PDF batch converter (LibreOffice headless)")
    parser.add_argument("--src", default="./teach", help="папка с .docx (по умолчанию ./teach)")
    parser.add_argument("--out", default=None, help="папка для PDF (по умолчанию = src)")
    parser.add_argument("--recursive", action="store_true", help="искать .docx рекурсивно")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not src.exists() or not src.is_dir():
        print(f"Папка-источник не найдена: {src}", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.out).resolve() if args.out else src
    outdir.mkdir(parents=True, exist_ok=True)

    soffice = find_soffice()
    if not soffice:
        print("Не найден 'soffice'. Установите LibreOffice:\n  sudo apt-get install -y libreoffice", file=sys.stderr)
        sys.exit(3)

    # Собираем список файлов
    pattern = "**/*.docx" if args.recursive else "*.docx"
    files = [p for p in src.glob(pattern) if p.is_file() and not p.name.startswith("~$")]
    if not files:
        print("DOCX-файлы не найдены.")
        sys.exit(0)

    # Разбиваем на батчи (LibreOffice нормально переварит и сотню, но на больших объёмах можно дробить)
    BATCH = 100
    total = 0
    for i in range(0, len(files), BATCH):
        batch = files[i:i+BATCH]
        print(f"Конвертация {i+1}-{i+len(batch)} из {len(files)}…")
        rc = convert_batch(soffice, batch, outdir)
        if rc != 0:
            print(f"Ошибка конвертации (код {rc}) на батче {i//BATCH + 1}", file=sys.stderr)
            # продолжаем, чтобы сконвертировать остальные
        total += len(batch)

    print(f"Готово. Обработано файлов: {total}. PDF в: {outdir}")
    sys.exit(0)

if __name__ == "__main__":
    main()
