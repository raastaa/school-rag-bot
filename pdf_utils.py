# pdf_utils.py
from __future__ import annotations
import os
from typing import List, Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.fonts import addMapping
from pypdf import PdfReader, PdfWriter

def _ensure_fonts():
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
        addMapping('DejaVu', 0, 0, 'DejaVu')
    except Exception:
        pass

def _doc(title: str, fname: str):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    _ensure_fonts()
    return SimpleDocTemplate(
        fname, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm
    )

def _styles():
    st = getSampleStyleSheet()
    st['Title'].fontName = 'DejaVu'
    st['Title'].fontSize = 16
    st['Normal'].fontName = 'DejaVu'
    st['Normal'].fontSize = 10
    return st

def build_local_pdf(question: str, cites: List[Dict[str,Any]], out_path: str):
    """
    PDF №1 (локальный): заголовок = вопрос, далее блоки: [источник | страницы] + выдержка.
    """
    doc = _doc(question, out_path)
    st = _styles()
    flow = [Paragraph(f"<b>Вопрос:</b> {question}", st['Title']), Spacer(1, 8)]
    if not cites:
        flow.append(Paragraph("В локальном справочнике релевантных фрагментов не найдено.", st['Normal']))
    else:
        for c in cites:
            head = f"{c.get('source','')}"
            pf, pt = c.get("page_from"), c.get("page_to")
            if pf:
                head += f" (стр. {pf}-{pt})"
            flow.append(Paragraph(f"<b>{head}</b>", st['Normal']))
            preview = (c.get("preview") or c.get("text") or "").replace("\n", "<br/>")
            flow.append(Paragraph(preview, st['Normal']))
            flow.append(Spacer(1, 6))
    doc.build(flow)


def slice_pdf_pages(src_path: str, start: int, end: int) -> str:
    """Сохраняет диапазон страниц [start, end] из PDF в новый файл.

    Возвращает путь к созданному файлу. Страницы нумеруются с 1.
    Если диапазон выходит за пределы, он будет скорректирован.
    Результат сохраняется в папке outputs/snippets/ с именем
    <basename>_<start>_<end>.pdf.
    """
    reader = PdfReader(src_path)
    total = len(reader.pages)
    if total == 0:
        return src_path
    s = max(1, start)
    e = min(total, end)
    writer = PdfWriter()
    for i in range(s - 1, e):
        writer.add_page(reader.pages[i])
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_dir = os.path.join("outputs", "snippets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_{s}_{e}.pdf")
    with open(out_path, "wb") as f:
        writer.write(f)
    return out_path
