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
