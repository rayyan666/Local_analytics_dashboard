# backend/reports/report_generator.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import io, base64

def make_pdf(title: str, dataframe_records: list, chart_png_bytes: bytes, out_path: str):
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles['Title']), Spacer(1,12)]
    if chart_png_bytes:
        img_buf = io.BytesIO(chart_png_bytes)
        story.append(Image(img_buf, width=400, height=300))
        story.append(Spacer(1,12))
    if dataframe_records:
        cols = list(dataframe_records[0].keys())
        data = [cols] + [[str(r.get(c,"")) for c in cols] for r in dataframe_records[:20]]
        t = Table(data)
        story.append(t)
    doc.build(story)
    return out_path

def decode_base64_png(b64: str) -> bytes:
    return base64.b64decode(b64)
