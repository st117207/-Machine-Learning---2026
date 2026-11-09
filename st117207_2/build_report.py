"""
Воспроизводимый скрипт сборки отчёта.

Берёт REPORT.md и собирает из него report.pdf (Markdown -> HTML -> PDF).
Не требует внешних бинарников кроме вложенного pandoc (tools/pandoc/pandoc-3.13/pandoc.exe)
и Python-пакета xhtml2pdf.

Запуск:
    python build_report.py
"""

from pathlib import Path
import subprocess
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
MD_FILE = ROOT / "REPORT.md"
HTML_FILE = ROOT / "report.html"
PDF_FILE = ROOT / "report.pdf"
PANDOC = ROOT / "tools" / "pandoc" / "pandoc-3.13" / "pandoc.exe"

if not PANDOC.exists():
    print(f"ОШИБКА: pandoc.exe не найден по пути {PANDOC}")
    print("Установите pandoc одним из способов:")
    print("  winget install JohnMacFarlane.Pandoc")
    print("  или скачайте с https://github.com/jgm/pandoc/releases")
    print("и положите executable в tools/pandoc/pandoc-3.13/")
    raise SystemExit(1)

CSS = """
@font-face {
    font-family: 'ArialCYR';
    src: url('fonts/arial.ttf');
    font-weight: normal;
    font-style: normal;
}
@font-face {
    font-family: 'ArialCYR';
    src: url('fonts/arialbd.ttf');
    font-weight: bold;
    font-style: normal;
}
@font-face {
    font-family: 'ArialCYR';
    src: url('fonts/ariali.ttf');
    font-weight: normal;
    font-style: italic;
}
* {
    font-family: 'ArialCYR', Arial, sans-serif !important;
}
@page {
    size: A4;
    margin: 2cm 1.8cm 2cm 1.8cm;
    @bottom-center {
        content: "str" counter(page) " / " counter(pages);
        font-family: 'ArialCYR', sans-serif;
        font-size: 8pt;
        color: #666;
    }
}
body {
    font-family: 'ArialCYR', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1, h2, h3, h4 {
    font-family: 'ArialCYR', Arial, sans-serif;
    color: #8b0000;
    margin-top: 1.2em;
}
h1 { font-size: 18pt; border-bottom: 2px solid #8b0000; padding-bottom: 4px; }
h2 { font-size: 14pt; border-bottom: 1px solid #8b0000; padding-bottom: 3px; }
h3 { font-size: 12pt; color: #a00000; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 8.5pt;
}
th, td {
    border: 1px solid #999;
    padding: 4px 6px;
    text-align: left;
    vertical-align: top;
}
th { background: #f2f2f2; }
img {
    max-width: 100%;
    height: auto;
    margin: 0.5em 0;
}
pre, code {
    font-family: 'ArialCYR', monospace;
    font-size: 8.5pt;
    background: #f6f6f6;
    padding: 2px 4px;
    border-radius: 3px;
}
ul, ol { margin: 0.5em 0 0.5em 1.5em; }
"""


def build():
    # Шаг 1: Markdown -> HTML через pandoc
    subprocess.run(
        [str(PANDOC), str(MD_FILE), "-t", "html", "-s", "-o", str(HTML_FILE)],
        check=True,
    )

    html_text = HTML_FILE.read_text(encoding="utf-8")

    # Исправляем символы, которых нет в выбранном шрифте
    html_text = html_text.replace("\u2080", "0")

    # Встраиваем CSS перед </head>
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", f"<style>{CSS}</style></head>")
    else:
        html_text = f"<head><style>{CSS}</style></head>{html_text}"

    HTML_FILE.write_text(html_text, encoding="utf-8")

    # Шаг 2: HTML -> PDF через xhtml2pdf
    with open(PDF_FILE, "wb") as pdf_out:
        pisa_status = pisa.CreatePDF(
            src=html_text,
            dest=pdf_out,
            encoding="utf-8",
            path=str(ROOT),
        )

    if pisa_status.err:
        raise RuntimeError(f"xhtml2pdf errors: {pisa_status.err}")

    print(f"[OK] {PDF_FILE} ({PDF_FILE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
