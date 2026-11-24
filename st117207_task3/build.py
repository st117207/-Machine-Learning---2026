# -*- coding: utf-8 -*-
"""Сборка проекта одной командой: python build.py

Этапы:
1. Данные: если в data/raw/channels распакованы исходные каналы UCI, из них
   пересчитываются 28 циклопризнаков; иначе используется проверенная таблица
   data/raw/hydraulic_aggregates.csv, входящая в архив. Дополнительно
   собирается единая таблица data/processed/hydraulic.csv.
2. Моделирование: аудит, PCA, факторный анализ, кластеризация, проверка по
   меткам, таблицы, рисунки, макросы и data/processed/results.json.
3. Компиляция отчёта (pdflatex, три прохода); итоговый report/report.pdf
   копируется в корень проекта — report.pdf рядом с README для сдачи.

Режимы:
    python build.py             полная сборка
    python build.py tables      только таблицы и макросы из results.json
    python build.py download    скачать и распаковать исходные каналы UCI
    python build.py features    пересчитать агрегаты из исходных каналов
    python build.py clean       удалить производные файлы

Зависимости описаны в requirements.txt; для компиляции нужен pdflatex
(MiKTeX или TeX Live). Полный пересчёт из сырых сигналов требует
`python build.py download` (≈325 МБ распакованных каналов).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REPORT = ROOT / "report" / "report.tex"
REPORT_PDF_INTERNAL = ROOT / "report" / "report.pdf"  # результат компиляции
REPORT_PDF = ROOT / "report.pdf"                      # итоговый файл для сдачи

AUX = [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".blg", ".bbl"]


def _run(cmd: list[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"Команда завершилась с кодом {r.returncode}: {' '.join(cmd)}")


def clean() -> None:
    """Удаление производных файлов LaTeX, таблиц, рисунков и результатов."""
    report_dir = ROOT / "report"
    for f in list((report_dir / "figures").glob("*.pdf")) + \
             list((report_dir / "tables").glob("*.tex")):
        f.unlink()
    for suf in AUX:
        f = report_dir / f"report{suf}"
        if f.exists():
            f.unlink()
    for pdf in (REPORT_PDF_INTERNAL, REPORT_PDF):
        if pdf.exists():
            pdf.unlink()
    processed = ROOT / "data" / "processed"
    for name in ("hydraulic.csv", "results.json"):
        f = processed / name
        if f.exists():
            f.unlink()
    print("Производные файлы удалены.")


def main() -> None:
    sys.path.insert(0, str(SRC))
    from data import build_processed
    import modeling

    print("=== Этап 1/3: данные и аудит")
    build_processed()

    print("=== Этап 2/3: моделирование, таблицы и рисунки")
    modeling.main()

    print("=== Этап 3/3: компиляция отчёта")
    if not shutil.which("pdflatex"):
        sys.exit("pdflatex не найден: установите MiKTeX или TeX Live.")
    for _ in range(3):
        _run(["pdflatex", "-interaction=nonstopmode",
              "-halt-on-error", "report.tex"], REPORT.parent)
    if not REPORT_PDF_INTERNAL.exists():
        sys.exit("report/report.pdf не создан.")
    shutil.copyfile(REPORT_PDF_INTERNAL, REPORT_PDF)
    print(f"Готово: {REPORT_PDF}")


def download() -> None:
    sys.path.insert(0, str(SRC))
    import download as dl
    dl.download()


def features() -> None:
    sys.path.insert(0, str(SRC))
    from data import build_processed
    build_processed()
    print("Агрегаты и таблица data/processed/hydraulic.csv пересчитаны.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "clean":
        clean()
    elif arg == "tables":
        sys.path.insert(0, str(SRC))
        import modeling
        modeling.main(only_tables=True)
    elif arg == "download":
        download()
    elif arg == "features":
        features()
    else:
        main()
