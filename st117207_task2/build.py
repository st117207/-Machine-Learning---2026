# -*- coding: utf-8 -*-
"""Сборка проекта одной командой: python build.py

Этапы:
1. Загрузка и распаковка исходных данных UCI Statlog Shuttle (если ещё
   не распакованы): shuttle.trn, shuttle.tst.
2. Сборка единой таблицы data/processed/shuttle.csv и аудит данных.
3. Моделирование: подбор гиперпараметров, эксперименты, таблицы, рисунки,
   макросы и data/processed/results.json.
4. Компиляция отчёта (pdflatex, три прохода); итоговый report/report.pdf
   копируется в корень проекта — report.pdf рядом с README для сдачи.

Зависимости описаны в requirements.txt; для распаковки shuttle.trn.Z
требуется 7-Zip (7z.exe в PATH или C:\\Program Files\\7-Zip\\7z.exe).
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
REPORT_PDF = ROOT / "report.pdf"                     # итоговый файл для сдачи

AUX = [".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".blg", ".bbl"]


def _run(cmd: list[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"Команда завершилась с кодом {r.returncode}: {' '.join(cmd)}")


def clean() -> None:
    """Удаление производных файлов LaTeX и результатов моделирования."""
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
    print("Производные файлы удалены.")


def main() -> None:
    sys.path.insert(0, str(SRC))
    from data import build_processed
    import modeling

    print("=== Этап 1/3: исходные данные UCI Statlog Shuttle")
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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    elif len(sys.argv) > 1 and sys.argv[1] == "tables":
        sys.path.insert(0, str(SRC))
        import modeling
        modeling.main(only_tables=True)
    else:
        main()
