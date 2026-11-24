# -*- coding: utf-8 -*-
"""Общие константы и пути проекта st117207_task2 (задача 2, вариант 18).

Задача: классификация состояний радиаторной системы космического аппарата
Shuttle по девяти анонимным числовым показаниям бортовых датчиков
(набор UCI Statlog Shuttle).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

# Воспроизводимость
RANDOM_STATE = 42
N_SPLITS = 5

# Признаки и цель. Имена A..I взяты из описания StatLog: физический смысл
# девяти показаний не раскрыт; первый признак в описании назван «время»,
# но порядок строк в выгрузке перемешан, поэтому время как порядок утрачено.
FEATURES = list("ABCDEFGHI")
TARGET = "y"
CLASSES = [1, 2, 3, 4, 5, 6, 7]
CLASS_NAMES = {
    1: "Rad Flow",
    2: "Fpv Close",
    3: "Fpv Open",
    4: "High",
    5: "Bypass",
    6: "Bpv Close",
    7: "Bpv Open",
}

# Файлы исходных данных (разбиение авторов набора: 43500 / 14500)
TRAIN_FILE = RAW_DIR / "shuttle.trn"
TEST_FILE = RAW_DIR / "shuttle.tst"

PROCESSED_CSV = PROC_DIR / "shuttle.csv"
RESULTS_JSON = PROC_DIR / "results.json"

# Официальный архив UCI (на случай, если data/raw пуст)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip"

# Доля обучающей части, откладываемая внутри обучения для методических
# экспериментов с деревьями (не тестовая часть авторов!)
VAL_SIZE = 0.2

# Подвыборка обучающей части для подбора гиперпараметров SVC. На этом наборе
# ядровой SVM обучается за секунды, поэтому 0 означает «использовать всю
# обучающую часть»; параметр оставлен для воспроизводимости на слабых машинах.
SVC_TUNE_N = 0
