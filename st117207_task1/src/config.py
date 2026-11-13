# -*- coding: utf-8 -*-
"""Общие константы и пути проекта st117207_task1 (вариант 13, деградация подшипника)."""
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
N_BOOTSTRAP = 300

# IMS: частота дискретизации 20 кГц, 1 с записи (20480 отсчётов) на канал
FS_HZ = 20000
N_SAMPLES = 20480

# Предварительно заданные широкие полосы спектра, Гц
BANDS_HZ = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 10000)]

# Испытания: 1-е — обучающее, 2-е — полностью отложенное (проверка переноса)
TRAIN_TEST_NAME = "1st_test"
HOLDOUT_TEST_NAME = "2nd_test"

# Длина временного ряда в файле: 1 с, интервал между записями определяется
# по отметкам времени в именах файлов

FEATURES_CSV = PROC_DIR / "features.csv"
RESULTS_JSON = PROC_DIR / "results.json"
