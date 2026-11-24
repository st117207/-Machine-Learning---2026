# -*- coding: utf-8 -*-
"""Общие константы и пути проекта st117207_task3 (задача 3, вариант 13).

Задача: снижение размерности и кластеризация многоканальных измерений
гидравлического испытательного стенда (набор UCI Condition Monitoring of
Hydraulic Systems).

Из семи фиксированных каналов PS1, PS2, PS3, FS1, TS1, EPS1, VS1 по каждому
из 2 205 циклов вычисляются четыре характеристики (среднее, СКО, минимум,
максимум). Итоговая матрица — 2 205 x 28. Метки состояний узлов из profile.txt
в несупервизорные признаки не входят и используются только для последующей
проверки состава найденных групп.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHANNEL_DIR = RAW_DIR / "channels"      # исходные сигналы UCI (не входят в ZIP)
PROC_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"

# Воспроизводимость
RANDOM_STATE = 42

# Ограничения вычислительной нагрузки из условий задачи
PCA_MAX_ROWS = 10000        # для общей PCA
FA_MAX_ROWS = 2000          # для FactorAnalysis
FA_MAX_FEATURES = 60        # исходных признаков у нас 28 < 60
CLUSTER_MAX_OBJ = 1200      # одна подвыборка для трёх алгоритмов
TSNE_MAX_OBJ = 1000         # для t-SNE/Isomap (в работе t-SNE не используется)

# Разбиение: хронологическое 70/30 по номерам циклов. Дополнительно —
# блочное разбиение по 50 последовательных циклов (проверка устойчивости
# выводов к смещению режимов во времени).
TRAIN_FRAC = 0.70
BLOCK_SIZE = 50

# Анализ
N_PCA_MAX = 10              # число главных компонент для спектра и ошибки
CLUSTER_PCS = 4             # число PCA-координат для кластеризации (2..8)
K_RANGE = [2, 3, 4, 5, 6]   # числа групп для KMeans и Ward
FA_COMPONENTS = [2, 3, 5]   # рассматриваемые числа факторов
DBSCAN_MIN_SAMPLES = 10     # 2 * размерность (4) с округлением вверх
DBSCAN_EPS_GRID = [round(0.2 + 0.1 * i, 1) for i in range(15)]  # 0.2..1.6
N_STABILITY = 20            # число бутстрэп-повторов для устойчивости

# Каналы: имя, физическая величина, единица, частота дискретизации (Гц),
# группа для графиков нагрузок.
CHANNELS = [
    ("PS1", "Давление", "бар", 100, "Давление"),
    ("PS2", "Давление", "бар", 100, "Давление"),
    ("PS3", "Давление", "бар", 100, "Давление"),
    ("FS1", "Объёмный расход", "л/мин", 10, "Расход"),
    ("TS1", "Температура", "°C", 1, "Температура"),
    ("EPS1", "Мощность двигателя", "Вт", 100, "Мощность"),
    ("VS1", "Вибрация", "мм/с", 1, "Вибрация"),
]
CHANNEL_NAMES = [c[0] for c in CHANNELS]
CHANNEL_UNITS = {c[0]: c[2] for c in CHANNELS}
CHANNEL_GROUP = {c[0]: c[4] for c in CHANNELS}
CHANNEL_DESC = {c[0]: c[1] for c in CHANNELS}
CHANNEL_RATE = {c[0]: c[3] for c in CHANNELS}

STATS = [("mean", "среднее"), ("std", "СКО"),
         ("min", "минимум"), ("max", "максимум")]

FEATURES = [f"{ch}_{st}" for ch in CHANNEL_NAMES for st, _ in STATS]
FEATURE_GROUP = {f"{ch}_{st}": CHANNEL_GROUP[ch]
                 for ch in CHANNEL_NAMES for st, _ in STATS}

# Метки состояний узлов (profile.txt). В признаки не входят.
LABELS = ["cooler", "valve", "pump", "accum", "stable"]
LABEL_NAMES = {
    "cooler": "Состояние охладителя, %",
    "valve": "Состояние клапана, %",
    "pump": "Внутренняя утечка насоса",
    "accum": "Давление гидроаккумулятора, бар",
    "stable": "Флаг установившегося режима",
}
COOLER_NAMES = {3: "близко к отказу", 20: "пониженная эффективность",
                100: "полная эффективность"}
VALVE_NAMES = {73: "близко к отказу", 80: "сильное запаздывание",
               90: "слабое запаздывание", 100: "оптимальное переключение"}
PUMP_NAMES = {0: "нет утечки", 1: "слабая утечка", 2: "сильная утечка"}
ACCUM_NAMES = {90: "близко к отказу", 100: "сильно понижено",
               115: "слегка понижено", 130: "оптимальное давление"}
STABLE_NAMES = {0: "режим установился", 1: "режим не установился"}

# Файлы
AGGREGATES_CSV = RAW_DIR / "hydraulic_aggregates.csv"
PROFILE_FILE = RAW_DIR / "profile.txt"
EXAMPLE_CYCLE = RAW_DIR / "example_cycle.npz"   # один цикл для рисунка сигналов
PROCESSED_CSV = PROC_DIR / "hydraulic.csv"
RESULTS_JSON = PROC_DIR / "results.json"

# Официальный архив UCI (проверен SHA-256 в src/download.py)
UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/447/"
               "condition+monitoring+of+hydraulic+systems.zip")
UCI_ZIP_SHA256 = ("24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13"
                  "bc69079af1f2")
