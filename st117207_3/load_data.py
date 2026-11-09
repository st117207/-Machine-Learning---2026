"""
Загрузка и подготовка данных UCI Condition Monitoring of Hydraulic Systems
для Задачи 3, вариант 13 (снижение размерности и кластеризация).

Источник: https://zenodo.org/records/1323611 (UCI Condition Monitoring of Hydraulic Systems)

Структура исходных файлов:
- Каждый файл канала — матрица 2205 строк (циклов) × N столбцов (точек внутри цикла), tab-delimited
- profile.txt — метки состояния компонентов по циклам (не используются как признаки)

Из 7 фиксированных каналов PS1, PS2, PS3, FS1, TS1, EPS1, VS1 вычисляются 4 характеристики
по каждому циклу: среднее, СКО, минимум, максимум. Итоговая матрица 2 205 × 28.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 7 каналов по требованию варианта
CHANNELS = ["PS1", "PS2", "PS3", "FS1", "TS1", "EPS1", "VS1"]
# Физические величины и единицы для отчёта
CHANNEL_INFO = {
    "PS1": ("Давление", "bar", 100),
    "PS2": ("Давление", "bar", 100),
    "PS3": ("Давление", "bar", 100),
    "FS1": {"name": "Расход", "unit": "л/мин", "fs": 10},
    "TS1": {"name": "Температура", "unit": "\u00b0C", "fs": 1},
    "EPS1": {"name": "Мощность двигателя", "unit": "Вт", "fs": 100},
    "VS1": {"name": "Вибрация", "unit": "мм/с", "fs": 1},
}

# Исправляем CHANNEL_INFO для удобства
CHANNEL_INFO = {
    "PS1": ("Давление", "bar", 100),
    "PS2": ("Давление", "bar", 100),
    "PS3": ("Давление", "bar", 100),
    "FS1": ("Расход", "л/мин", 10),
    "TS1": ("Температура", "\u00b0C", 1),
    "EPS1": ("Мощность двигателя", "Вт", 100),
    "VS1": ("Вибрация", "мм/с", 1),
}

STAT_FUNCS = ["mean", "std", "min", "max"]


def read_channel_matrix(path: Path) -> np.ndarray:
    """Читает tab-delimited матрицу канала (строки = циклы, столбцы = точки цикла)."""
    data = np.loadtxt(path, delimiter="\t")
    return data


def extract_features(data_dir: Path) -> pd.DataFrame:
    """Извлекает 4 характеристики по каждому из 7 каналов -> матрица 2205 x 28."""
    frames = {}
    n_cycles = None

    for ch in CHANNELS:
        path = data_dir / f"{ch}.txt"
        mat = read_channel_matrix(path)
        if n_cycles is None:
            n_cycles = mat.shape[0]
        assert mat.shape[0] == n_cycles, f"{ch}: ожидалось {n_cycles} циклов, получено {mat.shape[0]}"

        for stat in STAT_FUNCS:
            if stat == "mean":
                val = mat.mean(axis=1)
            elif stat == "std":
                val = mat.std(axis=1)
            elif stat == "min":
                val = mat.min(axis=1)
            elif stat == "max":
                val = mat.max(axis=1)
            col_name = f"{ch}_{stat}"
            frames[col_name] = val

    df = pd.DataFrame(frames)
    df.index.name = "cycle"
    df.reset_index(inplace=True)
    return df


def load_profile(data_dir: Path) -> pd.DataFrame:
    """Читает profile.txt с метками состояния компонентов (только для проверки, не для обучения)."""
    # profile.txt: 5 столбцов, tab-delimited, без заголовка
    profile = pd.read_csv(data_dir / "profile.txt", sep="\t", header=None)
    profile.columns = ["cooler", "valve", "pump", "accumulator", "stable"]
    profile.index.name = "cycle"
    profile.reset_index(inplace=True)
    return profile


def build_features():
    """Основная точка входа: извлекает признаки, загружает профиль, сохраняет CSV."""
    print("Чтение исходных каналов и извлечение признаков...")
    df = extract_features(DATA_DIR)
    print(f"Матрица признаков: {df.shape}")

    print("Чтение profile.txt (метки состояния для проверки)...")
    profile = load_profile(DATA_DIR)
    print(f"Профиль: {profile.shape}")

    # Объединяем для удобства, но профиль не используем как признаки
    df_with_profile = df.merge(profile, on="cycle", how="left")

    feat_path = DATA_DIR / "features.csv"
    prof_path = DATA_DIR / "profile.csv"
    df.to_csv(feat_path, index=False)
    df_with_profile.to_csv(prof_path, index=False)

    print(f"Сохранено: {feat_path}")
    print(f"Сохранено: {prof_path}")
    return df, df_with_profile


if __name__ == "__main__":
    build_features()
