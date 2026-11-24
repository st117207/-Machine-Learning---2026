# -*- coding: utf-8 -*-
"""Данные задачи 3 (вариант 13): гидравлический испытательный стенд.

Источник — UCI Condition Monitoring of Hydraulic Systems (DOI 10.24432/C5CW21).
Каждый файл канала содержит 2 205 строк-циклов; в строке — отсчёты одного
цикла длительностью 60 с с частотой канала. Из семи фиксированных каналов
PS1, PS2, PS3, FS1, TS1, EPS1, VS1 вычисляются четыре характеристики за цикл:
среднее, СКО (по генеральной совокупности, ddof=0), минимум и максимум.

Порядок работы:
* Если в data/raw/channels лежат исходные .txt каналы (их скачивает
  `python build.py download`), агрегаты пересчитываются из них.
* Иначе используется проверенная таблица data/raw/hydraulic_aggregates.csv,
  которая входит в архив и получена тем же кодом из тех же исходных файлов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (AGGREGATES_CSV, BLOCK_SIZE, CHANNEL_DIR, CHANNEL_NAMES,
                    EXAMPLE_CYCLE, FEATURES, LABELS, PROC_DIR, PROCESSED_CSV,
                    PROFILE_FILE, RANDOM_STATE, TRAIN_FRAC)

CHANNEL_FILES = {c: CHANNEL_DIR / f"{c}.txt" for c in CHANNEL_NAMES}


def _read_channel(path) -> np.ndarray:
    """Матрица циклопризнаков канала: строки — циклы, столбцы — отсчёты."""
    arr = pd.read_csv(path, sep="\t", header=None).to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 2205:
        raise ValueError(f"{path}: ожидалось 2205 циклов, получено {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: найдены нечисловые значения")
    return arr


def build_aggregates() -> None:
    """Пересчёт 28 агрегатов из исходных каналов, если они распакованы."""
    if not all(p.exists() for p in CHANNEL_FILES.values()):
        if not AGGREGATES_CSV.exists():
            raise FileNotFoundError(
                "Нет ни исходных каналов в data/raw/channels, ни таблицы "
                "агрегатов. Выполните `python build.py download`.")
        print("Исходные каналы не найдены: используется "
              "data/raw/hydraulic_aggregates.csv.")
        return

    parts = []
    for ch in CHANNEL_NAMES:
        a = _read_channel(CHANNEL_FILES[ch])
        parts.append(pd.DataFrame({
            f"{ch}_mean": a.mean(axis=1),
            f"{ch}_std": a.std(axis=1, ddof=0),
            f"{ch}_min": a.min(axis=1),
            f"{ch}_max": a.max(axis=1),
        }))
    agg = pd.concat(parts, axis=1)
    agg.insert(0, "cycle", np.arange(1, len(agg) + 1))
    agg.to_csv(AGGREGATES_CSV, index=False, float_format="%.6g")
    ex = {ch: _read_channel(CHANNEL_FILES[ch])[0].astype(np.float32)
          for ch in CHANNEL_NAMES}
    np.savez_compressed(EXAMPLE_CYCLE, **ex)
    print(f"Агрегаты пересчитаны из каналов: {AGGREGATES_CSV} "
          f"({agg.shape[0]} x {agg.shape[1] - 1})")


def load_example_cycle() -> dict:
    """Один полный цикл по семи каналам — для рисунка исходных сигналов."""
    with np.load(EXAMPLE_CYCLE) as z:
        return {k: z[k] for k in z.files}


def read_profile() -> pd.DataFrame:
    prof = pd.read_csv(PROFILE_FILE, sep="\t", header=None, names=LABELS,
                       dtype=int)
    if len(prof) != 2205:
        raise ValueError("profile.txt: ожидалось 2205 строк")
    return prof


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Возвращает X (2 205 x 28), метки узлов и номера циклов."""
    agg = pd.read_csv(AGGREGATES_CSV)
    if "cycle" not in agg.columns:
        raise ValueError("В таблице агрегатов нет столбца cycle")
    missing = [f for f in FEATURES if f not in agg.columns]
    if missing:
        raise ValueError(f"В таблице агрегатов нет признаков: {missing}")
    prof = read_profile()
    if len(agg) != len(prof):
        raise ValueError("Число циклов в агрегатах и profile.txt не совпадает")
    X = agg[FEATURES].astype(float)
    cycles = agg["cycle"].to_numpy(dtype=int)
    return X, prof, cycles


def audit(X: pd.DataFrame, prof: pd.DataFrame) -> dict:
    """Содержательный аудит: пропуски, константы, дубликаты, выбросы."""
    n, p = X.shape
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ((X < low) | (X > high)).sum()
    constant = [c for c in X.columns if X[c].nunique() == 1]
    near = []
    for c in X.columns:
        vals, counts = np.unique(X[c].to_numpy(), return_counts=True)
        if len(vals) <= 2 or counts.max() / n >= 0.999:
            near.append(c)
    dup_rows = int(X.duplicated().sum())
    n_unique_rows = int(n - dup_rows)
    return {
        "n_cycles": int(n),
        "n_features": int(p),
        "n_channels": len(CHANNEL_NAMES),
        "missing": int(X.isna().sum().sum()),
        "duplicates": dup_rows,
        "unique_rows": n_unique_rows,
        "constant_features": constant,
        "n_constant": len(constant),
        "near_constant_features": near,
        "n_near_constant": len(near),
        "outlier_total": int(outliers.sum()),
        "outlier_share": float(outliers.sum() / (n * p)),
        "outlier_by_feature": {c: int(v) for c, v in outliers.items()},
        "min_by_feature": {c: float(v) for c, v in X.min().items()},
        "max_by_feature": {c: float(v) for c, v in X.max().items()},
        "mean_by_feature": {c: float(v) for c, v in X.mean().items()},
        "median_by_feature": {c: float(v) for c, v in X.median().items()},
        "std_by_feature": {c: float(v) for c, v in X.std(ddof=0).items()},
        "label_counts": {k: {str(v): int(c) for v, c in
                             prof[k].value_counts().sort_index().items()}
                         for k in LABELS},
        "label_unique": {k: sorted(int(v) for v in prof[k].unique())
                         for k in LABELS},
    }


def chronological_split(n: int, frac: float = TRAIN_FRAC) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(round(frac * n))
    return np.arange(n_train), np.arange(n_train, n)


def blocked_split(n: int, block: int = BLOCK_SIZE, frac: float = TRAIN_FRAC,
                  seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Случайное разбиение целыми блоками последовательных циклов.

    Соседние циклы внутри блока не разрываются, поэтому проверка не страдает
    от утечки почти одинаковых окон, но сохраняет состав режимов.
    """
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    n_train_blocks = int(round(frac * n_blocks))
    train_blocks = set(order[:n_train_blocks].tolist())
    train_idx, test_idx = [], []
    for b in range(n_blocks):
        lo, hi = b * block, min((b + 1) * block, n)
        (train_idx if b in train_blocks else test_idx).extend(range(lo, hi))
    return np.array(train_idx), np.array(test_idx)


def cluster_subsample(n: int, size: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """Фиксированная подвыборка объектов для сравнения трёх алгоритмов."""
    size = min(size, n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def build_processed() -> None:
    """Единая таблица для просмотра: признаки, метки, хронологическое разбиение."""
    build_aggregates()
    X, prof, cycles = load_dataset()
    tr, _ = chronological_split(len(X))
    df = X.copy()
    df.insert(0, "cycle", cycles)
    for k in LABELS:
        df[k] = prof[k].to_numpy()
    split = np.full(len(df), "test", dtype=object)
    split[tr] = "train"
    df["split"] = split
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False, float_format="%.6g")
    print(f"Таблица обработана: {PROCESSED_CSV} ({df.shape[0]} x {df.shape[1]})")
