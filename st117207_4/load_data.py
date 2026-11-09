"""
Загрузка и подготовка набора UCI Gas Sensor Array Drift для Задачи 4, вариант 21.

Источник: https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset
(UCI ID 224, https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip)

Структура исходных файлов (libsvm/svm-light):
  каждая строка: <метка_газа> 1:val1 2:val2 ... 128:val128
  - метка: один из шести типов газа (1..6)
  - 128 признаков: готовые числовые характеристики 16 сенсоров

10 временных партий (batch1.dat ... batch10.dat).
По требованию варианта:
  партии 1-6  -> обучение
  партии 7-8  -> проверка (валидация, early stopping, подбор гиперпараметров)
  партии 9-10 -> окончательный тест (используется один раз в конце)
Номер партии не используется как признак.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = DATA_DIR / "Dataset"
N_FEATURES = 128
GAS_NAMES = {1: "gas_1", 2: "gas_2", 3: "gas_3", 4: "gas_4", 5: "gas_5", 6: "gas_6"}

# Соответствие партий к разделам (по требованию варианта)
PARTY_SPLIT = {
    1: "train", 2: "train", 3: "train", 4: "train", 5: "train", 6: "train",
    7: "val", 8: "val",
    9: "test", 10: "test",
}


def parse_libsvm_line(line: str):
    """Разбирает одну строку libsvm: метка + пары индекс:значение."""
    parts = line.strip().split()
    if not parts:
        return None
    label = int(parts[0])
    x = np.zeros(N_FEATURES, dtype=float)
    for tok in parts[1:]:
        idx_str, val_str = tok.split(":")
        idx = int(idx_str) - 1  # libsvm индексы с 1
        if 1 <= idx <= N_FEATURES:
            x[idx - 1] = float(val_str)
    return label, x


def load_batch(batch_no: int) -> pd.DataFrame:
    """Загружает один batch-файл -> DataFrame с колонками label, party, f0..f127."""
    path = DATASET_DIR / f"batch{batch_no}.dat"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_libsvm_line(line)
            if parsed is None:
                continue
            label, x = parsed
            row = {"label": label, "party": batch_no}
            for j in range(N_FEATURES):
                row[f"f{j}"] = x[j]
            rows.append(row)
    return pd.DataFrame(rows)


def build_features():
    """Собирает все партии, добавляет колонку split, сохраняет CSV."""
    frames = []
    for b in range(1, 11):
        df_b = load_batch(b)
        print(f"batch{b}: {len(df_b)} измерений, метки {sorted(df_b['label'].unique())}")
        frames.append(df_b)
    df = pd.concat(frames, ignore_index=True)
    df["split"] = df["party"].map(PARTY_SPLIT)
    df["gas"] = df["label"].map(GAS_NAMES)

    feat_path = DATA_DIR / "features.csv"
    df.to_csv(feat_path, index=False)

    print(f"\nВсего: {len(df)} измерений, признаков: {N_FEATURES}")
    print(df.groupby(["split", "gas"]).size().unstack(fill_value=0))
    print("\nРаспределение по сплитам:")
    print(df["split"].value_counts())

    return df


if __name__ == "__main__":
    build_features()
