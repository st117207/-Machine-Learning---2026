"""
Загрузка и подготовка набора UCI Statlog (Shuttle) для Задачи 2, вариант 18.

Набор: https://archive.ics.uci.edu/dataset/148/statlog+shuttle
Загружается через пакет ucimlrepo (официальный клиент UCI ML Repository).

Оригинальный набор содержит 9 числовых признаков + класс (7 классов).
В ucimlrepo представлено 58 000 объектов, 7 признаков (признак «time» исключён
из публичной версии). Физический смысл признаков в описании UCI не раскрыт —
это честно указано в отчёте.

Разбиение: используется оригинальное разделение авторов на обучающую и тестовую
части (примерно 75 % / 25 %). Подбор гиперпараметров выполняется только по
обучающей части. Тестовая выборка используется один раз в конце.
"""

from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Оригинальные имена классов Shuttle (из описания UCI)
CLASS_NAMES = {
    1: "Rad Flow",
    2: "Fpv Close",
    3: "Fpv Open",
    4: "High",
    5: "Bypass",
    6: "Bpv Close",
    7: "Bpv Open",
}

# Все 9 числовых признаков из оригинального описания UCI.
# В ucimlrepo доступно 7 (признак «time» исключён). Честно указываем это в отчёте.
FEATURE_NAMES_9 = [
    "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9",
]


def load_and_save():
    """Скачивает набор через ucimlrepo, делит на train/test, сохраняет CSV."""
    print("Загрузка UCI Statlog (Shuttle) через ucimlrepo...")
    ds = fetch_ucirepo(id=148)

    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    # Привести целевую переменную к целым числам
    y_col = y.columns[0]
    y[y_col] = y[y_col].astype(int)

    # Объединить в один фрейм
    df = pd.concat([X, y], axis=1)
    df.columns = [f"X{i+1}" for i in range(X.shape[1])] + ["class"]

    print(f"Всего объектов: {len(df)}, признаков: {X.shape[1]}, классов: {sorted(df['class'].unique())}")
    print("\nРаспределение классов (все данные):")
    print(df["class"].value_counts().sort_index())

    # Разделение на train/test в пропорции авторов (~75% / 25%)
    # Оригинал: train ≈ 43500, test ≈ 14500 (сумма 58000)
    train_size = int(0.75 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Перемешаем обучающую часть (тестовая остаётся в авторском порядке)
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path = DATA_DIR / "shuttle_train.csv"
    test_path = DATA_DIR / "shuttle_test.csv"
    all_path = DATA_DIR / "shuttle_all.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(all_path, index=False)

    print(f"\nОбучающая часть: {len(train_df)} объектов -> {train_path}")
    print(f"Тестовая часть: {len(test_df)} объектов -> {test_path}")
    print("\nРаспределение классов в train:")
    print(train_df["class"].value_counts().sort_index())
    print("\nРаспределение классов в test:")
    print(test_df["class"].value_counts().sort_index())

    return train_df, test_df, df


if __name__ == "__main__":
    load_and_save()
