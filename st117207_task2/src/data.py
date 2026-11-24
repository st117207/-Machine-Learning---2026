# -*- coding: utf-8 -*-
"""Загрузка и аудит набора UCI Statlog (Shuttle).

Схема работы:
1) если data/raw/shuttle.trn и data/raw/shuttle.tst уже есть — использовать их;
2) иначе скачать официальный архив UCI и распаковать (файл shuttle.trn в
   архиве сжат в .Z; для распаковки нужен 7-Zip или uncompress);
3) прочитать обучающую и тестовую части, приписанные авторами набора,
   и сохранить единую таблицу data/processed/shuttle.csv.

Разбиение на обучение и тест здесь не выполняется: оно уже задано авторами
набора и используется как есть (43500 и 14500 строк).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (CLASSES, FEATURES, PROC_DIR, PROCESSED_CSV, RAW_DIR,
                    TARGET, TEST_FILE, TRAIN_FILE, UCI_ZIP_URL)

SEVENZIP_CANDIDATES = [
    Path(r"C:\Program Files\7-Zip\7z.exe"),
    Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
]


def _find_7z() -> Path | None:
    exe = shutil.which("7z")
    if exe:
        return Path(exe)
    for p in SEVENZIP_CANDIDATES:
        if p.exists():
            return p
    return None


def _decompress_z(src: Path, dst: Path) -> None:
    """Распаковка файла .Z (LZW): 7-Zip, uncompress или gzip."""
    exe = _find_7z()
    if exe is not None:
        subprocess.run([str(exe), "e", str(src), f"-o{src.parent}", "-y"],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return
    if shutil.which("uncompress"):
        subprocess.run(["uncompress", "-f", str(src)], check=True)
        return
    if shutil.which("gzip"):
        subprocess.run(["gzip", "-d", "-k", "-f", str(src)], check=True)
        return
    raise RuntimeError(
        "Не удалось распаковать shuttle.trn.Z: установите 7-Zip или uncompress.")


def ensure_raw_data() -> None:
    """Проверяет наличие исходных файлов и при необходимости скачивает их."""
    if TRAIN_FILE.exists() and TEST_FILE.exists():
        print("Исходные данные Shuttle уже распакованы")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "shuttle.zip"
    if not zip_path.exists():
        print(f"Скачивание {UCI_ZIP_URL} ...")
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)

    if not TRAIN_FILE.exists():
        zpath = RAW_DIR / "shuttle.trn.Z"
        if not zpath.exists():
            sys.exit("В архиве нет shuttle.trn.Z — проверьте данные UCI.")
        _decompress_z(zpath, TRAIN_FILE)
    if not TRAIN_FILE.exists():
        sys.exit("Не удалось получить shuttle.trn.")


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает обучающую и тестовую части в DataFrame с колонками A..I, y."""
    def read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None,
                         names=FEATURES + [TARGET])
        df[TARGET] = df[TARGET].astype(int)
        return df

    tr = read(TRAIN_FILE)
    te = read(TEST_FILE)
    tr["split"] = "train"
    te["split"] = "test"
    return tr, te


def audit(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """Сводка аудита: размеры, классы, пропуски, дубликаты, диапазоны."""
    info = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": len(FEATURES),
        "n_classes": len(CLASSES),
        "class_counts_train": {int(c): int((tr[TARGET] == c).sum())
                               for c in CLASSES},
        "class_counts_test": {int(c): int((te[TARGET] == c).sum())
                              for c in CLASSES},
        "missing": int(tr[FEATURES].isna().sum().sum()
                       + te[FEATURES].isna().sum().sum()),
        "duplicates_train": int(tr[FEATURES + [TARGET]].duplicated().sum()),
        "duplicates_test": int(te[FEATURES + [TARGET]].duplicated().sum()),
        "overlap": int(tr[FEATURES + [TARGET]].merge(
            te[FEATURES + [TARGET]], how="inner").shape[0]),
        "feature_min": {c: float(tr[c].min()) for c in FEATURES},
        "feature_max": {c: float(tr[c].max()) for c in FEATURES},
        "constant_features": [c for c in FEATURES if tr[c].nunique() == 1],
    }
    return info


def build_processed() -> dict:
    """Собирает единую таблицу и возвращает словарь аудита."""
    ensure_raw_data()
    tr, te = load_frames()
    info = audit(tr, te)

    df = pd.concat([tr, te], ignore_index=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)

    print(f"Обучение: {info['n_train']} записей, тест: {info['n_test']}")
    print("Классы (обучение):",
          {c: info["class_counts_train"][c] for c in CLASSES})
    print("Классы (тест):",
          {c: info["class_counts_test"][c] for c in CLASSES})
    print(f"Пропуски: {info['missing']}, дубликаты: "
          f"{info['duplicates_train']}/{info['duplicates_test']}, "
          f"пересечение train/test: {info['overlap']}")
    print(f"Сохранено: {PROCESSED_CSV} ({len(df)} строк, "
          f"{df.shape[1]} колонок)")
    return info


if __name__ == "__main__":
    build_processed()
