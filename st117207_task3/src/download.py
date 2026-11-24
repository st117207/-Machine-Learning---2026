# -*- coding: utf-8 -*-
"""Загрузка исходных сигналов UCI (нужна только для полного пересчёта).

`python build.py download` скачивает официальный архив
(≈ 76 МБ), проверяет SHA-256 и распаковывает семь используемых каналов
PS1, PS2, PS3, FS1, TS1, EPS1, VS1, а также profile.txt и описание набора
в data/raw/. После этого `python build.py features` пересчитывает таблицу
агрегатов из сырых сигналов.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import (CHANNEL_DIR, CHANNEL_NAMES, RAW_DIR, UCI_ZIP_SHA256,
                    UCI_ZIP_URL)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    CHANNEL_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all((CHANNEL_DIR / f"{c}.txt").exists()
                         for c in CHANNEL_NAMES):
        print("Исходные каналы уже распакованы: загрузка не нужна.")
        return

    tmp = RAW_DIR / "_uci_hydraulic.zip"
    print(f"Загрузка {UCI_ZIP_URL}")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as resp, \
            open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)

    digest = sha256_of(tmp)
    if digest != UCI_ZIP_SHA256:
        tmp.unlink(missing_ok=True)
        sys.exit(f"SHA-256 не совпал: {digest}\nОжидалось {UCI_ZIP_SHA256}")
    print("SHA-256 совпал.")

    with zipfile.ZipFile(tmp) as z:
        for name in CHANNEL_NAMES:
            z.extract(f"{name}.txt", CHANNEL_DIR)
        z.extract("profile.txt", RAW_DIR)
        for name in ("description.txt", "documentation.txt"):
            z.extract(name, RAW_DIR)
    tmp.unlink()
    print(f"Готово: каналы в {CHANNEL_DIR}")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
