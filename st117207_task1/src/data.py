# -*- coding: utf-8 -*-
"""Получение исходных данных IMS и извлечение признаков по записям.

Схема работы:
1) если каталоги data/raw/1st_test и data/raw/2nd_test уже распакованы — пропустить;
2) иначе скачать архив NASA PCoE «4. Bearings» (URL фиксирован) и распаковать
   цепочку zip -> 7z -> rar;
3) для каждой записи (файла) вычислить по каждому из 4 виброканалов RMS,
   дисперсию, эксцесс, пик-фактор и доли энергии в заданных широких полосах;
4) сохранить таблицу признаков и целевой величины (время до конца испытания, ч).
"""
from __future__ import annotations

import io
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from config import (BANDS_HZ, FEATURES_CSV, FS_HZ, N_SAMPLES, PROC_DIR, RAW_DIR,
                    ROOT, TRAIN_TEST_NAME, HOLDOUT_TEST_NAME)

ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
ZIP_PATH = RAW_DIR / "IMS_Bearings.zip"
SEVENZIP_CANDIDATES = [
    Path(r"C:\Program Files\7-Zip\7z.exe"),
    Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
]


# ---------------------------------------------------------------------------
# Шаг 1-3. Загрузка и распаковка исходных данных
# ---------------------------------------------------------------------------

def _download_zip() -> None:
    """Скачивает архив набора данных, если он ещё не скачан."""
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 10**9:
        print("Архив уже скачан, пропускаю загрузку")
        return
    print(f"Скачивание {ZIP_URL} (~1.1 ГБ) ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(ZIP_PATH, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    print("Скачивание завершено")


def _unpack_outer_zip() -> None:
    """Распаковывает внешний zip -> «4. Bearings/IMS.7z»."""
    inner = RAW_DIR / "4. Bearings" / "IMS.7z"
    if inner.exists():
        return
    print("Распаковка внешнего zip ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)


def _unpack_7z() -> None:
    """Распаковывает IMS.7z -> *.rar (py7zr, при отсутствии — 7-Zip)."""
    rar1 = RAW_DIR / "1st_test.rar"
    rar2 = RAW_DIR / "2nd_test.rar"
    if rar1.exists() and rar2.exists():
        return
    src = RAW_DIR / "4. Bearings" / "IMS.7z"
    print("Распаковка IMS.7z ...")
    try:
        import py7zr
        with py7zr.SevenZipFile(src, mode="r") as z:
            z.extract(targets=["1st_test.rar", "2nd_test.rar", "3rd_test.rar"], path=RAW_DIR)
    except ImportError:
        exe = _find_7z()
        _run_tool([str(exe), "e", str(src), "1st_test.rar", "2nd_test.rar", "-y", f"-o{RAW_DIR}"])


def _unpack_rar(name: str) -> None:
    """Распаковывает один RAR в одноимённый каталог (rarfile/unrar -> 7-Zip)."""
    rar = RAW_DIR / name
    out = RAW_DIR / name.replace(".rar", "")
    if out.exists() and any(out.rglob("*")):
        return
    print(f"Распаковка {name} ...")
    try:
        import rarfile  # тип: требует системного unrar
        with rarfile.RarFile(str(rar)) as rf:
            rf.extractall(str(RAW_DIR))
    except Exception:
        exe = _find_7z()
        _run_tool([str(exe), "x", str(rar), f"-o{RAW_DIR}", "-y"])


def _find_7z() -> Path:
    for p in SEVENZIP_CANDIDATES:
        if p.exists():
            return p
    p = shutil.which("7z")
    if p:
        return Path(p)
    raise RuntimeError(
        "Не найден инструмент для распаковки RAR. Установите 7-Zip "
        "(https://www.7-zip.org) либо unrar и повторите сборку."
    )


def _run_tool(cmd: list[str]) -> None:
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Ошибка распаковки: {cmd}\n{r.stderr}")


def ensure_raw_data() -> None:
    _download_zip()
    _unpack_outer_zip()
    _unpack_7z()
    _unpack_rar("1st_test.rar")
    _unpack_rar("2nd_test.rar")


def raw_test_dirs() -> dict[str, Path]:
    return {
        TRAIN_TEST_NAME: RAW_DIR / TRAIN_TEST_NAME / TRAIN_TEST_NAME,
        HOLDOUT_TEST_NAME: RAW_DIR / HOLDOUT_TEST_NAME / HOLDOUT_TEST_NAME,
    }


# ---------------------------------------------------------------------------
# Шаг 4. Извлечение признаков
# ---------------------------------------------------------------------------

def _parse_stamp(stem: str) -> datetime:
    """Отметка времени из имени файла: 'ГГГГ.ММ.ДД.ЧЧ.ММ.СС' или без секунд."""
    for fmt in ("%Y.%m.%d.%H.%M.%S", "%Y.%m.%d.%H.%M"):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    raise ValueError(f"Нераспознано имя записи: {stem}")


def _read_record(path: Path) -> np.ndarray:
    """Читает одну запись: матрица (20480, n), В."""
    df = pd.read_csv(path, sep="\t", header=None, dtype=np.float32)
    return df.to_numpy()


def _band_energies(x: np.ndarray) -> list[float]:
    """Доли суммарной спектральной энергии в заданных широких полосах."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / FS_HZ)
    power = (spec.real**2 + spec.imag**2)
    total = power.sum()
    out = []
    for lo, hi in BANDS_HZ:
        out.append(float(power[(freqs >= lo) & (freqs < hi)].sum() / total))
    return out


def _channel_stats(sig: np.ndarray) -> dict[str, float]:
    """Статистики одной оси (канала), В: RMS, дисперсия, эксцесс, пик-фактор, полосы."""
    mean = sig.mean()
    var = float(((sig - mean) ** 2).mean())
    rms = float(np.sqrt((sig**2).mean()))
    out = {
        "rms": rms,
        "var": var,
        "kurt": float(kurtosis(sig, fisher=True, bias=True)),
        "crest": float(np.abs(sig).max() / rms),
    }
    for (lo, hi), e in zip(BANDS_HZ, _band_energies(sig)):
        out[f"e{lo//1000:g}_{hi//1000:g}khz"] = e
    return out


def _record_features(x: np.ndarray, n_bearings: int = 4) -> dict[str, float]:
    """Признаки записи на уровне подшипников.

    В наборе 1 каждый подшипник оснащён двумя акселерометрами (оси x и y,
    столбцы 2b-1 и 2b), в наборе 2 — одним (столбец b). Агрегация по осям:
    RMS — по энергии, дисперсия и доли энергии полос — среднее по осям,
    эксцесс и пик-фактор — максимум по осям. Такие определения дают
    одинаковый смысл признаков для обоих испытаний.
    """
    chans = list(range(x.shape[1]))
    if len(chans) == 8:      # набор 1: два акселерометра (x, y) на подшипник
        groups = [chans[2 * b: 2 * b + 2] for b in range(4)]
    else:                    # набор 2: один канал на подшипник
        groups = [[c] for c in chans[:4]]
    feats: dict[str, float] = {}
    for b, grp in enumerate(groups, start=1):
        stats = [_channel_stats(x[:, c].astype(np.float64)) for c in grp]
        feats[f"b{b}_rms"] = float(np.sqrt(np.mean([s["rms"] ** 2 for s in stats])))
        feats[f"b{b}_var"] = float(np.mean([s["var"] for s in stats]))
        feats[f"b{b}_kurt"] = float(max(s["kurt"] for s in stats))
        feats[f"b{b}_crest"] = float(max(s["crest"] for s in stats))
        for key in [k for k in stats[0] if k.startswith("e")]:
            feats[f"b{b}_{key}"] = float(np.mean([s[key] for s in stats]))
    return feats


def build_features() -> pd.DataFrame:
    """Формирует таблицу признаков для обоих испытаний с кэшированием."""
    if FEATURES_CSV.exists():
        print("Таблица признаков уже построена, использую кэш")
        return pd.read_csv(FEATURES_CSV, parse_dates=["t_start"])

    rows = []
    n_channels = {"1st_test": 8, "2nd_test": 4}
    for name, d in raw_test_dirs().items():
        files = sorted(f for f in d.iterdir() if f.is_file())
        print(f"{name}: {len(files)} записей")
        for i, f in enumerate(files):
            t = _parse_stamp(f.stem)
            x = _read_record(f)[:, :n_channels[name]]
            row = {"test": name, "fname": f.stem, "t_start": t}
            row.update(_record_features(x))
            rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"  {name}: обработано {i+1}/{len(files)}")

    df = pd.DataFrame(rows).sort_values(["test", "t_start"]).reset_index(drop=True)

    # Целевая величина — прокси-ресурс: время от текущей записи до конца
    # данного испытания (по отметкам времени имён файлов), ч.
    t_end = df.groupby("test")["t_start"].transform("max")
    df["rul_h"] = (t_end - df["t_start"]).dt.total_seconds() / 3600.0

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Сохранено: {FEATURES_CSV} ({df.shape[0]} записей, {df.shape[1]} колонок)")
    return df


if __name__ == "__main__":
    ensure_raw_data()
    build_features()
