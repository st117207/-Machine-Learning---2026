"""
Извлечение признаков из набора NASA IMS Bearing Dataset.
Задача 1 (регрессия), вариант 13 — деградация подшипника.

Признаки для каждой записи (1-секундный снимок вибрации, fs = 20 кГц, 20480 отсчётов):
  - RMS, дисперсия, эксцесс, пик-фактор (по каждому каналу, затем усреднено по каналам)
  - энергия в нескольких заранее заданных широких полосах спектра

Ограничение варианта: номер записи, абсолютное время и время до конца эксперимента
НЕ используются как входные признаки.
"""

from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

FS = 20_000  # частота дискретизации, Гц
N = 20_480   # количество отсчётов в файле

# Время окончания каждого испытания (из Readme)
END_TIMES = {
    1: datetime(2003, 11, 25, 23, 39, 56),
    2: datetime(2004, 2, 19, 6, 22, 39),
    3: datetime(2004, 4, 4, 19, 1, 57),
}

# Широкие полосы спектра, Гц (верхняя граница <= Найквиста 10 кГц)
BANDS = [(0, 500), (500, 1_000), (1_000, 2_000), (2_000, 4_000), (4_000, 10_000)]

FILENAME_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})$")


def parse_file_time(name: str) -> datetime:
    m = FILENAME_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse datetime from filename: {name}")
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def read_signal(path: Path) -> np.ndarray:
    """Читает ASCII-файл сигнала -> массив (20480, n_channels)."""
    data = np.loadtxt(path, delimiter="\t", ndmin=2)
    if data.shape[0] != N:
        # بعض файлы могут содержать лишнюю пустую строку в конце
        data = data[:N]
    return data


def channel_features(sig: np.ndarray) -> dict:
    """Признаки одного канала (одномерный сигнал длины N)."""
    x = np.asarray(sig, dtype=float).ravel()
    rms = float(np.sqrt(np.mean(x ** 2)))
    var = float(np.var(x))
    kr = float(kurtosis(x))  # избыточный эксцесс (нормальный -> 0)
    crest = float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0
    return {"rms": rms, "var": var, "kurt": kr, "crest": crest}


def band_energies(sig: np.ndarray) -> dict:
    """Энергия в заданных полосах спектра для одного канала."""
    x = np.asarray(sig, dtype=float).ravel()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    energy = np.abs(spec) ** 2
    out = {}
    for i, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        out[f"band_{lo}_{hi}"] = float(np.sum(energy[mask]))
    return out


def extract_features(sig2d: np.ndarray) -> dict:
    """Признаки всей записи (все каналы) — усреднение по каналам."""
    sig2d = np.asarray(sig2d, dtype=float)
    n_ch = sig2d.shape[1] if sig2d.ndim == 2 else 1
    if sig2d.ndim == 1:
        sig2d = sig2d.reshape(-1, 1)

    feats = {}
    # статистические признаки по каналам
    for name in ["rms", "var", "kurt", "crest"]:
        vals = []
        for c in range(n_ch):
            vals.append(channel_features(sig2d[:, c])[name])
        feats[name] = float(np.mean(vals))

    # энергия в полосах — усреднено по каналам
    for lo, hi in BANDS:
        key = f"band_{lo}_{hi}"
        vals = []
        for c in range(n_ch):
            vals.append(band_energies(sig2d[:, c])[key])
        feats[key] = float(np.mean(vals))
    return feats


def build_dataset(raw_root: Path, out_csv: Path):
    """Обходит все испытания, извлекает признаки и целевую переменную."""
    rows = []
    raw_root = Path(raw_root)
    for set_dir in sorted(raw_root.iterdir()):
        if not set_dir.is_dir():
            continue
        m = re.match(r"(\d+)", set_dir.name)
        if not m:
            continue
        set_no = int(m.group(1))
        end_time = END_TIMES[set_no]

        # файлы лежат во вложенной папке с таким же именем
        files = sorted(set_dir.rglob("*"))
        files = [f for f in files if f.is_file() and FILENAME_RE.search(f.name)]
        print(f"[set {set_no}] найдено файлов: {len(files)}")

        for f in files:
            try:
                sig = read_signal(f)
                file_time = parse_file_time(f.name)
                rul_hours = (end_time - file_time).total_seconds() / 3600.0
                feats = extract_features(sig)
                rows.append({
                    "set": set_no,
                    "file": f.name,
                    "file_time": file_time.isoformat(),
                    "rul_hours": rul_hours,
                    **feats,
                })
            except Exception as e:
                print(f"  SKIP {f}: {e}")

    df = pd.DataFrame(rows)
    # убираем записи с отрицательным RUL (после конца эксперимента)
    df = df[df["rul_hours"] >= 0].copy()
    df = df.sort_values(["set", "file_time"]).reset_index(drop=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Всего записей: {len(df)}, признаков: {df.shape[1] - 4}")
    print(df.groupby("set").agg(n=("rul_hours", "size"),
                                rul_min=("rul_hours", "min"),
                                rul_max=("rul_hours", "max")))
    return df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    RAW = ROOT / "data" / "IMS"
    OUT = ROOT / "data" / "features.csv"
    build_dataset(RAW, OUT)
