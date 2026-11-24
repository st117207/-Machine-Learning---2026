# -*- coding: utf-8 -*-
"""Снижение размерности и кластеризация данных гидравлического стенда.

Вариант 13: UCI Condition Monitoring of Hydraulic Systems. Из семи каналов
PS1, PS2, PS3, FS1, TS1, EPS1, VS1 по каждому из 2 205 циклов вычислены
среднее, СКО, минимум и максимум (28 признаков).

Методика
--------
* Хронологическое разбиение 70/30 по номерам циклов: обучение — циклы
  1–1544, тест — 1545–2205. Дополнительно блочное разбиение по 50
  последовательных циклов (проверка устойчивости к смене режимов).
* Из 28 циклопризнаков исключены PS2_min, PS3_min и FS1_min (постоянные или
  почти постоянные), в анализе 25 признаков.
* PCA сопоставлена в двух вариантах предобработки: только центрирование и
  стандартизация. Спектр, накопленная дисперсия и ошибка восстановления
  (inverse_transform) на отложенных циклах при k = 1..10.
* FactorAnalysis на тех же 25 исходных признаках (2, 3, 5 факторов):
  нагрузки, noise_variance_, правдоподобие на отложенных циклах.
* KMeans, AgglomerativeClustering (ward) и DBSCAN — на одной фиксированной
  подвыборке 1 200 циклов и одних и тех же четырёх PCA-координатах.
* Метки состояний узлов из profile.txt в признаки не входят; они используются
  только для послетаковой проверки состава групп.
"""
from __future__ import annotations

import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import (ACCUM_NAMES, BLOCK_SIZE, CHANNEL_DESC, CHANNEL_GROUP,
                    CHANNEL_NAMES, CHANNEL_RATE, CHANNEL_UNITS, CLUSTER_MAX_OBJ,
                    CLUSTER_PCS, COOLER_NAMES, DBSCAN_EPS_GRID,
                    DBSCAN_MIN_SAMPLES, FA_COMPONENTS, FA_MAX_ROWS, FEATURES,
                    FEATURE_GROUP, FIG_DIR, K_RANGE, LABELS, LABEL_NAMES,
                    N_PCA_MAX, N_STABILITY, PCA_MAX_ROWS, PROC_DIR,
                    PUMP_NAMES, RANDOM_STATE, RESULTS_JSON, STABLE_NAMES,
                    STATS, TAB_DIR, TRAIN_FRAC, VALVE_NAMES)
from data import (audit, blocked_split, build_aggregates,
                  chronological_split, cluster_subsample, load_dataset,
                  load_example_cycle)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "axes.unicode_minus": True,
})

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2"]
NOISE_COLOR = "#b0b0b0"
COOLER_COLORS = {3: "#d62728", 20: "#ff7f0e", 100: "#1f77b4"}
VALVE_COLORS = {73: "#d62728", 80: "#ff7f0e", 90: "#2ca02c", 100: "#1f77b4"}
ACCUM_COLORS = {90: "#d62728", 100: "#ff7f0e", 115: "#2ca02c", 130: "#1f77b4"}
PUMP_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#d62728"}

STAT_SHORT = {"mean": "среднее", "std": "СКО", "min": "минимум",
              "max": "максимум"}
STAT_TICK = {"mean": "ср", "std": "СКО", "min": "мин", "max": "макс"}
# Порядок строк в тепловых картах нагрузок — по физическим группам
GROUP_ORDER = ["Давление", "Расход", "Температура", "Мощность", "Вибрация"]


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def fmt(x, nd: int = 2) -> str:
    """Число в LaTeX: запятая как разделитель, минус через \\textminus."""
    if x is None:
        return "---"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "---"
    if abs(v) < 0.5 * 10.0 ** (-nd):
        v = 0.0
    s = f"{v:.{nd}f}".replace(".", ",")
    if s.startswith("-"):
        s = r"\textminus{}" + s[1:]
    return s


def tex_escape(s: str) -> str:
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in str(s))


def feature_tex(f: str) -> str:
    ch, st = f.split("_")
    return f"{ch}, {STAT_SHORT[st]}"


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# PCA и факторный анализ
# ---------------------------------------------------------------------------

def _fix_component_signs(loadings: np.ndarray) -> np.ndarray:
    """Знак компоненты выбирается так, чтобы сумма нагрузок была > 0."""
    L = loadings.copy()
    for j in range(L.shape[1]):
        if L[:, j].sum() < 0:
            L[:, j] *= -1
    return L


def pca_analysis(X: np.ndarray, tr: np.ndarray, te: np.ndarray,
                 variant: str) -> dict:
    """PCA на обучающих циклах; ошибка восстановления — на отложенных."""
    scaler = StandardScaler(with_std=(variant == "standard"))
    Atr = scaler.fit_transform(X[tr])
    Ate = scaler.transform(X[te])

    full = PCA(n_components=N_PCA_MAX, svd_solver="full",
               random_state=RANDOM_STATE).fit(Atr)
    loadings = _fix_component_signs(full.components_.T)

    std_tr = Atr.std(axis=0)
    std_tr[std_tr == 0] = 1.0
    std_orig = X[tr].std(axis=0)
    std_orig[std_orig == 0] = 1.0
    rmse, rrmse = [], []
    for k in range(1, N_PCA_MAX + 1):
        p = PCA(n_components=k, svd_solver="full",
                random_state=RANDOM_STATE).fit(Atr)
        rec = scaler.inverse_transform(p.inverse_transform(p.transform(Ate)))
        err = X[te] - rec
        rmse.append(float(np.sqrt(np.mean(err ** 2))))
        per = np.sqrt(np.mean(err ** 2, axis=0)) / std_orig
        rrmse.append(float(np.sqrt(np.mean(per ** 2))))

    return {
        "evr": [float(v) for v in full.explained_variance_ratio_],
        "cum": [float(v) for v in np.cumsum(full.explained_variance_ratio_)],
        "loadings": loadings,
        "scores": full.transform(scaler.transform(X)),
        "rmse": rmse,
        "rrmse": rrmse,
        "scaler": scaler,
    }


def fa_analysis(X: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    """FactorAnalysis на стандартизованных исходных признаках."""
    sc = StandardScaler().fit(X[tr])
    Atr = sc.transform(X[tr])
    Ate = sc.transform(X[te])
    out = {}
    for k in FA_COMPONENTS:
        fa = FactorAnalysis(n_components=k, random_state=RANDOM_STATE,
                            max_iter=2000).fit(Atr)
        L = _fix_component_signs(fa.components_.T)
        noise = fa.noise_variance_
        out[k] = {
            "loadings": L,
            "noise": noise,
            "comm": 1.0 - noise,
            "ll_train": float(fa.score(Atr)),
            "ll_test": float(fa.score(Ate)),
            "n_iter": int(fa.n_iter_),
        }
    return out


def stability_score(Z: np.ndarray, k: int, ref: np.ndarray, algo: str,
                    reps: int = N_STABILITY, seed: int = RANDOM_STATE):
    """Устойчивость разбиения: ARI между полным и бутстрэп-разбиениями."""
    rng = np.random.default_rng(seed)
    n = len(Z)
    vals = []
    for b in range(reps):
        sub = rng.choice(n, size=int(0.8 * n), replace=False)
        if algo == "kmeans":
            m = KMeans(n_clusters=k, n_init=10, random_state=seed + b)
        else:
            m = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = m.fit_predict(Z[sub])
        vals.append(adjusted_rand_score(ref[sub], lab))
    return float(np.mean(vals)), float(np.std(vals))


def cluster_quality(Z: np.ndarray, lab: np.ndarray) -> dict:
    """Силуэт и Дэвис—Болдин только там, где они определены."""
    uniq = np.unique(lab)
    uniq = uniq[uniq != -1]
    mask = lab != -1
    if len(uniq) < 2 or mask.sum() <= len(uniq):
        return {"silhouette": None, "db": None}
    return {
        "silhouette": float(silhouette_score(Z[mask], lab[mask])),
        "db": float(davies_bouldin_score(Z[mask], lab[mask])),
    }


def label_match(lab: np.ndarray, truth: np.ndarray) -> dict:
    """ARI и NMI; точки шума DBSCAN исключаются."""
    mask = lab != -1
    if len(np.unique(lab[mask])) < 2:
        return {"ari": None, "nmi": None, "n_used": int(mask.sum())}
    return {
        "ari": float(adjusted_rand_score(truth[mask], lab[mask])),
        "nmi": float(normalized_mutual_info_score(truth[mask], lab[mask])),
        "n_used": int(mask.sum()),
    }


def _relabel_by_median(lab: np.ndarray, idx: np.ndarray,
                       cycles: np.ndarray) -> tuple[np.ndarray, dict]:
    """Перенумеровать группы по возрастанию медианного номера цикла.

    Нумерация sklearn произвольна: без переупорядочивания «группа 1» могла
    оказаться как ранней, так и поздней. Порядок по времени делает таблицы и
    рисунки сопоставимыми между алгоритмами.
    """
    uniq = [int(c) for c in np.unique(lab) if c != -1]
    order = sorted(uniq, key=lambda c: float(np.median(cycles[idx][lab == c])))
    mapping = {old: new for new, old in enumerate(order)}
    out = lab.copy()
    for old, new in mapping.items():
        out[lab == old] = new
    return out, mapping


def cluster_profile(lab: np.ndarray, idx: np.ndarray, cycles: np.ndarray,
                    prof: pd.DataFrame) -> list[dict]:
    """Профиль кластера: размер, диапазон циклов, доля установившегося режима."""
    out = []
    cyc_sub = cycles[idx]
    for c in sorted(set(lab.tolist())):
        if c == -1:
            continue
        m = lab == c
        cc = cyc_sub[m]
        out.append({
            "cluster": int(c), "size": int(m.sum()),
            "cycle_min": int(cc.min()), "cycle_med": int(np.median(cc)),
            "cycle_max": int(cc.max()),
            "stable": float(prof["stable"].to_numpy()[idx][m].mean()),
            "cooler": {str(int(v)): float(np.mean(
                prof["cooler"].to_numpy()[idx][m] == v)) for v in (3, 20, 100)},
        })
    return out


def contingency(lab: np.ndarray, truth: np.ndarray, levels) -> dict:
    mask = lab != -1
    ct = pd.crosstab(lab[mask], truth[mask]).reindex(
        index=sorted(np.unique(lab[mask])), columns=list(levels), fill_value=0)
    return {
        "rows": [int(i) for i in ct.index],
        "cols": [int(c) for c in ct.columns],
        "counts": ct.to_numpy().astype(int).tolist(),
    }


def sweep_partition(Z: np.ndarray, algo: str, truth: dict) -> dict:
    """KMeans или Ward: k = 2..6, силуэт, Дэвис—Болдин, устойчивость."""
    rows = []
    for k in K_RANGE:
        if algo == "kmeans":
            m = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        else:
            m = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = m.fit_predict(Z)
        q = cluster_quality(Z, lab)
        sm, ss = stability_score(Z, k, lab, algo)
        sizes = np.bincount(lab).tolist()
        row = {
            "k": k, "sizes": sorted(sizes, reverse=True),
            "silhouette": q["silhouette"], "db": q["db"],
            "stab_mean": sm, "stab_std": ss,
            "ari_cooler": label_match(lab, truth["cooler"])["ari"],
            "ari_valve": label_match(lab, truth["valve"])["ari"],
            "ari_accum": label_match(lab, truth["accum"])["ari"],
            "ari_pump": label_match(lab, truth["pump"])["ari"],
        }
        rows.append(row)
    return {"rows": rows, "labels": None}


def select_k(rows: list[dict], n_obj: int) -> int:
    """Правило выбора k: устойчивость >= 0,90, все группы >= 5 % объектов,
    затем наименьшее k, силуэт которого не ниже 95 % от максимального.

    Ограничение на размер группы отсекает вырожденные группы из нескольких
    циклов, а порог 95 % не позволяет выбирать k только из-за монотонного
    роста силуэта с числом групп.
    """
    cand = [r for r in rows
            if r["stab_mean"] >= 0.90
            and min(r["sizes"]) >= 0.05 * n_obj
            and r["silhouette"] is not None]
    if not cand:
        return max(rows, key=lambda r: (r["stab_mean"], -r["k"]))["k"]
    smax = max(r["silhouette"] for r in cand)
    ok = [r for r in cand if r["silhouette"] >= 0.95 * smax]
    return min(r["k"] for r in ok)


def dbscan_sweep(Z: np.ndarray, truth: dict, min_samples: int) -> dict:
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    dist, _ = nn.kneighbors(Z)
    kdist = np.sort(dist[:, -1])
    eps_rule = float(np.percentile(kdist, 90))
    eps_chosen = min(e for e in DBSCAN_EPS_GRID if e >= eps_rule)

    rows = []
    for eps in DBSCAN_EPS_GRID:
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        n_cl = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float(np.mean(lab == -1))
        q = cluster_quality(Z, lab)
        rows.append({
            "eps": float(eps), "n_clusters": int(n_cl), "noise": noise,
            "silhouette": q["silhouette"], "db": q["db"],
            "ari_cooler": label_match(lab, truth["cooler"])["ari"],
        })
    return {
        "kdist": kdist, "eps_rule": eps_rule, "eps": eps_chosen,
        "min_samples": min_samples, "rows": rows,
    }


# ---------------------------------------------------------------------------
# Рисунки
# ---------------------------------------------------------------------------

def fig_signals(ex: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(6.9, 6.4), sharex=True)
    axes = axes.ravel()
    for ax, ch in zip(axes, CHANNEL_NAMES):
        y = ex[ch]
        t = np.arange(len(y)) / CHANNEL_RATE[ch]
        ax.plot(t, y, lw=0.7, color="#1f4e79")
        ax.set_title(f"{ch}: {CHANNEL_DESC[ch].lower()}, "
                     f"{CHANNEL_UNITS[ch]} ({CHANNEL_RATE[ch]} Гц)",
                     fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    for ax in axes[4:6]:
        ax.set_xlabel("Время цикла, с", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_signals.pdf")
    plt.close(fig)


def fig_correlation(X: np.ndarray, features: list[str]) -> None:
    corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [f"{f.split('_')[0]}·{STAT_TICK[f.split('_')[1]]}"
              for f in features]
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.2)
    ax.set_yticklabels(labels, fontsize=6.2)
    for b in _row_bounds(features):
        ax.axhline(b - 0.5, color="k", lw=0.6)
        ax.axvline(b - 0.5, color="k", lw=0.6)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Коэффициент корреляции Пирсона", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_correlation.pdf")
    plt.close(fig)


def fig_drift(cycles: np.ndarray, ts1: np.ndarray, cooler: np.ndarray,
              scores: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 4.8), sharex=True)
    cyc = cycles
    cool = cooler
    axes[0].scatter(cyc, ts1, s=3, c=[COOLER_COLORS[v] for v in cool])
    axes[0].set_ylabel("TS1, среднее за цикл, °C")
    handles = [Line2D([], [], marker="o", ls="", color=COOLER_COLORS[v],
                      label=f"{v} % ({COOLER_NAMES[v]})")
               for v in (3, 20, 100)]
    axes[0].legend(handles=handles, fontsize=7.2, loc="upper right")
    axes[1].scatter(cyc, scores[:, 0], s=3,
                    c=[COOLER_COLORS[v] for v in cool])
    axes[1].set_ylabel("PC1 (стандартизация)")
    axes[1].set_xlabel("Номер цикла")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_drift.pdf")
    plt.close(fig)


def fig_spectrum(pca_std: dict, pca_cent: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))
    for ax, d, title in ((axes[0], pca_std, "Стандартизация"),
                         (axes[1], pca_cent, "Только центрирование")):
        k = np.arange(1, len(d["evr"]) + 1)
        ax.bar(k, d["evr"], color="#4c72b0", width=0.65)
        ax.set_xlabel("Число компонент $k$")
        ax.set_ylabel("Доля объяснённой дисперсии")
        ax.set_xticks(k)
        ax.tick_params(labelsize=8)
        ax2 = ax.twinx()
        ax2.plot(k, d["cum"], color="#c44e52", marker="o", ms=3.2, lw=1.3)
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("Накопленная доля", fontsize=9)
        ax2.grid(False)
        ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_spectrum.pdf")
    plt.close(fig)


def fig_reconstruction(pca_std: dict, pca_cent: dict,
                       pca_std_bl: dict, pca_cent_bl: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))
    k = np.arange(1, N_PCA_MAX + 1)
    plots = [
        (pca_std["rmse"], "станд., хроно", "#1f77b4", "-"),
        (pca_std_bl["rmse"], "станд., блоки", "#1f77b4", "--"),
        (pca_cent["rmse"], "центр., хроно", "#ff7f0e", "-"),
        (pca_cent_bl["rmse"], "центр., блоки", "#ff7f0e", "--"),
    ]
    for y, lab, c, ls in plots:
        axes[0].plot(k, y, color=c, ls=ls, marker="o", ms=3, lw=1.2, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE восстановления")
    axes[0].legend(fontsize=6.6, loc="upper right")
    plots = [
        (pca_std["rrmse"], "станд., хроно", "#1f77b4", "-"),
        (pca_std_bl["rrmse"], "станд., блоки", "#1f77b4", "--"),
        (pca_cent["rrmse"], "центр., хроно", "#ff7f0e", "-"),
        (pca_cent_bl["rrmse"], "центр., блоки", "#ff7f0e", "--"),
    ]
    for y, lab, c, ls in plots:
        axes[1].plot(k, y, color=c, ls=ls, marker="o", ms=3, lw=1.2)
    axes[1].set_ylabel("Относительная RMSE")
    for ax in axes:
        ax.set_xlabel("Число компонент $k$")
        ax.set_xticks(k)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_reconstruction.pdf")
    plt.close(fig)


def _row_bounds(features: list[str]) -> list[int]:
    """Границы физических групп в списке признаков."""
    bounds = []
    prev = None
    for i, f in enumerate(features):
        g = FEATURE_GROUP[f]
        if prev is not None and g != prev:
            bounds.append(i)
        prev = g
    return bounds


def fig_loadings(pca_std: dict, fa: dict, features: list[str]) -> None:
    order = list(range(len(features)))
    pcaL = pca_std["loadings"][order, :4]
    faL = fa[3]["loadings"][order, :3]
    labels = [f"{f.split('_')[0]}·{STAT_TICK[f.split('_')[1]]}"
              for f in features]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 7.0),
                             gridspec_kw={"width_ratios": [4, 3],
                                          "wspace": 0.14})
    v0 = float(np.abs(pcaL).max())
    v1 = float(np.abs(faL).max())
    im0 = axes[0].imshow(pcaL, cmap="RdBu_r", vmin=-v0, vmax=v0,
                         aspect="auto")
    im1 = axes[1].imshow(faL, cmap="RdBu_r", vmin=-v1, vmax=v1,
                         aspect="auto")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7.4)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels([])
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["PC1", "PC2", "PC3", "PC4"], fontsize=8)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["F1", "F2", "F3"], fontsize=8)
    for ax in axes:
        for b in _row_bounds(features):
            ax.axhline(b - 0.5, color="k", lw=0.6)
        ax.grid(False)
    axes[0].set_title("Нагрузки PCA", fontsize=9.5)
    axes[1].set_title("Нагрузки FA (3 фактора)", fontsize=9.5)
    cb0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal",
                       fraction=0.05, pad=0.09)
    cb0.set_label("Нагрузка PC (своя шкала)", fontsize=7.5)
    cb0.ax.tick_params(labelsize=6.5)
    cb0.set_ticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    cb1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal",
                       fraction=0.05, pad=0.09)
    cb1.set_label("Нагрузка FA (своя шкала)", fontsize=7.5)
    cb1.ax.tick_params(labelsize=6.5)
    cb1.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    fig.savefig(FIG_DIR / "fig6_loadings.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_factor_analysis(fa: dict, features: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    labels = [f"{f.split('_')[0]}·{STAT_TICK[f.split('_')[1]]}"
              for f in features]
    x = np.arange(len(labels))
    width = 0.27
    for i, k in enumerate(FA_COMPONENTS):
        axes[0].bar(x + (i - 1) * width, fa[k]["noise"], width=width,
                    label=f"{k} фактора", color=CLUSTER_COLORS[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=90, fontsize=6.0)
    axes[0].set_ylabel("Шумовая дисперсия")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].tick_params(labelsize=7)

    ks = FA_COMPONENTS
    axes[1].plot(ks, [fa[k]["ll_train"] for k in ks], marker="o",
                 label="обучение", color="#1f77b4")
    axes[1].plot(ks, [fa[k]["ll_test"] for k in ks], marker="s",
                 label="тест (хроно)", color="#c44e52")
    axes[1].plot(ks, [fa[k]["ll_test_blocked"] for k in ks], marker="^",
                 label="тест (блоки)", color="#55a868")
    axes[1].set_xlabel("Число факторов")
    axes[1].set_ylabel("Среднее лог-правдоподобие")
    axes[1].set_xticks(ks)
    axes[1].legend(fontsize=6.6)
    axes[1].tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_fa.pdf")
    plt.close(fig)


def fig_k_selection(km: dict, ag: dict, best_k: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
    ks = [r["k"] for r in km["rows"]]
    axes[0].plot(ks, [r["silhouette"] for r in km["rows"]], marker="o",
                 label="KMeans", color="#1f77b4")
    axes[0].plot(ks, [r["silhouette"] for r in ag["rows"]], marker="s",
                 label="Ward", color="#ff7f0e")
    axes[0].set_xlabel("Число групп $k$")
    axes[0].set_ylabel("Коэффициент силуэта")
    axes[0].legend(fontsize=7.5)
    for rows, c, lab in ((km["rows"], "#1f77b4", "KMeans"),
                         (ag["rows"], "#ff7f0e", "Ward")):
        axes[1].errorbar([r["k"] for r in rows], [r["stab_mean"] for r in rows],
                         yerr=[r["stab_std"] for r in rows], marker="o",
                         capsize=2.5, color=c, label=lab, lw=1.2)
    axes[1].set_xlabel("Число групп $k$")
    axes[1].set_ylabel("Устойчивость, ARI")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7.5)
    for ax in axes:
        ax.axvline(best_k, color="k", ls=":", lw=0.9)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_k_selection.pdf")
    plt.close(fig)


def fig_kdist(db: dict) -> None:
    d = db["kdist"]
    fig, ax = plt.subplots(figsize=(4.9, 3.3))
    ax.plot(np.arange(1, len(d) + 1), d, color="#1f4e79", lw=1.2)
    ax.axhline(db["eps"], color="#c44e52", ls="--", lw=1.0,
               label=f"$\\varepsilon$ = {fmt(db['eps'], 1)}")
    ax.axhline(db["eps_rule"], color="#999999", ls=":", lw=0.9,
               label="90-й процентиль")
    ax.set_xlabel("Ранг точки")
    ax.set_ylabel(f"Расстояние до {db['min_samples']}-го соседа")
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_kdist.pdf")
    plt.close(fig)


def fig_dendrogram(Z4: np.ndarray, best_k: int) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    sub = rng.choice(len(Z4), size=80, replace=False)
    Zs = linkage(Z4[sub], method="ward")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    dendrogram(Zs, ax=axes[0], no_labels=True, color_threshold=0,
               above_threshold_color="#1f4e79")
    axes[0].set_xlabel("Наблюдения (80 циклов)")
    axes[0].set_ylabel("Расстояние объединения")
    axes[0].tick_params(labelsize=7)

    full = linkage(Z4, method="ward")
    dist = full[-60:, 2]
    kk = np.arange(len(Z4) - 59, len(Z4) + 1)
    axes[1].plot(kk, dist, marker="o", ms=2.6, color="#1f4e79", lw=1.0)
    axes[1].set_xlabel("Номер шага слияния")
    axes[1].set_ylabel("Расстояние объединения")
    axes[1].tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_dendrogram.pdf")
    plt.close(fig)


def _scatter_clusters(ax, Z2, lab, title, sizes=None):
    uniq = sorted(set(lab))
    for j, c in enumerate(uniq):
        m = lab == c
        if c == -1:
            ax.scatter(Z2[m, 0], Z2[m, 1], s=5, c=NOISE_COLOR, marker="x",
                       lw=0.6, label="шум")
        else:
            ax.scatter(Z2[m, 0], Z2[m, 1], s=5,
                       c=CLUSTER_COLORS[j % len(CLUSTER_COLORS)],
                       label=f"группа {c + 1}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.tick_params(labelsize=8)


def fig_clusters(Z2: np.ndarray, lab_km: np.ndarray, lab_ag: np.ndarray,
                 lab_db: np.ndarray, k: int, eps: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharex=True, sharey=True)
    _scatter_clusters(axes[0], Z2, lab_km, f"KMeans, k = {k}")
    _scatter_clusters(axes[1], Z2, lab_ag, f"Ward, k = {k}")
    _scatter_clusters(axes[2], Z2, lab_db,
                      f"DBSCAN, $\\varepsilon$ = {fmt(eps, 1)}")
    for ax in axes:
        ax.legend(fontsize=6, loc="best", markerscale=1.6, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_clusters.pdf")
    plt.close(fig)


def fig_label_plane(Z2: np.ndarray, lab_km: np.ndarray, lab_db: np.ndarray,
                    valve: np.ndarray, accum: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.4), sharex=True, sharey=True)
    _scatter_clusters(axes[0, 0], Z2, lab_km, "KMeans")
    _scatter_clusters(axes[0, 1], Z2, lab_db, "DBSCAN")
    for ax, vals, colors, names, title in (
            (axes[1, 0], valve, VALVE_COLORS, VALVE_NAMES, "Состояние клапана"),
            (axes[1, 1], accum, ACCUM_COLORS, ACCUM_NAMES,
             "Гидроаккумулятор")):
        for v in sorted(np.unique(vals)):
            m = vals == v
            ax.scatter(Z2[m, 0], Z2[m, 1], s=5, c=colors[v],
                       label=f"{v}: {names[v]}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=5.6, loc="best", markerscale=1.8, framealpha=0.9)
        ax.tick_params(labelsize=8)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.legend(fontsize=6, loc="best", markerscale=1.6, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_label_plane.pdf")
    plt.close(fig)


def fig_crosstab(cool: dict, valve: dict, accum: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
    for j, (ax, ct, names, title) in enumerate((
            (axes[0], cool, COOLER_NAMES, "Охладитель, %"),
            (axes[1], valve, VALVE_NAMES, "Клапан, %"),
            (axes[2], accum, ACCUM_NAMES, "Гидроаккумулятор, бар"))):
        counts = np.array(ct["counts"], dtype=float)
        share = counts / counts.sum(axis=1, keepdims=True)
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ct["cols"])))
        ax.set_xticklabels(ct["cols"], fontsize=7)
        ax.set_yticks(range(len(ct["rows"])))
        if j == 0:
            ax.set_yticklabels([f"группа {r + 1}" for r in ct["rows"]],
                               fontsize=7)
        for i in range(share.shape[0]):
            for k in range(share.shape[1]):
                v = share[i, k]
                ax.text(k, i, f"{int(counts[i, k])}", ha="center", va="center",
                        fontsize=6.5, color="w" if v > 0.55 else "k")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Уровень состояния", fontsize=8)
        ax.grid(False)
    fig.subplots_adjust(wspace=0.08)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="Доля строки")
    fig.savefig(FIG_DIR / "fig13_crosstab.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(sens: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    names = [s["name"] for s in sens]
    x = np.arange(len(names))
    axes[0].bar(x, [s["silhouette"] for s in sens], color="#4c72b0", width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("Коэффициент силуэта")
    w = 0.27
    for i, key, lab, c in ((0, "ari_cooler", "охладитель", "#1f77b4"),
                           (1, "ari_valve", "клапан", "#ff7f0e"),
                           (2, "ari_accum", "аккумулятор", "#2ca02c")):
        axes[1].bar(x + (i - 1) * w, [s[key] if s[key] is not None else 0
                                      for s in sens], width=w, color=c,
                    label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("ARI с состоянием узла")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig14_sensitivity.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------

def _table(name: str, lines: list[str]) -> None:
    (TAB_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _num(v, nd: int = 2, small: int = 3) -> str:
    """Число с запятой; для малых по модулю величин — больше знаков."""
    if v is None:
        return "---"
    return fmt(v, small if abs(float(v)) < 10 else nd)


def _sizes(sizes) -> str:
    return "/".join(str(int(v)) for v in sizes)


def make_tables(r: dict) -> None:
    a = r["audit"]
    sp = r["split"]
    pca = r["pca"]
    fa = r["fa"]
    cl = r["clustering"]
    fin = cl["final"]
    km = fin["kmeans"]
    dr = r["drift"]

    # --- 1. Аудит ----------------------------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Характеристика данных и результаты аудита}"
        r"\label{tab:data}",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Показатель & Значение \\", r"\midrule",
        f"Циклов (объектов) & {a['n_cycles']} \\\\",
        f"Каналов & {a['n_channels']} \\\\",
        f"Циклопризнаков (7 каналов $\\times$ 4) & {a['n_features']} \\\\",
        f"Постоянных признаков & {a['n_constant']} \\\\",
        f"Малоинформативных, вкл. постоянные & {a['n_near_constant']} \\\\",
        f"Признаков в анализе & {r['n_features_active']} \\\\",
        f"Пропусков & {a['missing']} \\\\",
        f"Полных дубликатов строк & {a['duplicates']} \\\\",
        f"Выбросов по правилу 1,5 IQR & {a['outlier_total']} "
        f"({fmt(100 * a['outlier_share'])}~\\%) \\\\",
        r"\midrule",
        f"Обучение: циклы & {sp['train_first']}--{sp['train_last']} "
        f"({sp['n_train']}) \\\\",
        f"Тест: циклы & {sp['test_first']}--{sp['test_last']} "
        f"({sp['n_test']}) \\\\",
        f"Блочное разбиение: обучение / тест & "
        f"{sp['n_train_blocked']} / {sp['n_test_blocked']} \\\\",
        f"Блок последовательных циклов & {sp['block_size']} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    _table("tab_data.tex", lines)

    # --- 2. Признаки -------------------------------------------------------
    med = a.get("median_by_feature", a["mean_by_feature"])
    lines = [
        r"\begin{table}[!t]", r"\centering", r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.93}",
        r"\caption{Циклопризнаки: единицы, диапазон, медиана, СКО, число "
        r"выбросов и участие в анализе}\label{tab:features}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Канал & Характеристика & Ед. & Мин. & Медиана & Макс. & СКО & "
        r"Выбросы \\", r"\midrule",
    ]
    prev = None
    for f in FEATURES:
        ch, st = f.split("_")
        if prev is not None and ch != prev:
            lines.append(r"\midrule")
        prev = ch
        marks = "" if f not in a["near_constant_features"] else r"$^{\ast}$"
        lines.append(
            f"{ch}{marks} & {STAT_SHORT[st]} & {CHANNEL_UNITS[ch]} & "
            f"{_num(a['min_by_feature'][f])} & {_num(med[f])} & "
            f"{_num(a['max_by_feature'][f])} & "
            f"{_num(a['std_by_feature'][f], small=4)} & "
            f"{a['outlier_by_feature'][f]} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}}",
        r"\par\smallskip{\scriptsize $^{\ast}$Признак исключён из анализа: "
        r"постоянный или почти постоянный.}",
        r"\end{table}",
    ]
    _table("tab_features.tex", lines)

    # --- 3. Метки состояний ------------------------------------------------
    names = {"cooler": COOLER_NAMES, "valve": VALVE_NAMES,
             "pump": PUMP_NAMES, "accum": ACCUM_NAMES,
             "stable": STABLE_NAMES}
    lines = [
        r"\begin{table}[!t]", r"\centering", r"\small",
        r"\caption{Уровни состояния узлов из \texttt{profile.txt} (в признаки "
        r"не входят)}\label{tab:labels}",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Показатель & Уровень & $n$ & \% \\", r"\midrule",
    ]
    for k in LABELS:
        lines.append(r"\multicolumn{4}{l}{\textit{" +
                     tex_escape(LABEL_NAMES[k]) + r"}} \\")
        counts = a["label_counts"][k]
        for v in sorted(int(x) for x in counts):
            lines.append(
                f" & {v}: {tex_escape(names[k][v])} & {counts[str(v)]} & "
                f"{fmt(100 * counts[str(v)] / a['n_cycles'])} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    _table("tab_labels.tex", lines)

    # --- 4. Спектр PCA -----------------------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Доли объяснённой дисперсии PCA при стандартизации и "
        r"только центрировании}\label{tab:pca}",
        r"\begin{tabular}{crrrr}", r"\toprule",
        r"$k$ & \multicolumn{2}{c}{Стандартизация} & "
        r"\multicolumn{2}{c}{Центрирование} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & EVR & Накопл. & EVR & Накопл. \\", r"\midrule",
    ]
    for i in range(N_PCA_MAX):
        lines.append(
            f"{i + 1} & {fmt(pca['standard']['evr'][i], 4)} & "
            f"{fmt(pca['standard']['cum'][i], 4)} & "
            f"{fmt(pca['centered']['evr'][i], 4)} & "
            f"{fmt(pca['centered']['cum'][i], 4)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _table("tab_pca.tex", lines)

    # --- 5. Ошибка восстановления -----------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\caption{Ошибка восстановления PCA на отложенных циклах: RMSE в "
        r"исходных единицах и относительная RMSE}\label{tab:recon}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{crrrrrr}", r"\toprule",
        r"$k$ & \multicolumn{3}{c}{Стандартизация} & "
        r"\multicolumn{3}{c}{Центрирование} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & RMSE & Отн. (хроно) & Отн. (блоки) & RMSE & Отн. (хроно) & "
        r"Отн. (блоки) \\", r"\midrule",
    ]
    for i in range(N_PCA_MAX):
        lines.append(
            f"{i + 1} & {_num(pca['standard']['rmse'][i])} & "
            f"{fmt(pca['standard']['rrmse'][i], 3)} & "
            f"{fmt(pca['standard_blocked']['rrmse'][i], 3)} & "
            f"{_num(pca['centered']['rmse'][i])} & "
            f"{fmt(pca['centered']['rrmse'][i], 3)} & "
            f"{fmt(pca['centered_blocked']['rrmse'][i], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _table("tab_recon.tex", lines)

    # --- 6. Факторный анализ ----------------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Факторный анализ: правдоподобие и остаточная структура}"
        r"\label{tab:fa}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{crrrrrr}", r"\toprule",
        r"Факторов & LL (обучение) & LL (хроно-тест) & LL (блочный тест) & "
        r"Средняя шум. дисп. & Средняя общность & Итераций \\",
        r"\midrule",
    ]
    for k in fa["components"]:
        v = fa["rows"].get(k, fa["rows"].get(str(k)))
        lines.append(
            f"{k} & {fmt(v['ll_train'])} & {fmt(v['ll_test'])} & "
            f"{fmt(v['ll_test_blocked'])} & {fmt(v['noise_mean'], 3)} & "
            f"{fmt(v['comm_mean'], 3)} & {v['n_iter']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _table("tab_fa.tex", lines)

    # --- 7. Перебор k ------------------------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\scriptsize",
        r"\caption{Перебор числа групп: размеры (по убыванию), силуэт, "
        r"Дэвис---Болдин, устойчивость и совпадение с состоянием охладителя}"
        r"\label{tab:ksweep}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{crrrrrrrrrr}", r"\toprule",
        r" & \multicolumn{5}{c}{KMeans} & \multicolumn{5}{c}{Ward} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        r"$k$ & Размеры & Силуэт & Д.--Б. & Устойч. & ARI охл. & Размеры & "
        r"Силуэт & Д.--Б. & Устойч. & ARI охл. \\", r"\midrule",
    ]
    for rk, ra in zip(cl["kmeans"]["rows"], cl["agglomerative"]["rows"]):
        lines.append(
            f"{rk['k']} & {_sizes(rk['sizes'])} & {fmt(rk['silhouette'], 3)} & "
            f"{fmt(rk['db'], 3)} & {fmt(rk['stab_mean'], 3)} & "
            f"{fmt(rk['ari_cooler'], 3)} & {_sizes(ra['sizes'])} & "
            f"{fmt(ra['silhouette'], 3)} & {fmt(ra['db'], 3)} & "
            f"{fmt(ra['stab_mean'], 3)} & {fmt(ra['ari_cooler'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _table("tab_ksweep.tex", lines)

    # --- 8. Сетка eps для DBSCAN ------------------------------------------
    ms = cl["dbscan"]["min_samples"]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Зависимость разбиения DBSCAN от радиуса "
        r"$\varepsilon$ при \texttt{min\_samples}~=~" + str(ms) +
        r"}\label{tab:dbscan}",
        r"\begin{tabular}{crrrr}", r"\toprule",
        r"$\varepsilon$ & Групп & Доля шума, \% & Силуэт & Д.--Б. \\",
        r"\midrule",
    ]
    for row in cl["dbscan"]["rows"]:
        e = fmt(row["eps"], 1)
        if abs(row["eps"] - cl["dbscan"]["eps"]) < 1e-9:
            e = r"\textbf{" + e + "}"
        sil = ("---" if row["silhouette"] is None
               else fmt(row["silhouette"], 3))
        dbi = "---" if row["db"] is None else fmt(row["db"], 3)
        lines.append(
            f"{e} & {row['n_clusters']} & {fmt(100 * row['noise'], 1)} & "
            f"{sil} & {dbi} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _table("tab_dbscan.tex", lines)

    # --- 9. Итоговое сравнение трёх алгоритмов -----------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\caption{Сравнение трёх алгоритмов на одной подвыборке "
        r"1~200 циклов и четырёх PCA-координатах}\label{tab:clust}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcrrrrrrr}", r"\toprule",
        r"Алгоритм & Групп & Размеры групп & Силуэт & Д.--Б. & Шум, \% & "
        r"ARI охл. & ARI клап. & ARI акк. \\", r"\midrule",
    ]
    rows = [("KMeans", km), ("Ward", fin["agglomerative"]),
            ("DBSCAN", fin["dbscan"])]
    for name, v in rows:
        prof = v["profile"]
        sizes = "/".join(str(p["size"]) for p in prof)
        stab = v.get("stab")
        stab_txt = fmt(stab[0], 2) if isinstance(stab, (list, tuple)) else "---"
        lines.append(
            f"{name} & {len(prof)} & {sizes} & "
            f"{fmt(v['quality']['silhouette'], 3)} & "
            f"{fmt(v['quality']['db'], 3)} & "
            f"{fmt(100 * v.get('noise', 0.0), 1)} & "
            f"{fmt(v['match']['cooler']['ari'], 3)} & "
            f"{fmt(v['match']['valve']['ari'], 3)} & "
            f"{fmt(v['match']['accum']['ari'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _table("tab_clust.tex", lines)

    # --- 10. Согласие с метками -------------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Согласие найденных групп с состояниями узлов: ARI / NMI "
        r"(для DBSCAN без точек шума)}\label{tab:match}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Состояние & KMeans & Ward & DBSCAN \\", r"\midrule",
    ]
    for k, name in (("cooler", "Охладитель"), ("valve", "Клапан"),
                    ("pump", "Насос"), ("accum", "Гидроаккумулятор")):
        cells = []
        for alg in ("kmeans", "agglomerative", "dbscan"):
            mm = fin[alg]["match"][k]
            cells.append(f"{fmt(mm['ari'], 3)} / {fmt(mm['nmi'], 3)}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _table("tab_match.tex", lines)

    # --- 11. Дрейф по третям ----------------------------------------------
    fm = dr["feature_means"]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Ранняя, средняя и поздняя трети кампании (доли групп — "
        r"по подвыборке 1~200 циклов)}"
        r"\label{tab:drift}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Показатель & Ранняя & Средняя & Поздняя \\", r"\midrule",
        r"Циклы & " + " & ".join(f"{x[0]}--{x[1]}"
                                 for x in dr["cycle_range"]) + r" \\",
        r"\midrule",
    ]
    for f, title in (("TS1_mean", r"TS1, среднее, $^\circ$C"),
                     ("VS1_mean", "VS1, среднее, мм/с"),
                     ("EPS1_mean", "EPS1, среднее, Вт"),
                     ("PS1_mean", "PS1, среднее, бар")):
        lines.append(f"{title} & " + " & ".join(
            fmt(v, 3) for v in fm[f]) + r" \\")
    lines.append(r"PC1, среднее & " + " & ".join(
        fmt(v) for v in dr["pc1_mean"]) + r" \\")
    lines.append(r"\midrule")
    for j in range(len(km["profile"])):
        lines.append(f"Доля группы {j + 1} (KMeans), \\% & " + " & ".join(
            fmt(100 * row[j], 1) for row in dr["km_composition"]) + r" \\")
    lines.append(r"\midrule")
    for i, v in enumerate((3, 20, 100)):
        lines.append(f"Доля охладителя {v}, \\% & " + " & ".join(
            fmt(100 * row[i], 1) for row in dr["label_share"]["cooler"])
            + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _table("tab_drift.tex", lines)

    # --- 12. Чувствительность к масштабу ----------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Чувствительность KMeans ($k = 3$) к выбору координат и "
        r"масштабированию}\label{tab:sensitivity}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Вариант координат & Силуэт & ARI охл. & ARI клап. & ARI акк. \\",
        r"\midrule",
    ]
    for s in r["sensitivity"]:
        lines.append(
            f"{tex_escape(s['name'])} & {fmt(s['silhouette'], 3)} & "
            f"{fmt(s['ari_cooler'], 3)} & {fmt(s['ari_valve'], 3)} & "
            f"{fmt(s['ari_accum'], 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _table("tab_sensitivity.tex", lines)

    # --- 13. Профили групп -------------------------------------------------
    lines = [
        r"\begin{table}[H]", r"\centering", r"\scriptsize",
        r"\caption{Профили итоговых групп: размер, положение во времени и "
        r"состав по охладителю}\label{tab:profile}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Алгоритм & Группа & $n$ & Циклы & Медиана & "
        r"Неустан., \% & Охл. 3, \% & Охл. 20, \% & Охл. 100, \% \\",
        r"\midrule",
    ]
    for alg, title in (("kmeans", "KMeans"), ("agglomerative", "Ward"),
                       ("dbscan", "DBSCAN")):
        for p in fin[alg]["profile"]:
            lines.append(
                f"{title} & {p['cluster'] + 1} & {p['size']} & "
                f"{p['cycle_min']}--{p['cycle_max']} & {p['cycle_med']} & "
                f"{fmt(100 * p['stable'], 1)} & "
                f"{fmt(100 * p['cooler']['3'], 1)} & "
                f"{fmt(100 * p['cooler']['20'], 1)} & "
                f"{fmt(100 * p['cooler']['100'], 1)} \\\\")
        if alg != "dbscan":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _table("tab_profile.tex", lines)

    # --- 14. Сопряжённость (приложение) -----------------------------------
    ct = cl["ct_kmeans"]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Матрицы сопряжённости групп KMeans и состояний узлов "
        r"(число циклов подвыборки)}\label{tab:crosstab}",
    ]
    for j, (key, title) in enumerate((("cooler", "Охладитель, \\%"),
                                      ("valve", "Клапан, \\%"),
                                      ("accum", "Гидроаккумулятор, бар"))):
        block = ct[key]
        if j:
            lines.append(r"\par\medskip")
        lines.append(r"\begin{center}\textit{" + title + r"}\end{center}")
        lines.append(r"\begin{tabular}{l" + "r" * len(block["cols"]) + "r}")
        lines.append(r"\toprule")
        lines.append("Группа & " + " & ".join(str(c) for c in block["cols"])
                     + r" & Всего \\")
        lines.append(r"\midrule")
        for i, row in enumerate(block["counts"]):
            cells = " & ".join(str(v) for v in row)
            lines.append(f"{block['rows'][i] + 1} & {cells} & "
                         f"{sum(row)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    _table("tab_crosstab.tex", lines)


# ---------------------------------------------------------------------------
# Макросы
# ---------------------------------------------------------------------------

DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three",
               "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
               "8": "Eight", "9": "Nine"}


def _macro_name(name: str) -> str:
    """Имя макроса без цифр (в LaTeX цифра завершает имя команды)."""
    return "".join(DIGIT_WORDS.get(ch, ch) for ch in name)


def make_macros(r: dict) -> None:
    a = r["audit"]
    sp = r["split"]
    pca = r["pca"]
    fa = r["fa"]
    cl = r["clustering"]
    fin = cl["final"]
    dr = r["drift"]
    km, ag, db = fin["kmeans"], fin["agglomerative"], fin["dbscan"]
    kprof, dprof = km["profile"], db["profile"]
    fa_rows = fa["rows"]

    n_fig = len(list(FIG_DIR.glob("*.pdf")))
    n_tab = len([p for p in TAB_DIR.glob("*.tex") if p.name != "macros.tex"])
    m = [("NSrc", "11"), ("NApp", "2"), ("NFig", str(n_fig)),
         ("NTab", str(n_tab))]
    m += [
        ("DataCycles", str(a["n_cycles"])),
        ("DataChannels", str(a["n_channels"])),
        ("DataFeaturesAll", str(a["n_features"])),
        ("DataFeaturesUsed", str(r["n_features_active"])),
        ("DataConstant", str(a["n_constant"])),
        ("DataNearConstant", str(a["n_near_constant"])),
        ("DataConstantList", tex_escape(", ".join(a["constant_features"]))),
        ("DataNearConstantList",
         tex_escape(", ".join(a["near_constant_features"]))),
        ("DataMissing", str(a["missing"])),
        ("DataDuplicates", str(a["duplicates"])),
        ("DataOutliers", str(a["outlier_total"])),
        ("DataOutlierShare", fmt(100 * a["outlier_share"])),
        ("DataTrain", str(sp["n_train"])),
        ("DataTest", str(sp["n_test"])),
        ("DataTrainFirst", str(sp["train_first"])),
        ("DataTrainLast", str(sp["train_last"])),
        ("DataTestFirst", str(sp["test_first"])),
        ("DataTestLast", str(sp["test_last"])),
        ("BlockTrain", str(sp["n_train_blocked"])),
        ("BlockTest", str(sp["n_test_blocked"])),
        ("BlockSize", str(sp["block_size"])),
        ("ValvePerLevel", str(a["label_counts"]["valve"]["73"])),
        ("ValveOptimal", str(a["label_counts"]["valve"]["100"])),
        ("CoolerPerLevel", str(a["label_counts"]["cooler"]["3"])),
        ("StableOne", str(a["label_counts"]["stable"]["1"])),
        ("CoolerLast", str(a["label_counts"]["cooler"]["100"])),
    ]

    evr = pca["standard"]["evr"]
    cum = pca["standard"]["cum"]
    cencum = pca["centered"]["cum"]
    for i, name in enumerate(("One", "Two", "Three", "Four", "Five", "Six")):
        m.append((f"PcaEvr{name}", fmt(evr[i], 4)))
    for i, name in ((1, "Two"), (2, "Three"), (3, "Four"), (4, "Five"),
                    (5, "Six"), (9, "Ten")):
        m.append((f"PcaCum{name}", fmt(cum[i], 4)))
    m += [
        ("PcaCentEvrOne", fmt(pca["centered"]["evr"][0], 4)),
        ("PcaCentCumTwo", fmt(cencum[1], 4)),
        ("PcaCentCumFour", fmt(cencum[3], 4)),
        ("PcaRmseStdTwo", _num(pca["standard"]["rmse"][1])),
        ("PcaRelStdTwo", fmt(pca["standard"]["rrmse"][1], 3)),
        ("PcaRelStdFour", fmt(pca["standard"]["rrmse"][3], 3)),
        ("PcaRelStdSix", fmt(pca["standard"]["rrmse"][5], 3)),
        ("PcaRelStdTen", fmt(pca["standard"]["rrmse"][9], 3)),
        ("PcaRmseCentTwo", _num(pca["centered"]["rmse"][1])),
        ("PcaRelCentTwo", fmt(pca["centered"]["rrmse"][1], 3)),
        ("PcaRelCentTen", fmt(pca["centered"]["rrmse"][9], 3)),
        ("PcaRelStdTwoBlocked", fmt(pca["standard_blocked"]["rrmse"][1], 3)),
        ("PcaRelCentTwoBlocked", fmt(pca["centered_blocked"]["rrmse"][1], 3)),
        ("PcaClusterShare", fmt(pca["cluster_share"], 4)),
        ("PcaPcOneCorr", fmt(pca["pc1_corr"], 3)),
    ]

    for k, name in ((2, "Two"), (3, "Three"), (5, "Five")):
        v = fa_rows.get(k, fa_rows.get(str(k)))
        m += [
            (f"FaLlTrain{name}", fmt(v["ll_train"])),
            (f"FaLlTest{name}", fmt(v["ll_test"])),
            (f"FaLlBlocked{name}", fmt(v["ll_test_blocked"])),
            (f"FaNoise{name}", fmt(v["noise_mean"], 3)),
            (f"FaComm{name}", fmt(v["comm_mean"], 3)),
            (f"FaIter{name}", str(v["n_iter"])),
        ]
    m.append(("FaBestK", str(fa["best_test_k"])))

    kmrow = {row["k"]: row for row in cl["kmeans"]["rows"]}
    m += [
        ("ClustN", str(cl["n_obj"])),
        ("ClustShareOfAll", fmt(100 * cl["n_obj"] / r["n_cycles"], 1)),
        ("ClustPcs", str(cl["n_pcs"])),
        ("ClustShare", fmt(cl["pca_share"], 4)),
        ("ClustSharePct", fmt(100 * cl["pca_share"], 1)),
        ("RuleStab", "0,90"),
        ("RuleShare", "5"),
        ("RuleSil", "95"),
        ("KmK", str(cl["kmeans"]["best_k"])),
        ("KmSil", fmt(km["quality"]["silhouette"], 3)),
        ("KmDb", fmt(km["quality"]["db"], 3)),
        ("KmStab", fmt(km["stab"][0], 2)),
        ("KmAriCooler", fmt(km["match"]["cooler"]["ari"], 3)),
        ("KmAriValve", fmt(km["match"]["valve"]["ari"], 3)),
        ("KmAriAccum", fmt(km["match"]["accum"]["ari"], 3)),
        ("KmAriPump", fmt(km["match"]["pump"]["ari"], 3)),
        ("KmAriAg", fmt(km["ari_km_ag"], 3)),
        ("KmAriDb", fmt(km["ari_km_db"], 3)),
        ("KmSizeOne", str(kprof[0]["size"])),
        ("KmSizeTwo", str(kprof[1]["size"])),
        ("KmSizeThree", str(kprof[2]["size"])),
        ("KmCycleOne", f"{kprof[0]['cycle_min']}--{kprof[0]['cycle_max']}"),
        ("KmCycleTwo", f"{kprof[1]['cycle_min']}--{kprof[1]['cycle_max']}"),
        ("KmCycleThree", f"{kprof[2]['cycle_min']}--{kprof[2]['cycle_max']}"),
        ("KmStableOne", fmt(100 * kprof[0]["stable"], 1)),
        ("KmStableTwo", fmt(100 * kprof[1]["stable"], 1)),
        ("KmStableThree", fmt(100 * kprof[2]["stable"], 1)),
        ("WardSil", fmt(ag["quality"]["silhouette"], 3)),
        ("WardDb", fmt(ag["quality"]["db"], 3)),
        ("WardStab", fmt(ag["stab"][0], 2)),
        ("WardAriCooler", fmt(ag["match"]["cooler"]["ari"], 3)),
        ("WardAriValve", fmt(ag["match"]["valve"]["ari"], 3)),
        ("WardAriAccum", fmt(ag["match"]["accum"]["ari"], 3)),
        ("WardAriPump", fmt(ag["match"]["pump"]["ari"], 3)),
        ("KFourSil", fmt(kmrow[4]["silhouette"], 3)),
        ("KFourAriCooler", fmt(kmrow[4]["ari_cooler"], 3)),
        ("KFiveSil", fmt(kmrow[5]["silhouette"], 3)),
        ("KSixSil", fmt(kmrow[6]["silhouette"], 3)),
        ("KSixSizeMin", str(min(kmrow[6]["sizes"]))),
        ("DbEps", fmt(db["eps"], 1)),
        ("DbEpsRule", fmt(cl["dbscan"]["eps_rule"], 3)),
        ("DbMinSamples", str(db["min_samples"])),
        ("DbNClusters", str(db["n_clusters"])),
        ("DbNoise", fmt(100 * db["noise"], 1)),
        ("DbSil", fmt(db["quality"]["silhouette"], 3)),
        ("DbDb", fmt(db["quality"]["db"], 3)),
        ("DbAriCooler", fmt(db["match"]["cooler"]["ari"], 3)),
        ("DbAriValve", fmt(db["match"]["valve"]["ari"], 3)),
        ("DbAriAccum", fmt(db["match"]["accum"]["ari"], 3)),
        ("DbSizeOne", str(dprof[0]["size"])),
        ("DbSizeTwo", str(dprof[1]["size"])),
        ("DbSizeThree", str(dprof[2]["size"])),
        ("DbSizeFour", str(dprof[3]["size"])),
        ("DbMsFiveNoise",
         fmt(100 * cl["dbscan"]["min_samples_sensitivity"]["5"]["noise"], 1)),
        ("DbMsTwentyNoise",
         fmt(100 * cl["dbscan"]["min_samples_sensitivity"]["20"]["noise"], 1)),
        ("KmSilTwo", fmt(kmrow[2]["silhouette"], 3)),
        ("KFiveSizeMin", str(min(kmrow[5]["sizes"]))),
    ]
    for i, name in enumerate(("One", "Two", "Three", "Four")):
        p = dprof[i]
        m += [
            (f"DbStable{name}", fmt(100 * p["stable"], 1)),
            (f"DbCycle{name}", f"{p['cycle_min']}--{p['cycle_max']}"),
        ]
    agprof = ag["profile"]
    for i, name in enumerate(("One", "Two", "Three")):
        m.append((f"WardSize{name}", str(agprof[i]["size"])))
    hp = a["high_pressure"]
    m += [
        ("DataHighPressN", str(hp["n"])),
        ("DataHighPressShare", fmt(100 * hp["n"] / a["n_cycles"], 1)),
        ("DataHighPressRange", f"{hp['first']}--{hp['last']}"),
        ("DataHighPressSub", str(hp["n_sub"])),
        ("HighFlagPs1", fmt(hp["means"]["PS1_mean"][0], 1)),
        ("HighRestPs1", fmt(hp["means"]["PS1_mean"][1], 1)),
        ("HighFlagFs1", fmt(hp["means"]["FS1_mean"][0], 2)),
        ("HighRestFs1", fmt(hp["means"]["FS1_mean"][1], 2)),
        ("HighFlagTs1", fmt(hp["means"]["TS1_mean"][0], 1)),
        ("HighRestTs1", fmt(hp["means"]["TS1_mean"][1], 1)),
        ("HighFlagPs3", fmt(hp["means"]["PS3_mean"][0], 2)),
        ("HighRestPs3", fmt(hp["means"]["PS3_mean"][1], 2)),
        ("HighFlagEps1", fmt(hp["means"]["EPS1_mean"][0], 0)),
        ("HighRestEps1", fmt(hp["means"]["EPS1_mean"][1], 0)),
    ]
    ct_cool = km["ct"]["cooler"]["counts"]
    m += [
        ("KmG2CoolerThree", str(ct_cool[1][0])),
        ("KmG2CoolerTwenty", str(ct_cool[1][1])),
        ("KmG2CoolerHundred", str(ct_cool[1][2])),
        ("KmG1CoolerThree", str(ct_cool[0][0])),
        ("KmG3CoolerHundred", str(ct_cool[2][2])),
    ]

    fm = dr["feature_means"]
    for i, name in ((0, "Early"), (1, "Mid"), (2, "Late")):
        m += [
            (f"DriftTs1{name}", fmt(fm["TS1_mean"][i], 1)),
            (f"DriftVs1{name}", fmt(fm["VS1_mean"][i], 3)),
            (f"DriftEps1{name}", fmt(fm["EPS1_mean"][i], 0)),
            (f"DriftPs1{name}", fmt(fm["PS1_mean"][i], 1)),
            (f"DriftPc1{name}", fmt(dr["pc1_mean"][i], 2)),
        ]
    m += [
        ("DriftPc1Corr", fmt(dr["pc1_corr"], 3)),
        ("DriftPc1Spearman", fmt(dr["pc1_spearman"], 3)),
        ("DriftTopFeature", tex_escape(dr["top_corr"][0][0])),
        ("DriftTopCorr", fmt(dr["top_corr"][0][1], 3)),
        ("DriftSecondFeature", tex_escape(dr["top_corr"][1][0])),
        ("DriftSecondCorr", fmt(dr["top_corr"][1][1], 3)),
        ("DriftCoolerThreeEarly",
         fmt(100 * dr["label_share"]["cooler"][0][0], 1)),
        ("DriftCoolerTwentyMid",
         fmt(100 * dr["label_share"]["cooler"][1][1], 1)),
        ("DriftValveEach",
         fmt(100 * dr["label_share"]["valve"][0][0], 1)),
        ("DriftValveOptimal",
         fmt(100 * dr["label_share"]["valve"][0][3], 1)),
        ("DriftCoolerHundredLate",
         fmt(100 * dr["label_share"]["cooler"][2][2], 1)),
    ]
    for j in range(3):
        for i, part in enumerate(("Early", "Mid", "Late")):
            m.append((f"DriftGroup{j + 1}{part}",
                      fmt(100 * dr["km_composition"][i][j], 1)))
    for i, part in enumerate(("Early", "Mid", "Late")):
        m.append((f"DriftAccumNinety{part}",
                  fmt(100 * dr["label_share"]["accum"][i][0], 1)))

    # нагрузки, на которые есть ссылки в тексте
    feat = r["features_used"]
    Ls = pca["loadings_standard"]
    Lc = pca["loadings_centered"]
    fa3 = fa["loadings"].get(3, fa["loadings"].get("3"))
    fa2 = fa["loadings"].get(2, fa["loadings"].get("2"))
    nz3 = fa["noise"].get(3, fa["noise"].get("3"))
    nz2 = fa["noise"].get(2, fa["noise"].get("2"))

    def _ld(L, f, j):
        return L[feat.index(f)][j]

    m += [
        ("PcaLoadVs1", fmt(_ld(Ls, "VS1_mean", 0), 2)),
        ("PcaLoadPs3", fmt(_ld(Ls, "PS3_mean", 0), 2)),
        ("PcaLoadPs1Std", fmt(_ld(Ls, "PS1_std", 0), 2)),
        ("PcaLoadEps1Std", fmt(_ld(Ls, "EPS1_std", 0), 2)),
        ("PcaLoadTs1", fmt(_ld(Ls, "TS1_mean", 0), 2)),
        ("PcaLoadFs1", fmt(_ld(Ls, "FS1_mean", 0), 2)),
        ("PcaLoad2Eps1", fmt(_ld(Ls, "EPS1_mean", 1), 2)),
        ("PcaLoad2Ps1", fmt(_ld(Ls, "PS1_mean", 1), 2)),
        ("PcaLoad2Ps2", fmt(_ld(Ls, "PS2_mean", 1), 2)),
        ("PcaLoad2Ts1", fmt(_ld(Ls, "TS1_mean", 1), 2)),
        ("PcaLoad2Vs1Std", fmt(_ld(Ls, "VS1_std", 1), 2)),
        ("PcaCentLoadEps1", fmt(_ld(Lc, "EPS1_mean", 0), 3)),
        ("FaLoadTs1", fmt(_ld(fa3, "TS1_mean", 0), 2)),
        ("FaLoadEps1Min", fmt(_ld(fa3, "EPS1_min", 0), 2)),
        ("FaLoadEps1Max", fmt(_ld(fa3, "EPS1_max", 0), 2)),
        ("FaLoad2Eps1", fmt(_ld(fa2, "EPS1_mean", 1), 2)),
        ("FaLoad2Ps1", fmt(_ld(fa2, "PS1_mean", 1), 2)),
        ("FaLoad2Ps2", fmt(_ld(fa2, "PS2_mean", 1), 2)),
        ("FaLoad2Fs1", fmt(_ld(fa2, "FS1_mean", 1), 2)),
        ("FaNoiseTs1Std", fmt(nz3[feat.index("TS1_std")], 2)),
        ("FaNoisePs3Max", fmt(nz3[feat.index("PS3_max")], 2)),
        ("FaNoiseFs1Max", fmt(nz3[feat.index("FS1_max")], 2)),
        ("FaNoiseVs1Std", fmt(nz3[feat.index("VS1_std")], 2)),
        ("DbAriPump", fmt(db["match"]["pump"]["ari"], 3)),
        ("KmNmiCooler", fmt(km["match"]["cooler"]["nmi"], 3)),
        ("WardNmiCooler", fmt(ag["match"]["cooler"]["nmi"], 3)),
        ("DbNmiCooler", fmt(db["match"]["cooler"]["nmi"], 3)),
        ("NmiValveMax", fmt(max(fin[a]["match"]["valve"]["nmi"]
                               for a in ("kmeans", "agglomerative",
                                         "dbscan")), 3)),
        ("DbNoiseSmall", fmt(100 * cl["dbscan"]["rows"][0]["noise"], 1)),
    ]

    sens_names = ["PcFourZ", "PcFourRaw", "PcTwoZ", "RawZ", "RawRaw",
                  "PcFourCentZ"]
    for name, s in zip(sens_names, r["sensitivity"]):
        m += [(f"SensSil{name}", fmt(s["silhouette"], 3)),
              (f"SensAriCooler{name}", fmt(s["ari_cooler"], 3))]

    lines = ["% Автоматически сгенерировано src/modeling.py; не редактировать.",
             ""]
    for name, value in m:
        # В имени команды LaTeX допускаются только буквы: цифра завершает
        # имя (\HighFlagPs1 читается как \HighFlagPs и текст «1»). Поэтому
        # цифры в именах макросов заменяются словами.
        lines.append(f"\\newcommand{{\\{_macro_name(name)}}}{{{value}}}")
    (TAB_DIR / "macros.tex").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


# ---------------------------------------------------------------------------
# Основной расчёт
# ---------------------------------------------------------------------------

def run_analysis() -> tuple[dict, dict]:
    """Считает все результаты; возвращает (results, контекст для рисунков)."""
    build_aggregates()
    Xdf, prof, cycles = load_dataset()
    X = Xdf.to_numpy(dtype=float)
    n = len(X)
    tr, te = chronological_split(n)
    btr, bte = blocked_split(n)
    truth = {k: prof[k].to_numpy() for k in LABELS}

    audit_res = audit(Xdf, prof)
    # Постоянные и почти постоянные признаки не несут информации, а после
    # стандартизации дают огромные z-оценки и делают ковариацию в FA
    # вырожденной, поэтому из анализа они исключаются.
    drop = set(audit_res["near_constant_features"])
    active = [j for j, f in enumerate(FEATURES) if f not in drop]
    feat_names = [FEATURES[j] for j in active]
    Xa = X[:, active]

    pca_std = pca_analysis(Xa, tr, te, "standard")
    pca_cent = pca_analysis(Xa, tr, te, "centered")
    pca_std_bl = pca_analysis(Xa, btr, bte, "standard")
    pca_cent_bl = pca_analysis(Xa, btr, bte, "centered")

    fa = fa_analysis(Xa, tr, te)
    fa_bl = fa_analysis(Xa, btr, bte)
    for k in fa:
        fa[k]["ll_test_blocked"] = fa_bl[k]["ll_test"]
        fa[k]["n_iter_blocked"] = fa_bl[k]["n_iter"]

    # --- кластеризация: одна подвыборка, одни и те же PCA-координаты --------
    scores = pca_std["scores"]
    sd = scores[tr].std(axis=0)
    sd[sd < 1e-12] = 1.0
    Zs = scores / sd
    Z4all = Zs[:, :CLUSTER_PCS]
    idx = cluster_subsample(n, CLUSTER_MAX_OBJ, RANDOM_STATE)
    # Аудит «выбросов»: у средних давлений и расхода 1,5-IQR отмечает один и
    # тот же непрерывный участок кампании. Проверяем, что это не разброс,
    # а другой рабочий режим, и что он же становится отдельной группой.
    ps1 = X[:, FEATURES.index("PS1_mean")]
    q1, q3 = np.percentile(ps1, [25, 75])
    hp = ps1 > q3 + 1.5 * (q3 - q1)
    audit_res["high_pressure"] = {
        "n": int(hp.sum()),
        "first": int(cycles[hp].min()),
        "last": int(cycles[hp].max()),
        "n_sub": int(hp[idx].sum()),
        "means": {f: [float(X[hp, FEATURES.index(f)].mean()),
                      float(X[~hp, FEATURES.index(f)].mean())]
                  for f in ("PS1_mean", "PS2_mean", "PS3_mean",
                            "FS1_mean", "TS1_mean", "EPS1_mean")},
    }
    Z4 = Z4all[idx]
    truth_sub = {k: truth[k][idx] for k in LABELS}

    km = sweep_partition(Z4, "kmeans", truth_sub)
    ag = sweep_partition(Z4, "agglomerative", truth_sub)
    best_k = select_k(km["rows"], len(idx))
    km_model = KMeans(n_clusters=best_k, n_init=20,
                      random_state=RANDOM_STATE).fit(Z4)
    lab_km = km_model.labels_
    ag_model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    lab_ag = ag_model.fit_predict(Z4)

    db = dbscan_sweep(Z4, truth_sub, DBSCAN_MIN_SAMPLES)
    db_model = DBSCAN(eps=db["eps"], min_samples=DBSCAN_MIN_SAMPLES).fit(Z4)
    lab_db = db_model.labels_
    lab_km, map_km = _relabel_by_median(lab_km, idx, cycles)
    lab_ag, map_ag = _relabel_by_median(lab_ag, idx, cycles)
    lab_db, map_db = _relabel_by_median(lab_db, idx, cycles)
    db.update({
        "sizes": sorted(np.bincount(lab_db[lab_db != -1]).tolist(),
                        reverse=True),
        "n_clusters": int(len(set(lab_db)) - (1 if -1 in lab_db else 0)),
        "noise": float(np.mean(lab_db == -1)),
    })
    q = cluster_quality(Z4, lab_db)
    db["silhouette"], db["db"] = q["silhouette"], q["db"]
    db["match"] = {k: label_match(lab_db, truth_sub[k])
                   for k in ("cooler", "valve", "pump", "accum")}
    # чувствительность DBSCAN к min_samples при выбранном eps
    ms_noise = {}
    for ms in (5, 10, 20):
        l = DBSCAN(eps=db["eps"], min_samples=ms).fit_predict(Z4)
        ms_noise[str(ms)] = {"noise": float(np.mean(l == -1)),
                        "n_clusters": int(len(set(l)) - (1 if -1 in l else 0))}
    db["min_samples_sensitivity"] = ms_noise

    final = {
        "kmeans": {"k": best_k,
                   "sizes": sorted(np.bincount(lab_km).tolist(), reverse=True),
                   "quality": cluster_quality(Z4, lab_km),
                   "stab": stability_score(Z4, best_k, lab_km, "kmeans"),
                   "match": {k: label_match(lab_km, truth_sub[k])
                             for k in ("cooler", "valve", "pump", "accum")},
                   "ari_km_ag": float(adjusted_rand_score(lab_km, lab_ag)),
                   "ari_km_db": float(adjusted_rand_score(
                       lab_km[lab_db != -1], lab_db[lab_db != -1])),
                   "ct": {k: contingency(lab_km, truth_sub[k], levels)
                          for k, levels in (("cooler", [3, 20, 100]),
                                            ("valve", [73, 80, 90, 100]),
                                            ("accum", [90, 100, 115, 130]),
                                            ("pump", [0, 1, 2]))},
                   },
        "agglomerative": {"k": best_k,
                          "sizes": sorted(np.bincount(lab_ag).tolist(),
                                          reverse=True),
                          "quality": cluster_quality(Z4, lab_ag),
                          "stab": stability_score(Z4, best_k, lab_ag,
                                                  "agglomerative"),
                          "match": {k: label_match(lab_ag, truth_sub[k])
                                    for k in ("cooler", "valve", "pump",
                                              "accum")}},
        "dbscan": {"eps": db["eps"], "min_samples": db["min_samples"],
                   "n_clusters": db["n_clusters"], "noise": db["noise"],
                   "sizes": db["sizes"], "quality": q,
                   "match": db["match"], "profile": None},
    }
    final["kmeans"]["predict_all"] = np.array(
        [map_km.get(int(v), -1) for v in km_model.predict(Z4all)])
    final["kmeans"]["profile"] = cluster_profile(lab_km, idx, cycles, prof)
    final["agglomerative"]["profile"] = cluster_profile(lab_ag, idx, cycles, prof)
    final["dbscan"]["profile"] = cluster_profile(lab_db, idx, cycles, prof)
    final["dbscan"]["ct"] = {k: contingency(lab_db, truth_sub[k], levels)
                             for k, levels in (("cooler", [3, 20, 100]),
                                               ("valve", [73, 80, 90, 100]),
                                               ("accum", [90, 100, 115, 130]))}

    # --- чувствительность к масштабу ---------------------------------------
    sc_raw = StandardScaler().fit(Xa[tr])
    raw28 = sc_raw.transform(Xa)[idx]
    cent_scores = pca_cent["scores"]
    sd_c = cent_scores[tr].std(axis=0)
    sd_c[sd_c < 1e-12] = 1.0
    variants = [
        ("PCA-4 (z)", Zs[idx, :4]),
        ("PCA-4 (без z)", scores[idx, :4]),
        ("PCA-2 (z)", Zs[idx, :2]),
        (f"{len(feat_names)} призн. (z)", raw28),
        (f"{len(feat_names)} призн. (без z)", Xa[idx]),
        ("PCA-4 центр. (z)", (cent_scores / sd_c)[idx, :4]),
    ]
    sens = []
    for name, Zv in variants:
        m = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
        lab = m.fit_predict(Zv)
        qq = cluster_quality(Zv, lab)
        sens.append({
            "name": name,
            "silhouette": qq["silhouette"],
            "ari_cooler": label_match(lab, truth_sub["cooler"])["ari"],
            "ari_valve": label_match(lab, truth_sub["valve"])["ari"],
            "ari_accum": label_match(lab, truth_sub["accum"])["ari"],
        })

    # --- ранняя / средняя / поздняя части кампании -------------------------
    parts = np.array_split(np.arange(n), 3)
    part_names = ["ранняя", "средняя", "поздняя"]
    drift_feats = ["PS1_mean", "TS1_mean", "VS1_mean", "EPS1_mean", "FS1_mean"]
    drift = {
        "parts": part_names,
        "cycle_range": [[int(p[0] + 1), int(p[-1] + 1)] for p in parts],
        "feature_means": {f: [float(X[p, FEATURES.index(f)].mean())
                              for p in parts] for f in drift_feats},
        "pc1_mean": [float(scores[p, 0].mean()) for p in parts],
        "pc1_corr": float(np.corrcoef(cycles, scores[:, 0])[0, 1]),
        "pc1_spearman": float(spearmanr(cycles, scores[:, 0]).statistic),
        "km_composition": [
            [float(np.mean(lab_km[np.isin(idx, p)] == c))
             for c in range(best_k)] for p in parts],
        "label_share": {k: [[float(np.mean(truth[k][p] == v)) for v in
                             sorted(np.unique(truth[k]))] for p in parts]
                        for k in ("cooler", "valve", "accum")},
    }
    corr_feat = {f: float(np.corrcoef(cycles, Xa[:, j])[0, 1])
                 for j, f in enumerate(feat_names)}
    drift["top_corr"] = sorted(corr_feat.items(), key=lambda kv: -abs(kv[1]))[:5]

    results = {
        "audit": audit_res,
        "split": {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "train_first": int(tr[0] + 1), "train_last": int(tr[-1] + 1),
            "test_first": int(te[0] + 1), "test_last": int(te[-1] + 1),
            "n_train_blocked": int(len(btr)), "n_test_blocked": int(len(bte)),
            "block_size": BLOCK_SIZE, "train_frac": TRAIN_FRAC,
        },
        "pca": {
            "standard": {k: pca_std[k] for k in ("evr", "cum", "rmse",
                                                 "rrmse")},
            "centered": {k: pca_cent[k] for k in ("evr", "cum", "rmse",
                                                 "rrmse")},
            "standard_blocked": {k: pca_std_bl[k] for k in ("evr", "cum",
                                                            "rmse", "rrmse")},
            "centered_blocked": {k: pca_cent_bl[k] for k in ("evr", "cum",
                                                             "rmse", "rrmse")},
            "loadings_standard": pca_std["loadings"].tolist(),
            "loadings_centered": pca_cent["loadings"].tolist(),
            "cluster_share": float(pca_std["cum"][CLUSTER_PCS - 1]),
            "pc1_corr": drift["pc1_corr"],
        },
        "fa": {
            "components": list(FA_COMPONENTS),
            "rows": {k: {"ll_train": fa[k]["ll_train"],
                         "ll_test": fa[k]["ll_test"],
                         "ll_test_blocked": fa[k]["ll_test_blocked"],
                         "noise_mean": float(np.mean(fa[k]["noise"])),
                         "noise_min": float(np.min(fa[k]["noise"])),
                         "noise_max": float(np.max(fa[k]["noise"])),
                         "comm_mean": float(np.mean(fa[k]["comm"])),
                         "n_iter": fa[k]["n_iter"]}
                     for k in FA_COMPONENTS},
            "loadings": {k: fa[k]["loadings"].tolist() for k in FA_COMPONENTS},
            "noise": {k: fa[k]["noise"].tolist() for k in FA_COMPONENTS},
            "best_test_k": int(max(FA_COMPONENTS,
                                   key=lambda k: fa[k]["ll_test"])),
        },
        "clustering": {
            "n_obj": int(len(idx)), "n_pcs": CLUSTER_PCS,
            "pca_share": float(pca_std["cum"][CLUSTER_PCS - 1]),
            "kmeans": {"rows": km["rows"], "best_k": best_k},
            "agglomerative": {"rows": ag["rows"], "best_k": best_k},
            "dbscan": {k: db[k] for k in ("eps", "eps_rule", "min_samples",
                                          "rows", "n_clusters", "noise",
                                          "sizes", "silhouette", "db",
                                          "match", "min_samples_sensitivity")},
            "final": {k: {kk: vv for kk, vv in final[k].items()
                          if kk != "predict_all"}
                      for k in ("kmeans", "agglomerative", "dbscan")},
            "match_final": {k: final[k]["match"] for k in
                            ("kmeans", "agglomerative", "dbscan")},
            "ct_kmeans": final["kmeans"]["ct"],
        },
        "drift": drift,
        "sensitivity": sens,
        "features_used": feat_names,
        "n_features_active": len(feat_names),
        "n_features": len(FEATURES),
        "n_cycles": int(n),
    }

    ctx = {
        "Xdf": Xdf, "prof": prof, "cycles": cycles, "X": X,
        "Xa": Xa, "features": feat_names,
        "tr": tr, "te": te, "idx": idx, "Z4all": Z4all, "Z4": Z4,
        "ex": load_example_cycle(), "scores": scores,
        "pca_std": pca_std, "pca_cent": pca_cent,
        "pca_std_bl": pca_std_bl, "pca_cent_bl": pca_cent_bl, "fa": fa,
        "lab_km": lab_km, "lab_ag": lab_ag, "lab_db": lab_db,
        "km": km, "ag": ag, "db": db, "best_k": best_k,
    }
    return results, ctx


def main(only_tables: bool = False) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    if only_tables and RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        make_tables(results)
        make_macros(results)
        print(f"Таблицы и макросы обновлены из {RESULTS_JSON}")
        return
    results, ctx = run_analysis()

    fig_signals(ctx["ex"])
    fig_correlation(ctx["Xa"], ctx["features"])
    fig_drift(ctx["cycles"], ctx["Xdf"]["TS1_mean"].to_numpy(),
              ctx["prof"]["cooler"].to_numpy(), ctx["scores"])
    fig_spectrum(ctx["pca_std"], ctx["pca_cent"])
    fig_reconstruction(ctx["pca_std"], ctx["pca_cent"],
                       ctx["pca_std_bl"], ctx["pca_cent_bl"])
    fig_loadings(ctx["pca_std"], ctx["fa"], ctx["features"])
    fig_factor_analysis(ctx["fa"], ctx["features"])
    fig_k_selection(ctx["km"], ctx["ag"], ctx["best_k"])
    fig_kdist(ctx["db"])
    fig_dendrogram(ctx["Z4"], ctx["best_k"])
    fig_clusters(ctx["Z4all"][ctx["idx"]][:, :2], ctx["lab_km"],
                 ctx["lab_ag"], ctx["lab_db"], ctx["best_k"],
                 ctx["db"]["eps"])
    fig_label_plane(ctx["Z4all"][ctx["idx"]][:, :2], ctx["lab_km"],
                    ctx["lab_db"],
                    ctx["prof"]["valve"].to_numpy()[ctx["idx"]],
                    ctx["prof"]["accum"].to_numpy()[ctx["idx"]])
    fig_crosstab(results["clustering"]["ct_kmeans"]["cooler"],
                 results["clustering"]["ct_kmeans"]["valve"],
                 results["clustering"]["ct_kmeans"]["accum"])
    fig_sensitivity(results["sensitivity"])

    make_tables(results)
    make_macros(results)

    RESULTS_JSON.write_text(
        json.dumps(jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"Готово: {RESULTS_JSON}")


if __name__ == "__main__":
    main(only_tables="--tables-only" in sys.argv)
