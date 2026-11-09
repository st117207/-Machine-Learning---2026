"""
Задача 3 (снижение размерности и кластеризация), вариант 18 — Общие режимы гидравлического испытательного стенда.

Обязательные элементы (общие требования к задаче 3 + вариант 13):
  * Аудит данных: источник, число объектов, признаков, единицы, пропуски, выбросы
  * PCA: центрирование vs стандартизация, спектр説明ленной дисперсии, ошибка восстановления inverse_transform
  * FactorAnalysis (sklearn): нагрузки, noise_variance_, логи-правдоподобие score на отложенных данных, 2/3/5 факторов
  * KMeans, AgglomerativeClustering (ward), DBSCAN — на одинаковых PCA-координатах, одна подвыборка
  * KMeans и иерархическая: k=2..6 по силуэту и устойчивости
  * AgglomerativeClustering: дерево слияний на 50-100 наблюдениях
  * DBSCAN: обоснование min_samples и eps по графику k-го соседа, доля шума
  * Сравнение silhouette, индекс Дэвиса-Болдина
  * Минимум 4 типа графиков: спектр PCA+ошибка восстановления; нагрузки компонент и FA; сравнение алгоритмов в PCA-плоскости; два специальных графика варианта
  * Ограничения: PCA ≤10 000 строк, FA ≤2 000 строк, кластеризация ≤1 200 объектов
  * Метки из profile.txt — только для проверки, не для обучения
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figure"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Ограничения вычислительной нагрузки
MAX_PCA_ROWS = 10_000
MAX_FA_ROWS = 2_000
MAX_CLUSTER_ROWS = 1_200

# 7 каналов по варианту
CHANNELS = ["PS1", "PS2", "PS3", "FS1", "TS1", "EPS1", "VS1"]
STAT_FUNCS = ["mean", "std", "min", "max"]
FEATURE_COLS = [f"{ch}_{stat}" for ch in CHANNELS for stat in STAT_FUNCS]

# Группировка каналов для предметных графиков
CHANNEL_GROUPS = {
    "Давление": ["PS1", "PS2", "PS3"],
    "Расход": ["FS1"],
    "Температура": ["TS1"],
    "Мощность": ["EPS1"],
    "Вибрация": ["VS1"],
}
CHANNEL_TO_GROUP = {}
for group, chs in CHANNEL_GROUPS.items():
    for ch in chs:
        CHANNEL_TO_GROUP[ch] = group


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def load_data():
    feat_path = DATA_DIR / "features.csv"
    prof_path = DATA_DIR / "profile.csv"
    df = pd.read_csv(feat_path)
    profile = pd.read_csv(prof_path)
    X = df[FEATURE_COLS].values
    return X, df, profile


def train_test_split_cycles(X, df, profile, test_fraction=0.3):
    """Разбиение по циклам: обучение и отложенные наблюдения."""
    n = len(df)
    n_test = int(n * test_fraction)
    # Последние циклы — отложенные (тестовые)
    train_idx = np.arange(n - n_test)
    test_idx = np.arange(n - n_test, n)
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def run_pca(X_train, X_test, feature_cols):
    results = {}

    # Центрирование
    scaler_center = StandardScaler(with_mean=True, with_std=False)
    X_train_c = scaler_center.fit_transform(X_train)
    X_test_c = scaler_center.transform(X_test)

    pca_center = PCA().fit(X_train_c)
    results["centered"] = {
        "explained_variance_ratio": pca_center.explained_variance_ratio_.copy(),
        "cumsum": np.cumsum(pca_center.explained_variance_ratio_).copy(),
        "singular_values": pca_center.singular_values_.copy(),
    }

    # Стандартизация
    scaler_std = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler_std.fit_transform(X_train)
    X_test_s = scaler_std.transform(X_test)

    pca_std = PCA().fit(X_train_s)
    results["standardized"] = {
        "explained_variance_ratio": pca_std.explained_variance_ratio_.copy(),
        "cumsum": np.cumsum(pca_std.explained_variance_ratio_).copy(),
        "singular_values": pca_std.singular_values_.copy(),
    }

    # Ошибка восстановления на отложенных данных при нескольких размерностях
    dims = [2, 3, 5, 10, 15, 20]
    recon_center, recon_std = [], []
    for d in dims:
        pc = PCA(n_components=d).fit(X_train_s)
        X_rec = pc.inverse_transform(pc.transform(X_test_s))
        recon_std.append(float(np.mean((X_test_s - X_rec) ** 2)))

        pc2 = PCA(n_components=d).fit(X_train_c)
        X_rec2 = pc2.inverse_transform(pc2.transform(X_test_c))
        recon_center.append(float(np.mean((X_test_c - X_rec2) ** 2)))

    results["recon_dims"] = dims
    results["recon_centered"] = recon_center
    results["recon_standardized"] = recon_std

    # Сохраняем масштабированные данные для дальнейшего использования
    results["X_train_scaled"] = X_train_s
    results["X_test_scaled"] = X_test_s
    results["pca_std"] = pca_std

    return results


def plot_pca_spectrum(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_comp = len(res["standardized"]["explained_variance_ratio"])
    x = np.arange(1, n_comp + 1)

    # Спектр
    axes[0].semilogy(x, res["standardized"]["explained_variance_ratio"], "o-", label="стандартизация")
    axes[0].semilogy(x, res["centered"]["explained_variance_ratio"], "s--", label="центрирование")
    axes[0].set_xlabel("Главная компонента")
    axes[0].set_ylabel("Доля объяснённой дисперсии (лог. шкала)")
    axes[0].set_title("Спектр PCA: доля объяснённой дисперсии")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Накопленная сумма
    axes[1].plot(x, res["standardized"]["cumsum"], "o-", label="стандартизация")
    axes[1].plot(x, res["centered"]["cumsum"], "s--", label="центрирование")
    axes[1].axhline(0.95, color="r", linestyle="--", label="95%")
    axes[1].set_xlabel("Количество компонент")
    axes[1].set_ylabel("Накопленная доля объяснённой дисперсии")
    axes[1].set_title("Накопленная объяснённая дисперсия PCA")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    savefig("pca_spectrum.png")


def plot_pca_reconstruction(res):
    plt.figure(figsize=(8, 5))
    plt.plot(res["recon_dims"], res["recon_standardized"], "o-", label="стандартизация")
    plt.plot(res["recon_dims"], res["recon_centered"], "s--", label="центрирование")
    plt.xlabel("Количество главных компонент")
    plt.ylabel("Средняя квадратичная ошибка восстановления")
    plt.title("Ошибка восстановления PCA (inverse_transform) на отложенных данных")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("pca_reconstruction.png")


# ---------------------------------------------------------------------------
# Factor Analysis
# ---------------------------------------------------------------------------
def run_factor_analysis(X_train, X_test):
    results = {}
    for n_factors in [2, 3, 5]:
        fa = FactorAnalysis(n_components=n_factors, random_state=RANDOM_STATE)
        fa.fit(X_train[:MAX_FA_ROWS])
        score = fa.score(X_test[:MAX_FA_ROWS])
        results[n_factors] = {
            "loadings": fa.components_.copy(),
            "noise_variance": fa.noise_variance_.copy(),
            "loglikelihood_train": float(fa.score(X_train[:MAX_FA_ROWS])),
            "loglikelihood_test": float(score),
        }
    return results


def plot_fa_loadings(fa_results, feature_cols):
    """Нагрузки факторов с группировкой по каналам."""
    n_factors = len(fa_results)
    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 3 * n_factors))
    if n_factors == 1:
        axes = [axes]

    # Цвета по группам каналов
    group_colors = {"Давление": "tab:red", "Расход": "tab:blue", "Температура": "tab:green",
                    "Мощность": "tab:orange", "Вибрация": "tab:purple"}

    for idx, (n_factors, res) in enumerate(fa_results.items()):
        loadings = res["loadings"]  # shape (n_factors, n_features)
        for f in range(n_factors):
            ax = axes[idx]
            vals = loadings[f]
            colors = [group_colors[CHANNEL_TO_GROUP[col.rsplit("_", 1)[0]]] for col in feature_cols]
            ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_ylabel(f"Нагрузка фактора {f+1}")
            ax.set_title(f"FA ({n_factors} факторов): нагрузки фактора {f+1}")
            ax.grid(True, alpha=0.3, axis="y")
        if idx == n_factors - 1:
            ax.set_xticks(range(len(feature_cols)))
            ax.set_xticklabels(feature_cols, rotation=90, fontsize=6)
        else:
            ax.set_xticks([])

    savefig("fa_loadings.png")


def plot_fa_noise(fa_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_factors_list = list(fa_results.keys())
    noise_vals = [fa_results[n]["noise_variance"] for n in n_factors_list]
    x = np.arange(len(noise_vals[0]))
    for n, nv in zip(n_factors_list, noise_vals):
        ax.plot(x, nv, "o-", label=f"{n} факторов")
    ax.set_xlabel("Признак (индекс)")
    ax.set_ylabel("Шумовая дисперсия noise_variance_")
    ax.set_title("FA: шумовые дисперсии по каналам")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("fa_noise.png")


# ---------------------------------------------------------------------------
# Кластеризация
# ---------------------------------------------------------------------------
def cluster_comparison(X_pca_subset, y_subset=None):
    """Сравнение KMeans, Agglomerative, DBSCAN на одних и тех же данных."""
    results = {}

    # KMeans для k=2..6
    k_range = range(2, 7)
    km_silhouette = []
    km_db = []
    km_labels = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca_subset)
        km_labels[k] = labels
        if len(np.unique(labels)) > 1:
            km_silhouette.append(silhouette_score(X_pca_subset, labels))
            km_db.append(davies_bouldin_score(X_pca_subset, labels))
        else:
            km_silhouette.append(np.nan)
            km_db.append(np.nan)

    results["kmeans"] = {"k_range": list(k_range), "silhouette": km_silhouette,
                         "davies_bouldin": km_db, "labels": km_labels}

    # AgglomerativeClustering для k=2..6
    agg_silhouette = []
    agg_db = []
    agg_labels = {}
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X_pca_subset)
        agg_labels[k] = labels
        if len(np.unique(labels)) > 1:
            agg_silhouette.append(silhouette_score(X_pca_subset, labels))
            agg_db.append(davies_bouldin_score(X_pca_subset, labels))
        else:
            agg_silhouette.append(np.nan)
            agg_db.append(np.nan)

    results["agglomerative"] = {"k_range": list(k_range), "silhouette": agg_silhouette,
                                "davies_bouldin": agg_db, "labels": agg_labels}

    # Дерево слияний на 50-100 наблюдениях
    n_dendro = min(100, len(X_pca_subset))
    agg_dendro = AgglomerativeClustering(n_clusters=2, linkage="ward", compute_distances=True)
    agg_dendro.fit(X_pca_subset[:n_dendro])

    results["dendrogram"] = {
        "children": agg_dendro.children_,
        "distances": agg_dendro.distances_,
        "n_leaves": agg_dendro.n_leaves_,
        "n_points": n_dendro,
    }

    # DBSCAN: подбор eps по графику k-го соседа
    from sklearn.neighbors import NearestNeighbors
    k_neighbors = min(5, len(X_pca_subset) - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_pca_subset)
    distances, _ = nn.kneighbors(X_pca_subset)
    k_dist = np.sort(distances[:, -1])

    results["dbscan_k_dist"] = {"k_dist": k_dist, "k": k_neighbors + 1}

    # Подбор eps и min_samples
    best_dbscan = None
    best_score = -1
    best_params = None
    for min_s in [3, 5, 10]:
        for eps in np.percentile(k_dist, [50, 70, 85, 90, 95]):
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X_pca_subset)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).mean()
            if n_clusters >= 2 and noise_ratio < 0.5:
                mask = labels != -1
                if mask.sum() > 1:
                    s = silhouette_score(X_pca_subset[mask], labels[mask])
                    if s > best_score:
                        best_score = s
                        best_dbscan = labels
                        best_params = {"eps": eps, "min_samples": min_s,
                                       "n_clusters": n_clusters, "noise_ratio": noise_ratio}

    results["dbscan"] = {"labels": best_dbscan, "params": best_params,
                         "best_silhouette": best_score}

    return results


def plot_k_selection(km_res, agg_res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    k_range = km_res["k_range"]
    axes[0].plot(k_range, km_res["silhouette"], "o-", label="KMeans")
    axes[0].plot(k_range, agg_res["silhouette"], "s--", label="Agglomerative")
    axes[0].set_xlabel("Число кластеров k")
    axes[0].set_ylabel("Коэффициент силуэта")
    axes[0].set_title("Силаэт vs число кластеров")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, km_res["davies_bouldin"], "o-", label="KMeans")
    axes[1].plot(k_range, agg_res["davies_bouldin"], "s--", label="Agglomerative")
    axes[1].set_xlabel("Число кластеров k")
    axes[1].set_ylabel("Индекс Дэвиса—Болдина")
    axes[1].set_title("Индекс Дэвиса—Болдина vs число кластеров")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    savefig("k_selection.png")


def plot_dendrogram(dendro_info, X_subset):
    import scipy.cluster.hierarchy as sch
    # Построим linkage матрицу напрямую через scipy
    Z = sch.linkage(X_subset, method="ward", metric="euclidean")
    plt.figure(figsize=(10, 5))
    sch.dendrogram(Z,
                   distance_sort="descending",
                   show_leaf_counts=True,
                   color_threshold=Z[:, 2].max() * 0.7)
    plt.xlabel("Наблюдение")
    plt.ylabel("Дистанция слияния")
    plt.title(f"Дерево слияний AgglomerativeClustering (ward, {dendro_info['n_points']} объектов)")
    plt.grid(True, alpha=0.3, axis="y")
    savefig("dendrogram.png")


def plot_dbscan_k_dist(dbscan_kdist_info, dbscan_params):
    plt.figure(figsize=(8, 5))
    plt.plot(dbscan_kdist_info["k_dist"], "o-", markersize=3)
    plt.axvline(dbscan_params["min_samples"], color="r", linestyle="--",
                label=f"min_samples={dbscan_params['min_samples']}")
    plt.xlabel("Индекс наблюдения (отсортировано по расстоянию)")
    plt.ylabel("Расстояние до k-го соседа")
    plt.title(f"График расстояния до {dbscan_kdist_info['k']}-го соседа (DBSCAN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("dbscan_k_dist.png")


def plot_clusters_in_pca(X_pca_subset, cluster_results, feature_cols):
    """Сопоставление KMeans/DBSCAN/агломеративного в одной PCA-плоскости."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = ["KMeans (k=3)", "Agglomerative (k=3)", "DBSCAN"]
    label_sets = [
        cluster_results["kmeans"]["labels"][3],
        cluster_results["agglomerative"]["labels"][3],
        cluster_results["dbscan"]["labels"],
    ]

    for ax, name, labels in zip(axes, names, label_sets):
        unique = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique))))
        for i, lab in enumerate(unique):
            mask = labels == lab
            if lab == -1:
                ax.scatter(X_pca_subset[mask, 0], X_pca_subset[mask, 1],
                           c="gray", marker="x", s=10, label="шум", alpha=0.5)
            else:
                ax.scatter(X_pca_subset[mask, 0], X_pca_subset[mask, 1],
                           c=[colors[i]], s=20, label=f"кластер {lab}", alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(name)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    savefig("cluster_comparison.png")


def plot_clusters_with_profile(X_pca_subset, cluster_results, profile_subset, profile_cols):
    """PCA-плоскость с раскраской по состоянию клапана/гидроаккумулятора."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col in zip(axes, profile_cols):
        vals = profile_subset[col].values
        unique_vals = np.unique(vals)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
        for i, v in enumerate(unique_vals):
            mask = vals == v
            ax.scatter(X_pca_subset[mask, 0], X_pca_subset[mask, 1],
                       c=[colors[i]], s=20, label=f"{col}={v}", alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA-плоскость, раскраска по {col}")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    savefig("clusters_with_profile.png")


# ---------------------------------------------------------------------------
# Главный запуск
# ---------------------------------------------------------------------------
def main():
    X, df, profile = load_data()
    print(f"Данные загружены: X={X.shape}, профиль={profile.shape}")

    # Аудит данных
    print("\n=== АУДИТ ДАННЫХ ===")
    print(f"Число независимых объектов (циклов): {len(df)}")
    print(f"Число признаков: {len(FEATURE_COLS)} (7 kanalov x 4 statistiki)")
    print(f"Признаки: {FEATURE_COLS}")
    print(f"Пропуски: {pd.DataFrame(X).isna().sum().sum()}")
    print(f"Постоянные столбцы: {np.sum(np.std(X, axis=0) == 0)}")

    # Распределение состояний из profile (для справки, не для обучения)
    print("\nРаспределение состояний компонентов (из profile.txt):")
    for col in ["cooler", "valve", "pump", "accumulator", "stable"]:
        print(f"  {col}: {profile[col].value_counts().to_dict()}")

    # Разбиение на обучение и отложенное
    train_idx, test_idx = train_test_split_cycles(X, df, profile)
    X_train = X[train_idx]
    X_test = X[test_idx]
    print(f"\nОбучающая часть: {len(train_idx)}, отложенная: {len(test_idx)}")

    # Масштабирование
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ==================== PCA ====================
    print("\n=== PCA ===")
    pca_results = run_pca(X_train_s, X_test_s, FEATURE_COLS)
    plot_pca_spectrum(pca_results)
    plot_pca_reconstruction(pca_results)

    # PCA координаты для кластеризации (берём первые 10 компонент)
    pca_full = PCA(n_components=10, random_state=RANDOM_STATE)
    X_train_pca = pca_full.fit_transform(X_train_s)
    X_test_pca = pca_full.transform(X_test_s)

    # ==================== Factor Analysis ====================
    print("\n=== Factor Analysis ===")
    fa_results = run_factor_analysis(X_train_s, X_test_s)
    plot_fa_loadings(fa_results, FEATURE_COLS)
    plot_fa_noise(fa_results)

    # ==================== Кластеризация ====================
    print("\n=== Кластеризация ===")
    # Подвыборка ≤ 1200 объектов для кластеризации
    n_cluster = min(MAX_CLUSTER_ROWS, len(X_train_pca))
    cluster_idx = np.random.choice(len(X_train_pca), size=n_cluster, replace=False)
    X_cluster = X_train_pca[cluster_idx]
    # profile имеет тот же индекс, что и df (0..2204), train_idx - это позиции в X
    profile_cluster = profile.iloc[train_idx].iloc[cluster_idx].reset_index(drop=True)

    cluster_res = cluster_comparison(X_cluster, profile_cluster.values if hasattr(profile_cluster, 'values') else None)

    plot_k_selection(cluster_res["kmeans"], cluster_res["agglomerative"])
    plot_dendrogram(cluster_res["dendrogram"], X_cluster)
    plot_dbscan_k_dist(cluster_res["dbscan_k_dist"], cluster_res["dbscan"]["params"])
    plot_clusters_in_pca(X_cluster, cluster_res, FEATURE_COLS)

    # Кластеры с профилем
    profile_cols_subset = ["cooler", "accumulator"]
    plot_clusters_with_profile(X_cluster, cluster_res, profile_cluster, profile_cols_subset)

    # Метрики кластеризации
    print("\n=== Метрики кластеризации ===")
    summary = {}
    for algo in ["kmeans", "agglomerative"]:
        best_k = None
        best_sil = -1
        for k, sil in zip(cluster_res[algo]["k_range"], cluster_res[algo]["silhouette"]):
            if not np.isnan(sil) and sil > best_sil:
                best_sil = sil
                best_k = k
        summary[algo] = {"best_k": best_k, "best_silhouette": best_sil}
        print(f"{algo}: лучшее k={best_k}, silhouette={best_sil:.3f}")

    if cluster_res["dbscan"]["params"]:
        print(f"DBSCAN: eps={cluster_res['dbscan']['params']['eps']:.4f}, "
              f"min_samples={cluster_res['dbscan']['params']['min_samples']}, "
              f"кластеры={cluster_res['dbscan']['params']['n_clusters']}, "
              f"шум={cluster_res['dbscan']['params']['noise_ratio']:.3f}")

    # Сохраняем результаты
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nГотово. Графики в figure/, метрики в results/")


if __name__ == "__main__":
    main()
