"""
Обучение и оценка классификаторов — Задача 2, вариант 18 (Shuttle).

Обязательные элементы (общие требования к задаче 2 + вариант 18):
  * DummyClassifier (тривиальная база) + ≥3 базовых классификатора (LogReg/LinearSVC, kNN, SVM)
  * DecisionTreeClassifier, RandomForestClassifier + градиентный бустинг (HistGradientBoosting)
  * Pipeline со стандартизацией для kNN/SVM/линейных моделей
  * Эксперименты с деревьями (max_depth, min_samples_leaf), лесом (n_estimators), бустингом (learning_rate, итерации)
  * GridSearchCV (подбор только на обучении)
  * Основная метрика: macro-F1 (не accuracy, т.к. данные сильно несбалансированы)
  * Метрики: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC (OvR macro), PR-AUC
  * Матрица ошибок для всех моделей; ROC- и PR-кривые; сравнение алгоритмов (столбчатая диаграмма CV);
    карта гиперпараметров; распределение классов; нормированная матрица ошибок и F1 по классам
  * Тестовая выборка — один раз в конце

Структура вывода: st117207_2/figure/ (все графики), st117207_2/results/ (метрики)
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figure"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CLASS_NAMES = {
    1: "Rad Flow", 2: "Fpv Close", 3: "Fpv Open", 4: "High",
    5: "Bypass", 6: "Bpv Close", 7: "Bpv Open",
}
CLASSES = sorted(CLASS_NAMES.keys())


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(DATA_DIR / "shuttle_train.csv")
    test = pd.read_csv(DATA_DIR / "shuttle_test.csv")
    feature_cols = [c for c in train.columns if c.startswith("X")]
    X_train = train[feature_cols].values
    y_train = train["class"].values
    X_test = test[feature_cols].values
    y_test = test["class"].values
    return X_train, y_train, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# Распределение классов (график до обучения)
# ---------------------------------------------------------------------------
def plot_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y, title in [(axes[0], y_train, "Обучающая часть"),
                         (axes[1], y_test, "Тестовая часть")]:
        vals, counts = np.unique(y, return_counts=True)
        ax.bar(vals.astype(str), counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("Класс")
        ax.set_ylabel("Число объектов")
        ax.set_title(f"Распределение классов — {title}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        for v, c in zip(vals, counts):
            ax.text(v, c, str(c), ha="center", va="bottom", fontsize=8)
    savefig("class_distribution.png")


# ---------------------------------------------------------------------------
# Кросс-валидация и сравнение алгоритмов
# ---------------------------------------------------------------------------
def cv_score(pipe, X, y, cv=5):
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    res = cross_validate(pipe, X, y, cv=cv_splitter, scoring=scoring,
                         return_train_score=False, n_jobs=1, error_score=np.nan)
    return {k: float(np.nanmean(v)) for k, v in res.items() if k.startswith("test_")}


def compare_algorithms(X_train, y_train):
    """Сравнение алгоритмов — столбчатая диаграмма macro-F1 по CV."""
    models = {
        "Базовый (стратифицированный)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Логистическая регрессия": Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
        "LinearSVC": Pipeline([("scale", StandardScaler()),
                               ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE))]),
        "kNN (k=5)": Pipeline([("scale", StandardScaler()),
                               ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "SVM (RBF)": Pipeline([("scale", StandardScaler()),
                               ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    rows = []
    for name, model in models.items():
        try:
            m = cv_score(model, X_train, y_train)
            rows.append({"model": name, **m})
        except Exception as e:
            print(f"  CV failed for {name}: {e}")
    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)

    # Столбчатая диаграмма macro-F1
    plt.figure(figsize=(10, 6))
    order = df_cv.sort_values("test_f1_macro", ascending=True)
    colors = ["crimson" if "Базовый" in n else "steelblue" for n in order["model"]]
    plt.barh(order["model"], order["test_f1_macro"], color=colors, edgecolor="black")
    for i, (_, r) in enumerate(order.iterrows()):
        plt.text(r["test_f1_macro"], i, f"{r['test_f1_macro']:.3f}", va="center")
    plt.xlabel("macro-F1 (кросс-валидация, 5 фолдов)")
    plt.title("Сравнение алгоритмов по macro-F1 (CV)")
    plt.grid(True, alpha=0.3, axis="x")
    savefig("algorithm_comparison.png")

    return df_cv, models


# ---------------------------------------------------------------------------
# Эксперименты с деревьями
# ---------------------------------------------------------------------------
def tree_experiments(X_train, y_train):
    """max_depth / min_samples_leaf для одного дерева — ошибка обучения и проверки."""
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                               random_state=RANDOM_STATE, stratify=y_train)

    depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, None]
    train_err, val_err = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=2,
                                     random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        train_err.append(1 - accuracy_score(y_tr, clf.predict(X_tr)))
        val_err.append(1 - accuracy_score(y_val, clf.predict(X_val)))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_err, marker="o", label="обучение")
    plt.plot(depths, val_err, marker="o", label="проверка")
    plt.xlabel("max_depth (None = без ограничения)")
    plt.ylabel("Ошибка (1 − точность)")
    plt.title("Дерево решений: ошибка от max_depth")
    plt.legend(); plt.grid(True, alpha=0.3)
    savefig("tree_max_depth.png")

    # min_samples_leaf
    leaves = [1, 2, 5, 10, 20, 50, 100]
    train_err_l, val_err_l = [], []
    for l in leaves:
        clf = DecisionTreeClassifier(max_depth=None, min_samples_leaf=l,
                                     random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        train_err_l.append(1 - accuracy_score(y_tr, clf.predict(X_tr)))
        val_err_l.append(1 - accuracy_score(y_val, clf.predict(X_val)))

    plt.figure(figsize=(8, 5))
    plt.plot(leaves, train_err_l, marker="o", label="обучение")
    plt.plot(leaves, val_err_l, marker="o", label="проверка")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Ошибка (1 − точность)")
    plt.title("Дерево решений: ошибка от min_samples_leaf")
    plt.xscale("log"); plt.legend(); plt.grid(True, alpha=0.3)
    savefig("tree_min_samples_leaf.png")


def forest_experiments(X_train, y_train):
    """Случайный лес: ошибка и разброс от числа деревьев."""
    n_trees = [10, 50, 100, 200, 400]
    mean_err, std_err = [], []
    for n in n_trees:
        scores = []
        for seed in [0, 1, 2, 3, 4]:
            clf = RandomForestClassifier(n_estimators=n, random_state=seed)
            m = cv_score(clf, X_train, y_train, cv=3)
            scores.append(1 - m["test_accuracy"])
        mean_err.append(np.mean(scores))
        std_err.append(np.std(scores))

    plt.figure(figsize=(8, 5))
    plt.errorbar(n_trees, mean_err, yerr=std_err, marker="o", capsize=4)
    plt.xlabel("Число деревьев (n_estimators)")
    plt.ylabel("Ошибка (1 − точность), среднее ± std по 5 seed")
    plt.title("Случайный лес: ошибка от числа деревьев")
    plt.grid(True, alpha=0.3)
    savefig("forest_n_estimators.png")

    # Одно дерево для сравнения
    single = DecisionTreeClassifier(random_state=RANDOM_STATE)
    m_single = cv_score(single, X_train, y_train, cv=3)
    print(f"Одно дерево: accuracy={m_single['test_accuracy']:.4f}, macro-F1={m_single['test_f1_macro']:.4f}")


def boosting_experiments(X_train, y_train):
    """Градиентный бустинг: кривая проверочной ошибки по итерациям для двух learning_rate."""
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                               random_state=RANDOM_STATE, stratify=y_train)

    for lr in [0.05, 0.2]:
        clf = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=300, random_state=RANDOM_STATE,
            early_stopping=False)
        clf.fit(X_tr, y_tr)
        # Ошибка по итерациям (loss не даёт классификацию, используем staged_predict)
        val_err = []
        for pred in clf.staged_predict(X_val):
            val_err.append(1 - accuracy_score(y_val, pred))

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(val_err) + 1), val_err, marker="o", markersize=2,
                 label=f"скорость обучения={lr}")
        plt.xlabel("Число итераций")
        plt.ylabel("Проверочная ошибка (1 − точность)")
        plt.title(f"Градиентный бустинг: ошибка от числа итераций (скорость={lr})")
        plt.legend(); plt.grid(True, alpha=0.3)
        savefig(f"boosting_lr{lr}.png")


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------
def grid_search(X_train, y_train):
    """GridSearchCV для RandomForest и HistGradientBoosting."""
    results = {}

    # Random Forest — компактная сетка
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    }
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs_rf = GridSearchCV(rf, rf_grid, cv=cv_splitter, scoring="f1_macro", n_jobs=1)
    gs_rf.fit(X_train, y_train)
    results["random_forest"] = {
        "best_params": gs_rf.best_params_,
        "best_cv_f1_macro": float(gs_rf.best_score_),
    }

    # HistGradientBoosting — компактная сетка
    hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    hgb_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300],
        "max_depth": [None, 5, 10],
    }
    gs_hgb = GridSearchCV(hgb, hgb_grid, cv=cv_splitter, scoring="f1_macro", n_jobs=1)
    gs_hgb.fit(X_train, y_train)
    results["hist_gradient_boosting"] = {
        "best_params": gs_hgb.best_params_,
        "best_cv_f1_macro": float(gs_hgb.best_score_),
    }

    with open(RESULTS_DIR / "grid_search.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("GridSearchCV результаты:", results)
    return gs_rf, gs_hgb


# ---------------------------------------------------------------------------
# Итоговая оценка на тесте
# ---------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        if df.ndim == 1:
            y_proba = np.column_stack([1 - df, df])
        else:
            y_proba = df

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    # F1 по классам
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    for cls, f1v in zip(CLASSES, f1_per_class):
        metrics[f"f1_class_{cls}"] = float(f1v)

    # ROC-AUC OvR macro
    try:
        y_bin = label_binarize(y_test, classes=CLASSES)
        if y_proba is not None:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr"))
            # PR-AUC macro
            ap_scores = []
            for i, cls in enumerate(CLASSES):
                y_bin_cls = (y_test == cls).astype(int)
                if y_bin_cls.sum() > 0:
                    ap_scores.append(average_precision_score(y_bin_cls, y_proba[:, i]))
            metrics["pr_auc_macro"] = float(np.mean(ap_scores)) if ap_scores else np.nan
    except Exception as e:
        print(f"  AUC/PR error for {name}: {e}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return metrics, cm, cm_norm, y_pred, y_proba


def plot_confusion_matrix(cm_norm, name, fname):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Нормированная матрица ошибок — {name}")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in CLASSES], rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in CLASSES])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    savefig(fname)


def plot_roc_pr(all_results, fname_roc, fname_pr):
    """ROC и PR кривые основных моделей."""
    plt.figure(figsize=(8, 7))
    for name, res in all_results.items():
        if res["y_proba"] is None:
            continue
        y_bin = label_binarize(res["y_test"], classes=CLASSES)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), res["y_proba"].ravel())
        auc = roc_auc_score(y_bin, res["y_proba"], average="macro", multi_class="ovr")
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Доля ложноположительных")
    plt.ylabel("Доля истинноположительных")
    plt.title("ROC-кривые (OvR, макроусреднение)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    savefig(fname_roc)

    plt.figure(figsize=(8, 7))
    for name, res in all_results.items():
        if res["y_proba"] is None:
            continue
        y_test_bin = (res["y_test"][:, None] == np.array(CLASSES)).astype(int)
        for i, cls in enumerate(CLASSES):
            if y_test_bin[:, i].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], res["y_proba"][:, i])
            ap = average_precision_score(y_test_bin[:, i], res["y_proba"][:, i])
            plt.plot(recall, precision, label=f"{CLASS_NAMES[cls]} (AP={ap:.2f})", lw=1.5, alpha=0.8)
    plt.xlabel("Полнота")
    plt.ylabel("Точность")
    plt.title("Кривые точность-полнота по классам")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, alpha=0.3)
    savefig(fname_pr)


# ---------------------------------------------------------------------------
# Основной запуск
# ---------------------------------------------------------------------------
def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    plot_class_distribution(y_train, y_test)

    print("\n=== Сравнение алгоритмов (CV) ===")
    df_cv, models = compare_algorithms(X_train, y_train)
    print(df_cv.round(4).to_string(index=False))

    print("\n=== Эксперименты с деревьями ===")
    tree_experiments(X_train, y_train)
    forest_experiments(X_train, y_train)
    boosting_experiments(X_train, y_train)

    print("\n=== GridSearchCV ===")
    gs_rf, gs_hgb = grid_search(X_train, y_train)

    # Итоговые модели на тесте
    print("\n=== Итоговая оценка на тестовой выборке ===")
    all_results = {}

    # Базовая и несколько 대표тельных моделей
    test_models = {
        "Базовый (стратифицированный)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Логистическая регрессия": Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
        "Случайный лес (лучший GS)": gs_rf.best_estimator_,
        "Градиентный бустинг (лучший GS)": gs_hgb.best_estimator_,
    }
    for name, model in test_models.items():
        model.fit(X_train, y_train)

    summary_rows = []
    for name, model in test_models.items():
        metrics, cm, cm_norm, y_pred, y_proba = evaluate_model(name, model, X_test, y_test)
        all_results[name] = {"y_test": y_test, "y_proba": y_proba, "metrics": metrics}
        plot_confusion_matrix(cm_norm, name, f"confusion_{name.split()[0].lower()}.png")
        row = {"model": name, **metrics}
        summary_rows.append(row)
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)

    plot_roc_pr(all_results, "roc_curves.png", "pr_curves.png")

    # Карта гиперпараметров (RandomForest: n_estimators vs max_depth, ф1 макро по CV)
    print("\n=== Карта гиперпараметров RandomForest ===")
    n_est_grid = [50, 100, 200, 400]
    depth_grid = [None, 10, 20, 30]
    heatmap = np.zeros((len(depth_grid), len(n_est_grid)))
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for i, d in enumerate(depth_grid):
        for j, n in enumerate(n_est_grid):
            clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=RANDOM_STATE)
            m = cv_score(clf, X_train, y_train, cv=5)
            heatmap[i, j] = m["test_f1_macro"]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, label="macro-F1 (CV)")
    plt.xticks(range(len(n_est_grid)), n_est_grid)
    plt.yticks(range(len(depth_grid)), [str(d) for d in depth_grid])
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.title("RandomForest: карта макро-F1 по гиперпараметрам (CV)")
    for i in range(len(depth_grid)):
        for j in range(len(n_est_grid)):
            plt.text(j, i, f"{heatmap[i, j]:.3f}", ha="center", va="center", fontsize=8)
    savefig("rf_hyperparam_heatmap.png")

    print("\nГотово. Все метрики в results/, графики в figure/")


if __name__ == "__main__":
    main()
