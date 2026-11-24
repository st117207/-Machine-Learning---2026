# -*- coding: utf-8 -*-
"""Моделирование задачи «Состояния космического аппарата Shuttle» (вариант 18).

Методика
--------
* Данные — UCI Statlog Shuttle: 43 500 обучающих и 14 500 тестовых записей,
  разбиение предложено авторами набора и используется как есть.
* Признаки — девять анонимных числовых показаний A..I. Их физический смысл
  в описании набора не раскрыт; первый признак документирован как «время»,
  но порядок строк перемешан, поэтому время как порядок утрачено.
* Основная метрика — macro-F1: класс 1 занимает около 80 %, поэтому обычная
  точность вводит в заблуждение.
* Подбор гиперпараметров — GridSearchCV / RandomizedSearchCV со
  StratifiedKFold(5) только по обучающей части; тест используется один раз
  после выбора модели.
* Модели: Dummy, логистическая регрессия, kNN, SVM (RBF), решающее дерево,
  случайный лес и HistGradientBoosting.
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_validate,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (CLASS_NAMES, CLASSES, FEATURES, FIG_DIR, N_SPLITS,
                    PROCESSED_CSV, RANDOM_STATE, RESULTS_JSON, SVC_TUNE_N,
                    TAB_DIR, TARGET, VAL_SIZE)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                     random_state=RANDOM_STATE)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

CLS_COLORS = {1: "#bdbdbd", 2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c",
              5: "#d62728", 6: "#9467bd", 7: "#8c564b"}
CLASS_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
               6: "Six", 7: "Seven"}
MODEL_WORDS = {"dummy": "Dummy", "logreg": "Logreg", "knn": "Knn",
               "svm": "Svm", "tree": "Tree", "rf": "Rf", "hgb": "Hgb"}
MODEL_ORDER = ["dummy", "logreg", "knn", "svm", "tree", "rf", "hgb"]
MODEL_LABELS = {"dummy": "Dummy", "logreg": "Лог. регрессия", "knn": "kNN",
                "svm": "SVM", "tree": "Дерево", "rf": "Случайный лес",
                "hgb": "Бустинг"}


# ---------------------------------------------------------------------------
# Метрики и форматирование
# ---------------------------------------------------------------------------

def scores_of(model, X) -> np.ndarray:
    """Непрерывные оценки для ROC: predict_proba, иначе decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def auc_per_class(y_true, score) -> list[float]:
    """AUC one-vs-rest по каждому классу (без требования сумм вероятностей)."""
    out = []
    for j, c in enumerate(CLASSES):
        ybin = (y_true == c).astype(int)
        out.append(float(roc_auc_score(ybin, score[:, j])))
    return out


def full_metrics(y_true, y_pred, score=None) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
    }
    if score is not None:
        aucs = auc_per_class(y_true, score)
        m["roc_auc_macro"] = float(np.mean(aucs))
        m["auc_per_class"] = {int(c): a for c, a in zip(CLASSES, aucs)}
    return m


def per_class_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0)
    return {int(c): {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)}


def per_class_of(results: dict, key: str, c: int) -> dict:
    """Метрики класса c для модели key (ключи JSON могут быть строками)."""
    pc = results[key]["test"]["per_class"]
    return pc[c] if c in pc else pc[str(c)]


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
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def params_tex(params: dict) -> str:
    """Компактная запись гиперпараметров для макроса."""
    if not params:
        return "---"
    parts = []
    for k, v in params.items():
        key = k.replace("model__", "")
        if v is None:
            val = "нет"
        elif isinstance(v, float):
            val = f"{v:.3g}".replace(".", ",")
        else:
            val = str(v)
        parts.append(f"{key}={val}")
    return tex_escape(", ".join(parts))


# ---------------------------------------------------------------------------
# Модели и подбор гиперпараметров
# ---------------------------------------------------------------------------

def make_specs() -> dict[str, dict]:
    """Модельная матрица: базовая, линейная, метрические, деревья, ансамбли."""
    rs = RANDOM_STATE
    specs: dict[str, dict] = {}

    specs["dummy"] = dict(
        name="Dummy (most frequent)",
        pipe=Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        search=None)

    specs["logreg"] = dict(
        name="Логистическая регрессия",
        pipe=Pipeline([("sc", StandardScaler()),
                       ("model", LogisticRegression(max_iter=3000,
                                                    random_state=rs))]),
        search=dict(kind="grid", params={
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__class_weight": [None, "balanced"]}))

    specs["knn"] = dict(
        name="kNN",
        pipe=Pipeline([("sc", StandardScaler()),
                       ("model", KNeighborsClassifier(n_jobs=-1))]),
        search=dict(kind="grid", params={
            "model__n_neighbors": [1, 3, 5, 9, 15],
            "model__weights": ["uniform", "distance"]}))

    specs["svm"] = dict(
        name="SVM (RBF)",
        pipe=Pipeline([("sc", StandardScaler()),
                       ("model", SVC(kernel="rbf", cache_size=1000))]),
        search=dict(kind="grid", params={
            "model__C": [1.0, 10.0, 100.0],
            "model__gamma": [0.01, 0.1],
            "model__class_weight": [None, "balanced"]}))

    specs["tree"] = dict(
        name="Решающее дерево",
        pipe=DecisionTreeClassifier(random_state=rs),
        search=dict(kind="grid", params={
            "max_depth": [3, 5, 7, 10, 15, 20, None],
            "min_samples_leaf": [1, 5, 20, 50]}))

    specs["rf"] = dict(
        name="Случайный лес",
        pipe=RandomForestClassifier(n_estimators=200, random_state=rs,
                                    n_jobs=-1),
        search=dict(kind="grid", params={
            "n_estimators": [200],
            "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", 0.5]}))

    specs["hgb"] = dict(
        name="Градиентный бустинг (HistGB)",
        pipe=HistGradientBoostingClassifier(random_state=rs,
                                            early_stopping=False),
        search=dict(kind="random", n_iter=24, params={
            "learning_rate": loguniform(0.03, 0.3),
            "max_iter": [100, 200, 300],
            "max_leaf_nodes": [15, 31, 63],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0],
            "class_weight": [None, "balanced"]}))
    return specs


def tune_model(spec: dict, X, y) -> tuple[object, dict, object | None]:
    """Подбор гиперпараметров только по обучающей части."""
    if spec["search"] is None:
        return clone(spec["pipe"]).fit(X, y), {}, None

    s = spec["search"]
    Xs, ys = X, y
    if SVC_TUNE_N and len(y) > SVC_TUNE_N:
        Xs, _, ys, _ = train_test_split(
            X, y, train_size=SVC_TUNE_N, stratify=y,
            random_state=RANDOM_STATE)

    if s["kind"] == "grid":
        search = GridSearchCV(spec["pipe"], s["params"], scoring="f1_macro",
                              cv=CV, n_jobs=-1, refit=True)
    else:
        search = RandomizedSearchCV(
            spec["pipe"], s["params"], n_iter=s["n_iter"],
            scoring="f1_macro", cv=CV, n_jobs=-1, refit=True,
            random_state=RANDOM_STATE)
    search.fit(Xs, ys)
    return search.best_estimator_, dict(search.best_params_), search


def evaluate_model(model, Xtr, ytr, Xte, yte) -> tuple[dict, object]:
    """Кросс-валидация на обучении + однократная проверка на тесте."""
    cvr = cross_validate(
        clone(model), Xtr, ytr, cv=CV, n_jobs=-1,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy",
                 "balanced_accuracy": "balanced_accuracy"})
    cv = {
        "macro_f1": float(cvr["test_f1_macro"].mean()),
        "macro_f1_std": float(cvr["test_f1_macro"].std()),
        "accuracy": float(cvr["test_accuracy"].mean()),
        "balanced_accuracy": float(cvr["test_balanced_accuracy"].mean()),
    }
    fitted = clone(model).fit(Xtr, ytr)
    pred = fitted.predict(Xte)
    score = scores_of(fitted, Xte)
    test = full_metrics(yte, pred, score)
    test["per_class"] = per_class_metrics(yte, pred)
    test["confusion"] = confusion_matrix(yte, pred, labels=CLASSES).tolist()
    return {"cv": cv, "test": test}, fitted


# ---------------------------------------------------------------------------
# Методические эксперименты с деревьями и ансамблями
# ---------------------------------------------------------------------------

def run_experiments(X, y) -> dict:
    Xa, Xv, ya, yv = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE)
    exp: dict = {}

    def err(model) -> float:
        return float(1.0 - accuracy_score(yv, model.predict(Xv)))

    # 1. Переобучение одного дерева: глубина и минимальный лист.
    # Кроме доли ошибок считаем логарифмическую потерю: на почти разделимых
    # данных доля ошибок быстро становится нулевой, а лог-потеря показывает
    # переуверенность глубокого дерева.
    def ll(model, X_, y_) -> float:
        return float(log_loss(y_, model.predict_proba(X_), labels=CLASSES))

    depths = list(range(1, 21))
    tr_err, va_err, tr_ll, va_ll = [], [], [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE).fit(Xa, ya)
        tr_err.append(float(1.0 - accuracy_score(ya, m.predict(Xa))))
        va_err.append(err(m))
        tr_ll.append(ll(m, Xa, ya))
        va_ll.append(ll(m, Xv, yv))
    exp["tree_depth"] = {"depth": depths, "train": tr_err, "val": va_err,
                         "train_ll": tr_ll, "val_ll": va_ll}

    leaves = [1, 2, 5, 10, 20, 50, 100]
    tr_err, va_err, tr_ll, va_ll = [], [], [], []
    for leaf in leaves:
        m = DecisionTreeClassifier(min_samples_leaf=leaf,
                                   random_state=RANDOM_STATE).fit(Xa, ya)
        tr_err.append(float(1.0 - accuracy_score(ya, m.predict(Xa))))
        va_err.append(err(m))
        tr_ll.append(ll(m, Xa, ya))
        va_ll.append(ll(m, Xv, yv))
    exp["tree_leaf"] = {"leaf": leaves, "train": tr_err, "val": va_err,
                        "train_ll": tr_ll, "val_ll": va_ll}

    # 2. Случайный лес: ошибка и разброс по random_state от числа деревьев
    ns = [1, 3, 5, 10, 20, 50, 100, 200, 300]
    states = [0, 1, 2]
    errs = np.zeros((len(states), len(ns)))
    lls = np.zeros((len(states), len(ns)))
    for i, rs in enumerate(states):
        for j, n in enumerate(ns):
            m = RandomForestClassifier(n_estimators=n, random_state=rs,
                                       n_jobs=-1).fit(Xa, ya)
            errs[i, j] = err(m)
            lls[i, j] = ll(m, Xv, yv)
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xa, ya)
    exp["rf_ntrees"] = {
        "n": ns, "mean": errs.mean(axis=0).tolist(),
        "std": errs.std(axis=0).tolist(), "tree_err": err(tree),
        "mean_ll": lls.mean(axis=0).tolist(),
        "std_ll": lls.std(axis=0).tolist(), "tree_ll": ll(tree, Xv, yv)}

    # 3. Градиентный бустинг: ошибка и лог-потеря по числу итераций.
    # class_weight='balanced' — как в окончательной модели; без балансировки
    # промежуточные итерации на сильно несбалансированных данных неустойчивы.
    iters = list(range(1, 301))
    lr_low, lr_high = 0.05, 0.2
    curves, curves_ll = {}, {}
    for lr in (lr_low, lr_high):
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=300, early_stopping=False,
            class_weight="balanced", random_state=RANDOM_STATE).fit(Xa, ya)
        curves[lr] = [float(1.0 - accuracy_score(yv, p))
                      for p in m.staged_predict(Xv)]
        curves_ll[lr] = [float(log_loss(yv, p, labels=CLASSES))
                         for p in m.staged_predict_proba(Xv)]
    exp["gb_iters"] = {
        "iters": iters, "lr_low": lr_low, "lr_high": lr_high,
        "err_low": curves[lr_low], "err_high": curves[lr_high],
        "ll_low": curves_ll[lr_low], "ll_high": curves_ll[lr_high]}

    # 4. Влияние трёх существенных гиперпараметров леса. Доля ошибок на этих
    # данных почти нулевая, поэтому параллельно считаем лог-потерю.
    rf_depths = [1, 2, 3, 5, 7, 10, 15, 20, None]
    rf_depth_err, rf_depth_ll = [], []
    for d in rf_depths:
        m = RandomForestClassifier(n_estimators=200, max_depth=d,
                                   random_state=RANDOM_STATE,
                                   n_jobs=-1).fit(Xa, ya)
        rf_depth_err.append(err(m))
        rf_depth_ll.append(ll(m, Xv, yv))
    rf_leaves = [1, 2, 5, 10, 20, 50, 100]
    rf_leaf_err, rf_leaf_ll = [], []
    for leaf in rf_leaves:
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf,
                                   random_state=RANDOM_STATE,
                                   n_jobs=-1).fit(Xa, ya)
        rf_leaf_err.append(err(m))
        rf_leaf_ll.append(ll(m, Xv, yv))
    rf_feats = [0.1, 0.2, 1 / 3, 0.5, 0.7, 1.0]
    rf_feat_err, rf_feat_ll = [], []
    for mf in rf_feats:
        m = RandomForestClassifier(n_estimators=200, max_features=mf,
                                   random_state=RANDOM_STATE,
                                   n_jobs=-1).fit(Xa, ya)
        rf_feat_err.append(err(m))
        rf_feat_ll.append(ll(m, Xv, yv))
    exp["rf_params"] = {
        "depth": rf_depths, "depth_err": rf_depth_err, "depth_ll": rf_depth_ll,
        "leaf": rf_leaves, "leaf_err": rf_leaf_err, "leaf_ll": rf_leaf_ll,
        "feat": rf_feats, "feat_err": rf_feat_err, "feat_ll": rf_feat_ll}

    # 5. Влияние трёх существенных гиперпараметров бустинга
    gb_lrs = [0.02, 0.05, 0.1, 0.2, 0.3]
    gb_lr_err = []
    for lr in gb_lrs:
        m = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=200, early_stopping=False,
            random_state=RANDOM_STATE).fit(Xa, ya)
        gb_lr_err.append(err(m))
    gb_iters = [20, 50, 100, 200, 300, 400]
    gb_iter_err = []
    for it in gb_iters:
        m = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=it, early_stopping=False,
            random_state=RANDOM_STATE).fit(Xa, ya)
        gb_iter_err.append(err(m))
    gb_leaves = [7, 15, 31, 63, 127]
    gb_leaf_err = []
    for leaf in gb_leaves:
        m = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, max_leaf_nodes=leaf,
            early_stopping=False, random_state=RANDOM_STATE).fit(Xa, ya)
        gb_leaf_err.append(err(m))
    exp["gb_params"] = {
        "lr": gb_lrs, "lr_err": gb_lr_err,
        "iters": gb_iters, "iters_err": gb_iter_err,
        "leaf": gb_leaves, "leaf_err": gb_leaf_err}

    # 6. Проверка ненужности масштабирования для деревьев и ансамблей:
    # одна и та же модель с StandardScaler и без него, кросс-валидация.
    scaling_models = {
        "tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                     random_state=RANDOM_STATE),
        "hgb": HistGradientBoostingClassifier(max_iter=100,
                                              early_stopping=False,
                                              random_state=RANDOM_STATE),
    }
    exp["scaling"] = {}
    for k, m in scaling_models.items():
        raw = cross_val_score(clone(m), X, y, cv=CV, scoring="f1_macro",
                              n_jobs=-1)
        scaled = cross_val_score(
            Pipeline([("sc", StandardScaler()), ("model", clone(m))]),
            X, y, cv=CV, scoring="f1_macro", n_jobs=-1)
        exp["scaling"][k] = {
            "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
            "scaled_mean": float(scaled.mean()),
            "scaled_std": float(scaled.std())}

    return exp


def search_summary(search, kind: str) -> dict:
    """Данные для рисунков подбора гиперпараметров."""
    if search is None:
        return {}
    cvr = pd.DataFrame(search.cv_results_)
    if kind == "svc":
        rows = []
        for _, r in cvr.iterrows():
            rows.append((float(r["param_model__C"]),
                         float(r["param_model__gamma"]),
                         float(r["mean_test_score"])))
        cs = sorted({r[0] for r in rows})
        gs = sorted({r[1] for r in rows})
        grid = np.full((len(gs), len(cs)), np.nan)
        for c, g, s in rows:
            i, j = gs.index(g), cs.index(c)
            grid[i, j] = s if np.isnan(grid[i, j]) else max(grid[i, j], s)
        return {"C": cs, "gamma": gs, "score": grid.tolist()}
    rows = []
    for _, r in cvr.iterrows():
        rows.append((float(r["param_learning_rate"]),
                     int(r["param_max_iter"]),
                     float(r["mean_test_score"])))
    return {"learning_rate": [r[0] for r in rows],
            "max_iter": [r[1] for r in rows],
            "score": [r[2] for r in rows]}


# ---------------------------------------------------------------------------
# Рисунки
# ---------------------------------------------------------------------------

def fig_class_distribution(tr: pd.DataFrame, te: pd.DataFrame) -> None:
    ctr = [int((tr[TARGET] == c).sum()) for c in CLASSES]
    cte = [int((te[TARGET] == c).sum()) for c in CLASSES]
    ntr, nte = len(tr), len(te)
    x = np.arange(len(CLASSES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.9, 3.4), constrained_layout=True)
    b1 = ax.bar(x - w / 2, ctr, w, label="обучение", color="#08519c")
    b2 = ax.bar(x + w / 2, cte, w, label="тест", color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in CLASSES], fontsize=8)
    ax.set_ylabel("Число записей (лог. шкала)")
    for bars, counts, n in ((b1, ctr, ntr), (b2, cte, nte)):
        for rect, v in zip(bars, counts):
            ax.text(rect.get_x() + rect.get_width() / 2, v * 1.18,
                    f"{v}\n({100 * v / n:.1f} %)".replace(".", ","),
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(1, max(ctr) * 6)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_classes.pdf")
    plt.close(fig)


def fig_domain(tr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0),
                             constrained_layout=True)
    for c in CLASSES:
        m = tr[TARGET] == c
        kw = dict(color=CLS_COLORS[c], label=f"{c} {CLASS_NAMES[c]}")
        if c == 1:
            axes[0].scatter(tr.loc[m, "A"], tr.loc[m, "C"], s=4, alpha=0.15, **kw)
            axes[1].scatter(tr.loc[m, "A"], tr.loc[m, "B"], s=4, alpha=0.15, **kw)
        else:
            axes[0].scatter(tr.loc[m, "A"], tr.loc[m, "C"], s=16, alpha=0.8,
                            edgecolor="k", linewidth=0.2, **kw)
            axes[1].scatter(tr.loc[m, "A"], tr.loc[m, "B"], s=16, alpha=0.8,
                            edgecolor="k", linewidth=0.2, **kw)
    axes[0].set_xlabel("Признак A, усл. ед.")
    axes[0].set_ylabel("Признак C, усл. ед.")
    axes[1].set_xlabel("Признак A, усл. ед.")
    axes[1].set_ylabel("Признак B, усл. ед. (symlog)")
    axes[1].set_yscale("symlog", linthresh=100)
    axes[0].legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    fig.savefig(FIG_DIR / "fig2_domain.pdf")
    plt.close(fig)


def fig_model_comparison(results: dict) -> None:
    keys = MODEL_ORDER
    means = [results[k]["cv"]["macro_f1"] for k in keys]
    stds = [results[k]["cv"]["macro_f1_std"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.9, 3.2), constrained_layout=True)
    x = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color="#08519c",
                  edgecolor="k", linewidth=0.4)
    for rect, v, s in zip(bars, means, stds):
        ax.text(rect.get_x() + rect.get_width() / 2, v + s + 0.03,
                f"{v:.2f}".replace(".", ","), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[k] for k in keys], rotation=20,
                       ha="right", fontsize=8)
    ax.set_ylabel("macro-F1 кросс-валидации")
    ax.set_ylim(0, 1.12)
    fig.savefig(FIG_DIR / "fig3_models.pdf")
    plt.close(fig)


def fig_tree_overfit(exp: dict) -> None:
    d = exp["tree_depth"]
    l = exp["tree_leaf"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8),
                             constrained_layout=True)
    axes[0].plot(d["depth"], d["train"], "o-", ms=3, label="обучение")
    axes[0].plot(d["depth"], d["val"], "s-", ms=3, label="проверка")
    axes[0].set_xlabel("Максимальная глубина")
    axes[0].set_ylabel("Доля ошибок")
    axes[0].set_title("Ошибка от глубины", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].plot(d["depth"], d["train_ll"], "o-", ms=3, label="обучение")
    axes[1].plot(d["depth"], d["val_ll"], "s-", ms=3, label="проверка")
    axes[1].set_xlabel("Максимальная глубина")
    axes[1].set_ylabel("Лог-потеря")
    axes[1].set_title("Лог-потеря от глубины", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].semilogx(l["leaf"], l["train"], "o-", ms=3, label="обучение")
    axes[2].semilogx(l["leaf"], l["val"], "s-", ms=3, label="проверка")
    axes[2].set_xlabel("Мин. объектов в листе")
    axes[2].set_ylabel("Доля ошибок")
    axes[2].set_title("Ошибка от размера листа", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.savefig(FIG_DIR / "fig4_tree_overfit.pdf")
    plt.close(fig)


def fig_rf_ntrees(exp: dict) -> None:
    d = exp["rf_ntrees"]
    n = np.array(d["n"])
    mean = np.array(d["mean"])
    std = np.array(d["std"])
    mean_ll = np.array(d["mean_ll"])
    std_ll = np.array(d["std_ll"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             constrained_layout=True)
    axes[0].semilogx(n, mean, "o-", ms=4, color="#08519c",
                     label="случайный лес")
    axes[0].fill_between(n, mean - std, mean + std, color="#08519c",
                         alpha=0.2, label="разброс по random_state")
    axes[0].axhline(d["tree_err"], color="#d62728", ls="--", lw=1.2,
                    label="одно дерево")
    axes[0].set_xlabel("Число деревьев в лесу")
    axes[0].set_ylabel("Доля ошибок на проверке")
    axes[0].set_title("Доля ошибок", fontsize=9)
    axes[0].legend(fontsize=7.5)
    axes[1].semilogx(n, mean_ll, "o-", ms=4, color="#08519c",
                     label="случайный лес")
    axes[1].fill_between(n, mean_ll - std_ll, mean_ll + std_ll,
                         color="#08519c", alpha=0.2,
                         label="разброс по random_state")
    axes[1].axhline(d["tree_ll"], color="#d62728", ls="--", lw=1.2,
                    label="одно дерево")
    axes[1].set_xlabel("Число деревьев в лесу")
    axes[1].set_ylabel("Логарифмическая потеря")
    axes[1].set_title("Лог-потеря", fontsize=9)
    axes[1].legend(fontsize=7.5)
    fig.savefig(FIG_DIR / "fig5_rf_ntrees.pdf")
    plt.close(fig)


def fig_gb_iters(exp: dict) -> None:
    d = exp["gb_iters"]
    it = np.array(d["iters"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1),
                             constrained_layout=True)
    for key, label in (("low", f"learning rate = {d['lr_low']:.2f}"),
                       ("high", f"learning rate = {d['lr_high']:.2f}")):
        lab = label.replace(".", ",")
        axes[0].plot(it, d[f"err_{key}"], lw=1.0, label=lab)
        axes[1].semilogy(it, d[f"ll_{key}"], lw=1.0, label=lab)
        ll = np.array(d[f"ll_{key}"])
        k = int(np.argmin(ll))
        axes[1].plot(it[k], ll[k], "o", ms=4, color="k")
        axes[1].axvline(it[k], color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("Число итераций бустинга")
    axes[0].set_ylabel("Доля ошибок на проверке")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Число итераций бустинга")
    axes[1].set_ylabel("Лог-потеря (лог. шкала)")
    axes[1].legend(fontsize=8)
    fig.savefig(FIG_DIR / "fig6_gb_iters.pdf")
    plt.close(fig)


def fig_rf_gb_params(exp: dict) -> None:
    rf = exp["rf_params"]
    gb = exp["gb_params"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9),
                             constrained_layout=True)

    # Верхний ряд — случайный лес: доля ошибок уже нулевая, поэтому лог-потеря
    depth_labels = [str(v) if v is not None else "∞" for v in rf["depth"]]
    axes[0, 0].plot(range(len(rf["depth"])), rf["depth_ll"], "o-", ms=3)
    axes[0, 0].set_xticks(range(len(rf["depth"])))
    axes[0, 0].set_xticklabels(depth_labels, fontsize=7)
    axes[0, 0].set_xlabel("max_depth")
    axes[0, 0].set_ylabel("Лог-потеря")
    axes[0, 1].semilogx(rf["leaf"], rf["leaf_ll"], "o-", ms=3)
    axes[0, 1].set_xticks(rf["leaf"])
    axes[0, 1].set_xticklabels([str(v) for v in rf["leaf"]], fontsize=7)
    axes[0, 1].set_xlabel("min_samples_leaf")
    axes[0, 2].plot(rf["feat"], rf["feat_ll"], "o-", ms=3)
    axes[0, 2].set_xlabel("max_features")
    axes[0, 0].set_title("Случайный лес", fontsize=9)
    axes[0, 1].set_title("Случайный лес", fontsize=9)
    axes[0, 2].set_title("Случайный лес", fontsize=9)

    # Нижний ряд — бустинг: доля ошибок чувствительнее и устойчивее лог-потери
    axes[1, 0].plot(range(len(gb["lr"])), gb["lr_err"], "s-", ms=3,
                    color="#d62728")
    axes[1, 0].set_xticks(range(len(gb["lr"])))
    axes[1, 0].set_xticklabels([f"{v:g}".replace(".", ",") for v in gb["lr"]],
                               fontsize=7, rotation=30)
    axes[1, 0].set_xlabel("learning_rate")
    axes[1, 1].plot(gb["iters"], gb["iters_err"], "s-", ms=3, color="#d62728")
    axes[1, 1].set_xlabel("max_iter")
    axes[1, 2].semilogx(gb["leaf"], gb["leaf_err"], "s-", ms=3,
                        color="#d62728")
    axes[1, 2].set_xticks(gb["leaf"])
    axes[1, 2].set_xticklabels([str(v) for v in gb["leaf"]], fontsize=7)
    axes[1, 2].set_xlabel("max_leaf_nodes")
    for ax in axes[1]:
        ax.set_ylabel("Доля ошибок")
    for ax in axes[1]:
        ax.set_title("Градиентный бустинг", fontsize=9)
    fig.savefig(FIG_DIR / "fig7_rf_gb_params.pdf")
    plt.close(fig)


def fig_search(svc: dict, hgb: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             constrained_layout=True)
    if svc:
        grid = np.array(svc["score"])
        im = axes[0].imshow(grid, cmap="viridis", aspect="auto")
        axes[0].set_xticks(range(len(svc["C"])))
        axes[0].set_xticklabels([f"{v:g}".replace(".", ",") for v in svc["C"]])
        axes[0].set_yticks(range(len(svc["gamma"])))
        axes[0].set_yticklabels([f"{v:g}".replace(".", ",")
                                 for v in svc["gamma"]])
        axes[0].set_xlabel("Параметр C")
        axes[0].set_ylabel(r"Параметр $\gamma$")
        lo, hi = float(grid.min()), float(grid.max())
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                axes[0].text(j, i, f"{grid[i, j]:.2f}".replace(".", ","),
                             ha="center", va="center", fontsize=7,
                             color="w" if grid[i, j] < lo + 0.55 * (hi - lo)
                             else "k")
        fig.colorbar(im, ax=axes[0], orientation="horizontal",
                     fraction=0.075, pad=0.12)
    if hgb:
        sc = axes[1].scatter(hgb["learning_rate"], hgb["score"],
                             c=hgb["max_iter"], cmap="plasma", s=32,
                             edgecolor="k", linewidth=0.3)
        axes[1].set_xlabel("learning_rate")
        axes[1].set_ylabel("macro-F1 (CV)")
        cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02)
        cb.set_label("max_iter", fontsize=8)
    fig.savefig(FIG_DIR / "fig8_search.pdf")
    plt.close(fig)


def fig_confusion_all(results: dict) -> None:
    """Нормированные матрицы ошибок всех моделей: 2 x 4."""
    fig, axes = plt.subplots(2, 4, figsize=(7.3, 4.5),
                             constrained_layout=True)
    flat = axes.ravel()
    for ax, k in zip(flat, MODEL_ORDER):
        cm = np.asarray(results[k]["test"]["confusion"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(MODEL_LABELS[k], fontsize=8)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=6)
        ax.set_yticklabels(CLASSES, fontsize=6)
        ax.grid(False)
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                if i != j and cmn[i, j] > 0.005:
                    ax.text(j, i, f"{cmn[i, j]:.2f}".replace(".", ","),
                            ha="center", va="center", fontsize=5.4,
                            color="w" if cmn[i, j] > 0.55 else "k")
    for ax in flat[len(MODEL_ORDER):]:
        ax.axis("off")
    fig.savefig(FIG_DIR / "fig13_confusion_all.pdf")
    plt.close(fig)


def fig_confusion(cm_norm: np.ndarray, cm_counts: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if i != j and v < 0.005:
                continue
            ax.text(j, i, f"{v:.2f}".replace(".", ","), ha="center",
                    va="center", fontsize=6.5,
                    color="w" if v > 0.55 else "k")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    labels = [f"{c} {CLASS_NAMES[c]}" for c in CLASSES]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_confusion.pdf")
    plt.close(fig)


def fig_per_class(per_class: dict) -> None:
    p = [per_class[c]["precision"] for c in CLASSES]
    r = [per_class[c]["recall"] for c in CLASSES]
    f = [per_class[c]["f1"] for c in CLASSES]
    s = [per_class[c]["support"] for c in CLASSES]
    x = np.arange(len(CLASSES))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.9, 3.5), constrained_layout=True)
    ax.bar(x - w, p, w, label="precision", color="#08519c")
    ax.bar(x, r, w, label="recall", color="#2ca02c")
    ax.bar(x + w, f, w, label="F1", color="#d62728")
    for xi, fi, si in zip(x, f, s):
        ax.text(xi + w, fi + 0.02, f"n={si}", ha="center", fontsize=6.5)
    ax.axhline(np.mean(f), color="k", ls="--", lw=1,
               label="macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} {CLASS_NAMES[c]}" for c in CLASSES],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.savefig(FIG_DIR / "fig10_per_class.pdf")
    plt.close(fig)


def fig_roc(y_true, score, per_class: dict) -> None:
    fpr, tpr, auc = {}, {}, {}
    for j, c in enumerate(CLASSES):
        ybin = (y_true == c).astype(int)
        fpr[c], tpr[c], _ = roc_curve(ybin, score[:, j])
        auc[c] = float(roc_auc_score(ybin, score[:, j]))
    all_fpr = np.unique(np.concatenate([fpr[c] for c in CLASSES]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in CLASSES:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= len(CLASSES)
    macro_auc = float(np.mean([auc[c] for c in CLASSES]))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1),
                             constrained_layout=True)
    for c in CLASSES:
        axes[0].plot(fpr[c], tpr[c], lw=0.8, alpha=0.6,
                     label=f"{c} (AUC {auc[c]:.2f})".replace(".", ","))
    axes[0].plot(all_fpr, mean_tpr, "k--", lw=1.6,
                 label=f"макро (AUC {macro_auc:.2f})".replace(".", ","))
    axes[0].set_xlabel("Доля ложных срабатываний (FPR)")
    axes[0].set_ylabel("Доля верных (TPR)")
    axes[0].legend(fontsize=5.8, loc="lower right")

    rare = [2, 3, 6, 7]
    for c in rare:
        axes[1].plot(fpr[c], tpr[c], lw=1.3,
                     label=f"{c} {CLASS_NAMES[c]} (AUC {auc[c]:.2f}, n={per_class[c]['support']})".replace(".", ","))
    axes[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[1].set_xlabel("Доля ложных срабатываний (FPR)")
    axes[1].set_ylabel("Доля верных (TPR)")
    axes[1].legend(fontsize=6.0, loc="lower right")
    fig.savefig(FIG_DIR / "fig11_roc.pdf")
    plt.close(fig)


def fig_importance(perm, feat_names, coef) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.2))
    mean = np.asarray(perm["mean"])
    std = np.asarray(perm["std"])
    order = np.argsort(mean)
    axes[0].barh(np.array(feat_names)[order], mean[order],
                 xerr=std[order], color="#08519c", capsize=2)
    axes[0].set_xlabel("Падение macro-F1 при перестановке")
    axes[0].set_title("Перестановочная важность", fontsize=9)

    vmax = np.abs(coef).max()
    im = axes[1].imshow(coef, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        aspect="auto")
    axes[1].set_xticks(range(len(feat_names)))
    axes[1].set_xticklabels(feat_names, fontsize=8)
    axes[1].set_yticks(range(len(CLASSES)))
    axes[1].set_yticklabels([f"{c}" for c in CLASSES], fontsize=8)
    axes[1].set_xlabel("Признак")
    axes[1].set_ylabel("Класс")
    axes[1].set_title("Коэффициенты логистической регрессии", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_importance.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------

def make_tables(results: dict, data: dict, exp: dict,
                per_class: dict, auc_final: dict, auc_logreg: dict,
                auc_rf: dict, perm: dict, cm_counts: np.ndarray,
                best_key: str) -> None:
    # Таблица 1 — характеристика данных
    ntr, nte = data["n_train"], data["n_test"]
    s_major_tr = 100 * data["class_counts_train"][1] / ntr
    s_major_te = 100 * data["class_counts_test"][1] / nte
    s_minor_tr = 100 * sum(data["class_counts_train"][c] for c in (2, 3, 6, 7)) / ntr
    s_minor_te = 100 * sum(data["class_counts_test"][c] for c in (2, 3, 6, 7)) / nte
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Характеристика набора данных}\label{tab:data}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Показатель & Обучение & Тест \\", r"\midrule",
        f"Число записей & {ntr} & {nte} \\\\",
        f"Число признаков & {data['n_features']} & {data['n_features']} \\\\",
        f"Число классов & {data['n_classes']} & {data['n_classes']} \\\\",
        f"Доля класса 1, \\% & {fmt(s_major_tr)} & {fmt(s_major_te)} \\\\",
        f"Доля классов 2, 3, 6, 7, \\% & {fmt(s_minor_tr)} & {fmt(s_minor_te)} \\\\",
        f"Пропуски & {data['missing']} & {data['missing']} \\\\",
        f"Дубликаты & {data['duplicates_train']} & {data['duplicates_test']} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (TAB_DIR / "tab_data.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 2 — распределение по классам
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Распределение классов в обучающей и тестовой частях}"
        r"\label{tab:classes}",
        r"\begin{tabular}{clrrrr}", r"\toprule",
        r"Класс & Состояние & \multicolumn{2}{c}{Обучение} & "
        r"\multicolumn{2}{c}{Тест} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & $n$ & \% & $n$ & \% \\", r"\midrule",
    ]
    for c in CLASSES:
        lines.append(
            f"{c} & {CLASS_NAMES[c]} & {data['class_counts_train'][c]} & "
            f"{fmt(100 * data['class_counts_train'][c] / ntr)} & "
            f"{data['class_counts_test'][c]} & "
            f"{fmt(100 * data['class_counts_test'][c] / nte)} \\\\")
    lines += [r"\midrule",
              f" & Всего & {ntr} & 100,00 & {nte} & 100,00 \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_classes.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 3 — сравнение моделей
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\caption{Качество моделей: кросс-валидация на обучении и "
        r"однократная проверка на тесте}\label{tab:models}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Модель & CV macro-F1 & Тест, accuracy & Bal. accuracy & "
        r"Macro-F1 & Weighted-F1 & ROC-AUC \\", r"\midrule",
    ]
    for k in MODEL_ORDER:
        r_ = results[k]
        lines.append(
            f"{r_['name']} & {fmt(r_['cv']['macro_f1'])} $\\pm$ "
            f"{fmt(r_['cv']['macro_f1_std'])} & {fmt(r_['test']['accuracy'], 4)} & "
            f"{fmt(r_['test']['balanced_accuracy'])} & "
            f"{fmt(r_['test']['macro_f1'])} & "
            f"{fmt(r_['test']['weighted_f1'])} & "
            f"{fmt(r_['test']['roc_auc_macro'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB_DIR / "tab_models.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 4 — пространства поиска и выбранные параметры
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Подбор гиперпараметров: поиск, выбранные значения и "
        r"оценка кросс-валидации}\label{tab:search}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.4cm}"
        r">{\raggedright\arraybackslash}p{1.5cm}"
        r">{\raggedright\arraybackslash}p{4.1cm}"
        r">{\raggedright\arraybackslash}p{4.2cm}r@{}}", r"\toprule",
        r"Модель & Поиск & Пространство & Лучшие параметры & CV macro-F1 \\",
        r"\midrule",
    ]
    spaces = {
        "dummy": ("---", "---"),
        "logreg": (r"Grid, $5\times2$",
                   r"$C\in\{0{,}01;\ldots;100\}$; class\_weight"),
        "knn": (r"Grid, $5\times2$", r"$k\in\{1;3;5;9;15\}$; weights"),
        "svm": (r"Grid, $3\times2\times2$",
                r"$C\in\{1;10;100\}$; $\gamma\in\{0{,}01;0{,}1\}$; class\_weight"),
        "tree": (r"Grid, $7\times4$", r"max\_depth; min\_samples\_leaf"),
        "rf": (r"Grid, $1\times3\times2\times2$",
               r"max\_depth; min\_samples\_leaf; max\_features"),
        "hgb": (r"Random, 24",
                r"learning\_rate; max\_iter; max\_leaf\_nodes; "
                r"min\_samples\_leaf; $l_2$"),
    }
    for k in MODEL_ORDER:
        r_ = results[k]
        sp, desc = spaces[k]
        lines.append(
            f"{r_['name']} & {sp} & {desc} & {params_tex(r_['params'])} & "
            f"{fmt(r_['cv']['macro_f1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_search.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 5 — метрики по классам для окончательной модели
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Метрики окончательной модели по классам (тест)}"
        r"\label{tab:class}",
        r"\begin{tabular}{clrrrr}", r"\toprule",
        r"Класс & Состояние & Precision & Recall & F1 & Поддержка \\",
        r"\midrule",
    ]
    for c in CLASSES:
        pc = per_class[c]
        lines.append(
            f"{c} & {CLASS_NAMES[c]} & {fmt(pc['precision'])} & "
            f"{fmt(pc['recall'])} & {fmt(pc['f1'])} & {pc['support']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_class.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 6 — F1 по классам для всех моделей
    cols = ["logreg", "knn", "svm", "tree", "rf", "hgb"]
    heads = ["Лог. регр.", "kNN", "SVM", "Дерево", "Лес", "Бустинг"]
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\caption{F1 по классам на тесте для всех моделей}"
        r"\label{tab:classmodels}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{clrrrrrr}", r"\toprule",
        r"Класс & Состояние & " + " & ".join(heads) + r" \\",
        r"\midrule",
    ]
    for c in CLASSES:
        vals = [fmt(per_class_of(results, k, c)["f1"]) for k in cols]
        lines.append(f"{c} & {CLASS_NAMES[c]} & " + " & ".join(vals)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB_DIR / "tab_classmodels.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")

    # Таблица 7 — AUC по классам
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{AUC one-vs-rest по классам на тесте}\label{tab:auc}",
        r"\begin{tabular}{clccc}", r"\toprule",
        r"Класс & Состояние & Окончательная & Лог. регрессия & "
        r"Случайный лес \\", r"\midrule",
    ]
    for c in CLASSES:
        lines.append(
            f"{c} & {CLASS_NAMES[c]} & {fmt(auc_final[c])} & "
            f"{fmt(auc_logreg[c])} & {fmt(auc_rf[c])} \\\\")
    lines += [r"\midrule",
              f" & Macro & {fmt(results[best_key]['test']['roc_auc_macro'])} & "
              f"{fmt(results['logreg']['test']['roc_auc_macro'])} & "
              f"{fmt(results['rf']['test']['roc_auc_macro'])} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_auc.tex").write_text("\n".join(lines), encoding="utf-8")

    # Приложение А — матрица ошибок в абсолютных числах
    lines = [
        r"\begin{table}[H]", r"\centering", r"\footnotesize",
        r"\caption{Матрица ошибок окончательной модели, абсолютные числа}"
        r"\label{tab:confcounts}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Истинный & \multicolumn{7}{c}{Предсказанный класс} & Всего \\",
        r"\cmidrule(lr){2-8}",
        r"класс & 1 & 2 & 3 & 4 & 5 & 6 & 7 & \\", r"\midrule",
    ]
    for i, c in enumerate(CLASSES):
        row = " & ".join(str(int(v)) for v in cm_counts[i])
        lines.append(f"{c} & {row} & {int(cm_counts[i].sum())} \\\\")
    lines += [r"\midrule",
              r"Всего & " + " & ".join(str(int(v)) for v in cm_counts.sum(axis=0))
              + f" & {int(cm_counts.sum())} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_confcounts.tex").write_text("\n".join(lines),
                                                encoding="utf-8")

    # Приложение Б — перестановочная важность
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Перестановочная важность признаков окончательной модели "
        r"(падение macro-F1)}\label{tab:importance}",
        r"\begin{tabular}{clrr}", r"\toprule",
        r"Место & Признак & Среднее падение & Стандартное отклонение \\",
        r"\midrule",
    ]
    order = np.argsort(perm["mean"])[::-1]
    for rank, j in enumerate(order, 1):
        lines.append(f"{rank} & {FEATURES[j]} & {fmt(perm['mean'][j])} & "
                     f"{fmt(perm['std'][j])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_importance.tex").write_text("\n".join(lines),
                                                encoding="utf-8")
    # Копия для приложения: своя метка, чтобы не дублировать \label
    lines_app = [ln.replace(r"\label{tab:importance}",
                            r"\label{tab:importanceapp}") for ln in lines]
    (TAB_DIR / "tab_importance_app.tex").write_text("\n".join(lines_app),
                                                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Макросы
# ---------------------------------------------------------------------------

def make_macros(results: dict, data: dict, exp: dict, per_class: dict,
                auc_final: dict, perm: dict, cm_norm: np.ndarray,
                best_key: str, n_fig: int, n_tab: int, n_src: int,
                n_app: int) -> None:
    ntr, nte = data["n_train"], data["n_test"]
    s_major_tr = 100 * data["class_counts_train"][1] / ntr
    s_major_te = 100 * data["class_counts_test"][1] / nte
    s_minor_tr = 100 * sum(data["class_counts_train"][c] for c in (2, 3, 6, 7)) / ntr
    s_minor_te = 100 * sum(data["class_counts_test"][c] for c in (2, 3, 6, 7)) / nte

    m = []
    m += [
        ("NTrain", str(ntr)), ("NTest", str(nte)), ("NTotal", str(ntr + nte)),
        ("NFeatures", str(data["n_features"])), ("NClasses", str(data["n_classes"])),
        ("ShareMajorTrain", fmt(s_major_tr)), ("ShareMajorTest", fmt(s_major_te)),
        ("ShareMinorTrain", fmt(s_minor_tr)), ("ShareMinorTest", fmt(s_minor_te)),
        ("NDupTrain", str(data["duplicates_train"])),
        ("NDupTest", str(data["duplicates_test"])),
        ("NOverlap", str(data["overlap"])), ("NMissing", str(data["missing"])),
        ("NFig", str(n_fig)), ("NTab", str(n_tab)), ("NSrc", str(n_src)),
        ("NApp", str(n_app)),
    ]
    for c in CLASSES:
        w = CLASS_WORDS[c]
        m.append((f"NClass{w}Train", str(data["class_counts_train"][c])))
        m.append((f"NClass{w}Test", str(data["class_counts_test"][c])))
        m.append((f"ShareClass{w}Train", fmt(100 * data["class_counts_train"][c] / ntr)))
        m.append((f"ShareClass{w}Test", fmt(100 * data["class_counts_test"][c] / nte)))

    for k in MODEL_ORDER:
        w = MODEL_WORDS[k]
        r_ = results[k]
        m += [
            (f"CvF{w}", fmt(r_["cv"]["macro_f1"])),
            (f"CvF{w}Std", fmt(r_["cv"]["macro_f1_std"])),
            (f"Acc{w}", fmt(r_["test"]["accuracy"], 4)),
            (f"Bal{w}", fmt(r_["test"]["balanced_accuracy"])),
            (f"FOne{w}", fmt(r_["test"]["macro_f1"])),
            (f"Wf{w}", fmt(r_["test"]["weighted_f1"])),
            (f"Auc{w}", fmt(r_["test"]["roc_auc_macro"])),
            (f"Params{w}", params_tex(r_["params"])),
        ]

    b = results[best_key]
    m += [
        ("BestName", tex_escape(b["name"])),
        ("BestCvF", fmt(b["cv"]["macro_f1"])),
        ("BestCvFStd", fmt(b["cv"]["macro_f1_std"])),
        ("BestAcc", fmt(b["test"]["accuracy"], 4)),
        ("BestBal", fmt(b["test"]["balanced_accuracy"])),
        ("BestFOne", fmt(b["test"]["macro_f1"])),
        ("BestWf", fmt(b["test"]["weighted_f1"])),
        ("BestAuc", fmt(b["test"]["roc_auc_macro"])),
        ("BestParams", params_tex(b["params"])),
    ]

    for c in CLASSES:
        w = CLASS_WORDS[c]
        pc = per_class[c]
        m += [
            (f"PrecClass{w}", fmt(pc["precision"])),
            (f"RecClass{w}", fmt(pc["recall"])),
            (f"FOneClass{w}", fmt(pc["f1"])),
            (f"SupportClass{w}", str(pc["support"])),
            (f"AucClass{w}", fmt(auc_final[c])),
        ]

    # Метрики по классам для всех моделей — для обсуждения редких классов
    for k in MODEL_ORDER:
        w = MODEL_WORDS[k]
        for c in CLASSES:
            cw = CLASS_WORDS[c]
            pc = per_class_of(results, k, c)
            m += [
                (f"Prec{w}Class{cw}", fmt(pc["precision"])),
                (f"Rec{w}Class{cw}", fmt(pc["recall"])),
                (f"FOne{w}Class{cw}", fmt(pc["f1"])),
            ]

    # Матрица ошибок: все 49 нормированных элементов и абсолютных чисел
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            m.append((f"Conf{CLASS_WORDS[ci]}{CLASS_WORDS[cj]}",
                      fmt(cm_norm[i, j])))
    cm_counts = np.asarray(results[best_key]["test"]["confusion"], dtype=float)
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            m.append((f"ConfCnt{CLASS_WORDS[ci]}{CLASS_WORDS[cj]}",
                      str(int(cm_counts[i, j]))))

    # Матрицы ошибок всех моделей: нормированные доли и абсолютные числа
    for k in MODEL_ORDER:
        w = MODEL_WORDS[k]
        cmk = np.asarray(results[k]["test"]["confusion"], dtype=float)
        cmkn = cmk / cmk.sum(axis=1, keepdims=True)
        for i, ci in enumerate(CLASSES):
            for j, cj in enumerate(CLASSES):
                m.append((f"Conf{w}{CLASS_WORDS[ci]}{CLASS_WORDS[cj]}",
                          fmt(cmkn[i, j])))
                m.append((f"ConfCnt{w}{CLASS_WORDS[ci]}{CLASS_WORDS[cj]}",
                          str(int(cmk[i, j]))))

    # Эксперименты
    d = exp["tree_depth"]
    bi = int(np.argmin(d["val"]))
    deep = len(d["depth"]) - 1
    m += [
        ("TreeBestDepth", str(d["depth"][bi])),
        ("TreeBestTrainErr", fmt(d["train"][bi], 3)),
        ("TreeBestValErr", fmt(d["val"][bi], 3)),
        ("TreeBestTrainLl", fmt(d["train_ll"][bi], 4)),
        ("TreeBestValLl", fmt(d["val_ll"][bi], 4)),
        ("TreeDeepTrainErr", fmt(d["train"][deep], 3)),
        ("TreeDeepValErr", fmt(d["val"][deep], 3)),
        ("TreeDeepTrainLl", fmt(d["train_ll"][deep], 4)),
        ("TreeDeepValLl", fmt(d["val_ll"][deep], 4)),
    ]
    l = exp["tree_leaf"]
    bl = int(np.argmin(l["val"]))
    m += [
        ("TreeBestLeaf", str(l["leaf"][bl])),
        ("TreeBestLeafValErr", fmt(l["val"][bl], 3)),
        ("TreeBestLeafValLl", fmt(l["val_ll"][bl], 4)),
        ("TreeLeafTrainErr", fmt(l["train"][bl], 3)),
    ]
    rfn = exp["rf_ntrees"]
    m += [
        ("RfTreeErr", fmt(rfn["tree_err"], 4)),
        ("RfTreeLl", fmt(rfn["tree_ll"], 4)),
        ("RfNfinal", str(rfn["n"][-1])),
        ("RfErrFinal", fmt(rfn["mean"][-1], 4)),
        ("RfStdFinal", fmt(rfn["std"][-1], 3)),
        ("RfLlFinal", fmt(rfn["mean_ll"][-1], 4)),
        ("RfLlStdFinal", fmt(rfn["std_ll"][-1], 4)),
        ("RfErrHundred", fmt(rfn["mean"][rfn["n"].index(100)], 3)),
        ("RfLlHundred", fmt(rfn["mean_ll"][rfn["n"].index(100)], 4)),
    ]
    gbi = exp["gb_iters"]
    low = np.array(gbi["err_low"])
    high = np.array(gbi["err_high"])
    low_ll = np.array(gbi["ll_low"])
    high_ll = np.array(gbi["ll_high"])
    m += [
        ("GbLrLow", fmt(gbi["lr_low"], 2)),
        ("GbLrHigh", fmt(gbi["lr_high"], 2)),
        ("GbErrLowFinal", fmt(low[-1], 4)),
        ("GbErrHighFinal", fmt(high[-1], 4)),
        ("GbMinIterLow", str(int(np.argmin(low)) + 1)),
        ("GbMinIterHigh", str(int(np.argmin(high)) + 1)),
        ("GbMinErrLow", fmt(low.min(), 4)),
        ("GbMinErrHigh", fmt(high.min(), 4)),
        ("GbLlMinLow", fmt(low_ll.min(), 4)),
        ("GbLlMinHigh", fmt(high_ll.min(), 4)),
        ("GbLlFinalLow", fmt(low_ll[-1], 4)),
        ("GbLlFinalHigh", fmt(high_ll[-1], 4)),
        ("GbLlIterLow", str(int(np.argmin(low_ll)) + 1)),
        ("GbLlIterHigh", str(int(np.argmin(high_ll)) + 1)),
    ]
    rf = exp["rf_params"]
    m += [
        ("RfExpBestDepth", str(rf["depth"][int(np.argmin(rf["depth_err"]))])),
        ("RfExpBestLeaf", str(rf["leaf"][int(np.argmin(rf["leaf_err"]))])),
        ("RfExpBestFeat", fmt(rf["feat"][int(np.argmin(rf["feat_err"]))], 2)),
        ("RfExpErrDepth", fmt(min(rf["depth_err"]), 3)),
        ("RfExpErrLeaf", fmt(min(rf["leaf_err"]), 3)),
        ("RfExpErrFeat", fmt(min(rf["feat_err"]), 3)),
        ("RfExpLlBestDepth", str(rf["depth"][int(np.argmin(rf["depth_ll"]))])),
        ("RfExpLlBestLeaf", str(rf["leaf"][int(np.argmin(rf["leaf_ll"]))])),
        ("RfExpLlBestFeat", fmt(rf["feat"][int(np.argmin(rf["feat_ll"]))], 2)),
        ("RfExpLlDepth", fmt(min(rf["depth_ll"]), 4)),
        ("RfExpLlLeaf", fmt(min(rf["leaf_ll"]), 4)),
        ("RfExpLlFeat", fmt(min(rf["feat_ll"]), 4)),
    ]
    gbp = exp["gb_params"]
    m += [
        ("GbExpBestLr", fmt(gbp["lr"][int(np.argmin(gbp["lr_err"]))], 2)),
        ("GbExpBestIter", str(gbp["iters"][int(np.argmin(gbp["iters_err"]))])),
        ("GbExpBestLeaf", str(gbp["leaf"][int(np.argmin(gbp["leaf_err"]))])),
        ("GbExpErrLr", fmt(min(gbp["lr_err"]), 3)),
        ("GbExpErrIter", fmt(min(gbp["iters_err"]), 3)),
        ("GbExpErrLeaf", fmt(min(gbp["leaf_err"]), 3)),
    ]
    # Проверка масштабирования для деревьев
    for k, w in (("tree", "Tree"), ("rf", "Rf"), ("hgb", "Hgb")):
        sc = exp.get("scaling", {}).get(k)
        if sc:
            m += [
                (f"Scale{w}Raw", fmt(sc["raw_mean"])),
                (f"Scale{w}RawStd", fmt(sc["raw_std"])),
                (f"Scale{w}Scaled", fmt(sc["scaled_mean"])),
                (f"Scale{w}ScaledStd", fmt(sc["scaled_std"])),
            ]
    # Параметры, выбранные поиском
    svc = results["svm"]["params"]
    hgb = results["hgb"]["params"]
    m += [
        ("SvcBestC", fmt(svc.get("model__C"), 0)),
        ("SvcBestGamma", fmt(svc.get("model__gamma"), 2)),
        ("SvcSearchF", fmt(results["svm"]["cv"]["macro_f1"])),
        ("HgbBestLr", fmt(hgb.get("learning_rate"), 2)),
        ("HgbBestIter", str(hgb.get("max_iter", "---"))),
        ("HgbBestLeaf", str(hgb.get("max_leaf_nodes", "---"))),
        ("HgbBestLTwo", fmt(hgb.get("l2_regularization"), 2)),
        ("HgbClassWeight",
         tex_escape("нет" if hgb.get("class_weight") is None
                    else str(hgb.get("class_weight")))),
        ("HgbSearchF", fmt(results["hgb"]["cv"]["macro_f1"])),
    ]
    # Важность
    order = np.argsort(perm["mean"])[::-1]
    j1, j2 = int(order[0]), int(order[1])
    m += [
        ("ImpTopFeature", FEATURES[j1]),
        ("ImpTopValue", fmt(perm["mean"][j1])),
        ("ImpTopStd", fmt(perm["std"][j1])),
        ("ImpSecondFeature", FEATURES[j2]),
        ("ImpSecondValue", fmt(perm["mean"][j2])),
    ]

    lines = ["% Автоматически сгенерировано src/modeling.py; не редактировать.",
             ""]
    for name, value in m:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (TAB_DIR / "macros.tex").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------

def main(only_tables: bool = False) -> None:
    np.random.seed(RANDOM_STATE)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_CSV)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    Xtr = tr[FEATURES].to_numpy(dtype=float)
    ytr = tr[TARGET].to_numpy(dtype=int)
    Xte = te[FEATURES].to_numpy(dtype=float)
    yte = te[TARGET].to_numpy(dtype=int)

    data = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": len(FEATURES), "n_classes": len(CLASSES),
        "class_counts_train": {int(c): int((tr[TARGET] == c).sum())
                               for c in CLASSES},
        "class_counts_test": {int(c): int((te[TARGET] == c).sum())
                              for c in CLASSES},
        "missing": int(df[FEATURES].isna().sum().sum()),
        "duplicates_train": int(tr[FEATURES + [TARGET]].duplicated().sum()),
        "duplicates_test": int(te[FEATURES + [TARGET]].duplicated().sum()),
        "overlap": int(tr[FEATURES + [TARGET]].merge(
            te[FEATURES + [TARGET]], how="inner").shape[0]),
    }

    if only_tables:
        d = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        data = dict(d["data"])
        data["class_counts_train"] = {int(c): v for c, v in
                                      data["class_counts_train"].items()}
        data["class_counts_test"] = {int(c): v for c, v in
                                     data["class_counts_test"].items()}
        per_class = {int(c): v for c, v in d["per_class"].items()}
        auc_final = {int(c): v for c, v in d["auc_final"].items()}
        auc_logreg = {int(c): v for c, v in d["auc_logreg"].items()}
        auc_rf = {int(c): v for c, v in d["auc_rf"].items()}
        make_tables(d["results"], data, d["experiments"],
                    per_class, auc_final, auc_logreg,
                    auc_rf, d["perm_importance"],
                    np.array(d["confusion_counts"]), d["best_key"])
        make_macros(d["results"], data, d["experiments"], per_class,
                    auc_final, d["perm_importance"],
                    np.array(d["confusion_norm"]), d["best_key"],
                    d["n_fig"], d["n_tab"], d["n_src"], d["n_app"])
        print("Готово: таблицы и макросы перезаписаны по results.json.")
        return

    specs = make_specs()
    results: dict = {}
    fitted: dict = {}
    searches: dict = {}

    print("=== Подбор гиперпараметров и оценка моделей")
    for k in MODEL_ORDER:
        spec = specs[k]
        print(f"  {k}: {spec['name']}")
        model, params, search = tune_model(spec, Xtr, ytr)
        res, fit = evaluate_model(model, Xtr, ytr, Xte, yte)
        res.update(name=spec["name"], params=params)
        results[k] = res
        fitted[k] = fit
        if search is not None:
            searches[k] = search
        print(f"    CV macro-F1={res['cv']['macro_f1']:.4f} "
              f"± {res['cv']['macro_f1_std']:.4f}; "
              f"тест macro-F1={res['test']['macro_f1']:.4f}, "
              f"accuracy={res['test']['accuracy']:.4f}")

    best_key = max(MODEL_ORDER, key=lambda k: results[k]["cv"]["macro_f1"])
    print(f"=== Окончательная модель по CV macro-F1: {best_key} "
          f"({results[best_key]['name']})")

    print("=== Методические эксперименты")
    exp = run_experiments(Xtr, ytr)
    exp["search_svc"] = search_summary(searches.get("svm"), "svc")
    exp["search_hgb"] = search_summary(searches.get("hgb"), "hgb")

    best_fit = fitted[best_key]
    pred_te = best_fit.predict(Xte)
    score_te = scores_of(best_fit, Xte)
    per_class = per_class_metrics(yte, pred_te)
    cm_counts = confusion_matrix(yte, pred_te, labels=CLASSES)
    cm_norm = confusion_matrix(yte, pred_te, labels=CLASSES, normalize="true")
    auc_final = {int(c): a for c, a in
                 zip(CLASSES, auc_per_class(yte, score_te))}
    auc_logreg = {int(c): a for c, a in zip(
        CLASSES, auc_per_class(yte, scores_of(fitted["logreg"], Xte)))}
    auc_rf = {int(c): a for c, a in zip(
        CLASSES, auc_per_class(yte, scores_of(fitted["rf"], Xte)))}

    # Перестановочная важность — на отложенной части обучения, не на тесте
    Xa, Xv, ya, yv = train_test_split(
        Xtr, ytr, test_size=VAL_SIZE, stratify=ytr,
        random_state=RANDOM_STATE)
    imp_model = clone(best_fit).fit(Xa, ya)
    perm = permutation_importance(imp_model, Xv, yv, scoring="f1_macro",
                                  n_repeats=10, random_state=RANDOM_STATE,
                                  n_jobs=-1)
    perm_dict = {"mean": perm.importances_mean.tolist(),
                 "std": perm.importances_std.tolist()}

    coef = fitted["logreg"].named_steps["model"].coef_

    print("=== Рисунки")
    fig_class_distribution(tr, te)
    fig_domain(tr)
    fig_model_comparison(results)
    fig_tree_overfit(exp)
    fig_rf_ntrees(exp)
    fig_gb_iters(exp)
    fig_rf_gb_params(exp)
    fig_search(exp["search_svc"], exp["search_hgb"])
    fig_confusion_all(results)
    fig_confusion(cm_norm, cm_counts)
    fig_per_class(per_class)
    fig_roc(yte, score_te, per_class)
    fig_importance(perm_dict, FEATURES, coef)

    n_fig, n_tab, n_src, n_app = 13, 11, 9, 2
    make_tables(results, data, exp, per_class, auc_final, auc_logreg, auc_rf,
                perm_dict, cm_counts, best_key)
    make_macros(results, data, exp, per_class, auc_final, perm_dict, cm_norm,
                best_key, n_fig, n_tab, n_src, n_app)

    out = {
        "data": data,
        "results": results,
        "best_key": best_key,
        "experiments": exp,
        "per_class": {int(c): per_class[c] for c in CLASSES},
        "auc_final": {int(c): auc_final[c] for c in CLASSES},
        "auc_logreg": {int(c): auc_logreg[c] for c in CLASSES},
        "auc_rf": {int(c): auc_rf[c] for c in CLASSES},
        "perm_importance": perm_dict,
        "confusion_counts": cm_counts.tolist(),
        "confusion_norm": cm_norm.tolist(),
        "n_fig": n_fig, "n_tab": n_tab, "n_src": n_src, "n_app": n_app,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"Готово: {RESULTS_JSON}")


if __name__ == "__main__":
    import sys
    main(only_tables="tables" in sys.argv)
