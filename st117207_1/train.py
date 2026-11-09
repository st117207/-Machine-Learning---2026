"""
Обучение и оценка регрессионных моделей.
Задача 1 (регрессия), вариант 13 — деградация подшипника (NASA IMS Bearing Dataset).

Цель: по вибрационным признакам оценить время, остающееся до конца эксперимента (RUL).

Стратегия разбиения (только временное):
  - Набор 1 (октябрь-ноябрь 2003) — обучение + кросс-валидация по времени (TimeSeriesSplit).
  - Набор 2 (февраль 2004) — отложенное испытание между экспериментами (проверка переноса).
  - Набор 3 (март-апрель 2004) — полностью отложенный объект (финальная проверка).

Обязательные элементы варианта 13 и общих требований:
  * базовая модель + МНК;
  * физически мотивированная линейная постановка;
  * полиномиальные признаки 2-й степени;
  * Ridge + Lasso/Elastic Net (регуляризация);
  * стандартизация внутри конвейера, подбор гиперпараметров только на обучении + CV;
  * метрики MAE/RMSE/R² (обучение, CV, тест); графики измеренное/предсказанное, остатки;
  * устойчивость коэффициентов (бутстрэп/повторная CV);
  * проверка на отложенном объекте vs интерполяционная оценка.
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "data" / "features.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Признаки (без целевой и служебных колонок)
FEATURE_COLS = [
    "rms", "var", "kurt", "crest",
    "band_0_500", "band_500_1000", "band_1000_2000",
    "band_2000_4000", "band_4000_10000",
]
TARGET = "rul_hours"
SERVE_COLS = ["set", "file", "file_time", TARGET]

# Физически мотивированный поднабор признаков (интерпретируемые деградацией):
# общий уровень вибрации (RMS) и высокочастотная энергия (растёт при износе подшипника).
PHYS_FEATURES = ["rms", "band_4000_10000"]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df["file_time"] = pd.to_datetime(df["file_time"])
    return df


def split_sets(df):
    sets = {s: g.sort_values("file_time").reset_index(drop=True) for s, g in df.groupby("set")}
    return sets


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def build_pipeline(model, degree=1):
    steps = [("scale", StandardScaler())]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("model", model))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------
def model_baseline(y_train):
    """Базовая модель: константа (среднее RUL по обучению)."""
    return float(np.mean(y_train))


def predict_baseline(value, n):
    return np.full(n, value)


def model_ols(X, y):
    return build_pipeline(LinearRegression(), degree=1).fit(X, y)


def model_ols_phys(X, y):
    return build_pipeline(LinearRegression(), degree=1).fit(X[PHYS_FEATURES], y)


def model_poly2(X, y):
    return build_pipeline(LinearRegression(), degree=2).fit(X, y)


def model_ridge(X, y, alpha):
    return build_pipeline(Ridge(alpha=alpha, random_state=0), degree=2).fit(X, y)


def model_lasso(X, y, alpha):
    return build_pipeline(Lasso(alpha=alpha, random_state=0, max_iter=10000), degree=2).fit(X, y)


def model_elasticnet(X, y, alpha):
    return build_pipeline(ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=0, max_iter=10000), degree=2).fit(X, y)


# ---------------------------------------------------------------------------
# Кросс-валидация по времени на обучающем наборе (набор 1)
# ---------------------------------------------------------------------------
def ts_cv_score(pipe_builder, X, y, X_cols=None, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {"MAE": [], "RMSE": [], "R2": []}
    for train_idx, val_idx in tscv.split(X):
        X_tr_full, X_val_full = X.iloc[train_idx], X.iloc[val_idx]
        X_tr = X_tr_full[X_cols] if X_cols is not None else X_tr_full
        X_val = X_val_full[X_cols] if X_cols is not None else X_val_full
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe = pipe_builder(X_tr, y_tr)
        pred = pipe.predict(X_val)
        m = metrics(y_val.values, pred)
        for k in scores:
            scores[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in scores.items()}


def ts_cv_predict(pipe_builder, X, y, n_splits=5):
    """Предсказания на валидационных фолдах (для графиков и метрик CV)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        pipe = pipe_builder(X_tr, y_tr)
        oof[val_idx] = pipe.predict(X_val)
    return oof


# ---------------------------------------------------------------------------
# Бутстрэп устойчивости коэффициентов (линейная модель)
# ---------------------------------------------------------------------------
def bootstrap_coefs(X, y, n_boots=200):
    """Линейная регрессия (OLS со стандартизацией) + бутстрэп."""
    from sklearn.linear_model import LinearRegression as LR
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    base = LR().fit(Xs, y)
    coefs = []
    rng = np.random.RandomState(0)
    n = len(y)
    for _ in range(n_boots):
        idx = rng.randint(0, n, n)
        m = LR().fit(Xs[idx], y.values[idx])
        coefs.append(m.coef_)
    coefs = np.array(coefs)
    return {
        "feature": list(X.columns),
        "coef_mean": coefs.mean(axis=0),
        "coef_std": coefs.std(axis=0),
        "coef_min": coefs.min(axis=0),
        "coef_max": coefs.max(axis=0),
        "sign_stable": np.all(coefs >= 0, axis=0) | np.all(coefs <= 0, axis=0),
        "intercept_mean": float(np.mean([LR().fit(Xs[idx], y.values[idx]).intercept_ for idx in [rng.randint(0, n, n) for _ in range(n_boots)]])),
    }


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()


def plot_history(sets):
    """RUL vs время для каждого испытания (история деградации)."""
    plt.figure(figsize=(12, 5))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for s, g in sets.items():
        plt.plot(g["file_time"], g[TARGET], color=colors.get(s, "gray"),
                 label=f"Испытание {s}", alpha=0.7, linewidth=1)
    plt.xlabel("Время записи")
    plt.ylabel("RUL (часы до конца эксперимента)")
    plt.title("Динамика остаточного ресурса (RUL) по времени испытаний")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("history_rul.png")


def plot_pred_vs_true(y_true, y_pred, title, fname):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6, edgecolor="k", linewidth=0.3)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="идеал")
    plt.xlabel("Измеренное RUL (ч)")
    plt.ylabel("Предсказанное RUL (ч)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(fname)


def plot_residuals(y_true, y_pred, title, fname):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(7, 4))
    plt.scatter(y_pred, resid, s=12, alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Предсказанное RUL (ч)")
    plt.ylabel("Остаток (измеренное − предсказанное)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    savefig(fname)


def plot_validation_curve(alphas, train_err, cv_err, fname, ylabel="RMSE (ч)"):
    plt.figure(figsize=(7, 4))
    plt.semilogx(alphas, train_err, marker="o", label="обучение")
    plt.semilogx(alphas, cv_err, marker="o", label="CV (TimeSeriesSplit)")
    plt.xlabel("alpha (сила регуляризации)")
    plt.ylabel(ylabel)
    plt.title("Зависимость ошибки от силы регуляризации (Ridge)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(fname)


# ---------------------------------------------------------------------------
# Основной запуск
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    sets = split_sets(df)

    # ---------- обучающий набор (испытание 1) ----------
    train = sets[1]
    X_tr = train[FEATURE_COLS]
    y_tr = train[TARGET]

    # ---------- отложенные наборы ----------
    test2 = sets[2]  # проверка переноса между испытаниями
    X_te2 = test2[FEATURE_COLS]
    y_te2 = test2[TARGET]

    test3 = sets[3]  # полностью отложенный объект (финальный тест)
    X_te3 = test3[FEATURE_COLS]
    y_te3 = test3[TARGET]

    results = {}

    # --- 1. Базовая модель (константа) ---
    base_val = model_baseline(y_tr.values)
    res = {}
    res["train"] = metrics(y_tr.values, predict_baseline(base_val, len(y_tr)))
    res["cv"] = metrics(y_tr.values, predict_baseline(base_val, len(y_tr)))
    res["test2"] = metrics(y_te2.values, predict_baseline(base_val, len(y_te2)))
    res["test3"] = metrics(y_te3.values, predict_baseline(base_val, len(y_te3)))
    results["baseline"] = res
    print("baseline (const):", res)

    # --- 2. МНК (линейная регрессия) ---
    ols = model_ols(X_tr, y_tr)
    res = {}
    res["train"] = metrics(y_tr.values, ols.predict(X_tr))
    res["cv"] = ts_cv_score(lambda X, y: model_ols(X, y), X_tr, y_tr)
    res["test2"] = metrics(y_te2.values, ols.predict(X_te2))
    res["test3"] = metrics(y_te3.values, ols.predict(X_te3))
    results["ols"] = res
    print("ols:", res)

    # OLS predictions for plots
    ols_cv_pred = ts_cv_predict(lambda X, y: model_ols(X, y), X_tr, y_tr)
    mask = ~np.isnan(ols_cv_pred)

    # --- 3. Физически мотивированная линейная постановка ---
    ols_phys = model_ols_phys(X_tr, y_tr)
    res = {}
    res["train"] = metrics(y_tr.values, ols_phys.predict(X_tr[PHYS_FEATURES]))
    res["cv"] = ts_cv_score(lambda X, y: model_ols_phys(X, y), X_tr, y_tr, X_cols=PHYS_FEATURES)
    res["test2"] = metrics(y_te2.values, ols_phys.predict(X_te2[PHYS_FEATURES]))
    res["test3"] = metrics(y_te3.values, ols_phys.predict(X_te3[PHYS_FEATURES]))
    results["ols_phys"] = res
    print("ols_phys:", res)

    # --- 4. Полиномиальная регрессия 2-й степени ---
    poly2 = model_poly2(X_tr, y_tr)
    res = {}
    res["train"] = metrics(y_tr.values, poly2.predict(X_tr))
    res["cv"] = ts_cv_score(lambda X, y: model_poly2(X, y), X_tr, y_tr)
    res["test2"] = metrics(y_te2.values, poly2.predict(X_te2))
    res["test3"] = metrics(y_te3.values, poly2.predict(X_te3))
    results["poly2"] = res
    print("poly2:", res)

    # --- 5. Ridge с подбором alpha по CV на обучении ---
    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    cv_rmse = []
    train_rmse = []
    for a in alphas:
        pipe = build_pipeline(Ridge(alpha=a, random_state=0), degree=2)
        pipe.fit(X_tr, y_tr)
        train_rmse.append(np.sqrt(mean_squared_error(y_tr, pipe.predict(X_tr))))
        # CV RMSE
        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        for tr_i, val_i in tscv.split(X_tr):
            p2 = build_pipeline(Ridge(alpha=a, random_state=0), degree=2)
            p2.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
            rmses.append(np.sqrt(mean_squared_error(y_tr.iloc[val_i], p2.predict(X_tr.iloc[val_i]))))
        cv_rmse.append(float(np.mean(rmses)))
    best_a = alphas[int(np.argmin(cv_rmse))]
    ridge = model_ridge(X_tr, y_tr, best_a)
    res = {}
    res["train"] = metrics(y_tr.values, ridge.predict(X_tr))
    res["cv"] = ts_cv_score(lambda X, y: model_ridge(X, y, best_a), X_tr, y_tr)
    res["test2"] = metrics(y_te2.values, ridge.predict(X_te2))
    res["test3"] = metrics(y_te3.values, ridge.predict(X_te3))
    results["ridge"] = {**res, "best_alpha": best_a}
    print(f"ridge (alpha={best_a}):", res)
    plot_validation_curve(alphas, train_rmse, cv_rmse, "ridge_validation_curve.png")

    # --- 6. Lasso с подбором alpha ---
    lasso_alphas = [0.001, 0.01, 0.1, 1, 10]
    lasso_cv = []
    for a in lasso_alphas:
        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        for tr_i, val_i in tscv.split(X_tr):
            p2 = build_pipeline(Lasso(alpha=a, random_state=0, max_iter=20000), degree=2)
            p2.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i])
            rmses.append(np.sqrt(mean_squared_error(y_tr.iloc[val_i], p2.predict(X_tr.iloc[val_i]))))
        lasso_cv.append(float(np.mean(rmses)))
    best_la = lasso_alphas[int(np.argmin(lasso_cv))]
    lasso = model_lasso(X_tr, y_tr, best_la)
    res = {}
    res["train"] = metrics(y_tr.values, lasso.predict(X_tr))
    res["cv"] = ts_cv_score(lambda X, y: model_lasso(X, y, best_la), X_tr, y_tr)
    res["test2"] = metrics(y_te2.values, lasso.predict(X_te2))
    res["test3"] = metrics(y_te3.values, lasso.predict(X_te3))
    results["lasso"] = {**res, "best_alpha": best_la}
    print(f"lasso (alpha={best_la}):", res)

    # ---------- графики ----------
    plot_history(sets)

    plot_pred_vs_true(y_tr.values, ols.predict(X_tr),
                      "МНК: измеренное vs предсказанное (обучение, испытание 1)",
                      "ols_train_pred.png")
    plot_pred_vs_true(y_tr.values[mask], ols_cv_pred[mask],
                      "МНК: измеренное vs предсказанное (CV фолды, испытание 1)",
                      "ols_cv_pred.png")
    plot_pred_vs_true(y_te3.values, ridge.predict(X_te3),
                      f"Ridge (α={best_a}): измеренное vs предсказанное (испытание 3, отложенный объект)",
                      "ridge_test3_pred.png")
    plot_pred_vs_true(y_te2.values, ridge.predict(X_te2),
                      f"Ridge (α={best_a}): измеренное vs предсказанное (испытание 2, перенос)",
                      "ridge_test2_pred.png")

    plot_residuals(y_te3.values, ridge.predict(X_te3),
                   f"Ridge (α={best_a}): остатки (испытание 3)",
                   "ridge_test3_resid.png")

    # ---------- устойчивость коэффициентов (бутстрэп) ----------
    boot = bootstrap_coefs(X_tr, y_tr, n_boots=200)
    boot_df = pd.DataFrame({
        "feature": boot["feature"],
        "coef_mean": boot["coef_mean"],
        "coef_std": boot["coef_std"],
        "coef_min": boot["coef_min"],
        "coef_max": boot["coef_max"],
        "sign_stable": boot["sign_stable"],
    })
    boot_df.to_csv(RESULTS / "bootstrap_coefs.csv", index=False)
    print("\nБутстрэп коэффициентов (OLS на испытании 1):")
    print(boot_df.round(4))

    # ---------- сохранение метрик ----------
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # таблица метрик для отчёта
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for split in ["train", "cv", "test2", "test3"]:
            if split in r:
                for k, v in r[split].items():
                    row[f"{split}_{k}"] = round(v, 4)
        if "best_alpha" in r:
            row["best_alpha"] = r["best_alpha"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "metrics_table.csv", index=False)
    print("\nМетрики сохранены в results/")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
