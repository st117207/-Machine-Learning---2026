# -*- coding: utf-8 -*-
"""Моделирование задачи «Деградация подшипника» (вариант 13).

Методика
--------
* Обучающее испытание — набор 1 IMS (2156 записей, полный жизненный цикл).
  Отложенное испытание — набор 2 (984 записей), хронологически более поздний:
  разбиение только временное, одно испытание отложено полностью.
* Подбор гиперпараметров — GridSearchCV с TimeSeriesSplit(5) только по
  обучающему испытанию; тест (набор 2) используется один раз.
* Признаки по записи (по каждому подшипнику): RMS, дисперсия, эксцесс,
  пик-фактор и доли энергии в пяти широких полосах. Амплитудные признаки
  нормируются на здоровый базовый уровень начала испытания (медиана первых
  10 % записей) и логарифмируются. Дополнительно — агрегат «худшего
  подшипника» (поэлементный максимум по четырём подшипникам).
* Физически мотивированная постановка: кусочно-линейная цель
  min(RUL, C) — плато исправного состояния; порог C подбирается по КВ.
* Область применимости: признаки вне диапазона обучения ограничиваются его
  границами (конвейерный преобразователь RangeClipper).
* Модели: базовая (медиана), МНК, физическая постановка (МНК с усечённой
  целью), полином + гребень, Ridge, Lasso, Elastic Net.

Результаты записываются в report/tables/*.tex, report/figures/*.pdf и
data/processed/results.json.
"""
from __future__ import annotations

import json
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import (FIG_DIR, FEATURES_CSV, N_BOOTSTRAP, RANDOM_STATE,
                    RESULTS_JSON, TAB_DIR, TRAIN_TEST_NAME, HOLDOUT_TEST_NAME)

CV_FOLDS = TimeSeriesSplit(n_splits=5)
ALPHAS = list(np.logspace(-3, 4, 29))
L1_RATIOS = [0.1, 0.5, 0.9]
CAP_GRID = [160, 200, 250, 300]

AMP = ["rms", "var", "crest"]
BANDS = ["e0_1khz", "e1_2khz", "e2_3khz", "e3_5khz", "e5_10khz"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Добавляет нормированные и агрегированные признаки.

    Возвращает (df, признаки конфигурации A «по подшипникам» (36),
    признаки конфигурации B «худший подшипник» (9)).
    """
    df = df.copy().sort_values(["test", "t_start"]).reset_index(drop=True)
    acols = [f"b{b}_{s}" for b in (1, 2, 3, 4) for s in AMP]
    # здоровый базовый уровень: медиана первых 10 % записей каждого испытания
    baseline = df.groupby("test")[acols].transform(
        lambda s: s.iloc[: max(1, int(0.1 * len(s)))].median())
    for c in acols:
        df[c + "_logrel"] = np.log10(df[c] / baseline[c])

    feats_a = [f"b{b}_{s}_logrel" for b in (1, 2, 3, 4) for s in AMP]
    feats_a += [f"b{b}_{s}" for b in (1, 2, 3, 4) for s in ["kurt"] + BANDS]

    feats_b = []
    for s in AMP:
        name = f"max_{s}logrel"
        df[name] = df[[f"b{b}_{s}_logrel" for b in (1, 2, 3, 4)]].max(axis=1)
        feats_b.append(name)
    for s in ["kurt"] + BANDS:
        name = f"max_{s}"
        df[name] = df[[f"b{b}_{s}" for b in (1, 2, 3, 4)]].max(axis=1)
        feats_b.append(name)
    return df, feats_a, feats_b


class RangeClipper(BaseEstimator, TransformerMixin):
    """Ограничение признаков диапазоном обучающих данных.

    Значения вне [min, max] обучающей выборки заменяются границами:
    модель не экстраполирует по признакам за пределы области, на которой
    она определена. Границы запоминаются при обучении конвейера.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = X.min(axis=0)
        self.hi_ = X.max(axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _make_pipe(feats: list[str], model) -> Pipeline:
    return Pipeline([("clip", RangeClipper()), ("sc", StandardScaler()),
                     ("model", model)])


def _make_poly_pipe(feats: list[str], degree: int) -> Pipeline:
    """Конвейер с полиномиальным расширением: clip → станд. → полином → станд. → модель.

    После PolynomialFeatures обязательна повторная стандартизация (sc2):
    квадраты и попарные произведения имеют иной масштаб, чем исходные признаки.
    """
    return Pipeline([("clip", RangeClipper()), ("sc", StandardScaler()),
                     ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("sc2", StandardScaler()),
                     ("model", Ridge(random_state=RANDOM_STATE))])


def _identity(t):
    """Тождественное обратное преобразование (target уже в исходном масштабе)."""
    return t


def _capped(pipe: Pipeline, C: float) -> TransformedTargetRegressor:
    """Оборачивает конвейер в физически мотивированную цель min(RUL, C)."""
    return TransformedTargetRegressor(
        regressor=pipe, func=partial(_cap, c=C), inverse_func=_identity,
        check_inverse=False)


def _cap(y, c):
    return np.minimum(y, c)


def build_models(feats_a: list[str], feats_b: list[str]) -> dict[str, dict]:
    """Модельная матрица: 2 конфигурации признаков × обязательные модели."""
    m: dict[str, dict] = {}

    m["base"] = dict(
        name="Базовая модель (медиана)", cfg="A", feats=feats_a,
        pipe=Pipeline([("model", DummyRegressor(strategy="median"))]), grid=None)

    for cfg_key, cfg_name, feats in [("A", "по подшипникам (36)", feats_a),
                                     ("B", "худший подшипник (9)", feats_b)]:
        tag = "" if cfg_key == "A" else "_b"
        m[f"ols{tag}"] = dict(
            name=f"МНК, наивная постановка ({cfg_name})", cfg=cfg_key, feats=feats,
            pipe=_make_pipe(feats, LinearRegression()), grid=None)

        m[f"phys{tag}"] = dict(
            name=f"Физическая постановка, МНК с целью min(RUL, C) ({cfg_name})",
            cfg=cfg_key, feats=feats,
            pipe=_capped(_make_pipe(feats, LinearRegression()), CAP_GRID[0]),
            grid={"func": [partial(_cap, c=C) for C in CAP_GRID]})

        # Третья степень допускается только если КВ подтверждает улучшение:
        # для набора А (36 признаков) полином 3-й степени порождает ~9 тыс.
        # столбцов, поэтому сетка степеней ограничена [1, 2]; для набора Б
        # (9 признаков) разрешены степени 1–3.
        deg_grid = [1, 2] if cfg_key == "A" else [1, 2, 3]
        m[f"ridge_poly{tag}"] = dict(
            name=f"Полином + гребень, цель min(RUL, C) ({cfg_name})",
            cfg=cfg_key, feats=feats,
            pipe=_capped(_make_poly_pipe(feats, deg_grid[0]), CAP_GRID[0]),
            grid={"func": [partial(_cap, c=C) for C in CAP_GRID],
                  "regressor__poly__degree": deg_grid,
                  "regressor__model__alpha": ALPHAS})

        m[f"ridge{tag}"] = dict(
            name=f"Гребневая регрессия, цель min(RUL, C) ({cfg_name})",
            cfg=cfg_key, feats=feats,
            pipe=_capped(_make_pipe(feats, Ridge(random_state=RANDOM_STATE)), CAP_GRID[0]),
            grid={"func": [partial(_cap, c=C) for C in CAP_GRID],
                  "regressor__model__alpha": ALPHAS})

        m[f"lasso{tag}"] = dict(
            name=f"Лассо-регрессия, цель min(RUL, C) ({cfg_name})",
            cfg=cfg_key, feats=feats,
            pipe=_capped(_make_pipe(
                feats, Lasso(max_iter=200000, random_state=RANDOM_STATE)), CAP_GRID[0]),
            grid={"func": [partial(_cap, c=C) for C in CAP_GRID],
                  "regressor__model__alpha": ALPHAS})

        m[f"enet{tag}"] = dict(
            name=f"Упругая сеть, цель min(RUL, C) ({cfg_name})",
            cfg=cfg_key, feats=feats,
            pipe=_capped(_make_pipe(
                feats, ElasticNet(max_iter=200000, random_state=RANDOM_STATE)), CAP_GRID[0]),
            grid={"func": [partial(_cap, c=C) for C in CAP_GRID],
                  "regressor__model__alpha": ALPHAS,
                  "regressor__model__l1_ratio": L1_RATIOS})
    return m


# ---------------------------------------------------------------------------
# Метрики и форматирование
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred) -> dict[str, float]:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "R2": float(r2_score(y_true, y_pred))}


def _spearman_safe(y_pred, y_true) -> float:
    """Ранговая корреляция; NaN при постоянном прогнозе (например, медиана)."""
    if np.std(np.asarray(y_pred, dtype=float)) < 1e-12:
        return float("nan")
    return float(spearmanr(y_pred, y_true).statistic)


def _dof(model) -> int:
    """Эффективное число параметров: ненулевые коэффициенты + свободный член."""
    inner = model.regressor_ if hasattr(model, "regressor_") else model
    est = inner.steps[-1][1]
    if hasattr(est, "coef_"):
        return int(np.sum(np.abs(np.asarray(est.coef_)) > 1e-10)) + 1
    return 1


def fmt(x: float, nd: int = 2) -> str:
    """Десятичная запятая и типографский минус (для ГОСТ); NaN → «—»."""
    if x is None or not np.isfinite(x):
        return "---"
    s = f"{x:.{nd}f}"
    if float(s.replace(",", ".")) == 0 and s.startswith("-"):
        s = s[1:]
    return s.replace("-", r"\textminus{}").replace(".", ",")


def _pyval(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _describe_params(key: str, params: dict) -> str:
    """Человекочитаемая строка гиперпараметров для таблицы."""
    bits = []
    C = params.get("cap_hours")
    for k, v in params.items():
        if k == "cap_hours":
            continue
        if k.endswith("alpha"):
            bits.append(f"$\\alpha$={fmt(float(v), 3)}")
        elif k.endswith("l1_ratio"):
            bits.append(f"$l_1$={v}")
        elif k.endswith("degree"):
            bits.append(f"$d$={v}")
    if C is not None:
        bits.insert(1, f"$C$={C:g}")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------

def main(only_tables: bool = False) -> None:
    if only_tables:
        emit_tables()
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(FIG_DIR.glob("*.pdf")) + list(TAB_DIR.glob("*.tex")):
        old.unlink()

    df_raw = pd.read_csv(FEATURES_CSV, parse_dates=["t_start"])
    df, feats_a, feats_b = prepare_features(df_raw)
    tr = df[df["test"] == TRAIN_TEST_NAME].reset_index(drop=True)
    ho = df[df["test"] == HOLDOUT_TEST_NAME].reset_index(drop=True)

    models = build_models(feats_a, feats_b)
    results: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for key, spec in models.items():
        print(f"=== {key}: {spec['name']}")
        Xtr = tr[spec["feats"]].to_numpy()
        ytr = tr["rul_h"].to_numpy()
        Xho = ho[spec["feats"]].to_numpy()
        yho = ho["rul_h"].to_numpy()
        if spec["grid"] is not None:
            gs = GridSearchCV(clone(spec["pipe"]), spec["grid"], cv=CV_FOLDS,
                              scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
            gs.fit(Xtr, ytr)
            best = gs.best_estimator_
            params = {}
            for k, v in gs.best_params_.items():
                if k == "func":
                    params["cap_hours"] = float(v.keywords["c"])
                else:
                    params[k] = _pyval(v)
            cv_mae = float(-gs.best_score_)
            cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
            curves[key] = {"params": [dict(p) for p in gs.cv_results_["params"]],
                           "mae": (-gs.cv_results_["mean_test_score"]).tolist()}
        else:
            best = clone(spec["pipe"]).fit(Xtr, ytr)
            params = {}
            cv_mae = cv_std = None
            curves[key] = None
        cvr = cross_validate(best, Xtr, ytr, cv=CV_FOLDS,
                             scoring={"MAE": "neg_mean_absolute_error",
                                      "RMSE": "neg_root_mean_squared_error",
                                      "R2": "r2"})
        if cv_mae is None:
            cv_mae = float(-cvr["test_MAE"].mean())
            cv_std = float(cvr["test_MAE"].std())
        results[key] = dict(
            name=spec["name"], cfg=spec["cfg"], feats=len(spec["feats"]), params=params,
            train=metrics(ytr, best.predict(Xtr)),
            cv={"MAE": cv_mae, "MAE_std": cv_std,
                "RMSE": float(-cvr["test_RMSE"].mean()),
                "R2": float(cvr["test_R2"].mean())},
            test=metrics(yho, best.predict(Xho)),
            test_spearman=_spearman_safe(best.predict(Xho), yho),
            dof=_dof(best),
        )
        fitted[key] = (best, Xtr, ytr, Xho, yho)
        r_ = results[key]
        print(f"   CV MAE={cv_mae:.1f} ± {cv_std:.1f}  test MAE={r_['test']['MAE']:.1f} "
              f"R2={r_['test']['R2']:.2f} rho={r_['test_spearman']:.2f}")

    # ------------------------------------------------------------------
    # Окончательная модель: правило одного стандартного отклонения.
    # Среди моделей, чей MAE КВ не превышает минимум более чем на одно
    # стандартное отклонение по складкам КВ, выбирается самая простая —
    # с наименьшим эффективным числом параметров (число ненулевых
    # коэффициентов); при равенстве — лучший MAE КВ. Правило мотивировано
    # риском переноса: статистически неразличимые модели предпочтительно
    # брать разреженными и устойчивыми (правило 1SE, Hastie et al.).
    # ------------------------------------------------------------------
    k_min = min(results, key=lambda k: results[k]["cv"]["MAE"])
    thr = results[k_min]["cv"]["MAE"] + results[k_min]["cv"]["MAE_std"]
    within = [k for k in results if results[k]["cv"]["MAE"] <= thr]
    best_key = min(within, key=lambda k: (results[k]["dof"],
                                          results[k]["cv"]["MAE"]))
    print(f"Порог 1SE: {thr:.1f} ч; кандидаты: {within}")
    best_model, Xtr_b, ytr_b, Xho_b, yho_b = fitted[best_key]
    spec_b = models[best_key]
    print(f"Окончательная модель: {best_key} — {spec_b['name']}")

    # ------------------------------------------------------------------
    # Бутстрэп коэффициентов окончательной модели
    # ------------------------------------------------------------------
    inner = best_model.regressor_ if hasattr(best_model, "regressor_") else best_model
    final_est = inner.steps[-1][1]
    boot_summary = []
    coefs = np.zeros((0, 0))
    if hasattr(final_est, "coef_"):
        rng = np.random.default_rng(RANDOM_STATE)
        n = len(Xtr_b)
        coefs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            mdl = clone(best_model).fit(Xtr_b[idx], ytr_b[idx])
            reg = mdl.regressor_ if hasattr(mdl, "regressor_") else mdl
            coefs.append(np.asarray(reg.steps[-1][1].coef_))
        coefs = np.asarray(coefs)
        names = list(inner[:-1].get_feature_names_out(spec_b["feats"]))
        for j, nm in enumerate(names):
            c = coefs[:, j]
            boot_summary.append(dict(
                feature=str(nm), coef_median=float(np.median(c)),
                ci_lo=float(np.quantile(c, 0.025)), ci_hi=float(np.quantile(c, 0.975)),
                sign_pos=float((c > 0).mean())))
        boot_summary.sort(key=lambda d: -abs(d["coef_median"]))
        feat_names_final = names
    else:
        feat_names_final = []

    # ------------------------------------------------------------------
    # Диагностика остатков на отложенном испытании
    # ------------------------------------------------------------------
    pred_ho = best_model.predict(Xho_b)
    res_ho = yho_b - pred_ho
    # Диагностика остатков ведётся против признаков «худшего подшипника»
    # (набор Б) независимо от конфигурации итоговой модели
    Xw = ho[feats_b].to_numpy()
    resid_corr = dict(
        rho_pred=float(spearmanr(res_ho, pred_ho).statistic),
        rho_max_rms_logrel=float(spearmanr(res_ho, Xw[:, 0]).statistic),
        rho_max_kurt=float(spearmanr(res_ho, Xw[:, 3]).statistic),
    )

    # Минимальный прогноз наивной МНК на отложенном испытании (для текста)
    ols_spec = models["ols"]
    ols_min_pred = float(np.min(fitted["ols"][0].predict(ho[ols_spec["feats"]].to_numpy())))

    # Сводка по данным
    t1 = tr["t_start"]
    t2 = ho["t_start"]
    data_summary = {
        "n_train": int(len(tr)), "n_holdout": int(len(ho)),
        "t_start_train": t1.iloc[0].strftime("%d.%m.%Y"),
        "t_end_train": t1.iloc[-1].strftime("%d.%m.%Y"),
        "t_start_holdout": t2.iloc[0].strftime("%d.%m.%Y"),
        "t_end_holdout": t2.iloc[-1].strftime("%d.%m.%Y"),
        "rul_max_train": float(tr["rul_h"].max()),
        "rul_max_holdout": float(ho["rul_h"].max()),
        "dur_train_h": float((t1.iloc[-1] - t1.iloc[0]).total_seconds() / 3600),
        "dur_holdout_h": float((t2.iloc[-1] - t2.iloc[0]).total_seconds() / 3600),
    }

    out = dict(results=results, best_key=best_key, boot_summary=boot_summary,
               resid_corr=resid_corr, data_summary=data_summary,
               ols_min_pred=ols_min_pred)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    make_figures(df, tr, ho, results, curves, best_key, models, fitted,
                 boot_summary, feat_names_final, coefs, pred_ho, res_ho, Xw)
    make_tables(results, best_key, boot_summary, resid_corr, data_summary)
    make_macros(results, best_key, resid_corr, data_summary, ols_min_pred)
    print("Готово: таблицы, рисунки и макросы записаны.")


# ---------------------------------------------------------------------------
# Рисунки
# ---------------------------------------------------------------------------

def _t_hours(t: pd.Series) -> np.ndarray:
    return (t - t.iloc[0]).dt.total_seconds().to_numpy() / 3600.0


def make_figures(df, tr, ho, results, curves, best_key, models, fitted,
                 boot_summary, feat_names_final, coefs, pred_ho, res_ho, Xho_raw) -> None:
    t1 = _t_hours(tr["t_start"])
    t2 = _t_hours(ho["t_start"])

    # 1. Деградация: RMS по подшипникам в обоих испытаниях
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8), sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for b in range(1, 5):
        axes[0].plot(t1, tr[f"b{b}_rms"], color=colors[b - 1], lw=0.9,
                     label=f"подшипник {b}")
        axes[1].plot(t2, ho[f"b{b}_rms"], color=colors[b - 1], lw=0.9,
                     label=f"подшипник {b}")
    for ax, ttl in zip(axes, ["Испытание 1", "Испытание 2"]):
        ax.set_xlabel("Время испытания, ч")
        ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel("RMS, В")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_degradation.pdf")
    plt.close(fig)

    # 2. Перераспределение энергии по полосам (отказавший подшипник)
    bands = ["e0_1khz", "e1_2khz", "e2_3khz", "e3_5khz", "e5_10khz"]
    labels = ["0–1 кГц", "1–2 кГц", "2–3 кГц", "3–5 кГц", "5–10 кГц"]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), sharey=True)
    axes[0].stackplot(t1, *[tr[f"b3_{bd}"].to_numpy() for bd in bands],
                      labels=labels, alpha=0.85, linewidth=0.2)
    axes[1].stackplot(t2, *[ho[f"b1_{bd}"].to_numpy() for bd in bands],
                      alpha=0.85, linewidth=0.2)
    axes[0].set_title("Испытание 1, подшипник 3", fontsize=10)
    axes[1].set_title("Испытание 2, подшипник 1", fontsize=10)
    axes[0].set_ylabel("Доля энергии полосы")
    axes[0].set_xlabel("Время испытания, ч")
    axes[1].set_xlabel("Время испытания, ч")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_bands.pdf")
    plt.close(fig)

    # 3. Зависимость ошибки проверки от регуляризации и сложности
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    for key, label, color in [("lasso_b", "Lasso", "#d62728"),
                              ("ridge_b", "Ridge", "#1f77b4"),
                              ("enet_b", "Elastic Net", "#2ca02c")]:
        if curves.get(key):
            by_alpha: dict[float, list[float]] = {}
            for p, v in zip(curves[key]["params"], curves[key]["mae"]):
                by_alpha.setdefault(p["regressor__model__alpha"], []).append(v)
            xs = sorted(by_alpha)
            ys = [min(by_alpha[x]) for x in xs]
            axes[0].semilogx(xs, ys, marker="o", ms=3, lw=1.2, label=label, color=color)
    axes[0].set_xlabel(r"Коэффициент регуляризации $\alpha$")
    axes[0].set_ylabel("MAE кросс-валидации, ч")
    axes[0].legend(fontsize=8)
    dc = curves.get("ridge_poly_b")
    if dc:
        deg_best: dict[int, list[float]] = {}
        for p, v in zip(dc["params"], dc["mae"]):
            deg_best.setdefault(p["regressor__poly__degree"], []).append(v)
        ds = sorted(deg_best)
        axes[1].plot(ds, [min(deg_best[d]) for d in ds], marker="s", ms=4, lw=1.2,
                     color="#1f77b4")
        axes[1].set_xticks(ds)
    axes[1].set_xlabel("Степень полинома $d$")
    axes[1].set_ylabel("MAE кросс-валидации, ч")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_validation.pdf")
    plt.close(fig)

    # 4. Измеренное — предсказанное
    best_model, Xtr_b, ytr_b, Xho_b, yho_b = fitted[best_key]
    pred_tr = best_model.predict(Xtr_b)
    fig, ax = plt.subplots(figsize=(4.7, 4.4))
    lim = max(ytr_b.max(), pred_tr.max(), yho_b.max(), pred_ho.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="идеальный прогноз")
    ax.scatter(pred_tr, ytr_b, s=5, alpha=0.25, c="#9ecae1", label="обучение (испытание 1)")
    ax.scatter(pred_ho, yho_b, s=9, alpha=0.75, c="#d62728", marker="^",
               label="тест (испытание 2)")
    ax.set_xlabel("Прогноз времени до конца испытания, ч")
    ax.set_ylabel("Измеренное время до конца испытания, ч")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_pred_actual.pdf")
    plt.close(fig)

    # 5. Остатки на отложенном испытании
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9))
    axes[0].scatter(pred_ho, res_ho, s=9, alpha=0.6, c="#08519c")
    axes[0].axhline(0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("Прогноз, ч")
    axes[0].set_ylabel("Остаток, ч")
    axes[1].scatter(Xho_raw[:, 0], res_ho, s=9, alpha=0.6, c="#2ca02c")
    axes[1].axhline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel(r"lg(RMS/RMS$_{баз}$) худшего подшипника")
    axes[1].set_ylabel("Остаток, ч")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_residuals.pdf")
    plt.close(fig)

    # 6. Бутстрэп коэффициентов (ненулевые, топ-8 по модулю медианы)
    if coefs.size and boot_summary:
        name_to_idx = {nm: j for j, nm in enumerate(feat_names_final)}
        nz = [d for d in boot_summary if abs(d["coef_median"]) > 1e-9]
        order = [name_to_idx[d["feature"]] for d in (nz or boot_summary)[:8]]
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        data = [coefs[:, j] for j in order]
        names = [feat_names_final[j] for j in order]
        vp = ax.violinplot(data, showextrema=False, widths=0.85)
        for pc in vp["bodies"]:
            pc.set_facecolor("#9ecae1")
            pc.set_alpha(0.7)
        ax.boxplot(data, widths=0.18, showfliers=False, medianprops={"color": "k"})
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("Коэффициент")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig6_bootstrap.pdf")
        plt.close(fig)

    # 7. Интерполяционная оценка и перенос
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    keys = ["base", best_key]
    labels = ["Базовая модель", "Окончательная модель"]
    cv_mae = [results[k]["cv"]["MAE"] for k in keys]
    te_mae = [results[k]["test"]["MAE"] for k in keys]
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, cv_mae, w, label="Кросс-валидация (интерполяция)", color="#08519c")
    ax.bar(x + w / 2, te_mae, w, label="Перенос (испытание 2)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE, ч")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_transfer.pdf")
    plt.close(fig)

    # Д1. Эксцесс по подшипникам (приложение)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8), sharey=False)
    for b in range(1, 5):
        axes[0].plot(t1, tr[f"b{b}_kurt"], color=colors[b - 1], lw=0.9, label=f"подшипник {b}")
        axes[1].plot(t2, ho[f"b{b}_kurt"], color=colors[b - 1], lw=0.9)
    axes[0].set_ylabel("Эксцесс")
    axes[0].set_xlabel("Время испытания 1, ч")
    axes[1].set_xlabel("Время испытания 2, ч")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figD1_kurtosis.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Таблицы и макросы
# ---------------------------------------------------------------------------

ORDER = ["base", "ols", "phys", "ridge_poly", "ridge", "lasso", "enet",
         "ols_b", "phys_b", "ridge_poly_b", "ridge_b", "lasso_b", "enet_b"]
SHORT = {"base": "Базовая (медиана)", "ols": "МНК (наивная)", "phys": "МНК + цель min(RUL, C)",
         "ridge_poly": "Полином + гребень", "ridge": "Ridge", "lasso": "Lasso",
         "enet": "Elastic Net"}


def _short(key: str) -> str:
    base_key = key[:-2] if key.endswith("_b") else key
    return SHORT[base_key]


def make_tables(results, best_key, boot_summary, resid_corr, data_summary) -> None:
    # Таблица 1 — сводка по испытаниям
    lines = [
        r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Характеристика исходных испытаний}\label{tab:data}",
        r"\begin{tabular}{p{3.9cm}p{4.3cm}p{4.3cm}}", r"\toprule",
        r"Параметр & Испытание 1 (обучение) & Испытание 2 (тест) \\", r"\midrule",
        f"Число записей & {data_summary['n_train']} & {data_summary['n_holdout']} \\\\",
        f"Период & {data_summary['t_start_train']} --- {data_summary['t_end_train']} & "
        f"{data_summary['t_start_holdout']} --- {data_summary['t_end_holdout']} \\\\",
        f"Длительность, ч & {fmt(data_summary['dur_train_h'], 1)} & "
        f"{fmt(data_summary['dur_holdout_h'], 1)} \\\\",
        f"Максимум прокси-ресурса, ч & {fmt(data_summary['rul_max_train'], 1)} & "
        f"{fmt(data_summary['rul_max_holdout'], 1)} \\\\",
        f"Отказавший подшипник & №3 (внутр. кольцо), №4 (ролики) & №1 (наружн. кольцо) \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (TAB_DIR / "tab_data.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 2 — гиперпараметры
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Подобранные по кросс-валидации гиперпараметры моделей}\label{tab:hyper}",
             r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.7cm}c"
             r">{\raggedright\arraybackslash}p{3.3cm}cc}", r"\toprule",
             r"Модель & Призн. & Гиперпараметры & $p_{\text{эф}}$ & MAE КВ, ч \\",
             r"\midrule"]
    for k in ORDER:
        if k not in results:
            continue
        r_ = results[k]
        lines.append(f"{_short(k)} & {('Б' if r_['cfg'] == 'B' else 'А')} & "
                     f"{_describe_params(k, r_['params'])} & {r_['dof']} & "
                     f"{fmt(r_['cv']['MAE'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_hyper.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 3 — метрики
    header = [
        r" & & \multicolumn{3}{c}{Обучение} & \multicolumn{3}{c}{Кросс-валидация} & "
        r"\multicolumn{3}{c}{Тест (испытание 2)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Модель & Призн. & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ \\",
        r"\midrule"]
    body = []
    for k in ORDER:
        if k not in results:
            continue
        r_ = results[k]
        row = [_short(k), ("Б" if r_["cfg"] == "B" else "А")]
        for part in ("train", "cv", "test"):
            row += [fmt(r_[part]["MAE"], 1), fmt(r_[part]["RMSE"], 1),
                    fmt(r_[part]["R2"], 2)]
        body.append(" & ".join(row) + r" \\")
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Метрики качества моделей: MAE, RMSE, $R^2$ (часы до "
             r"конца испытания)}\label{tab:metrics}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{p{2.9cm}c rrr rrr rrr}", r"\toprule"] \
            + header + body \
            + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB_DIR / "tab_metrics.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 4 — R2 и ранговая корреляция на тесте
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Качество прогнозов на отложенном испытании}\label{tab:test}",
             r"\begin{tabular}{p{3.4cm}c ccc}", r"\toprule",
             r"Модель & Призн. & $R^2$ (тест) & MAE (тест), ч & $\rho$ Спирмена \\", r"\midrule"]
    for k in ORDER:
        if k not in results:
            continue
        r_ = results[k]
        lines.append(f"{_short(k)} & {('Б' if r_['cfg'] == 'B' else 'А')} & "
                     f"{fmt(r_['test']['R2'], 2)} & {fmt(r_['test']['MAE'], 1)} & "
                     f"{fmt(r_['test_spearman'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_test_quality.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 5 — коэффициенты и бутстрэп: только ненулевые коэффициенты
    nz = [d for d in boot_summary if abs(d["coef_median"]) > 1e-9]
    rows = nz if nz else boot_summary
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Бутстрэп-оценки ненулевых коэффициентов итоговой "
             r"модели (95\,\%-ные доверительные интервалы)}\label{tab:coef}",
             r"\begin{tabular}{p{5.6cm}rrrr}", r"\toprule",
             r"Признак & Медиана & 2,5\,\% & 97,5\,\% & Доля $\beta>0$ \\", r"\midrule"]
    for d in rows[:9]:
        nm = str(d["feature"]).replace("_", r"\_")
        lines.append(f"{nm} & {fmt(d['coef_median'], 3)} & {fmt(d['ci_lo'], 3)} & "
                     f"{fmt(d['ci_hi'], 3)} & {fmt(d['sign_pos'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_coef.tex").write_text("\n".join(lines), encoding="utf-8")

    # Таблица 6 — корреляции остатков
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Ранговые корреляции остатков отложенного испытания "
             r"с факторами}\label{tab:resid}",
             r"\begin{tabular}{p{8.5cm}c}", r"\toprule",
             r"Фактор & Коэффициент Спирмена \\", r"\midrule",
             f"Прогноз модели & {fmt(resid_corr['rho_pred'], 2)} \\\\",
             f"lg(RMS/RMS$_{{\\text{{баз}}}}$) худшего подшипника & {fmt(resid_corr['rho_max_rms_logrel'], 2)} \\\\",
             f"Эксцесс худшего подшипника & {fmt(resid_corr['rho_max_kurt'], 2)} \\\\",
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB_DIR / "tab_resid.tex").write_text("\n".join(lines), encoding="utf-8")


def make_macros(results, best_key, resid_corr, data_summary, ols_min_pred: float) -> None:
    r = results[best_key]
    # Гиперпараметры окончательной модели одной строкой
    bits = []
    for k, v in r["params"].items():
        if k.endswith("alpha"):
            bits.append(f"$\\alpha$={fmt(float(v), 3)}")
        elif k.endswith("l1_ratio"):
            bits.append(f"$l_1$={v}")
        elif k.endswith("degree"):
            bits.append(f"$d$={v}")
        elif k == "cap_hours":
            bits.append(f"$C$={v:g}")
    hyper = ", ".join(bits) if bits else "---"
    macros = [
        ("BestModelName", results[best_key]["name"]),
        ("BestHyper", hyper),
        ("BestNFeat", str(r["feats"])),
        ("BestDof", str(r["dof"])),
        ("BestThrOneSE", fmt(results[min(results, key=lambda k: results[k]["cv"]["MAE"])]
                             ["cv"]["MAE"] + results[min(results, key=lambda k: results[k]["cv"]["MAE"])]
                             ["cv"]["MAE_std"], 1)),
        ("BestMAECV", fmt(r["cv"]["MAE"], 1)),
        ("BestMAECVstd", fmt(r["cv"]["MAE_std"], 1)),
        ("BestRMSECV", fmt(r["cv"]["RMSE"], 1)),
        ("BestRtwoCV", fmt(r["cv"]["R2"], 2)),
        ("BestMAETest", fmt(r["test"]["MAE"], 1)),
        ("BestRMSETest", fmt(r["test"]["RMSE"], 1)),
        ("BestRtwoTest", fmt(r["test"]["R2"], 2)),
        ("BestMAETrain", fmt(r["train"]["MAE"], 1)),
        ("BestRMSETrain", fmt(r["train"]["RMSE"], 1)),
        ("BestRtwoTrain", fmt(r["train"]["R2"], 2)),
        ("BestRhoTest", fmt(r["test_spearman"], 2)),
        ("BaseMAETest", fmt(results["base"]["test"]["MAE"], 1)),
        ("BaseRMSETest", fmt(results["base"]["test"]["RMSE"], 1)),
        ("OlsMAETest", fmt(results["ols"]["test"]["MAE"], 1)),
        ("OlsMAECV", fmt(results["ols"]["cv"]["MAE"], 1)),
        ("OlsMAETrain", fmt(results["ols"]["train"]["MAE"], 1)),
        ("OlsRtwoTrain", fmt(results["ols"]["train"]["R2"], 2)),
        ("OlsMinPred", fmt(ols_min_pred, 0)),
        ("PhysAMAETest", fmt(results["phys"]["test"]["MAE"], 1)),
        ("PhysBMAETest", fmt(results["phys_b"]["test"]["MAE"], 1)),
        ("RidgeAMAETest", fmt(results["ridge"]["test"]["MAE"], 1)),
        ("LassoAMAETest", fmt(results["lasso"]["test"]["MAE"], 1)),
        ("OlsBMAETest", fmt(results["ols_b"]["test"]["MAE"], 1)),
        ("RidgePolyBMAETest", fmt(results["ridge_poly_b"]["test"]["MAE"], 1)),
        ("RidgeBMAETest", fmt(results["ridge_b"]["test"]["MAE"], 1)),
        ("ENetBMAETest", fmt(results["enet_b"]["test"]["MAE"], 1)),
        ("ENetAMAETest", fmt(results["enet"]["test"]["MAE"], 1)),
        ("RhoResidPred", fmt(resid_corr["rho_pred"], 2)),
        ("RhoResidRMS", fmt(resid_corr["rho_max_rms_logrel"], 2)),
        ("RhoResidKurt", fmt(resid_corr["rho_max_kurt"], 2)),
        ("NTrain", str(data_summary["n_train"])),
        ("NHoldout", str(data_summary["n_holdout"])),
        ("DurTrain", fmt(data_summary["dur_train_h"], 0)),
        ("DurHoldout", fmt(data_summary["dur_holdout_h"], 0)),
        ("RulMaxTrain", fmt(data_summary["rul_max_train"], 0)),
        ("RulMaxHoldout", fmt(data_summary["rul_max_holdout"], 0)),
    ]
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros]
    (TAB_DIR / "macros.tex").write_text("\n".join(lines), encoding="utf-8")


def emit_tables() -> None:
    """Перезапись таблиц и макросов по сохранённому results.json (без обучения)."""
    d = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    make_tables(d["results"], d["best_key"], d["boot_summary"],
                d["resid_corr"], d["data_summary"])
    make_macros(d["results"], d["best_key"], d["resid_corr"],
                d["data_summary"], d["ols_min_pred"])
    print("Готово: таблицы и макросы перезаписаны по results.json.")


if __name__ == "__main__":
    import sys
    main(only_tables="tables" in sys.argv)
