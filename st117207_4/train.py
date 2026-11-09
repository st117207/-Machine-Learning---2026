"""
Задача 4 (нейронные сети, PyTorch), вариант 21 — Дрейф отклика массива химических сенсоров.

Обязательные элементы (общие требования к задаче 4 + вариант 21):
  * PyTorch (torch.nn, torch.optim, мини-батчи); сравнение с Ridge/LogisticRegression и RandomForest/HistGradientBoosting
  * Явный цикл обучения: инициализация, мини-батчи, прямой проход, функция потерь,
    loss.backward(), optimizer.step(), optimizer.zero_grad(), фиксированные random_state
  * Проверочная выборка (партии 7-8) для early stopping (patience) и сохранения лучших весов
  * Сравнение активаций identity / ReLU / tanh при одной архитектуре
  • Перебор глубины (1-4 скрытых слоя) и ширины (8-256 нейронов), число обучаемых параметров
  • Исследование оптимизатора (SGD с моментом, Adam, AdamW), LR (лог. сетка, >=4 значения),
    размера мини-батча, weight decay, dropout
  * Заполнение пропусков, стандартизация внутри обучающего конвейера (тест не участвует)
  * Метрики классификации: матрица ошибок, сбалансированная точность, макро-F1, F1 по классам,
    макро-ROC-AUC один-против-остальных (где есть оба класса)
  * Два предметных графика варианта: макро-F1 по партиям; матрица ошибок на поздних vs ранних партиях
  * Эксперимент с физическим априорным знанием и сравнение с обычной сетью
  * Воспроизводимый отчёт: python build_report.py -> report.pdf
"""

from pathlib import Path
import json
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figure"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cpu")
N_FEATURES = 128
N_CLASSES = 6
CLASS_NAMES = [f"gas_{i}" for i in range(1, 7)]

# split по партиям (требование варианта)
SPLIT_OF_PARTY = {1: "train", 2: "train", 3: "train", 4: "train", 5: "train", 6: "train",
                  7: "val", 8: "val", 9: "test", 10: "test"}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_DIR / "features.csv")
    feature_cols = [f"f{i}" for i in range(N_FEATURES)]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64) - 1  # 0..5
    split = df["split"].values
    party = df["party"].values.astype(np.int64)

    # стандартизация внутри обучающего конвейера (только по train)
    scaler = StandardScaler()
    X_train_raw = X[split == "train"]
    scaler.fit(X_train_raw)
    X_s = scaler.transform(X)

    masks = {s: (split == s) for s in ["train", "val", "test"]}
    data = {}
    for s in ["train", "val", "test"]:
        m = masks[s]
        data[s] = {
            "X": torch.tensor(X_s[m], dtype=torch.float32, device=DEVICE),
            "y": torch.tensor(y[m], dtype=torch.long, device=DEVICE),
            "party": party[m],
        }
    return data, feature_cols, scaler


# ---------------------------------------------------------------------------
# Архитектура сети
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, n_in, n_hidden_layers, n_hidden_units, activation, n_out, dropout=0.0):
        super().__init__()
        layers = []
        act = {"identity": nn.Identity(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        in_dim = n_in
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, n_hidden_units))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = n_hidden_units
        layers.append(nn.Linear(in_dim, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Обучение (явный цикл с early stopping)
# ---------------------------------------------------------------------------
def train_model(model, Xtr, ytr, Xval, yval, optimizer_name, lr, batch_size,
                weight_decay, epochs, patience, dropout=0.0):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(Xtr, ytr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "grad_norm": [], "weight_norm": []}

    for epoch in range(epochs):
        # ---- обучение ----
        model.train()
        running_loss = 0.0
        n_batches = 0
        grad_norms = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # норма градиента
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        # норма весов
        w_norm = 0.0
        for p in model.parameters():
            w_norm += p.data.norm(2).item() ** 2
        w_norm = w_norm ** 0.5

        # ---- проверка ----
        model.eval()
        with torch.no_grad():
            val_logits = model(Xval)
            val_loss = criterion(val_logits, yval).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == yval).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(grad_norm)
        history["weight_norm"].append(w_norm)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # загрузим лучшие веса
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_loss


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    return pred, probs, y_true


# ---------------------------------------------------------------------------
# Метрики классификации (с учётом отсутствующих классов в партии)
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_prob, labels):
    res = {}
    res["accuracy"] = float((y_true == y_pred).mean())
    res["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    res["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    res["f1_per_class"] = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    res["roc_auc_ovr_macro"] = float("nan")
    if y_prob is None:
        return res
    # macro ROC-AUC один-против-остальных, только где есть оба класса
    aucs = []
    for c in labels:
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or (y_bin == 0).sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_bin, y_prob[:, c]))
        except ValueError:
            pass
    res["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else float("nan")
    return res


# ---------------------------------------------------------------------------
# Сохранение метрик
# ---------------------------------------------------------------------------
def save_metrics(name, metrics_dict):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)


# ===========================================================================
# Базовые модели (сравнение с сетью)
# ===========================================================================
def run_baseline_models(data):
    results = {}
    Xtr = data["train"]["X"].cpu().numpy()
    ytr = data["train"]["y"].cpu().numpy()
    Xval = data["val"]["X"].cpu().numpy()
    yval = data["val"]["y"].cpu().numpy()
    Xte = data["test"]["X"].cpu().numpy()
    yte = data["test"]["y"].cpu().numpy()
    labels = np.arange(N_CLASSES)

    # LogisticRegression
    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)
    lr.fit(Xtr, ytr)
    p_val = lr.predict(Xval)
    p_te = lr.predict(Xte)
    pr_te = lr.predict_proba(Xte) if hasattr(lr, "predict_proba") else None
    results["LogisticRegression"] = {
        "val": classification_metrics(yval, p_val, None, labels),
        "test": classification_metrics(yte, p_te, pr_te, labels),
        "pred_test": p_te,
        "prob_test": pr_te,
    }

    # RandomForest
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=1)
    rf.fit(Xtr, ytr)
    p_val = rf.predict(Xval)
    p_te = rf.predict(Xte)
    pr_te = rf.predict_proba(Xte)
    results["RandomForest"] = {
        "val": classification_metrics(yval, p_val, None, labels),
        "test": classification_metrics(yte, p_te, pr_te, labels),
        "pred_test": p_te,
        "prob_test": pr_te,
    }

    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_iter=200)
    hgb.fit(Xtr, ytr)
    p_val = hgb.predict(Xval)
    p_te = hgb.predict(Xte)
    pr_te = hgb.predict_proba(Xte)
    results["HistGradientBoosting"] = {
        "val": classification_metrics(yval, p_val, None, labels),
        "test": classification_metrics(yte, p_te, pr_te, labels),
        "pred_test": p_te,
        "prob_test": pr_te,
    }
    return results


# ===========================================================================
# Архитектурные эксперименты
# ===========================================================================
def architecture_experiment(data):
    """Сравнение глубины/ширины/активации."""
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)

    results = []
    for depth in [1, 2, 3, 4]:
        for width in [8, 32, 128, 256]:
            for act in ["identity", "relu", "tanh"]:
                model = MLP(N_FEATURES, depth, width, act, N_CLASSES, dropout=0.0)
                n_params = count_params(model)
                model, history, best_val_loss = train_model(
                    model, Xtr, ytr, Xval, yval,
                    optimizer_name="adam", lr=1e-3, batch_size=64,
                    weight_decay=1e-4, epochs=60, patience=8)
                pred, prob, y_true = evaluate(model, Xte, yte)
                m = classification_metrics(y_true, pred, prob, labels)
                m["model"] = f"MLP_d{depth}_w{width}_{act}"
                m["depth"] = depth
                m["width"] = width
                m["activation"] = act
                m["n_params"] = n_params
                m["val_loss"] = best_val_loss
                results.append(m)
                print(f"  d={depth} w={width} act={act} params={n_params} "
                      f"test_macroF1={m['macro_f1']:.3f} test_balAcc={m['balanced_accuracy']:.3f}")

    df_arch = pd.DataFrame(results)
    df_arch.to_csv(RESULTS_DIR / "architecture.csv", index=False)
    return df_arch


def plot_architecture(df_arch):
    # ошибка против ширины и глубины (макро-F1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for act in ["identity", "relu", "tanh"]:
        sub = df_arch[df_arch["activation"] == act]
        axes[0].plot(sub["width"], sub["macro_f1"], "o-", label=f"{act}")
    axes[0].set_xlabel("Ширина скрытого слоя")
    axes[0].set_ylabel("Тестовый макро-F1")
    axes[0].set_title("Макро-F1 против ширины")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for width in [8, 32, 128, 256]:
        sub = df_arch[df_arch["width"] == width]
        axes[1].plot(sub["depth"], sub["macro_f1"], "o-", label=f"w={width}")
    axes[1].set_xlabel("Глубина (скрытых слоёв)")
    axes[1].set_ylabel("Тестовый макро-F1")
    axes[1].set_title("Макро-F1 против глубины")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    savefig("architecture_width_depth.png")


def plot_activations(df_arch):
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth in [1, 2, 3, 4]:
        sub = df_arch[df_arch["depth"] == depth].groupby("activation")["macro_f1"].mean()
        ax.plot(sub.index, sub.values, "o-", label=f"глубина {depth}")
    ax.set_xlabel("Активация")
    ax.set_ylabel("Средний тестовый макро-F1")
    ax.set_title("Сравнение identity / ReLU / tanh")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("activations.png")


def lr_experiment(data):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)
    results = []
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        model = MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1)
        model, history, best_val_loss = train_model(
            model, Xtr, ytr, Xval, yval, optimizer_name="adam",
            lr=lr, batch_size=64, weight_decay=1e-4, epochs=50, patience=7)
        pred, prob, y_true = evaluate(model, Xte, yte)
        m = classification_metrics(y_true, pred, prob, labels)
        m["lr"] = lr
        results.append(m)
        print(f"  lr={lr} test_macroF1={m['macro_f1']:.3f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "lr_experiment.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.semilogx(df["lr"], df["macro_f1"], "o-")
    plt.xlabel("Скорость обучения (lr)")
    plt.ylabel("Тестовый макро-F1")
    plt.title("Влияние скорости обучения")
    plt.grid(True, alpha=0.3)
    savefig("lr_experiment.png")
    return df


def batch_experiment(data):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)
    results = []
    for bs in [16, 64, 256, 1024]:
        model = MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1)
        model, history, best_val_loss = train_model(
            model, Xtr, ytr, Xval, yval, optimizer_name="adam",
            lr=1e-3, batch_size=bs, weight_decay=1e-4, epochs=50, patience=7)
        pred, prob, y_true = evaluate(model, Xte, yte)
        m = classification_metrics(y_true, pred, prob, labels)
        m["batch_size"] = bs
        results.append(m)
        print(f"  batch={bs} test_macroF1={m['macro_f1']:.3f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "batch_experiment.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(df["batch_size"], df["macro_f1"], "o-")
    plt.xlabel("Размер мини-батча")
    plt.ylabel("Тестовый макро-F1")
    plt.title("Влияние размера мини-батча")
    plt.grid(True, alpha=0.3)
    savefig("batch_experiment.png")
    return df


def optimizer_experiment(data):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)
    results = []
    for opt in ["sgd", "adam", "adamw"]:
        model = MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1)
        model, history, best_val_loss = train_model(
            model, Xtr, ytr, Xval, yval, optimizer_name=opt,
            lr=1e-3, batch_size=64, weight_decay=1e-4, epochs=50, patience=7)
        pred, prob, y_true = evaluate(model, Xte, yte)
        m = classification_metrics(y_true, pred, prob, labels)
        m["optimizer"] = opt
        results.append(m)
        print(f"  opt={opt} test_macroF1={m['macro_f1']:.3f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "optimizer_experiment.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.bar(df["optimizer"], df["macro_f1"])
    plt.ylabel("Тестовый макро-F1")
    plt.title("Влияние оптимизатора")
    plt.grid(True, alpha=0.3, axis="y")
    savefig("optimizer_experiment.png")
    return df


def weight_decay_experiment(data):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)
    results = []
    for wd in [0.0, 1e-4, 1e-3, 1e-2]:
        model = MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1)
        model, history, best_val_loss = train_model(
            model, Xtr, ytr, Xval, yval, optimizer_name="adam",
            lr=1e-3, batch_size=64, weight_decay=wd, epochs=50, patience=7)
        pred, prob, y_true = evaluate(model, Xte, yte)
        m = classification_metrics(y_true, pred, prob, labels)
        m["weight_decay"] = wd
        results.append(m)
        print(f"  wd={wd} test_macroF1={m['macro_f1']:.3f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "weight_decay_experiment.csv", index=False)
    plt.figure(figsize=(8, 5))
    x = [max(wd, 1e-5) for wd in df["weight_decay"]]
    plt.semilogx(x, df["macro_f1"], "o-")
    plt.xlabel("Weight decay")
    plt.ylabel("Тестовый макро-F1")
    plt.title("Влияние weight decay")
    plt.grid(True, alpha=0.3)
    savefig("weight_decay_experiment.png")
    return df


def learning_curves(data):
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)
    all_histories = []
    best_model = None
    best_test_f1 = -1
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1)
        model, history, best_val_loss = train_model(
            model, Xtr, ytr, Xval, yval, optimizer_name="adam",
            lr=1e-3, batch_size=64, weight_decay=1e-4, epochs=60, patience=8)
        pred, prob, y_true = evaluate(model, Xte, yte)
        m = classification_metrics(y_true, pred, prob, labels)
        all_histories.append((seed, history, m))
        if m["macro_f1"] > best_test_f1:
            best_test_f1 = m["macro_f1"]
            best_model = model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for seed, history, m in all_histories:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], "o-", alpha=0.6,
                     label=f"seed={seed} train")
        axes[1].plot(epochs, history["val_loss"], "o-", alpha=0.6,
                     label=f"seed={seed} val")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Ошибка обучения (CrossEntropy)")
    axes[0].set_title("Ошибка обучения по эпохам")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Ошибка проверки (CrossEntropy)")
    axes[1].set_title("Ошибка проверки по эпохам (ранняя остановка)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    savefig("learning_curves.png")
    torch.manual_seed(42)
    np.random.seed(42)
    best_model, history, _ = train_model(
        MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1),
        Xtr, ytr, Xval, yval, optimizer_name="adam",
        lr=1e-3, batch_size=64, weight_decay=1e-4, epochs=60, patience=8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["grad_norm"]) + 1)
    axes[0].plot(epochs, history["grad_norm"], "o-")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Норма градиента (L2)")
    axes[0].set_title("Норма градиента по эпохам")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["weight_norm"], "o-")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Норма весов (L2)")
    axes[1].set_title("Норма весов по эпохам")
    axes[1].grid(True, alpha=0.3)
    savefig("gradient_weight_norms.png")
    pred, prob, y_true = evaluate(best_model, Xte, yte)
    cm = confusion_matrix(y_true, pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.set_title("Нормированная матрица ошибок — MLP (тест, партии 9-10)")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels([f"gas {i+1}" for i in range(N_CLASSES)])
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels([f"gas {i+1}" for i in range(N_CLASSES)])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    savefig("confusion_mlp_test.png")
    m_test = classification_metrics(y_true, pred, prob, labels)
    print(f"\nЛучшая MLP на тесте: macroF1={m_test['macro_f1']:.3f} "
          f"balAcc={m_test['balanced_accuracy']:.3f} ROC-AUC={m_test['roc_auc_ovr_macro']:.3f}")
    return best_model, all_histories, m_test


def variant_subject_graphs(best_model, data):
    """Предметные графики варианта 21:
    (1) макро-F1 по временным партиям;
    (2) матрица ошибок на поздних партиях vs ранних."""
    X_all = data["train"]["X"]  # placeholder, пересоберём ниже
    # Соберём все данные по партиям
    df = pd.read_csv(DATA_DIR / "features.csv")
    feature_cols = [f"f{i}" for i in range(N_FEATURES)]
    X = df[feature_cols].values.astype(np.float32)
    y = (df["label"].values.astype(np.int64) - 1)
    party = df["party"].values.astype(np.int64)
    scaler = StandardScaler()
    train_mask = df["split"].values == "train"
    scaler.fit(X[train_mask])
    Xs = scaler.transform(X)

    labels = np.arange(N_CLASSES)
    f1_by_party = []
    parties = sorted(df["party"].unique())
    for p in parties:
        m = party == p
        Xp = torch.tensor(Xs[m], dtype=torch.float32, device=DEVICE)
        yp = torch.tensor(y[m], dtype=torch.long, device=DEVICE)
        pred, prob, y_true = evaluate(best_model, Xp, yp)
        present = np.unique(y_true)
        lab = [c for c in labels if c in present]
        mtr = classification_metrics(y_true, pred, prob, lab)
        f1_by_party.append((p, mtr["macro_f1"], len(yp)))

    plt.figure(figsize=(8, 5))
    ps = [t[0] for t in f1_by_party]
    f1s = [t[1] for t in f1_by_party]
    ns = [t[2] for t in f1_by_party]
    plt.bar(ps, f1s)
    for i, (p, f, n) in enumerate(f1_by_party):
        plt.text(p, f, f"n={n}", ha="center", va="bottom", fontsize=7)
    plt.xlabel("Временная партия")
    plt.ylabel("Макро-F1")
    plt.title("Макро-F1 по временным партиям (дрейф сенсоров)")
    plt.grid(True, alpha=0.3, axis="y")
    savefig("macro_f1_by_party.png")

    # Матрица ошибок: ранние (1-6) vs поздние (9-10)
    early_mask = np.isin(party, [1, 2, 3, 4, 5, 6])
    late_mask = np.isin(party, [9, 10])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, mask, title in [(axes[0], early_mask, "Ранние партии (1-6)"),
                            (axes[1], late_mask, "Поздние партии (9-10)")]:
        Xp = torch.tensor(Xs[mask], dtype=torch.float32, device=DEVICE)
        yp = torch.tensor(y[mask], dtype=torch.long, device=DEVICE)
        pred, prob, y_true = evaluate(best_model, Xp, yp)
        present = np.unique(y_true)
        lab = [c for c in labels if c in present]
        cm = confusion_matrix(y_true, pred, labels=lab)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = axes[1 if title.startswith("Поздние") else 0].imshow(
            cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        axes[1 if title.startswith("Поздние") else 0].set_title(title)
        axes[1 if title.startswith("Поздние") else 0].set_xlabel("Предсказанный класс")
        axes[1 if title.startswith("Поздние") else 0].set_ylabel("Истинный класс")
        axes[1 if title.startswith("Поздние") else 0].set_xticks(range(len(lab)))
        axes[1 if title.startswith("Поздние") else 0].set_xticklabels([f"g{c+1}" for c in lab])
        axes[1 if title.startswith("Поздние") else 0].set_yticks(range(len(lab)))
        axes[1 if title.startswith("Поздние") else 0].set_yticklabels([f"g{c+1}" for c in lab])
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    savefig("confusion_early_vs_late.png")
    return f1_by_party


def physical_prior_experiment(data):
    """Физический априорный опыт: добавить на вход общую энергию сигнала
    (сумма по всем 128 признакам) — физически осмысленная характеристика,
    коррелирующая с концентрацией газа. Сравнить с сетью без неё."""
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]
    labels = np.arange(N_CLASSES)

    def add_energy(X):
        energy = X.sum(dim=1, keepdim=True)
        return torch.cat([X, energy], dim=1)

    results = {}
    for tag, use_energy in [("без прирания", False), ("с энергией", True)]:
        Xtr2 = add_energy(Xtr) if use_energy else Xtr
        Xval2 = add_energy(Xval) if use_energy else Xval
        Xte2 = add_energy(Xte) if use_energy else Xte
        n_in = N_FEATURES + 1 if use_energy else N_FEATURES
        torch.manual_seed(42)
        np.random.seed(42)
        model = MLP(n_in, 2, 64, "relu", N_CLASSES, dropout=0.1)
        model, history, best_val_loss = train_model(
            model, Xtr2, ytr, Xval2, yval, optimizer_name="adam",
            lr=1e-3, batch_size=64, weight_decay=1e-4, epochs=50, patience=7)
        pred, prob, y_true = evaluate(model, Xte2, yte)
        m = classification_metrics(y_true, pred, prob, labels)
        results[tag] = m
        print(f"  {tag}: test_macroF1={m['macro_f1']:.3f} balAcc={m['balanced_accuracy']:.3f}")

    tags = list(results.keys())
    f1s = [results[t]["macro_f1"] for t in tags]
    plt.figure(figsize=(6, 5))
    plt.bar(tags, f1s)
    plt.ylabel("Тестовый макро-F1")
    plt.title("Физический априорный опыт: энергия сигнала на входе")
    plt.grid(True, alpha=0.3, axis="y")
    savefig("physical_prior.png")
    return results


def main():
    print("Загрузка данных...")
    data, feature_cols, scaler = load_data()
    print(f"train={len(data['train']['X'])} val={len(data['val']['X'])} test={len(data['test']['X'])}")

    print("\n=== Базовые модели ===")
    base = run_baseline_models(data)
    for name, r in base.items():
        print(f"  {name}: test_macroF1={r['test']['macro_f1']:.3f} "
              f"balAcc={r['test']['balanced_accuracy']:.3f}")

    print("\n=== Архитектурные эксперименты ===")
    df_arch = architecture_experiment(data)
    plot_architecture(df_arch)
    plot_activations(df_arch)

    print("\n=== Гиперпараметры ===")
    lr_experiment(data)
    batch_experiment(data)
    optimizer_experiment(data)
    weight_decay_experiment(data)

    print("\n=== Кривые обучения выбранной сети ===")
    best_model, histories, m_test = learning_curves(data)

    print("\n=== Предметные графики варианта ===")
    f1_by_party = variant_subject_graphs(best_model, data)

    print("\n=== Физический априорный опыт ===")
    phys = physical_prior_experiment(data)

    print("\nГотово. Графики в figure/, метрики в results/")


if __name__ == "__main__":
    main()
