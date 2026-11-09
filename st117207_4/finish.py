"""
Дообучение лучшей модели и построение предметных графиков варианта 21.
Запускается после основного train.py (чтобы не переобучать архитектуру заново).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figure"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from train import (
    load_data, MLP, train_model, evaluate, classification_metrics,
    savefig, variant_subject_graphs, physical_prior_experiment,
    N_FEATURES, N_CLASSES, DEVICE, RANDOM_SEED,
)

import warnings
warnings.filterwarnings("ignore")


def main():
    print("Загрузка данных...")
    data, feature_cols, scaler = load_data()
    Xtr = data["train"]["X"]; ytr = data["train"]["y"]
    Xval = data["val"]["X"]; yval = data["val"]["y"]
    Xte = data["test"]["X"]; yte = data["test"]["y"]

    print("\n=== Обучение лучшей модели (d=2, w=64, relu, adam) ===")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    best_model, history, best_val_loss = train_model(
        MLP(N_FEATURES, 2, 64, "relu", N_CLASSES, dropout=0.1),
        Xtr, ytr, Xval, yval, optimizer_name="adam",
        lr=1e-3, batch_size=64, weight_decay=1e-4, epochs=60, patience=8)

    from sklearn.metrics import confusion_matrix
    labels = np.arange(N_CLASSES)
    pred, prob, y_true = evaluate(best_model, Xte, yte)
    m_test = classification_metrics(y_true, pred, prob, labels)
    print(f"Тест: macroF1={m_test['macro_f1']:.3f} balAcc={m_test['balanced_accuracy']:.3f} "
          f"ROC-AUC={m_test['roc_auc_ovr_macro']:.3f}")

    # матрица ошибок на тесте
    cm = confusion_matrix(y_true, pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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

    # кривые обучения
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "o-", label="train")
    axes[1].plot(epochs, history["val_loss"], "o-", label="val")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Ошибка обучения")
    axes[0].set_title("Ошибка обучения по эпохам"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("Ошибка проверки")
    axes[1].set_title("Ошибка проверки по эпохам"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    savefig("learning_curves.png")

    # нормы
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["grad_norm"], "o-")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Норма градиента (L2)")
    axes[0].set_title("Норма градиента по эпохам"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["weight_norm"], "o-")
    axes[1].set_xlabel("Эпоха"); axes[1].set_ylabel("Норма весов (L2)")
    axes[1].set_title("Норма весов по эпохам"); axes[1].grid(True, alpha=0.3)
    savefig("gradient_weight_norms.png")

    print("\n=== Предметные графики варианта ===")
    f1_by_party = variant_subject_graphs(best_model, data)
    for p, f1, n in f1_by_party:
        print(f"  партия {p}: макро-F1={f1:.3f} (n={n})")

    print("\n=== Физический априорный опыт ===")
    phys = physical_prior_experiment(data)

    print("\nГотово.")


if __name__ == "__main__":
    main()
