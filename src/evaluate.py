"""
evaluate.py
------------
Computes the metrics and figures used throughout the thesis:
    - Accuracy, Precision, Recall, F1-score (Fig. 3.3-style)
    - MSE, MAE (Fig. 3.4-style)
    - Confusion matrix (Fig. 3.5-3.8-style)
    - Optimization comparison bar chart (Fig. 3.9-style)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_optimization_comparison(scores_dict, save_path, ylabel="Accuracy (%)"):
    """scores_dict e.g. {"Bayesian": 95.0, "Hyperband": 95.7, "BOHB": 96.5}"""
    plt.figure(figsize=(6, 5))
    names = list(scores_dict.keys())
    vals = [scores_dict[n] for n in names]
    plt.bar(names, vals, color=["#4C72B0", "#DD8452", "#8C8C8C"][: len(names)])
    plt.ylabel(ylabel)
    plt.title("Optimization Technique Comparison")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.1, f"{v:.2f}", ha="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
