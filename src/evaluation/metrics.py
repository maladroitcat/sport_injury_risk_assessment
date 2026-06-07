import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "artifacts/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "artifacts/cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data.metadata import LABEL_ORDER, LABEL_TO_ID


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "high_recall": float(recall[LABEL_TO_ID["high"]]),
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(LABEL_ORDER)
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABEL_ORDER))),
            target_names=LABEL_ORDER,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(LABEL_ORDER)), labels=LABEL_ORDER)
    ax.set_yticks(np.arange(len(LABEL_ORDER)), labels=LABEL_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
