"""
Evaluation module for produce freshness classification.
Computes metrics, generates confusion matrices, and creates visualizations.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_loader import LABEL_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (n_samples, n_classes). Optional.

    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def per_produce_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    produce_types: List[str],
) -> Dict[str, Dict]:
    """Compute metrics broken down by produce type.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        produce_types: List of produce type names.

    Returns:
        Dictionary mapping produce type to metrics dict.
    """
    types_arr = np.array(produce_types)
    unique_types = sorted(set(produce_types))
    results = {}

    for ptype in unique_types:
        mask = types_arr == ptype
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        results[ptype] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, average="binary", zero_division=0)),
            "fresh_count": int((yt == 0).sum()),
            "rotten_count": int((yt == 1).sum()),
        }

    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str,
):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_path: str,
):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc = roc_auc_score(y_true, y_proba[:, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def plot_per_produce_performance(
    produce_metrics: Dict[str, Dict],
    model_name: str,
    output_path: str,
):
    """Plot per-produce-type performance bar chart."""
    types = list(produce_metrics.keys())
    accuracies = [produce_metrics[t]["accuracy"] for t in types]
    f1_scores = [produce_metrics[t]["f1"] for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score", color="#FF9800")

    ax.set_xlabel("Produce Type", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Produce Performance - {model_name}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-produce performance saved to {output_path}")


def find_mispredictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    produce_types: List[str],
    n: int = 5,
) -> List[Dict]:
    """Find specific mispredictions for error analysis.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        produce_types: Produce types.
        n: Number of mispredictions to return.

    Returns:
        List of dicts with misprediction details.
    """
    mismatches = np.where(y_true != y_pred)[0]

    if len(mismatches) == 0:
        return []

    # Sample diverse mispredictions across produce types
    rng = np.random.RandomState(42)
    selected_idx = rng.choice(mismatches, size=min(n, len(mismatches)), replace=False)

    mispredictions = []
    for idx in selected_idx:
        mispredictions.append({
            "index": int(idx),
            "true_label": LABEL_NAMES[y_true[idx]],
            "predicted_label": LABEL_NAMES[y_pred[idx]],
            "produce_type": produce_types[idx],
        })

    return mispredictions


def evaluate_model(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    produce_types: List[str],
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
    output_dir: str = "data/outputs",
) -> Dict:
    """Full evaluation of a model.

    Args:
        y_pred: Predicted labels.
        y_true: True labels.
        produce_types: Produce type for each sample.
        y_proba: Predicted probabilities. Optional.
        model_name: Name of the model for plots.
        output_dir: Directory to save outputs.

    Returns:
        Dictionary of all results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*50}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):     {metrics['f1_weighted']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    # Per-produce metrics
    prod_metrics = per_produce_metrics(y_true, y_pred, produce_types)

    # Mispredictions
    mispredictions = find_mispredictions(y_true, y_pred, produce_types)

    # Plots
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plot_confusion_matrix(
        y_true, y_pred, model_name,
        os.path.join(output_dir, f"cm_{safe_name}.png"),
    )

    if y_proba is not None and metrics.get("auc_roc") is not None:
        plot_roc_curve(
            y_true, y_proba, model_name,
            os.path.join(output_dir, f"roc_{safe_name}.png"),
        )

    plot_per_produce_performance(
        prod_metrics, model_name,
        os.path.join(output_dir, f"produce_perf_{safe_name}.png"),
    )

    # Combine all results
    results = {
        **metrics,
        "model_name": model_name,
        "per_produce": prod_metrics,
        "mispredictions": mispredictions,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
    }

    return results


def generate_comparison_report(
    all_results: Dict[str, Dict],
    output_dir: str = "data/outputs",
):
    """Generate a comparison table and plot across all models.

    Args:
        all_results: Dict mapping model key to results dict.
        output_dir: Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<35} {'Accuracy':>10} {'F1 (W)':>10} {'F1 (M)':>10} {'AUC-ROC':>10}")
    print("-" * 70)

    model_names = []
    accuracies = []
    f1_scores = []

    for key, res in all_results.items():
        name = res.get("model_name", key)
        acc = res["accuracy"]
        f1w = res["f1_weighted"]
        f1m = res["f1_macro"]
        auc = res.get("auc_roc", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)

        print(f"{name:<35} {acc:>10.4f} {f1w:>10.4f} {f1m:>10.4f} {auc_str:>10}")

        model_names.append(name)
        accuracies.append(acc)
        f1_scores.append(f1w)

    # Comparison bar chart
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#2196F3")
    ax.bar(x + width / 2, f1_scores, width, label="F1 (Weighted)", color="#4CAF50")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nComparison chart saved to {output_path}")


def plot_training_history(history: Dict[str, list], output_dir: str = "data/outputs"):
    """Plot training history (loss and accuracy curves) for the deep model.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        output_dir: Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training history saved to {output_path}")
