"""
Experiment: Training set size sensitivity analysis.
Trains the deep learning model on varying fractions of the training data
and measures how performance changes as data availability increases.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_loader import create_dataloaders
from scripts.deep_model import DeepModel
from scripts.evaluate import compute_metrics
from scripts.utils import set_seed, ensure_dirs, save_results


def run_experiment(
    dataset_dir: str = "Dataset",
    fractions: list = None,
    batch_size: int = 32,
    num_workers: int = 4,
    head_epochs: int = 3,
    finetune_epochs: int = 10,
    patience: int = 3,
    seed: int = 42,
    output_dir: str = "data/outputs",
):
    """Run the training set size sensitivity experiment.

    Trains EfficientNet-B0 on {10%, 25%, 50%, 75%, 100%} of the training data
    and records performance on the fixed test set.

    Args:
        dataset_dir: Path to Dataset/ directory.
        fractions: List of training data fractions to test.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        head_epochs: Head-only training epochs (fewer for experiment speed).
        finetune_epochs: Fine-tuning epochs (fewer for experiment speed).
        patience: Early stopping patience.
        seed: Random seed.
        output_dir: Directory for outputs.
    """
    if fractions is None:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.0]

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "fractions": fractions,
        "n_train_samples": [],
        "accuracy": [],
        "f1_weighted": [],
        "f1_macro": [],
        "precision_macro": [],
        "recall_macro": [],
        "auc_roc": [],
    }

    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: Training with {frac*100:.0f}% of training data")
        print(f"{'='*60}")

        set_seed(seed)

        train_loader, val_loader, test_loader, info = create_dataloaders(
            dataset_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            train_fraction=frac,
        )

        n_train = info["n_train"]
        print(f"Training samples: {n_train}")
        results["n_train_samples"].append(n_train)

        # Train model
        model = DeepModel()
        model.fit(
            train_loader, val_loader, info["class_weights"],
            head_epochs=head_epochs,
            finetune_epochs=finetune_epochs,
            patience=patience,
        )

        # Evaluate on test set
        preds, labels, _ = model.predict(test_loader)
        probas, _ = model.get_proba(test_loader)
        metrics = compute_metrics(labels, preds, probas)

        for key in ["accuracy", "f1_weighted", "f1_macro", "precision_macro", "recall_macro"]:
            results[key].append(metrics[key])
        results["auc_roc"].append(metrics.get("auc_roc"))

        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"  AUC-ROC:     {metrics.get('auc_roc', 'N/A')}")

    # Save results
    save_results(results, os.path.join(output_dir, "experiment_results.json"))

    # Plot results
    _plot_experiment_results(results, output_dir)

    # Print summary
    _print_summary(results)

    return results


def _plot_experiment_results(results: dict, output_dir: str):
    """Create plots for the experiment results."""
    fractions = results["fractions"]
    n_samples = results["n_train_samples"]
    pct_labels = [f"{f*100:.0f}%" for f in fractions]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Accuracy vs training fraction
    axes[0].plot(fractions, results["accuracy"], "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Training Data Fraction", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Test Accuracy vs Training Data Size", fontsize=13)
    axes[0].set_xticks(fractions)
    axes[0].set_xticklabels(pct_labels)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Plot 2: F1 scores vs training fraction
    axes[1].plot(fractions, results["f1_weighted"], "go-", linewidth=2, markersize=8, label="F1 Weighted")
    axes[1].plot(fractions, results["f1_macro"], "ro-", linewidth=2, markersize=8, label="F1 Macro")
    axes[1].set_xlabel("Training Data Fraction", fontsize=12)
    axes[1].set_ylabel("F1 Score", fontsize=12)
    axes[1].set_title("F1 Score vs Training Data Size", fontsize=13)
    axes[1].set_xticks(fractions)
    axes[1].set_xticklabels(pct_labels)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Plot 3: AUC-ROC vs training fraction
    auc_vals = [v if v is not None else 0 for v in results["auc_roc"]]
    axes[2].plot(fractions, auc_vals, "mo-", linewidth=2, markersize=8)
    axes[2].set_xlabel("Training Data Fraction", fontsize=12)
    axes[2].set_ylabel("AUC-ROC", fontsize=12)
    axes[2].set_title("AUC-ROC vs Training Data Size", fontsize=13)
    axes[2].set_xticks(fractions)
    axes[2].set_xticklabels(pct_labels)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "experiment_learning_curves.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nExperiment plots saved to {output_path}")

    # Additional plot: Accuracy vs absolute number of samples
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_samples, results["accuracy"], "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy vs Number of Training Samples", fontsize=13)
    ax.grid(alpha=0.3)

    for i, (x, y) in enumerate(zip(n_samples, results["accuracy"])):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "experiment_absolute_samples.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Absolute samples plot saved to {output_path}")


def _print_summary(results: dict):
    """Print a formatted summary of experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY: Training Set Size Sensitivity")
    print("=" * 70)
    print(f"{'Fraction':<12} {'N Samples':<12} {'Accuracy':<12} {'F1 (W)':<12} {'AUC-ROC':<12}")
    print("-" * 70)

    for i, frac in enumerate(results["fractions"]):
        auc = results["auc_roc"][i]
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{frac*100:>6.0f}%     {results['n_train_samples'][i]:<12} "
              f"{results['accuracy'][i]:<12.4f} {results['f1_weighted'][i]:<12.4f} {auc_str:<12}")

    # Compute marginal gains
    print("\nMarginal gains (accuracy improvement per additional 15% data):")
    for i in range(1, len(results["fractions"])):
        gain = results["accuracy"][i] - results["accuracy"][i - 1]
        frac_diff = results["fractions"][i] - results["fractions"][i - 1]
        print(f"  {results['fractions'][i-1]*100:.0f}% -> {results['fractions'][i]*100:.0f}%: "
              f"+{gain:.4f} accuracy ({gain/frac_diff:.4f} per 1% data)")


def parse_args():
    parser = argparse.ArgumentParser(description="Run training set size sensitivity experiment.")
    parser.add_argument("--dataset-dir", type=str, default="Dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 1.0],
        help="Training data fractions to test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()

    run_experiment(
        dataset_dir=args.dataset_dir,
        fractions=args.fractions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        patience=args.patience,
        seed=args.seed,
    )
