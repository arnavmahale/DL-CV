"""
Training orchestrator for produce freshness classification.
Supports training all three model types: baseline, classical ML, and deep learning.
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_loader import create_dataloaders
from scripts.baseline_model import train_baseline
from scripts.classical_model import train_classical
from scripts.deep_model import train_deep
from scripts.evaluate import evaluate_model, generate_comparison_report
from scripts.utils import set_seed, ensure_dirs, save_results, get_device


def run_baseline(dataset_dir: str, batch_size: int, num_workers: int, seed: int):
    """Train and evaluate the naive baseline model."""
    print("=" * 60)
    print("TRAINING: Naive Baseline (Majority Class Classifier)")
    print("=" * 60)

    train_loader, val_loader, test_loader, info = create_dataloaders(
        dataset_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    model = train_baseline(train_loader)

    # Evaluate on test set
    preds, labels, types = model.predict(test_loader)
    probas, _ = model.get_proba(test_loader)

    results = evaluate_model(
        preds, labels, types, probas,
        model_name="Baseline (Majority Class)",
        output_dir=os.path.join("data", "outputs"),
    )

    save_results(results, os.path.join("data", "outputs", "baseline_results.json"))
    print(f"\nBaseline Test Accuracy: {results['accuracy']:.4f}")
    print(f"Baseline Test F1:       {results['f1_weighted']:.4f}")

    return results


def run_classical(
    dataset_dir: str, batch_size: int, num_workers: int, seed: int,
    classifier_type: str = "random_forest", tune: bool = True,
):
    """Train and evaluate the classical ML model."""
    print("=" * 60)
    print(f"TRAINING: Classical ML ({classifier_type})")
    print("=" * 60)

    train_loader, val_loader, test_loader, info = create_dataloaders(
        dataset_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    model = train_classical(train_loader, classifier_type=classifier_type, tune_hyperparams=tune)

    # Save model
    model.save(os.path.join("models", f"classical_{classifier_type}.pkl"))

    # Evaluate on test set
    preds, labels, types = model.predict(test_loader)
    probas, _ = model.get_proba(test_loader)

    results = evaluate_model(
        preds, labels, types, probas,
        model_name=f"Classical ({classifier_type})",
        output_dir=os.path.join("data", "outputs"),
    )

    save_results(results, os.path.join("data", "outputs", "classical_results.json"))
    print(f"\nClassical Test Accuracy: {results['accuracy']:.4f}")
    print(f"Classical Test F1:       {results['f1_weighted']:.4f}")

    return results


def run_deep(
    dataset_dir: str, batch_size: int, num_workers: int, seed: int,
    head_epochs: int = 5, finetune_epochs: int = 15, patience: int = 5,
):
    """Train and evaluate the deep learning model."""
    print("=" * 60)
    print("TRAINING: Deep Learning (EfficientNet-B0)")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, info = create_dataloaders(
        dataset_dir, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    model = train_deep(
        train_loader, val_loader, info["class_weights"],
        head_epochs=head_epochs,
        finetune_epochs=finetune_epochs,
        patience=patience,
    )

    # Save model
    model.save(os.path.join("models", "deep_efficientnet_b0.pth"))

    # Evaluate on test set
    preds, labels, types = model.predict(test_loader)
    probas, _ = model.get_proba(test_loader)

    results = evaluate_model(
        preds, labels, types, probas,
        model_name="Deep Learning (EfficientNet-B0)",
        output_dir=os.path.join("data", "outputs"),
    )

    # Save training history
    results["training_history"] = model.history
    save_results(results, os.path.join("data", "outputs", "deep_results.json"))
    print(f"\nDeep Learning Test Accuracy: {results['accuracy']:.4f}")
    print(f"Deep Learning Test F1:       {results['f1_weighted']:.4f}")

    return results


def run_all(args):
    """Train and evaluate all three models, then generate comparison."""
    results = {}

    results["baseline"] = run_baseline(
        args.dataset_dir, args.batch_size, args.num_workers, args.seed
    )
    results["classical"] = run_classical(
        args.dataset_dir, args.batch_size, args.num_workers, args.seed,
        classifier_type=args.classifier_type, tune=not args.no_tune,
    )
    results["deep"] = run_deep(
        args.dataset_dir, args.batch_size, args.num_workers, args.seed,
        head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs,
        patience=args.patience,
    )

    # Generate comparison report
    generate_comparison_report(results, output_dir=os.path.join("data", "outputs"))

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train produce freshness classification models."
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["baseline", "classical", "deep", "all"],
        help="Which model to train (default: all).",
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="Dataset",
        help="Path to the Dataset/ directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for data loaders.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    # Classical model args
    parser.add_argument(
        "--classifier-type", type=str, default="random_forest",
        choices=["random_forest", "gradient_boosting", "svm"],
        help="Classical classifier type.",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip hyperparameter tuning for classical model.",
    )
    # Deep model args
    parser.add_argument(
        "--head-epochs", type=int, default=5,
        help="Epochs for head-only training phase.",
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=15,
        help="Epochs for full fine-tuning phase.",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()

    if args.model == "all":
        run_all(args)
    elif args.model == "baseline":
        run_baseline(args.dataset_dir, args.batch_size, args.num_workers, args.seed)
    elif args.model == "classical":
        run_classical(
            args.dataset_dir, args.batch_size, args.num_workers, args.seed,
            classifier_type=args.classifier_type, tune=not args.no_tune,
        )
    elif args.model == "deep":
        run_deep(
            args.dataset_dir, args.batch_size, args.num_workers, args.seed,
            head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs,
            patience=args.patience,
        )
