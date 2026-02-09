"""
Setup script for Fresh or Rotten Produce Classification project.

This script:
- Creates necessary directories (models/, data/outputs/, notebooks/)
- Verifies dataset presence
- Checks dependencies
- Optionally trains models

Usage:
    python setup.py                    # Setup directories only
    python setup.py --install-deps     # Install dependencies
    python setup.py --train-all        # Setup + train all models
    python setup.py --check            # Check environment
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.absolute()


def create_directories():
    """Create necessary project directories."""
    root = get_project_root()
    directories = [
        "models",
        "data/outputs",
        "data/raw",
        "data/processed",
        "notebooks",
        "app",
        "static/css",
        "static/js",
    ]

    print("Creating project directories...")
    for directory in directories:
        dir_path = root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}/")

    print("\n✅ Directory structure created successfully!")


def check_dataset():
    """Verify that the dataset exists."""
    root = get_project_root()
    dataset_dir = root / "Dataset"

    print("\nChecking dataset...")
    if not dataset_dir.exists():
        print("  ⚠️  Dataset directory not found!")
        print(f"  Expected location: {dataset_dir}")
        print("\n  Please download the dataset and extract it to the Dataset/ directory.")
        print("  Directory structure should be:")
        print("    Dataset/")
        print("      ├── Fresh/")
        print("      │   ├── FreshApple/")
        print("      │   ├── FreshBanana/")
        print("      │   └── ...")
        print("      └── Rotten/")
        print("          ├── RottenApple/")
        print("          ├── RottenBanana/")
        print("          └── ...")
        return False

    # Check for Fresh and Rotten subdirectories
    fresh_dir = dataset_dir / "Fresh"
    rotten_dir = dataset_dir / "Rotten"

    if not fresh_dir.exists() or not rotten_dir.exists():
        print("  ⚠️  Dataset structure incomplete!")
        print(f"  Missing: {'Fresh/' if not fresh_dir.exists() else ''} "
              f"{'Rotten/' if not rotten_dir.exists() else ''}")
        return False

    # Count produce types
    fresh_types = len([d for d in fresh_dir.iterdir() if d.is_dir()])
    rotten_types = len([d for d in rotten_dir.iterdir() if d.is_dir()])

    print(f"  ✓ Dataset found!")
    print(f"  Fresh produce types: {fresh_types}")
    print(f"  Rotten produce types: {rotten_types}")

    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")

    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "flask": "Flask",
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)

    if missing:
        print(f"\n  ⚠️  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed!")
    return True


def install_dependencies():
    """Install dependencies from requirements.txt."""
    root = get_project_root()
    requirements_file = root / "requirements.txt"

    if not requirements_file.exists():
        print("  ⚠️  requirements.txt not found!")
        return False

    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("\n✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing dependencies: {e}")
        return False


def update_requirements():
    """Update requirements.txt with Flask dependencies."""
    root = get_project_root()
    requirements_file = root / "requirements.txt"

    # Read existing requirements
    with open(requirements_file, "r") as f:
        existing = f.read()

    # Add Flask dependencies if not present
    flask_deps = ["flask>=2.3.0", "flask-cors>=4.0.0"]
    needs_update = False

    for dep in flask_deps:
        package_name = dep.split(">=")[0]
        if package_name not in existing.lower():
            existing += f"{dep}\n"
            needs_update = True

    if needs_update:
        with open(requirements_file, "w") as f:
            f.write(existing)
        print("  ✓ Updated requirements.txt with Flask dependencies")


def train_models():
    """Train all three models."""
    root = get_project_root()
    train_script = root / "scripts" / "train.py"

    if not train_script.exists():
        print("  ⚠️  Training script not found!")
        return False

    print("\nTraining all models (this may take a while)...")
    try:
        subprocess.check_call([
            sys.executable, str(train_script),
            "--model", "all",
            "--dataset-dir", "Dataset"
        ])
        print("\n✅ All models trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error training models: {e}")
        return False


def check_environment():
    """Check the entire environment setup."""
    print("="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)

    # Python version
    print(f"\nPython version: {sys.version}")

    # Check CUDA/GPU
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: Yes (GPU: {torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available: Yes")
        else:
            print("GPU: Not available (CPU only)")
    except ImportError:
        print("\nPyTorch: Not installed")

    # Check disk space
    import shutil
    root = get_project_root()
    total, used, free = shutil.disk_usage(root)
    print(f"\nDisk space:")
    print(f"  Total: {total // (2**30)} GB")
    print(f"  Free:  {free // (2**30)} GB")

    if free < 10 * (2**30):  # Less than 10GB
        print("  ⚠️  Low disk space (recommended: 10GB+ free)")


def main():
    parser = argparse.ArgumentParser(
        description="Setup script for Fresh or Rotten Produce Classification"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies from requirements.txt"
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all models after setup"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment and dependencies"
    )

    args = parser.parse_args()

    print("="*60)
    print("FRESH OR ROTTEN - PROJECT SETUP")
    print("="*60)

    # Always create directories
    create_directories()

    # Update requirements.txt with Flask
    update_requirements()

    # Check dataset
    dataset_ok = check_dataset()

    # Check or install dependencies
    if args.install_deps:
        deps_ok = install_dependencies()
    else:
        deps_ok = check_dependencies()
        if not deps_ok:
            print("\n💡 Tip: Run with --install-deps to install missing packages")

    # Check environment if requested
    if args.check:
        check_environment()

    # Train models if requested
    if args.train_all:
        if not dataset_ok:
            print("\n❌ Cannot train models: Dataset not found!")
        elif not deps_ok:
            print("\n❌ Cannot train models: Dependencies not installed!")
        else:
            train_models()

    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    print(f"Directories:   {'✅' if True else '❌'}")
    print(f"Dataset:       {'✅' if dataset_ok else '⚠️  (see instructions above)'}")
    print(f"Dependencies:  {'✅' if deps_ok else '⚠️  (run --install-deps)'}")

    if dataset_ok and deps_ok:
        print("\n✅ Setup complete! You can now:")
        print("   - Train models: python scripts/train.py --model all")
        print("   - Run web app: python main.py")
        print("   - Run tests: python scripts/predict.py <image_path>")
    else:
        print("\n⚠️  Setup incomplete. Please resolve issues above.")

    print("="*60)


if __name__ == "__main__":
    main()
