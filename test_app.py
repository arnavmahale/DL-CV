"""
Quick test script to verify the web app works locally.

Usage:
    python test_app.py
"""

import os
import sys
import requests
from pathlib import Path


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import flask
        import flask_cors
        print("  ✓ All core dependencies imported successfully")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Flask version: {flask.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_model_exists():
    """Check if the trained model exists."""
    print("\nChecking model file...")
    model_path = Path("models/deep_efficientnet_b0.pth")

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ Model not found: {model_path}")
        print("  Run: python scripts/train.py --model deep")
        return False


def test_app_structure():
    """Verify app structure."""
    print("\nChecking app structure...")

    required_files = [
        "app/__init__.py",
        "app/routes.py",
        "app/inference.py",
        "static/index.html",
        "static/css/style.css",
        "static/js/app.js",
        "main.py",
        "setup.py",
    ]

    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            all_exist = False

    return all_exist


def test_app_starts():
    """Test if app can be imported."""
    print("\nTesting app initialization...")
    try:
        from app import create_app
        app = create_app()
        print("  ✓ Flask app created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error creating app: {e}")
        return False


def test_server_running():
    """Test if server is running (if started externally)."""
    print("\nChecking if server is running...")
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Server is running!")
            print(f"  Status: {data.get('status')}")
            print(f"  Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"  Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  Server not running (start with: python main.py)")
        return False
    except Exception as e:
        print(f"  Error checking server: {e}")
        return False


def main():
    print("="*60)
    print("FRESH OR ROTTEN - APP TEST")
    print("="*60)

    results = {
        "imports": test_imports(),
        "model": test_model_exists(),
        "structure": test_app_structure(),
        "app_init": test_app_starts(),
        "server": test_server_running(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.ljust(20)}: {status}")

    print("="*60)

    if all(results.values()):
        print("\n🎉 All tests passed! App is ready to run.")
        print("\nNext steps:")
        print("  1. Start the app: python main.py")
        print("  2. Open http://localhost:5000 in your browser")
        print("  3. Grant camera permissions")
        print("  4. Test with your phone on the same WiFi")
    else:
        print("\n⚠️  Some tests failed. Please fix issues above.")

    if not results["model"]:
        print("\n💡 To train the model:")
        print("  python scripts/train.py --model deep")


if __name__ == "__main__":
    main()
