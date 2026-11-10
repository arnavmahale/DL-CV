"""
Shared utilities for produce freshness classification project.
"""

import os
import json
import random
from datetime import datetime

import torch
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_results(results: dict, filepath: str):
    """Save results dictionary to a JSON file.

    Args:
        results: Dictionary of results to save.
        filepath: Output file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert non-serializable types
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            serializable[key] = int(value)
        elif isinstance(value, (np.floating,)):
            serializable[key] = float(value)
        elif isinstance(value, torch.Tensor):
            serializable[key] = value.cpu().numpy().tolist()
        else:
            serializable[key] = value

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


def load_results(filepath: str) -> dict:
    """Load results dictionary from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_project_root() -> str:
    """Get the project root directory (parent of scripts/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs():
    """Ensure all necessary output directories exist."""
    root = get_project_root()
    for d in ["models", "data/outputs"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)
