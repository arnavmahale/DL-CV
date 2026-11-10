"""
Data loading and preprocessing module for produce freshness classification.
Handles dataset loading, train/val/test splitting, and image transforms.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np


# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Label mapping
LABEL_MAP = {"Fresh": 0, "Rotten": 1}
LABEL_NAMES = ["Fresh", "Rotten"]

# Normalize inconsistent folder names in the dataset
PRODUCE_NAME_MAP = {
    "Capciscum": "Capsicum",
    "Okara": "Okra",
    "Bittergroud": "Bittergourd",
}


def get_image_paths_and_labels(dataset_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """Walk the dataset directory and collect image paths, binary labels, and produce types.

    Args:
        dataset_dir: Path to the Dataset/ directory containing Fresh/ and Rotten/ subdirs.

    Returns:
        Tuple of (image_paths, labels, produce_types).
    """
    image_paths = []
    labels = []
    produce_types = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for category in ["Fresh", "Rotten"]:
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_dir):
            raise FileNotFoundError(f"Category directory not found: {category_dir}")

        label = LABEL_MAP[category]

        for produce_folder in sorted(os.listdir(category_dir)):
            produce_path = os.path.join(category_dir, produce_folder)
            if not os.path.isdir(produce_path):
                continue

            # Extract produce type name (e.g., "FreshApple" -> "Apple")
            produce_name = produce_folder.replace("Fresh", "").replace("Rotten", "")
            produce_name = PRODUCE_NAME_MAP.get(produce_name, produce_name)

            for img_file in os.listdir(produce_path):
                ext = os.path.splitext(img_file)[1].lower()
                if ext in valid_extensions:
                    image_paths.append(os.path.join(produce_path, img_file))
                    labels.append(label)
                    produce_types.append(produce_name)

    return image_paths, labels, produce_types


def get_transforms(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """Get image transforms for the given split.

    Args:
        split: One of 'train', 'val', or 'test'.
        img_size: Target image size.

    Returns:
        A torchvision transforms.Compose pipeline.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class ProduceDataset(Dataset):
    """PyTorch Dataset for produce freshness classification."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        produce_types: List[str],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.produce_types = produce_types
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        produce_type = self.produce_types[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, produce_type


def create_data_splits(
    dataset_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[ProduceDataset, ProduceDataset, ProduceDataset]:
    """Create stratified train/val/test splits.

    Args:
        dataset_dir: Path to the Dataset/ directory.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    image_paths, labels, produce_types = get_image_paths_and_labels(dataset_dir)

    # Convert to numpy for sklearn splitting
    paths_arr = np.array(image_paths)
    labels_arr = np.array(labels)
    types_arr = np.array(produce_types)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, valtest_idx = train_test_split(
        np.arange(len(labels_arr)),
        test_size=val_test_ratio,
        stratify=labels_arr,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=relative_test_ratio,
        stratify=labels_arr[valtest_idx],
        random_state=seed,
    )

    train_dataset = ProduceDataset(
        paths_arr[train_idx].tolist(),
        labels_arr[train_idx].tolist(),
        types_arr[train_idx].tolist(),
        transform=get_transforms("train"),
    )
    val_dataset = ProduceDataset(
        paths_arr[val_idx].tolist(),
        labels_arr[val_idx].tolist(),
        types_arr[val_idx].tolist(),
        transform=get_transforms("val"),
    )
    test_dataset = ProduceDataset(
        paths_arr[test_idx].tolist(),
        labels_arr[test_idx].tolist(),
        types_arr[test_idx].tolist(),
        transform=get_transforms("test"),
    )

    return train_dataset, val_dataset, test_dataset


def get_class_weights(dataset: ProduceDataset) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.

    Args:
        dataset: A ProduceDataset instance.

    Returns:
        Tensor of class weights [weight_fresh, weight_rotten].
    """
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels, minlength=2)
    total = len(labels)
    weights = total / (2.0 * class_counts)
    return torch.FloatTensor(weights)


def get_weighted_sampler(dataset: ProduceDataset) -> WeightedRandomSampler:
    """Create a weighted random sampler to handle class imbalance during training.

    Args:
        dataset: A ProduceDataset instance.

    Returns:
        A WeightedRandomSampler.
    """
    class_weights = get_class_weights(dataset)
    sample_weights = [class_weights[label].item() for label in dataset.labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    train_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create train, val, and test DataLoaders.

    Args:
        dataset_dir: Path to the Dataset/ directory.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        seed: Random seed.
        train_fraction: Fraction of training data to use (for experiments).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict).
    """
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset_dir, seed=seed
    )

    # Subsample training data if needed
    if train_fraction < 1.0:
        n_train = len(train_dataset)
        n_subset = max(1, int(n_train * train_fraction))
        rng = np.random.RandomState(seed)
        subset_indices = rng.choice(n_train, size=n_subset, replace=False)

        subset_paths = [train_dataset.image_paths[i] for i in subset_indices]
        subset_labels = [train_dataset.labels[i] for i in subset_indices]
        subset_types = [train_dataset.produce_types[i] for i in subset_indices]

        train_dataset = ProduceDataset(
            subset_paths, subset_labels, subset_types,
            transform=get_transforms("train"),
        )

    sampler = get_weighted_sampler(train_dataset)
    class_weights = get_class_weights(train_dataset)

    # Disable pin_memory on MPS (not supported)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    info = {
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "class_weights": class_weights,
        "label_names": LABEL_NAMES,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "Dataset"
    train_loader, val_loader, test_loader, info = create_dataloaders(dataset_dir, batch_size=16)

    print(f"Train samples: {info['n_train']}")
    print(f"Val samples:   {info['n_val']}")
    print(f"Test samples:  {info['n_test']}")
    print(f"Class weights: {info['class_weights']}")

    # Test loading a batch
    images, labels, types = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Types: {types[:5]}")
