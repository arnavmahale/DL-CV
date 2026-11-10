"""
Naive baseline model: Majority class classifier.
Always predicts the most frequent class in the training set.
"""

import numpy as np
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from scripts.data_loader import LABEL_NAMES


class MajorityClassifier:
    """Predicts the majority class from training data for all inputs."""

    def __init__(self):
        self.majority_class = None
        self.class_distribution = None

    def fit(self, train_loader: DataLoader):
        """Determine the majority class from training data.

        Args:
            train_loader: Training DataLoader.
        """
        all_labels = []
        for _, labels, _ in train_loader:
            all_labels.extend(labels.numpy().tolist())

        all_labels = np.array(all_labels)
        class_counts = np.bincount(all_labels, minlength=2)
        self.majority_class = int(np.argmax(class_counts))
        self.class_distribution = class_counts / len(all_labels)

        print(f"Majority class: {LABEL_NAMES[self.majority_class]}")
        print(f"Class distribution: Fresh={self.class_distribution[0]:.3f}, "
              f"Rotten={self.class_distribution[1]:.3f}")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Predict majority class for all samples in the loader.

        Args:
            data_loader: DataLoader to predict on.

        Returns:
            Tuple of (predictions, true_labels, produce_types).
        """
        all_labels = []
        all_types = []

        for _, labels, types in data_loader:
            all_labels.extend(labels.numpy().tolist())
            all_types.extend(types)

        true_labels = np.array(all_labels)
        predictions = np.full_like(true_labels, self.majority_class)

        return predictions, true_labels, all_types

    def get_proba(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get probability estimates (constant for baseline).

        Args:
            data_loader: DataLoader to predict on.

        Returns:
            Tuple of (probabilities, true_labels).
        """
        all_labels = []
        for _, labels, _ in data_loader:
            all_labels.extend(labels.numpy().tolist())

        true_labels = np.array(all_labels)
        n = len(true_labels)

        # Probability is just the class distribution repeated
        probas = np.tile(self.class_distribution, (n, 1))

        return probas, true_labels


def train_baseline(train_loader: DataLoader) -> MajorityClassifier:
    """Train the majority class baseline.

    Args:
        train_loader: Training DataLoader.

    Returns:
        Fitted MajorityClassifier.
    """
    model = MajorityClassifier()
    model.fit(train_loader)
    return model
