"""
Classical ML model for produce freshness classification.
Extracts features using a pretrained ResNet-18 backbone, then trains
a Random Forest classifier on the extracted feature vectors.
"""

import os
import pickle
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from scripts.utils import get_device


class FeatureExtractor:
    """Extract features from images using a pretrained CNN backbone."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()

        # Load pretrained ResNet-18 without the final FC layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.feature_dim = 512  # ResNet-18 output dimension

    @torch.no_grad()
    def extract(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract feature vectors from all images in the DataLoader.

        Args:
            data_loader: DataLoader with images.

        Returns:
            Tuple of (features, labels, produce_types).
        """
        all_features = []
        all_labels = []
        all_types = []

        for images, labels, types in tqdm(data_loader, desc="Extracting features"):
            images = images.to(self.device)
            features = self.model(images).squeeze(-1).squeeze(-1)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_types.extend(types)

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels)

        return features, labels, all_types


class ClassicalModel:
    """Classical ML classifier using CNN-extracted features."""

    def __init__(self, classifier_type: str = "random_forest"):
        self.classifier_type = classifier_type
        self.feature_extractor = None
        self.pipeline = None
        self.best_params = None

    def _create_classifier(self):
        """Create the sklearn classifier based on type."""
        if self.classifier_type == "random_forest":
            return RandomForestClassifier(random_state=42, n_jobs=-1)
        elif self.classifier_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        elif self.classifier_type == "svm":
            return SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def _get_param_grid(self) -> dict:
        """Get hyperparameter search space for the classifier."""
        if self.classifier_type == "random_forest":
            return {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [10, 20, 30, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            }
        elif self.classifier_type == "gradient_boosting":
            return {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
            }
        elif self.classifier_type == "svm":
            return {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
            }
        return {}

    def fit(
        self,
        train_loader: DataLoader,
        tune_hyperparams: bool = True,
        n_iter: int = 20,
    ):
        """Extract features and train the classical ML model.

        Args:
            train_loader: Training DataLoader.
            tune_hyperparams: Whether to perform hyperparameter search.
            n_iter: Number of random search iterations.
        """
        print("Extracting training features...")
        self.feature_extractor = FeatureExtractor()
        X_train, y_train, _ = self.feature_extractor.extract(train_loader)

        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")

        # Create pipeline with scaling and classifier
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self._create_classifier()),
        ])

        if tune_hyperparams:
            print(f"Running hyperparameter search ({n_iter} iterations)...")
            search = RandomizedSearchCV(
                self.pipeline,
                self._get_param_grid(),
                n_iter=n_iter,
                cv=3,
                scoring="f1_weighted",
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train, y_train)
            self.pipeline = search.best_estimator_
            self.best_params = search.best_params_
            print(f"Best params: {self.best_params}")
            print(f"Best CV F1: {search.best_score_:.4f}")
        else:
            self.pipeline.fit(X_train, y_train)

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Predict on data from a DataLoader.

        Args:
            data_loader: DataLoader to predict on.

        Returns:
            Tuple of (predictions, true_labels, produce_types).
        """
        print("Extracting features for prediction...")
        X, y_true, produce_types = self.feature_extractor.extract(data_loader)
        y_pred = self.pipeline.predict(X)
        return y_pred, y_true, produce_types

    def get_proba(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get probability estimates.

        Args:
            data_loader: DataLoader to predict on.

        Returns:
            Tuple of (probabilities, true_labels).
        """
        X, y_true, _ = self.feature_extractor.extract(data_loader)
        probas = self.pipeline.predict_proba(X)
        return probas, y_true

    def save(self, filepath: str):
        """Save the trained pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "classifier_type": self.classifier_type,
                "best_params": self.best_params,
            }, f)
        print(f"Classical model saved to {filepath}")

    def load(self, filepath: str):
        """Load a trained pipeline from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.classifier_type = data["classifier_type"]
        self.best_params = data.get("best_params")
        self.feature_extractor = FeatureExtractor()
        print(f"Classical model loaded from {filepath}")


def train_classical(
    train_loader: DataLoader,
    classifier_type: str = "random_forest",
    tune_hyperparams: bool = True,
) -> ClassicalModel:
    """Train a classical ML model.

    Args:
        train_loader: Training DataLoader.
        classifier_type: Type of classifier ('random_forest', 'gradient_boosting', 'svm').
        tune_hyperparams: Whether to tune hyperparameters.

    Returns:
        Trained ClassicalModel.
    """
    model = ClassicalModel(classifier_type=classifier_type)
    model.fit(train_loader, tune_hyperparams=tune_hyperparams)
    return model
