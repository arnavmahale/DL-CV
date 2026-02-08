"""
Deep learning model for produce freshness classification.
Fine-tunes EfficientNet-B0 pretrained on ImageNet for binary classification.
"""

import os
import copy
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from scripts.utils import get_device


class FreshnessClassifier(nn.Module):
    """EfficientNet-B0 based binary classifier for produce freshness."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters except the classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


class DeepModel:
    """Wrapper for training and using the deep learning model."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        self.device = get_device()
        self.model = FreshnessClassifier(num_classes=num_classes, dropout=dropout)
        self.model = self.model.to(self.device)
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        head_epochs: int = 5,
        finetune_epochs: int = 15,
        head_lr: float = 1e-3,
        finetune_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        patience: int = 5,
    ) -> Dict[str, list]:
        """Train the model with two-phase strategy.

        Phase 1: Train only the classifier head with frozen backbone.
        Phase 2: Fine-tune entire model with lower learning rate.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            class_weights: Tensor of class weights for loss function.
            head_epochs: Epochs for phase 1 (head only).
            finetune_epochs: Epochs for phase 2 (full fine-tuning).
            head_lr: Learning rate for phase 1.
            finetune_lr: Learning rate for phase 2.
            weight_decay: Weight decay for optimizer.
            patience: Early stopping patience.

        Returns:
            Training history dictionary.
        """
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        # Phase 1: Train classifier head only
        print("\n--- Phase 1: Training classifier head ---")
        self.model.freeze_backbone()

        head_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)

        for epoch in range(head_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model, train_loader, criterion, optimizer, self.device
            )
            val_loss, val_acc = validate(self.model, val_loader, criterion, self.device)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{head_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Phase 2: Fine-tune entire model
        print("\n--- Phase 2: Fine-tuning full model ---")
        self.model.unfreeze_backbone()

        optimizer = torch.optim.AdamW([
            {"params": self.model.backbone.features.parameters(), "lr": finetune_lr},
            {"params": self.model.backbone.classifier.parameters(), "lr": finetune_lr * 10},
        ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=finetune_epochs, eta_min=1e-7
        )

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(finetune_epochs):
            train_loss, train_acc = train_one_epoch(
                self.model, train_loader, criterion, optimizer, self.device
            )
            val_loss, val_acc = validate(self.model, val_loader, criterion, self.device)
            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{finetune_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                print(f"  -> New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model (val_loss={best_val_loss:.4f})")

        return self.history

    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Predict on data from a DataLoader.

        Returns:
            Tuple of (predictions, true_labels, produce_types).
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_types = []

        for images, labels, types in tqdm(data_loader, desc="Predicting", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_types.extend(types)

        return np.array(all_preds), np.array(all_labels), all_types

    @torch.no_grad()
    def get_proba(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get probability estimates using softmax.

        Returns:
            Tuple of (probabilities, true_labels).
        """
        self.model.eval()
        all_probas = []
        all_labels = []

        for images, labels, _ in data_loader:
            images = images.to(self.device)
            outputs = self.model(images)
            probas = torch.softmax(outputs, dim=1)

            all_probas.append(probas.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

        probas = np.concatenate(all_probas, axis=0)
        labels = np.array(all_labels)
        return probas, labels

    def save(self, filepath: str):
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }, filepath)
        print(f"Deep model saved to {filepath}")

    def load(self, filepath: str):
        """Load model weights from disk."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})
        print(f"Deep model loaded from {filepath}")


def train_deep(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    head_epochs: int = 5,
    finetune_epochs: int = 15,
    patience: int = 5,
) -> DeepModel:
    """Train the deep learning model.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_weights: Tensor of class weights.
        head_epochs: Epochs for head-only training.
        finetune_epochs: Epochs for full fine-tuning.
        patience: Early stopping patience.

    Returns:
        Trained DeepModel.
    """
    model = DeepModel()
    model.fit(
        train_loader, val_loader, class_weights,
        head_epochs=head_epochs,
        finetune_epochs=finetune_epochs,
        patience=patience,
    )
    return model
