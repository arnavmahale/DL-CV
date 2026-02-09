"""
Model inference utilities for real-time prediction.

Handles loading the trained model and running inference on images.
"""

import os
import io
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from scripts.data_loader import get_transforms, LABEL_NAMES
from scripts.deep_model import DeepModel
from scripts.utils import get_device


class ModelPredictor:
    """Singleton class for model inference."""

    _instance = None
    _model = None
    _device = None
    _transform = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize predictor (only once)."""
        if self._model is None:
            self._device = get_device()
            self._transform = get_transforms("test")
            print("Model predictor initialized (lazy loading)")

    def load_model(self, model_path: str):
        """Load the trained model.

        Args:
            model_path: Path to the model weights (.pth file).
        """
        if self._model is not None:
            print("Model already loaded, skipping...")
            return

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}...")
        self._model = DeepModel()
        self._model.load(model_path)
        self._model.model.eval()
        self._model.model.to(self._device)
        print(f"Model loaded successfully on {self._device}")

    def predict_image(self, image_data: bytes) -> Dict:
        """Predict freshness from image bytes.

        Args:
            image_data: Image data as bytes (e.g., from request.files).

        Returns:
            Dictionary with prediction results:
            {
                "prediction": "Fresh" or "Rotten",
                "confidence": float (0-100),
                "probabilities": {
                    "Fresh": float (0-100),
                    "Rotten": float (0-100)
                }
            }
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image from bytes
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess
        input_tensor = self._transform(image).unsqueeze(0).to(self._device)

        # Predict
        with torch.no_grad():
            output = self._model.model(input_tensor)
            probas = torch.softmax(output, dim=1)[0]
            predicted_idx = probas.argmax().item()

        # Format results
        prediction = LABEL_NAMES[predicted_idx]
        confidence = float(probas[predicted_idx].item() * 100)
        probabilities = {
            "Fresh": float(probas[0].item() * 100),
            "Rotten": float(probas[1].item() * 100)
        }

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                k: round(v, 2) for k, v in probabilities.items()
            }
        }

    def predict_from_path(self, image_path: str) -> Dict:
        """Predict freshness from image file path.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary with prediction results.
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return self.predict_image(image_data)


# Global predictor instance
_predictor = None


def get_predictor() -> ModelPredictor:
    """Get the global model predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ModelPredictor()
    return _predictor
