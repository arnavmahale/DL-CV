"""
Run inference on a single image using the trained deep learning model.

Usage:
    python scripts/predict.py <image_path>
    python scripts/predict.py Dataset/Fresh/FreshApple/some_image.jpg
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from scripts.data_loader import get_transforms, LABEL_NAMES
from scripts.deep_model import DeepModel
from scripts.utils import get_device


def predict_image(image_path: str, model_path: str = "models/deep_efficientnet_b0.pth"):
    """Predict freshness of a single produce image.

    Args:
        image_path: Path to the image file.
        model_path: Path to the saved model weights.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train the model first: python scripts/train.py --model deep")
        sys.exit(1)

    # Load model
    model = DeepModel()
    model.load(model_path)
    model.model.eval()

    device = get_device()

    # Load and transform image
    transform = get_transforms("test")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model.model(input_tensor)
        probas = torch.softmax(output, dim=1)[0]
        predicted_class = probas.argmax().item()

    label = LABEL_NAMES[predicted_class]
    confidence = probas[predicted_class].item() * 100

    print(f"\nImage:      {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"  Fresh:  {probas[0].item()*100:.1f}%")
    print(f"  Rotten: {probas[1].item()*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict freshness of a produce image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", default="models/deep_efficientnet_b0.pth", help="Path to model weights.")
    args = parser.parse_args()

    predict_image(args.image, args.model)
