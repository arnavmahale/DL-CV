"""
Flask application factory for Fresh or Rotten web app.
"""

import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS


def create_app(model_path="models/deep_efficientnet_b0.pth"):
    """
    Create and configure the Flask application.

    Args:
        model_path: Path to the trained PyTorch model.

    Returns:
        Configured Flask app instance.
    """
    # Get absolute path to static folder
    project_root = Path(__file__).parent.parent
    static_folder = str(project_root / "static")

    app = Flask(
        __name__,
        static_folder=static_folder,
        static_url_path="/static"
    )

    # Enable CORS for API endpoints
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Configuration
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
    app.config["MODEL_PATH"] = model_path
    app.config["UPLOAD_FOLDER"] = "data/uploads"

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register blueprints/routes
    from app.routes import register_routes
    register_routes(app)

    # Preload model at startup to avoid slow first request
    with app.app_context():
        from app.inference import get_predictor
        predictor = get_predictor()
        if predictor._model is None and os.path.exists(model_path):
            print(f"Preloading model from {model_path}...")
            predictor.load_model(model_path)
            print("Model preloaded successfully!")

    return app
