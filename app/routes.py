"""
API routes for the Fresh or Rotten web application.
"""

import os
import time
from pathlib import Path
from flask import request, jsonify, send_from_directory, current_app

from app.inference import get_predictor


def register_routes(app):
    """Register all routes with the Flask app.

    Args:
        app: Flask application instance.
    """

    # Get the project root directory (parent of app/)
    project_root = Path(__file__).parent.parent
    static_folder = project_root / "static"

    @app.route("/")
    def index():
        """Serve the main web interface."""
        return send_from_directory(static_folder, "index.html")

    @app.route("/health")
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "model_loaded": get_predictor()._model is not None
        })

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """
        Predict produce freshness from an uploaded image.

        Expects:
            - multipart/form-data with 'image' field

        Returns:
            JSON with prediction results:
            {
                "success": true,
                "prediction": "Fresh" or "Rotten",
                "confidence": 95.3,
                "probabilities": {
                    "Fresh": 95.3,
                    "Rotten": 4.7
                },
                "inference_time": 0.15
            }
        """
        start_time = time.time()

        # Check if image is in request
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400

        # Validate file type
        allowed_extensions = {"jpg", "jpeg", "png", "bmp", "webp"}
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            }), 400

        try:
            # Load model if not loaded
            predictor = get_predictor()
            if predictor._model is None:
                model_path = current_app.config["MODEL_PATH"]
                predictor.load_model(model_path)

            # Read image data
            image_data = file.read()

            # Run inference
            result = predictor.predict_image(image_data)

            # Calculate inference time
            inference_time = time.time() - start_time

            # Return results
            return jsonify({
                "success": True,
                **result,
                "inference_time": round(inference_time, 3)
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    @app.route("/api/model-info")
    def model_info():
        """Get information about the loaded model."""
        predictor = get_predictor()
        return jsonify({
            "model_loaded": predictor._model is not None,
            "device": str(predictor._device) if predictor._device else None,
            "model_path": current_app.config.get("MODEL_PATH"),
            "supported_produce": [
                "Apple", "Banana", "Bellpepper", "Bittergourd", "Capsicum",
                "Carrot", "Cucumber", "Mango", "Okra", "Orange",
                "Potato", "Strawberry", "Tomato"
            ],
            "classes": ["Fresh", "Rotten"]
        })

    @app.errorhandler(413)
    def file_too_large(e):
        """Handle file size too large error."""
        return jsonify({
            "success": False,
            "error": "File too large. Maximum size: 16MB"
        }), 413

    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        return jsonify({
            "success": False,
            "error": "Endpoint not found"
        }), 404

    @app.errorhandler(500)
    def internal_error(e):
        """Handle internal server errors."""
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
