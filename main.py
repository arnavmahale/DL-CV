"""
Main entry point for the Fresh or Rotten web application.

Starts the Flask web server for real-time produce freshness classification.

Usage:
    python main.py                    # Run in development mode
    python main.py --port 8080        # Run on custom port
    python main.py --host 0.0.0.0     # Allow external connections
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

# Create app instance for Gunicorn
app = create_app()


def main():
    parser = argparse.ArgumentParser(
        description="Fresh or Rotten Web Application"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 for external access."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with auto-reload"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/deep_efficientnet_b0.pth",
        help="Path to the trained model weights"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("\nPlease train the model first:")
        print("  python scripts/train.py --model deep")
        print("\nOr specify a different model path:")
        print(f"  python main.py --model-path <path-to-model>")
        sys.exit(1)

    # Create Flask app
    app = create_app(model_path=args.model_path)

    print("="*60)
    print("🍎 FRESH OR ROTTEN - WEB APPLICATION")
    print("="*60)
    print(f"\n📱 Starting web server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Model: {args.model_path}")
    print(f"   Debug: {args.debug}")
    print(f"\n🌐 Access the app at:")

    if args.host == "0.0.0.0":
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"   Local:    http://localhost:{args.port}")
        print(f"   Network:  http://{local_ip}:{args.port}")
        print(f"\n📱 On your phone: Connect to the same WiFi and visit:")
        print(f"   http://{local_ip}:{args.port}")
    else:
        print(f"   http://{args.host}:{args.port}")

    print(f"\n🎥 Grant camera permissions when prompted!")
    print("="*60 + "\n")

    # Run the app
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
