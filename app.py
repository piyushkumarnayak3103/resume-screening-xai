"""
app.py
─────────────────────────────────────────────────────────────────────────────
Flask Application Factory.

Usage:
    python app.py               # development server
    gunicorn app:create_app()   # production
─────────────────────────────────────────────────────────────────────────────
"""

import os
from flask import Flask
from config import config


def create_app() -> Flask:
    """
    Application factory — creates and configures the Flask app.
    Using a factory function makes the app easier to test and deploy.
    """
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # ── Apply config ──────────────────────────────────────────────────────
    app.secret_key            = config.SECRET_KEY
    app.config["DEBUG"]       = config.DEBUG
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"]      = config.UPLOAD_FOLDER

    # ── Ensure required directories exist ────────────────────────────────
    for directory in [
        config.UPLOAD_FOLDER,
        config.MODEL_DIR,
        config.DATASET_DIR,
        config.SHAP_PLOT_DIR,
        os.path.join("static", "img"),
    ]:
        os.makedirs(directory, exist_ok=True)

    # ── Register Blueprint ────────────────────────────────────────────────
    from routes import bp
    app.register_blueprint(bp)

    # ── Request / response logging (development only) ────────────────────
    if config.DEBUG:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        app.logger.setLevel(logging.DEBUG)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Development entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = config.DEBUG,
    )
