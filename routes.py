"""
routes.py
─────────────────────────────────────────────────────────────────────────────
Flask Blueprint containing all application routes.

Routes:
  GET  /                →  Landing / upload page
  GET  /demo            →  Pre-filled demo page
  POST /api/screen      →  Main screening endpoint (JSON)
  GET  /api/health      →  Health check
  GET  /api/roles       →  List all job role profiles
─────────────────────────────────────────────────────────────────────────────
"""

import os
import uuid
import traceback

from flask import (
    Blueprint, request, jsonify, render_template,
    current_app, send_from_directory,
)
from werkzeug.utils import secure_filename

from parser    import parse_resume, parse_job_description
from features  import (TFIDFExtractor, compute_all_similarities,
                        bert_cosine_similarity)
from skill_gap import analyze_skill_gap, recommend_jobs, JOB_ROLE_PROFILES
from xai       import explain_prediction
from config    import config

import joblib

bp = Blueprint("main", __name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (lazy – loaded once per worker on first request)
# ─────────────────────────────────────────────────────────────────────────────

_models: dict = {}


def _load_models() -> bool:
    """
    Load all serialised model artefacts into the _models dict.
    Returns True on success, False if any file is missing.
    """
    global _models
    if _models:          # already loaded
        return True

    paths = {
        "rf"   : os.path.join(config.MODEL_DIR, config.RF_MODEL_FILE),
        "lr"   : os.path.join(config.MODEL_DIR, config.LR_MODEL_FILE),
        "tfidf": os.path.join(config.MODEL_DIR, config.TFIDF_FILE),
        "le"   : os.path.join(config.MODEL_DIR, config.LABEL_ENC_FILE),
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            current_app.logger.warning(f"Model file not found: {path}")
            return False
        _models[key] = joblib.load(path)

    # Wrap tfidf sklearn vectorizer in our TFIDFExtractor helper
    extractor = TFIDFExtractor()
    extractor.vectorizer = _models["tfidf"]
    extractor._fitted    = True
    _models["extractor"] = extractor

    current_app.logger.info("All models loaded successfully.")
    return True


def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in config.ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    """Landing page with the resume upload form."""
    return render_template("index.html")


@bp.route("/demo")
def demo():
    """Demo result page with pre-computed sample data (no file upload needed)."""
    sample_data = {
        "candidate"     : "Arjun Sharma",
        "category"      : "Data Science",
        "confidence"    : 91.4,
        "similarity"    : {"tfidf": 0.72, "bert": 0.84, "hybrid": 0.79},
        "skill_gap"     : {
            "match_percentage" : 66.7,
            "matched_skills"   : ["python", "machine learning", "sql", "pandas"],
            "missing_skills"   : ["docker", "aws"],
            "extra_skills"     : ["flask", "git"],
            "recommendations"  : {
                "docker": {"name": "Docker Mastery – Udemy",
                           "url" : "https://www.udemy.com/course/docker-mastery/"},
                "aws"   : {"name": "AWS Cloud Practitioner – AWS Training",
                           "url" : "https://aws.amazon.com/training/"},
            },
        },
        "recommendations": [
            {"role": "Data Scientist",  "match_pct": 80.0,
             "matched_skills": ["python", "machine learning", "pandas", "sql"]},
            {"role": "ML Engineer",     "match_pct": 60.0,
             "matched_skills": ["python", "machine learning"]},
            {"role": "Data Analyst",    "match_pct": 55.0,
             "matched_skills": ["python", "sql", "pandas"]},
        ],
        "shap_plot"     : "/static/img/shap/waterfall.png",
        "top_features"  : ["python", "machine learning", "sql",
                           "neural network", "pandas"],
        "models_ready"  : False,
    }
    return render_template("results.html", **sample_data)


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/api/health")
def health():
    """Health check — tells the caller whether models are loaded."""
    models_ok = _load_models()
    return jsonify({
        "status"      : "ok",
        "models_ready": models_ok,
    })


@bp.route("/api/roles")
def list_roles():
    """Return all available job role profiles."""
    roles = {role: skills for role, skills in JOB_ROLE_PROFILES.items()}
    return jsonify({"roles": roles})


@bp.route("/api/screen", methods=["POST"])
def screen_resume():
    """
    Main screening endpoint.

    Accepts multipart/form-data:
        resume           – PDF or DOCX file
        job_description  – plain text string

    Returns JSON:
        {
          "candidate"      : str,
          "category"       : str,
          "confidence"     : float,
          "similarity"     : {tfidf, bert, hybrid},
          "skill_gap"      : {...},
          "recommendations": [...],
          "shap_plot"      : str  (URL),
          "top_features"   : [str],
          "models_ready"   : bool,
        }
    """
    # ── Validate input ────────────────────────────────────────────────────
    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided."}), 400

    file    = request.files["resume"]
    jd_text = request.form.get("job_description", "").strip()

    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Please upload a PDF or DOCX."
        }), 415

    if not jd_text:
        return jsonify({"error": "Job description cannot be empty."}), 400

    # ── Save uploaded file ────────────────────────────────────────────────
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    ext       = file.filename.rsplit(".", 1)[-1].lower()
    fname     = f"{uuid.uuid4().hex}.{ext}"           # unique filename
    file_path = os.path.join(config.UPLOAD_FOLDER, fname)
    file.save(file_path)

    try:
        # ── Parse resume ──────────────────────────────────────────────────
        resume_data = parse_resume(file_path)
        jd_data     = parse_job_description(jd_text)

        # ── Classification (ML) ───────────────────────────────────────────
        models_ready = _load_models()
        category, confidence, shap_result, top_features = (
            "N/A", 0.0, None, []
        )

        if models_ready:
            extractor  = _models["extractor"]
            rf_model   = _models["rf"]
            le         = _models["le"]

            resume_vec = extractor.transform([resume_data["raw_text"]])
            pred_enc   = rf_model.predict(resume_vec)[0]
            category   = le.inverse_transform([pred_enc])[0]
            confidence = float(rf_model.predict_proba(resume_vec).max()) * 100

            # ── SHAP explanation ──────────────────────────────────────────
            try:
                # Use a small in-memory background (the current sample)
                shap_result = explain_prediction(
                    model           = rf_model,
                    tfidf_extractor = extractor,
                    resume_text     = resume_data["raw_text"],
                    X_background    = resume_vec,
                    save_dir        = config.SHAP_PLOT_DIR,
                    filename        = f"waterfall_{fname}.png",
                )
                top_features = shap_result["top_features"][:5]
            except Exception as shap_err:
                current_app.logger.warning(
                    f"SHAP failed (non-fatal): {shap_err}")

        # ── Similarity scores ─────────────────────────────────────────────
        sim_scores = {"tfidf": 0.0, "bert": 0.0, "hybrid": 0.0}
        if models_ready:
            try:
                sim_scores = compute_all_similarities(
                    resume_data["raw_text"], jd_text, extractor
                )
                print(sim_scores)
            except Exception as sim_err:
                current_app.logger.warning(f"Similarity failed: {sim_err}")

        # ── Skill gap & recommendations ───────────────────────────────────
        gap_report    = analyze_skill_gap(
            resume_data["skills"], jd_data["required_skills"]
        )
        job_recs      = recommend_jobs(resume_data["skills"], top_n=5)

        # ── Build response ────────────────────────────────────────────────
        shap_url = None
        if shap_result:
            filename = os.path.basename(
                shap_result["plot_path"]
            )
            shap_url = f"/static/img/shap/{filename}"

        response = {
            "candidate"     : resume_data["name"] or "Unknown",
            "email"         : resume_data["email"],
            "skills_found"  : resume_data["skills"],
            "education"     : resume_data["education"],
            "experience_yrs": resume_data["experience"],
            "category"      : category,
            "confidence"    : round(confidence, 2),
            "similarity"    : sim_scores,
            "skill_gap"     : gap_report,
            "recommendations": job_recs,
            "shap_plot"     : shap_url,
            "top_features"  : top_features,
            "models_ready"  : models_ready,
        }
        return jsonify(response), 200

    except Exception as exc:
        current_app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Processing failed: {str(exc)}"}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
