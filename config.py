"""
config.py
─────────────────────────────────────────────────────────────
Centralised configuration for the Resume Screening System.
All file paths, model hyper-parameters, and feature flags
are kept here so that nothing is hard-coded in other modules.
─────────────────────────────────────────────────────────────
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── Flask ────────────────────────────────────────────────────────────
    SECRET_KEY        = os.environ.get("SECRET_KEY", "xai-resume-2026-secret")
    DEBUG             = False
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024          # 5 MB upload limit

    # ── File Upload ──────────────────────────────────────────────────────
    UPLOAD_FOLDER     = os.path.join(BASE_DIR, "data", "uploads")
    ALLOWED_EXTENSIONS = {"pdf", "docx"}

    # ── Paths ────────────────────────────────────────────────────────────
    MODEL_DIR         = os.path.join(BASE_DIR, "models")
    DATASET_DIR       = os.path.join(BASE_DIR, "dataset")
    STATIC_DIR        = os.path.join(BASE_DIR, "static")
    SHAP_PLOT_DIR     = os.path.join(BASE_DIR, "static", "img", "shap")

    # ── Saved model file names ───────────────────────────────────────────
    RF_MODEL_FILE     = "random_forest.pkl"
    LR_MODEL_FILE     = "logistic_regression.pkl"
    TFIDF_FILE        = "tfidf_vectorizer.pkl"
    LABEL_ENC_FILE    = "label_encoder.pkl"

    # ── TF-IDF ───────────────────────────────────────────────────────────
    TFIDF_MAX_FEATURES = 5_000
    TFIDF_NGRAM_RANGE  = (1, 2)

    # ── BERT ─────────────────────────────────────────────────────────────
    BERT_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
    # Set to False to skip BERT (faster, useful when GPU not available)
    USE_BERT          = True

    # ── Hybrid similarity weights ────────────────────────────────────────
    HYBRID_WEIGHT_TFIDF = 0.40
    HYBRID_WEIGHT_BERT  = 0.60

    # ── Model Training ───────────────────────────────────────────────────
    TEST_SIZE       = 0.20
    RANDOM_STATE    = 42
    CV_FOLDS        = 5

    # ── Random Forest hyper-params ───────────────────────────────────────
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH    = None
    RF_N_JOBS       = -1

    # ── Logistic Regression hyper-params ────────────────────────────────
    LR_C            = 1.0
    LR_MAX_ITER     = 1_000
    LR_SOLVER       = "lbfgs"
    LR_MULTI_CLASS  = "auto"

    # ── SHAP ─────────────────────────────────────────────────────────────
    SHAP_TOP_N_FEATURES = 15


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


# Active config — switch to ProductionConfig before deployment
config = DevelopmentConfig()
