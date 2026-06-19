"""
model_training.py
─────────────────────────────────────────────────────────────────────────────
Complete ML training pipeline.

Steps:
  1. Load the Kaggle Resume Dataset  (UpdatedResumeDataSet.csv)
  2. Encode class labels with LabelEncoder
  3. Extract TF-IDF features          → TFIDFExtractor
  4. Train Logistic Regression        → cross-validate + evaluate
  5. Train Random Forest              → cross-validate + evaluate
  6. Print a side-by-side comparison table
  7. Save all models to disk          → models/ directory

Run:
    python model_training.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import joblib
import numpy  as np
import pandas as pd

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

from features import TFIDFExtractor
from config   import config

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def evaluate(model, X_test, y_test, label_names: list[str]) -> dict:
    """
    Evaluate a fitted model and return a metrics dictionary.
    Also prints the full sklearn classification_report.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted",
                                     zero_division=0),
        "recall"   : recall_score(y_test, y_pred, average="weighted",
                                  zero_division=0),
        "f1"       : f1_score(y_test, y_pred, average="weighted",
                              zero_division=0),
    }
    print(classification_report(y_test, y_pred,
                                 target_names=label_names,
                                 zero_division=0))
    return metrics


def cross_val_summary(model, X, y, cv: int) -> dict:
    """
    Run stratified k-fold cross-validation and return mean ± std for
    accuracy, precision, recall, and F1.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True,
                          random_state=config.RANDOM_STATE)
    scoring = ["accuracy", "precision_weighted",
               "recall_weighted", "f1_weighted"]
    results = cross_validate(model, X, y, cv=skf, scoring=scoring,
                              n_jobs=-1, return_train_score=False)
    return {
        "cv_accuracy" : (results["test_accuracy"].mean(),
                         results["test_accuracy"].std()),
        "cv_precision": (results["test_precision_weighted"].mean(),
                         results["test_precision_weighted"].std()),
        "cv_recall"   : (results["test_recall_weighted"].mean(),
                         results["test_recall_weighted"].std()),
        "cv_f1"       : (results["test_f1_weighted"].mean(),
                         results["test_f1_weighted"].std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the Kaggle UpdatedResumeDataSet.csv file.
    Expected columns: 'Resume', 'Category'
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            "Download from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset\n"
            "and place it in the dataset/ folder as 'UpdatedResumeDataSet.csv'"
        )
    df = pd.read_csv(csv_path)
    # Validate expected columns
    required_cols = {"Resume", "Category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. "
                         f"Found: {list(df.columns)}")
    # Drop any nulls and reset index
    df = df.dropna(subset=["Resume", "Category"]).reset_index(drop=True)
    print(f"[Data] Loaded {len(df)} rows, {df['Category'].nunique()} categories.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(csv_path: str | None = None) -> dict:
    """
    Full training pipeline.  Returns a dict of all metrics for both models.

    Args:
        csv_path: path to the resume dataset CSV.
                  Defaults to config.DATASET_DIR / 'UpdatedResumeDataSet.csv'.
    """
    if csv_path is None:
        csv_path = os.path.join(config.DATASET_DIR, "UpdatedResumeDataSet.csv")

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    _banner("STEP 1 — Loading Dataset")
    df = load_dataset(csv_path)
    print(df["Category"].value_counts().to_string())

    # ── 2. Encode labels ──────────────────────────────────────────────────
    _banner("STEP 2 — Encoding Labels")
    le = LabelEncoder()
    y  = le.fit_transform(df["Category"])
    label_names = list(le.classes_)
    print(f"Classes ({len(label_names)}): {label_names}")

    le_path = os.path.join(config.MODEL_DIR, config.LABEL_ENC_FILE)
    joblib.dump(le, le_path)
    print(f"[Label Encoder] Saved → {le_path}")

    # ── 3. TF-IDF Feature Extraction ─────────────────────────────────────
    _banner("STEP 3 — TF-IDF Feature Extraction")
    tfidf_extractor = TFIDFExtractor(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range =config.TFIDF_NGRAM_RANGE,
    )
    X = tfidf_extractor.fit_transform(df["Resume"].tolist())
    print(f"[TF-IDF] Feature matrix shape: {X.shape}")

    tfidf_path = os.path.join(config.MODEL_DIR, config.TFIDF_FILE)
    tfidf_extractor.save(tfidf_path)

    # ── 4. Train / Test Split ─────────────────────────────────────────────
    _banner("STEP 4 — Train / Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config.TEST_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y,          # preserve class proportions
    )
    print(f"Train size : {X_train.shape[0]}")
    print(f"Test size  : {X_test.shape[0]}")

    results = {}

    # ── 5. Logistic Regression ────────────────────────────────────────────
    _banner("STEP 5 — Logistic Regression")
    lr = LogisticRegression(
        C           = config.LR_C,
        max_iter    = config.LR_MAX_ITER,
        solver      = config.LR_SOLVER,
        multi_class = config.LR_MULTI_CLASS,
        random_state= config.RANDOM_STATE,
    )
    lr.fit(X_train, y_train)

    print("\n[LR] Test Set Metrics:")
    lr_metrics = evaluate(lr, X_test, y_test, label_names)

    print(f"\n[LR] {config.CV_FOLDS}-Fold Cross-Validation:")
    lr_cv = cross_val_summary(lr, X, y, config.CV_FOLDS)
    for k, (mean, std) in lr_cv.items():
        print(f"  {k:18s}: {mean:.4f} ± {std:.4f}")
    lr_metrics.update(lr_cv)

    lr_path = os.path.join(config.MODEL_DIR, config.LR_MODEL_FILE)
    joblib.dump(lr, lr_path)
    print(f"[LR] Saved → {lr_path}")
    results["Logistic Regression"] = lr_metrics

    # ── 6. Random Forest ──────────────────────────────────────────────────
    _banner("STEP 6 — Random Forest")
    rf = RandomForestClassifier(
        n_estimators = config.RF_N_ESTIMATORS,
        max_depth    = config.RF_MAX_DEPTH,
        random_state = config.RANDOM_STATE,
        n_jobs       = config.RF_N_JOBS,
    )
    rf.fit(X_train, y_train)

    print("\n[RF] Test Set Metrics:")
    rf_metrics = evaluate(rf, X_test, y_test, label_names)

    print(f"\n[RF] {config.CV_FOLDS}-Fold Cross-Validation:")
    rf_cv = cross_val_summary(rf, X, y, config.CV_FOLDS)
    for k, (mean, std) in rf_cv.items():
        print(f"  {k:18s}: {mean:.4f} ± {std:.4f}")
    rf_metrics.update(rf_cv)

    rf_path = os.path.join(config.MODEL_DIR, config.RF_MODEL_FILE)
    joblib.dump(rf, rf_path)
    print(f"[RF] Saved → {rf_path}")
    results["Random Forest"] = rf_metrics

    # ── 7. Comparison Table ───────────────────────────────────────────────
    _banner("STEP 7 — Model Comparison Table")
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Model"       : model_name,
            "Accuracy"    : f"{m['accuracy']:.4f}",
            "Precision"   : f"{m['precision']:.4f}",
            "Recall"      : f"{m['recall']:.4f}",
            "F1-Score"    : f"{m['f1']:.4f}",
            "CV F1 (mean)": f"{m['cv_f1'][0]:.4f}",
            "CV F1 (std)" : f"{m['cv_f1'][1]:.4f}",
        })
    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))

    # Save comparison table
    table_path = os.path.join(config.MODEL_DIR, "comparison_table.csv")
    comparison_df.to_csv(table_path, index=False)
    print(f"\n[Comparison] Table saved → {table_path}")

    _banner("TRAINING COMPLETE — All models saved to models/")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_evaluate()
