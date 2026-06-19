"""
xai.py
─────────────────────────────────────────────────────────────────────────────
Explainable AI (XAI) module — SHAP-based feature importance.

Two functions are exposed:

  explain_prediction()  →  waterfall plot for a single resume/prediction
  plot_summary()        →  global feature importance bar chart (test set)

Both functions:
  • Work with both RandomForest (TreeExplainer) and
    LogisticRegression (LinearExplainer)
  • Save plots as PNGs to static/img/shap/
  • Return a structured dict for the Flask API to consume
─────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend (server safe)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import shap
import joblib
from scipy.sparse import issparse
from config import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dense(X) -> np.ndarray:
    """Convert a sparse matrix to dense numpy array if necessary."""
    return X.toarray() if issparse(X) else np.asarray(X)


def _build_explainer(model, X_background):
    """
    Build the appropriate SHAP explainer based on model type.

    RandomForestClassifier  → TreeExplainer  (exact, fast)
    LogisticRegression      → LinearExplainer (exact for linear models)
    Others                  → KernelExplainer  (model-agnostic, slower)
    """
    from sklearn.ensemble     import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if isinstance(model, RandomForestClassifier):
        return shap.TreeExplainer(model), "tree"

    if isinstance(model, LogisticRegression):
        X_bg = _dense(X_background)
        return shap.LinearExplainer(model, X_bg), "linear"

    # Fallback: KernelExplainer (use a small background sample)
    bg = shap.sample(_dense(X_background), 100)
    return shap.KernelExplainer(model.predict_proba, bg), "kernel"


def _top_n_features(
    shap_vals   : np.ndarray,
    feat_names  : np.ndarray,
    n           : int = config.SHAP_TOP_N_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-N features by absolute SHAP value."""
    top_idx   = np.argsort(np.abs(shap_vals))[::-1][:n]
    return feat_names[top_idx], shap_vals[top_idx]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def explain_prediction(
    model,
    tfidf_extractor,
    resume_text  : str,
    X_background,                    # sparse/dense matrix for background ref
    save_dir     : str = config.SHAP_PLOT_DIR,
    filename     : str = "waterfall.png",
) -> dict:
    """
    Generate a SHAP waterfall / bar chart for a single resume prediction.

    Args:
        model            : Fitted sklearn model (RF or LR).
        tfidf_extractor  : Fitted TFIDFExtractor instance.
        resume_text      : Raw resume text string.
        X_background     : Training feature matrix used as SHAP background.
        save_dir         : Directory to write the PNG.
        filename         : Output filename.

    Returns:
        {
          "predicted_class"   : int,
          "predicted_label"   : str  (if label_encoder provided),
          "confidence"        : float [0,1],
          "top_features"      : [str, ...],
          "shap_values"       : [float, ...],
          "plot_path"         : str,
          "explainer_type"    : str,
        }
    """
    _ensure_dir(save_dir)

    # ── Transform resume text ─────────────────────────────────────────────
    X_sample     = tfidf_extractor.transform([resume_text])
    X_sample_d   = _dense(X_sample)
    feat_names   = tfidf_extractor.get_feature_names()

    # ── Build explainer ───────────────────────────────────────────────────
    explainer, exp_type = _build_explainer(model, X_background)

    # ── Compute SHAP values ───────────────────────────────────────────────
    if exp_type == "tree":
        sv_all = explainer.shap_values(X_sample_d)
        # multi-class: list[classes] → pick predicted class
        pred_class = int(model.predict(X_sample_d)[0])
        if isinstance(sv_all, list):
            sv = sv_all[pred_class][0]
        else:
            sv = sv_all[0]
    elif exp_type == "linear":
        sv_all     = explainer.shap_values(X_sample_d)
        pred_class = int(model.predict(X_sample_d)[0])
        if sv_all.ndim == 3:           # (1, features, classes)
            sv = sv_all[0, :, pred_class]
        elif sv_all.ndim == 2:
            sv = sv_all[0]
        else:
            sv = sv_all.ravel()
    else:                              # kernel
        sv_all     = explainer.shap_values(X_sample_d)
        pred_class = int(model.predict(X_sample_d)[0])
        sv = sv_all[pred_class][0] if isinstance(sv_all, list) else sv_all[0]

    # ── Confidence (max predict_proba) ────────────────────────────────────
    proba      = model.predict_proba(X_sample)[0]
    confidence = float(proba.max())

    # ── Top-N features ────────────────────────────────────────────────────
    top_feats, top_vals = _top_n_features(sv, feat_names)

    # ── Waterfall bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#0F8B8D" if v > 0 else "#B85042" for v in top_vals]

    # Plot reversed so highest magnitude is at the top
    bars = ax.barh(
        top_feats[::-1],
        top_vals[::-1],
        color=colors[::-1],
        edgecolor="none",
        height=0.65,
    )

    ax.axvline(0, color="#333333", linewidth=0.8, zorder=5)
    ax.set_xlabel("SHAP Value  (positive → supports prediction,  "
                  "negative → opposes prediction)", fontsize=10)
    ax.set_title("Feature Importance — SHAP Explanation\n"
                 f"(Predicted class: {pred_class}   "
                 f"Confidence: {confidence*100:.1f}%)",
                 fontsize=12, fontweight="bold", pad=12)

    pos_patch = mpatches.Patch(color="#0F8B8D", label="Positive impact")
    neg_patch = mpatches.Patch(color="#B85042", label="Negative impact")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[SHAP] Waterfall plot saved → {plot_path}")

    return {
        "predicted_class" : pred_class,
        "confidence"      : round(confidence, 4),
        "top_features"    : top_feats.tolist(),
        "shap_values"     : [round(v, 6) for v in top_vals.tolist()],
        "plot_path"       : plot_path,
        "explainer_type"  : exp_type,
    }


def plot_summary(
    model,
    tfidf_extractor,
    X_test,
    save_dir : str = config.SHAP_PLOT_DIR,
    filename : str = "summary.png",
    max_display: int = 20,
) -> str:
    """
    Generate a global SHAP feature importance bar chart across the test set.
    Useful for the viva — shows which features matter most overall.

    Args:
        model            : Fitted sklearn model.
        tfidf_extractor  : Fitted TFIDFExtractor.
        X_test           : Test feature matrix.
        save_dir         : Output directory.
        filename         : Output filename.
        max_display      : Maximum number of features to show.

    Returns:
        Path to the saved PNG.
    """
    _ensure_dir(save_dir)

    X_test_d   = _dense(X_test)
    feat_names = tfidf_extractor.get_feature_names()

    explainer, exp_type = _build_explainer(model, X_test)

    if exp_type == "tree":
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):
            # Sum absolute SHAP values across all classes
            sv = np.sum([np.abs(v) for v in sv], axis=0)
        else:
            sv = np.abs(sv)
    else:
        sv = np.abs(explainer.shap_values(X_test_d))
        if sv.ndim == 3:
            sv = sv.sum(axis=2)

    # Mean absolute SHAP per feature
    mean_abs = sv.mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:max_display]
    top_feat = feat_names[top_idx]
    top_val  = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_feat[::-1], top_val[::-1],
            color="#0D1B2A", edgecolor="none", height=0.65)
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output magnitude)",
                  fontsize=10)
    ax.set_title("Global Feature Importance — SHAP Summary\n"
                 f"(Top {max_display} features across test set)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SHAP] Summary plot saved → {plot_path}")
    return plot_path
