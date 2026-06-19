"""
features.py
─────────────────────────────────────────────────────────────────────────────
Dual-mode feature extraction engine.

Mode A – TF-IDF (sparse, fast)
    • sklearn TfidfVectorizer with bigrams and sublinear TF scaling
    • Used for ML classification (Logistic Regression / Random Forest)
    • Serialised to disk so the same vectorizer is used at inference time

Mode B – BERT Sentence Embeddings (dense, semantic)
    • HuggingFace sentence-transformers/all-MiniLM-L6-v2
    • 384-dimensional embeddings; lightweight enough for CPU inference
    • Singleton pattern avoids reloading the model on every request

Mode C – Hybrid Similarity Score
    • Weighted combination: 40% TF-IDF cosine + 60% BERT cosine
    • Used for resume ↔ JD matching and ranking
─────────────────────────────────────────────────────────────────────────────
"""

import os
import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text
from config import config

# ── Optional BERT import (graceful degradation if not installed) ─────────────
_BERT_AVAILABLE = False
_bert_model = None

try:
    from sentence_transformers import SentenceTransformer
    _BERT_AVAILABLE = True
except ImportError:
    print("[features.py] sentence-transformers not installed. "
          "BERT embeddings will be disabled. "
          "Run: pip install sentence-transformers")


# ─────────────────────────────────────────────────────────────────────────────
# BERT Singleton Loader
# ─────────────────────────────────────────────────────────────────────────────

def _get_bert_model() -> "SentenceTransformer | None":
    """
    Return the BERT model, loading it on first call (singleton).
    Returns None if sentence-transformers is not available.
    """
    global _bert_model
    if not _BERT_AVAILABLE or not config.USE_BERT:
        return None
    if _bert_model is None:
        print(f"[BERT] Loading model: {config.BERT_MODEL_NAME}  (first call only)")
        _bert_model = SentenceTransformer(config.BERT_MODEL_NAME)
    return _bert_model


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Extractor Class
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFExtractor:
    """
    Wraps sklearn TfidfVectorizer with fit / transform / save / load helpers.

    Usage (training):
        extractor = TFIDFExtractor()
        X = extractor.fit_transform(corpus)          # list of raw strings
        extractor.save("models/tfidf_vectorizer.pkl")

    Usage (inference):
        extractor = TFIDFExtractor().load("models/tfidf_vectorizer.pkl")
        vec = extractor.transform(["Python ML engineer with 3 years…"])
    """

    def __init__(
        self,
        max_features: int = config.TFIDF_MAX_FEATURES,
        ngram_range: tuple = config.TFIDF_NGRAM_RANGE,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # log(1 + tf) — reduces impact of high-freq terms
            min_df=2,                    # ignore terms that appear in < 2 documents
            analyzer="word",
            token_pattern=r"[a-zA-Z+#]{2,}",  # keeps c++, c# etc.
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TFIDFExtractor":
        """Fit the vectorizer on a list of raw text strings."""
        cleaned = [clean_text(t) for t in corpus]
        self.vectorizer.fit(cleaned)
        self._fitted = True
        return self

    def transform(self, texts: list[str]):
        """
        Transform a list of raw strings to a sparse TF-IDF matrix.
        Preprocessing (clean_text) is applied automatically.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before transform().")
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def fit_transform(self, corpus: list[str]):
        """Fit and transform in one step."""
        self.fit(corpus)
        return self.transform(corpus)

    def get_feature_names(self) -> np.ndarray:
        """Return the vocabulary array (useful for SHAP feature labelling)."""
        return self.vectorizer.get_feature_names_out()

    def save(self, path: str) -> None:
        """Persist the fitted vectorizer to disk using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        print(f"[TF-IDF] Saved → {path}")

    def load(self, path: str) -> "TFIDFExtractor":
        """Load a previously saved vectorizer from disk."""
        self.vectorizer = joblib.load(path)
        self._fitted = True
        print(f"[TF-IDF] Loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# BERT Embedding Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_bert_embedding(text: str) -> np.ndarray:
    """
    Return a 384-dim BERT sentence embedding for a single text string.
    Falls back to a zero vector if BERT is unavailable.
    """
    model = _get_bert_model()
    if model is None:
        return np.zeros(384, dtype=np.float32)
    cleaned = clean_text(text)
    return model.encode(cleaned, show_progress_bar=False)


def get_bert_embeddings_batch(texts: list[str]) -> np.ndarray:
    """
    Batch encode a list of texts.
    Returns an (N, 384) float32 array.
    Batch processing is significantly faster than calling get_bert_embedding N times.
    """
    model = _get_bert_model()
    if model is None:
        return np.zeros((len(texts), 384), dtype=np.float32)
    cleaned = [clean_text(t) for t in texts]
    return model.encode(cleaned, batch_size=32, show_progress_bar=True)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Functions
# ─────────────────────────────────────────────────────────────────────────────

def tfidf_cosine_similarity(vec_a, vec_b) -> float:
    """
    Cosine similarity between two TF-IDF vectors (sparse or dense).
    Returns a float in [0, 1].
    """
    if issparse(vec_a):
        sim = cosine_similarity(vec_a, vec_b)
    else:
        sim = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))
    return float(sim[0][0])


def bert_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Semantic cosine similarity between two texts using BERT embeddings.
    Returns a float in [0, 1] (normalised by L2 norms).
    Falls back to 0.0 if BERT is unavailable.
    """
    emb_a = get_bert_embedding(text_a)
    emb_b = get_bert_embedding(text_b)
    norm  = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
    if norm == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / norm)


def hybrid_similarity(
    tfidf_sim: float,
    bert_sim : float,
    w_tfidf  : float = config.HYBRID_WEIGHT_TFIDF,
    w_bert   : float = config.HYBRID_WEIGHT_BERT,
) -> float:
    """
    Weighted linear combination of TF-IDF and BERT similarity scores.

    Default weights (from config):
        TF-IDF : 0.40  (captures keyword overlap)
        BERT   : 0.60  (captures semantic meaning)

    Returns a float in [0, 1].
    """
    return round(w_tfidf * tfidf_sim + w_bert * bert_sim, 4)


def compute_all_similarities(resume_text: str, jd_text: str,
                              tfidf_extractor: TFIDFExtractor) -> dict:
    """
    Compute all three similarity scores for a resume–JD pair.

    Args:
        resume_text      : raw resume text
        jd_text          : raw job description text
        tfidf_extractor  : fitted TFIDFExtractor instance

    Returns:
        dict with keys: tfidf, bert, hybrid  (all floats in [0,1])
    """
    r_vec     = tfidf_extractor.transform([resume_text])
    jd_vec    = tfidf_extractor.transform([jd_text])
    t_sim     = tfidf_cosine_similarity(r_vec, jd_vec)
    b_sim     = bert_cosine_similarity(resume_text, jd_text)
    h_sim     = hybrid_similarity(t_sim, b_sim)
    return {"tfidf": round(t_sim, 4), "bert": round(b_sim, 4), "hybrid": h_sim}
