"""
preprocessing.py
─────────────────────────────────────────────────────────────────────────────
NLP text-cleaning pipeline used by both the feature extractor and the
model training notebook.

Steps applied (in order):
  1. Lowercase
  2. URL / email / number removal
  3. Special-character stripping (keep alphanumerics + spaces)
  4. Tokenisation (simple whitespace split)
  5. Stopword removal  (NLTK English stopword list)
  6. Lemmatisation     (WordNet lemmatizer)
  7. Short-token filtering (tokens with len < 2 are dropped)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
from typing      import List

# ── Auto-download required NLTK data ─────────────────────────────────────────
for resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_STOP_WORDS   = set(stopwords.words("english"))
_LEMMATIZER   = WordNetLemmatizer()

# Domain words that should NOT be removed even if they appear in stopwords
_DOMAIN_KEEP  = {
    "not", "no", "but", "however", "although", "against",
    "above", "below", "up", "down",
}
EFFECTIVE_STOPWORDS = _STOP_WORDS - _DOMAIN_KEEP


# ─────────────────────────────────────────────────────────────────────────────
# Core cleaning functions
# ─────────────────────────────────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Strip http/https URLs and bare www addresses."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    """Strip email addresses (already extracted by parser.py)."""
    return re.sub(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}", " ", text)


def remove_numbers(text: str) -> str:
    """Replace standalone numbers with a space."""
    return re.sub(r"\b\d+\b", " ", text)


def remove_punctuation(text: str) -> str:
    """
    Replace all punctuation with spaces.
    Preserves the '+' in 'c++' by temporarily substituting it.
    """
    text = text.replace("c++", "cplusplus").replace("c#", "csharp")
    text = text.translate(str.maketrans(string.punctuation,
                                        " " * len(string.punctuation)))
    return text.replace("cplusplus", "c++").replace("csharp", "c#")


def tokenise(text: str) -> List[str]:
    """Split on whitespace and return non-empty tokens."""
    return [tok for tok in text.split() if tok]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out English stopwords (keeping domain-specific exceptions)."""
    return [t for t in tokens if t not in EFFECTIVE_STOPWORDS]


def lemmatise(tokens: List[str]) -> List[str]:
    """Lemmatise each token using NLTK WordNetLemmatizer."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def filter_short_tokens(tokens: List[str], min_len: int = 2) -> List[str]:
    """Drop tokens shorter than min_len characters."""
    return [t for t in tokens if len(t) >= min_len]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline → returns a single cleaned string.

    This is the function called by TFIDFExtractor and model training.
    """
    text = text.lower()
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)

    tokens = tokenise(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatise(tokens)
    tokens = filter_short_tokens(tokens)

    return " ".join(tokens)


def clean_corpus(texts: List[str]) -> List[str]:
    """
    Apply clean_text to a list of texts.
    Useful for vectorizer fitting during training.
    """
    return [clean_text(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "  Experienced Python developer with 3 years of Machine Learning "
        "and NLP. Proficient in c++, TensorFlow, and Docker (CI/CD). "
        "Reach me at john@example.com or visit https://portfolio.io   "
    )
    print("Input :", sample[:80], "...")
    print("Output:", clean_text(sample))
