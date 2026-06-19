"""
parser.py
─────────────────────────────────────────────────────────────────────────────
Handles all text extraction and entity recognition for resumes (PDF / DOCX)
and job descriptions (plain text).

Pipeline:
  1. Extract raw text  (PyMuPDF for PDF, python-docx for DOCX)
  2. SpaCy NER         – person name
  3. Regex patterns    – email, phone, education qualifications
  4. Keyword matching  – skills from a predefined taxonomy
─────────────────────────────────────────────────────────────────────────────
"""

import re
import fitz                  # PyMuPDF  —  pip install pymupdf
import docx                  # python-docx
import spacy

# Load SpaCy small English model (run: python -m spacy download en_core_web_sm)
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   check=True)
    _nlp = spacy.load("en_core_web_sm")


# ─────────────────────────────────────────────────────────────────────────────
# Skill Taxonomy
# Add / remove skills to fit your target domain.
# Keys are category labels; values are canonical skill strings (lowercase).
# ─────────────────────────────────────────────────────────────────────────────
SKILLS_TAXONOMY: dict[str, list[str]] = {
    "programming_languages": [
        "python", "java", "c++", "c#", "javascript", "typescript",
        "r", "scala", "kotlin", "swift", "go", "rust", "php", "ruby",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "neural network",
        "natural language processing", "nlp", "computer vision",
        "reinforcement learning", "bert", "gpt", "transformers",
        "transfer learning", "time series",
    ],
    "ml_libraries": [
        "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost",
        "lightgbm", "catboost", "huggingface", "fastai", "spacy", "nltk",
    ],
    "data_engineering": [
        "pandas", "numpy", "spark", "hadoop", "kafka", "airflow",
        "dbt", "etl", "data pipeline",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis",
        "elasticsearch", "sqlite", "oracle",
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        "ci/cd", "jenkins", "github actions", "linux",
    ],
    "web_frameworks": [
        "flask", "django", "fastapi", "react", "angular", "vue",
        "node.js", "express", "spring boot",
    ],
    "visualization": [
        "matplotlib", "seaborn", "plotly", "tableau", "power bi",
        "d3.js",
    ],
    "soft_skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "project management", "agile", "scrum",
    ],
}

# Flat list for O(1) membership test and easy iteration
ALL_SKILLS: list[str] = [
    skill for skills in SKILLS_TAXONOMY.values() for skill in skills
]


# ─────────────────────────────────────────────────────────────────────────────
# Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    Concatenates text from every page with newline separators.
    """
    text_parts: list[str] = []
    with fitz.open(file_path) as pdf_doc:
        for page in pdf_doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract all paragraph text from a DOCX file using python-docx.
    """
    document = docx.Document(file_path)
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs).strip()


def extract_text(file_path: str) -> str:
    """
    Auto-detect file format (PDF / DOCX) and extract raw text.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    if ext == "docx":
        return extract_text_from_docx(file_path)
    raise ValueError(f"Unsupported file type: .{ext}  (only PDF and DOCX are accepted)")


# ─────────────────────────────────────────────────────────────────────────────
# Entity Extraction Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_name(text: str) -> str:
    """
    Use SpaCy NER to find the most likely candidate name (PERSON entity).
    Falls back to an empty string if none is found.
    Only the first 3 000 characters are processed for speed.
    """
    doc = _nlp(text[:3_000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""


def extract_email(text: str) -> str:
    """Return the first email address found in text, or empty string."""
    match = re.search(r"[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else ""


def extract_phone(text: str) -> str:
    """
    Return the first phone number found in text.
    Supports formats: +91-XXXXXX, (XXX) XXX-XXXX, XXXXXXXXXX, etc.
    """
    pattern = r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}"
    match = re.search(pattern, text)
    return match.group().strip() if match else ""


def extract_skills(text: str) -> list[str]:
    """
    Match skills from ALL_SKILLS against the document text (case-insensitive).
    Returns a deduplicated, sorted list of matched skills.
    """
    text_lower = text.lower()
    found = {skill for skill in ALL_SKILLS if skill in text_lower}
    return sorted(found)


def extract_education(text: str) -> list[str]:
    """
    Identify education qualification mentions using regex patterns.
    Returns a deduplicated list.
    """
    patterns = [
        r"\b(B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?|MCA|BCA|B\.?Sc|M\.?Sc|"
        r"PhD|Ph\.?D|MBA|BBA|B\.?Com|M\.?Com)\b",
        r"\b(Bachelor(?:\'s)?|Master(?:\'s)?|Doctorate|Diploma|Associate)\b",
    ]
    qualifications: list[str] = []
    for pattern in patterns:
        qualifications.extend(re.findall(pattern, text, re.IGNORECASE))
    return sorted(set(q.strip() for q in qualifications))


def extract_years_of_experience(text: str) -> float:
    """
    Heuristic: scan for 'X year(s) of experience' patterns and return
    the maximum value found. Returns 0.0 if nothing is found.
    """
    pattern = r"(\d+(?:\.\d+)?)\s*\+?\s*year[s]?\s*(?:of)?\s*experience"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return max(float(m) for m in matches)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_resume(file_path: str) -> dict:
    """
    Full resume parsing pipeline.

    Returns a structured dictionary containing:
        raw_text     – complete extracted text
        name         – candidate name (SpaCy NER)
        email        – email address
        phone        – phone number
        skills       – matched skills list
        education    – qualification mentions
        experience   – estimated years of experience
    """
    raw_text = extract_text(file_path)
    return {
        "raw_text"   : raw_text,
        "name"       : extract_name(raw_text),
        "email"      : extract_email(raw_text),
        "phone"      : extract_phone(raw_text),
        "skills"     : extract_skills(raw_text),
        "education"  : extract_education(raw_text),
        "experience" : extract_years_of_experience(raw_text),
    }


def parse_job_description(jd_text: str) -> dict:
    """
    Parse a job description string.

    Returns:
        raw_text         – original JD text
        required_skills  – skills mentioned in the JD
        education_req    – required qualification mentions
    """
    return {
        "raw_text"        : jd_text,
        "required_skills" : extract_skills(jd_text),
        "education_req"   : extract_education(jd_text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    John Doe  |  john.doe@email.com  |  +91-9876543210
    B.Tech Computer Science, 2023

    Skills: Python, Machine Learning, Flask, SQL, Docker, NLP
    3 years of experience in data science and ML engineering.
    """
    result = parse_job_description(sample)
    print("Skills found:", result["required_skills"])
    print("Education   :", result["education_req"])
