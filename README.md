# Multi-Modal Explainable AI Framework
## Intelligent Resume Screening, Skill Gap Analysis & Job Recommendation System

**Group 41_12 · Section 2241041 · ITER, CSE · SOA University · 2026**
**Supervisor: Dr. Prativa Das**
**Team Lead: Piyush Kumar Nayak (2241014099)**

---

## 📁 Project Structure

```
project/
├── app.py                  # Flask application factory (entry point)
├── routes.py               # All Flask routes and API endpoints
├── parser.py               # Resume/JD text extraction (PDF + DOCX)
├── preprocessing.py        # NLP cleaning pipeline
├── features.py             # TF-IDF + BERT feature extraction
├── model_training.py       # ML training pipeline (run once)
├── skill_gap.py            # Skill gap analysis + job recommendation
├── xai.py                  # SHAP explainability module
├── config.py               # Centralised configuration
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Main web UI
├── static/
│   ├── css/                # Stylesheets (optional overrides)
│   ├── js/                 # Client-side scripts
│   └── img/shap/           # SHAP plot outputs
├── models/                 # Saved model artefacts (auto-created)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── comparison_table.csv
└── dataset/
    ├── sample_dataset.csv          # Included sample (5 rows)
    └── UpdatedResumeDataSet.csv    # Download from Kaggle (2400+ rows)
```

---

## ⚡ Quick Start (Step-by-Step)

### Step 1 — Clone / download the project
```bash
cd Desktop
mkdir resume_screening && cd resume_screening
# Place all project files here
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download SpaCy model
```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Download the Kaggle dataset
1. Go to: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
2. Download `UpdatedResumeDataSet.csv`
3. Place it in the `dataset/` folder

### Step 6 — Train the models (run once)
```bash
python model_training.py
```
This will:
- Load and preprocess 2,400+ resumes
- Train Logistic Regression and Random Forest classifiers
- Print evaluation metrics (Accuracy, Precision, Recall, F1)
- Save all models to `models/`
- Print a comparison table

**Expected output:**
```
Model                   Accuracy  Precision  Recall  F1-Score  CV F1 (mean)
Logistic Regression     0.9521    0.9534     0.9521  0.9519    0.9487
Random Forest           0.9687    0.9701     0.9687  0.9689    0.9651
```

### Step 7 — Start the Flask web app
```bash
python app.py
```
Open your browser: **http://localhost:5000**

---

## 🖥️ Using the Web Application

1. **Upload** a resume PDF or DOCX file
2. **Paste** the job description in the text area
3. Click **"Analyse Resume"**
4. View results:
   - Candidate name, email, extracted skills
   - ML classification category and confidence %
   - TF-IDF / BERT / Hybrid similarity scores
   - Skill match percentage with matched/missing/extra skills
   - Top job role recommendations
   - Learning resources for missing skills
   - SHAP feature importance waterfall chart

---

## 🧪 API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | Landing page                       |
| GET    | `/demo`           | Pre-loaded demo results            |
| POST   | `/api/screen`     | Main screening endpoint (JSON)     |
| GET    | `/api/health`     | Health check / model status        |
| GET    | `/api/roles`      | List all job role profiles         |

**POST `/api/screen` — example using curl:**
```bash
curl -X POST http://localhost:5000/api/screen \
  -F "resume=@/path/to/resume.pdf" \
  -F "job_description=Looking for Python ML engineer with Docker and AWS experience"
```

---

## 🏗️ System Architecture

```
Input Layer          Processing Layer                    Output Layer
─────────────        ──────────────────────────────      ─────────────────────
Resume (PDF/DOCX) →  1. Text Extraction (PyMuPDF)    →   Candidate Score/Rank
Job Description   →  2. NLP Preprocessing            →   Skill Gap Report
                      3. TF-IDF Vectorization         →   Job Recommendations
                      4. BERT Embeddings (384-dim)    →   SHAP Explanation Plot
                      5. RF / LR Classification       →   Category + Confidence
                      6. Cosine Similarity (Hybrid)   →
                      7. Skill Gap Analysis            →
                      8. SHAP Explainability           →
```

---

## 📊 Model Evaluation Metrics

| Metric    | Description                                        | Target  |
|-----------|----------------------------------------------------|---------|
| Accuracy  | Correct predictions / total samples                | > 85%   |
| Precision | TP / (TP + FP) — weighted across all classes       | > 82%   |
| Recall    | TP / (TP + FN) — weighted across all classes       | > 82%   |
| F1-Score  | Harmonic mean of Precision and Recall              | > 83%   |
| CV F1     | 5-fold cross-validation weighted F1                | > 82%   |

---
## 🔧 Author
Piyush Kumar Nayak
B.Tech Computer Science & Engineering
Institute of Technical Education and Research (ITER)
Siksha 'O' Anusandhan University
Project Supervisor: Dr. Prativa Das

---

## Contributions

- Designed and developed the complete Resume Screening Framework
- Implemented TF-IDF and BERT based feature extraction
- Trained and evaluated Machine Learning models
- Integrated SHAP Explainable AI visualizations
- Developed Skill Gap Analysis and Job Recommendation modules
- Built the Flask web application and user interface
- Prepared project documentation and deployment workflow

---

## 🔧 Configuration

Edit `config.py` to change:
- **`USE_BERT = False`** — disable BERT for faster testing on CPU
- **`TFIDF_MAX_FEATURES`** — vocabulary size (default 5000)
- **`RF_N_ESTIMATORS`** — Random Forest trees (default 200)
- **`BERT_MODEL_NAME`** — swap for a larger/smaller model

---

## 📦 Dataset

- **Kaggle Resume Dataset**: 2,484 resumes across 25 job categories
- URL: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- Columns: `Resume` (text), `Category` (label)
- Sample included at `dataset/sample_dataset.csv`

---

*Built with Python · Flask · Scikit-learn · HuggingFace Transformers · SHAP*
