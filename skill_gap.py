"""
skill_gap.py
─────────────────────────────────────────────────────────────────────────────
Skill Gap Analysis and Job Recommendation module.

Responsibilities:
  1. Compare resume skills vs job-description requirements
  2. Return matched, missing, and extra skills with match percentage
  3. Map each missing skill to a curated learning resource
  4. Recommend top job roles based on the candidate's skill profile
─────────────────────────────────────────────────────────────────────────────
"""

from typing import List, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Learning Resources Map
# {skill (lowercase): (resource_name, url)}
# ─────────────────────────────────────────────────────────────────────────────
COURSE_MAP: Dict[str, Tuple[str, str]] = {
    "python"               : ("Python for Everybody – Coursera",
                               "https://www.coursera.org/specializations/python"),
    "machine learning"     : ("ML Specialization – Andrew Ng (Coursera)",
                               "https://www.coursera.org/specializations/machine-learning-introduction"),
    "deep learning"        : ("Deep Learning Specialization – deeplearning.ai",
                               "https://www.coursera.org/specializations/deep-learning"),
    "bert"                 : ("HuggingFace NLP Course (free)",
                               "https://huggingface.co/learn/nlp-course"),
    "transformers"         : ("HuggingFace NLP Course (free)",
                               "https://huggingface.co/learn/nlp-course"),
    "natural language processing": ("NLP with Python – Udemy",
                               "https://www.udemy.com/course/nlp-natural-language-processing-with-python/"),
    "nlp"                  : ("NLP with Python – Udemy",
                               "https://www.udemy.com/course/nlp-natural-language-processing-with-python/"),
    "computer vision"      : ("CS231n – Stanford (free)",
                               "http://cs231n.stanford.edu/"),
    "tensorflow"           : ("TensorFlow Developer Certificate – Google",
                               "https://www.tensorflow.org/certificate"),
    "pytorch"              : ("PyTorch for Deep Learning – Zero to Mastery",
                               "https://www.zerotomastery.io/courses/learn-pytorch/"),
    "scikit-learn"         : ("Scikit-learn Official Documentation",
                               "https://scikit-learn.org/stable/user_guide.html"),
    "sql"                  : ("SQL for Data Science – Coursera",
                               "https://www.coursera.org/learn/sql-for-data-science"),
    "docker"               : ("Docker Mastery – Udemy",
                               "https://www.udemy.com/course/docker-mastery/"),
    "kubernetes"           : ("Kubernetes for Beginners – KodeKloud",
                               "https://kodekloud.com/courses/kubernetes-for-the-absolute-beginners-hands-on/"),
    "aws"                  : ("AWS Cloud Practitioner – AWS Training (free)",
                               "https://aws.amazon.com/training/digital/aws-cloud-practitioner-essentials/"),
    "git"                  : ("Git & GitHub Crash Course – freeCodeCamp",
                               "https://www.youtube.com/watch?v=RGOj5yH7evk"),
    "flask"                : ("Flask Mega-Tutorial – Miguel Grinberg",
                               "https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world"),
    "django"               : ("Django Official Tutorial",
                               "https://docs.djangoproject.com/en/stable/intro/tutorial01/"),
    "pandas"               : ("Pandas Documentation",
                               "https://pandas.pydata.org/docs/getting_started/index.html"),
    "numpy"                : ("NumPy Official Documentation",
                               "https://numpy.org/doc/stable/user/absolute_beginners.html"),
    "spark"                : ("Apache Spark with Python – Udemy",
                               "https://www.udemy.com/course/spark-and-python-for-big-data-with-pyspark/"),
    "agile"                : ("Agile Fundamentals – Coursera",
                               "https://www.coursera.org/learn/agile-development"),
    "scrum"                : ("Professional Scrum Master I – Scrum.org",
                               "https://www.scrum.org/assessments/professional-scrum-master-i-certification"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Job Role Profiles
# Maps job title → required skills (used for recommendation)
# ─────────────────────────────────────────────────────────────────────────────
JOB_ROLE_PROFILES: Dict[str, List[str]] = {
    "Data Scientist": [
        "python", "machine learning", "deep learning", "pandas", "numpy",
        "scikit-learn", "sql", "matplotlib", "statistics",
    ],
    "ML Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "docker", "kubernetes", "aws", "flask", "git",
    ],
    "NLP Engineer": [
        "python", "nlp", "bert", "transformers", "spacy", "nltk",
        "machine learning", "deep learning", "huggingface",
    ],
    "Data Analyst": [
        "python", "sql", "pandas", "numpy", "matplotlib", "seaborn",
        "tableau", "power bi", "excel",
    ],
    "Backend Developer": [
        "python", "flask", "django", "sql", "postgresql", "redis",
        "docker", "git", "linux", "aws",
    ],
    "Full-Stack Developer": [
        "python", "javascript", "react", "flask", "sql", "docker",
        "git", "aws", "html", "css",
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "terraform", "linux",
        "ci/cd", "git", "jenkins", "github actions",
    ],
    "Computer Vision Engineer": [
        "python", "computer vision", "deep learning", "pytorch", "tensorflow",
        "numpy", "scikit-learn", "git",
    ],
    "Research Scientist": [
        "python", "machine learning", "deep learning", "bert", "transformers",
        "pytorch", "numpy", "pandas", "statistics",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Analysis Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_skill_gap(
    resume_skills   : List[str],
    required_skills : List[str],
) -> Dict:
    """
    Compare candidate's skills against job requirements.

    Args:
        resume_skills   : Skills extracted from the candidate's resume.
        required_skills : Skills required by the target job description.

    Returns a structured dict:
        match_percentage  – float [0, 100]
        matched_skills    – skills present in both lists
        missing_skills    – required skills absent from resume
        extra_skills      – skills the candidate has beyond what's required
        recommendations   – {skill: {name, url}} for each missing skill
        total_required    – int
        total_resume      – int
    """
    resume_set   = {s.lower().strip() for s in resume_skills}
    required_set = {s.lower().strip() for s in required_skills}

    matched = sorted(resume_set & required_set)
    missing = sorted(required_set - resume_set)
    extra   = sorted(resume_set - required_set)

    match_pct = (len(matched) / len(required_set) * 100.0
                 if required_set else 0.0)

    # Build recommendations for missing skills
    recommendations: Dict[str, Dict[str, str]] = {}
    for skill in missing:
        if skill in COURSE_MAP:
            name, url = COURSE_MAP[skill]
        else:
            name = f'Search "{skill}" on Coursera / Udemy / YouTube'
            url  = f"https://www.google.com/search?q={skill.replace(' ','+%20')}+tutorial"
        recommendations[skill] = {"name": name, "url": url}

    return {
        "match_percentage" : round(match_pct, 2),
        "matched_skills"   : matched,
        "missing_skills"   : missing,
        "extra_skills"     : extra,
        "recommendations"  : recommendations,
        "total_required"   : len(required_set),
        "total_resume"     : len(resume_set),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Job Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def recommend_jobs(
    resume_skills : List[str],
    top_n         : int = 5,
) -> List[Dict]:
    """
    Rank job roles by skill overlap with the candidate's resume.

    Each result contains:
        role            – job title
        match_score     – float [0, 1] Jaccard similarity
        match_pct       – percentage (0–100)
        matched_skills  – skills the candidate already has for this role
        gap_skills      – skills the candidate is missing for this role

    Returns a sorted list (best match first) of top_n roles.
    """
    resume_set = {s.lower().strip() for s in resume_skills}
    scored: List[Dict] = []

    for role, required in JOB_ROLE_PROFILES.items():
        req_set     = set(required)
        matched     = resume_set & req_set
        union       = resume_set | req_set
        jaccard     = len(matched) / len(union) if union else 0.0
        match_pct   = (len(matched) / len(req_set) * 100) if req_set else 0.0

        scored.append({
            "role"          : role,
            "match_score"   : round(jaccard, 4),
            "match_pct"     : round(match_pct, 2),
            "matched_skills": sorted(matched),
            "gap_skills"    : sorted(req_set - resume_set),
        })

    # Sort descending by Jaccard score, then alphabetically for ties
    scored.sort(key=lambda x: (-x["match_score"], x["role"]))
    return scored[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    my_skills  = ["python", "machine learning", "flask", "pandas", "git", "sql"]
    jd_skills  = ["python", "machine learning", "bert", "docker", "aws", "sql"]

    print("=== Skill Gap Analysis ===")
    gap = analyze_skill_gap(my_skills, jd_skills)
    print(f"Match  : {gap['match_percentage']}%")
    print(f"Matched: {gap['matched_skills']}")
    print(f"Missing: {gap['missing_skills']}")
    print(f"Extra  : {gap['extra_skills']}")
    print("\nRecommendations:")
    for skill, res in gap["recommendations"].items():
        print(f"  {skill:30s} → {res['name']}")

    print("\n=== Job Recommendations ===")
    recs = recommend_jobs(my_skills, top_n=3)
    for r in recs:
        print(f"  {r['role']:30s}  {r['match_pct']:5.1f}%  gap={r['gap_skills']}")
