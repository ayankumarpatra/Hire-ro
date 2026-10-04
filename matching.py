"""Local, explainable resume-to-job matching for HireLens AI.

No generative AI is used here. The score is a heuristic based on TF-IDF text
similarity, skill coverage, and basic evidence sections in the resume.
It is intended as a screening aid, not a prediction of hiring success.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Common technical skills/keywords. This is deliberately small and transparent;
# mandatory skills supplied by the recruiter are always included separately.
KNOWN_SKILLS = [
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "sql", "html", "css", "flask", "django", "fastapi", "spring", "react", "angular",
    "vue", "node.js", "express", "git", "github", "docker", "kubernetes", "aws", "azure",
    "gcp", "linux", "mongodb", "mysql", "postgresql", "sqlite", "redis", "firebase",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "machine learning",
    "deep learning", "natural language processing", "nlp", "data analysis", "data science",
    "rest api", "restful api", "api", "oop", "object oriented programming", "dsa",
    "data structures", "algorithms", "gitlab", "ci/cd", "power bi", "tableau",
]

_ALIAS = {
    "node js": "node.js",
    "nodejs": "node.js",
    "react js": "react",
    "reactjs": "react",
    "angularjs": "angular",
    "postgres": "postgresql",
    "scikit learn": "scikit-learn",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "natural-language-processing": "natural language processing",
    "restful": "rest api",
    "rest api": "rest api",
    "object-oriented programming": "object oriented programming",
    "data structures and algorithms": "data structures",
}


def _normalise(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _skill_pattern(skill):
    escaped = re.escape(_normalise(skill))
    # Allow spaces/hyphens to vary slightly, while avoiding substring matches.
    escaped = escaped.replace(r"\ ", r"[\s-]+")
    return re.compile(r"(?<![a-z0-9+#.])" + escaped + r"(?![a-z0-9+#.])", re.IGNORECASE)


def _contains_skill(text, skill):
    normalised_text = _normalise(text)
    normalised_skill = _normalise(skill)
    canonical = _ALIAS.get(normalised_skill, normalised_skill)
    return bool(_skill_pattern(canonical).search(normalised_text))


def detect_skills(text):
    """Return known skills explicitly mentioned in text."""
    found = []
    for skill in KNOWN_SKILLS:
        if _contains_skill(text, skill):
            found.append(skill)
    return found


def _dedupe_skills(skills):
    result = []
    seen = set()
    for skill in skills:
        canonical = _ALIAS.get(_normalise(skill), _normalise(skill))
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _text_similarity(job_description, resume_text):
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
        matrix = vectorizer.fit_transform([job_description, resume_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0.0


def _evidence_score(info):
    """Small bonus for useful resume evidence sections, capped at 1."""
    evidence = 0
    if getattr(info, "experience", None):
        evidence += 0.40
    if getattr(info, "projects", None):
        evidence += 0.30
    if getattr(info, "skills", None):
        evidence += 0.20
    if getattr(info, "education", None):
        evidence += 0.10
    return min(evidence, 1.0)


def calculate_match(job_description, mandatory_skills, resume_text, info):
    """Calculate one deterministic, explainable profile-fit result."""
    jd_skills = _dedupe_skills(mandatory_skills)
    for skill in detect_skills(job_description):
        if skill not in jd_skills:
            jd_skills.append(skill)

    resume_skills = _dedupe_skills(getattr(info, "skills", []))
    for skill in detect_skills(resume_text):
        if skill not in resume_skills:
            resume_skills.append(skill)

    matched = [skill for skill in jd_skills if _contains_skill(resume_text, skill)]
    missing = [skill for skill in jd_skills if skill not in matched]

    mandatory = _dedupe_skills(mandatory_skills)
    mandatory_matched = [skill for skill in mandatory if _contains_skill(resume_text, skill)]
    mandatory_missing = [skill for skill in mandatory if skill not in mandatory_matched]

    similarity = _text_similarity(job_description, resume_text)
    if jd_skills:
        skill_coverage = len(matched) / len(jd_skills)
    else:
        skill_coverage = 0.0

    evidence = _evidence_score(info)
    score = (similarity * 0.60) + (skill_coverage * 0.30) + (evidence * 0.10)
    score = round(max(0.0, min(1.0, score)) * 100, 1)

    if score >= 70:
        suitability = "Strong match"
    elif score >= 50:
        suitability = "Review"
    else:
        suitability = "Low match"

    return {
        "score": score,
        "suitability": suitability,
        "semantic_score": round(similarity * 100, 1),
        "skill_score": round(skill_coverage * 100, 1),
        "evidence_score": round(evidence * 100, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "mandatory_matched": mandatory_matched,
        "mandatory_missing": mandatory_missing,
        "job_skills": jd_skills,
        "candidate_skills": resume_skills,
        "method": "60% job-description similarity + 30% skill coverage + 10% resume evidence",
    }
