"""
skills_extractor.py
-------------------
Extracts technical and professional skills from text using taxonomy matching
and regex patterns with word boundaries. Performs skill gap analysis
between Job Descriptions and Resumes.
"""

from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple


# Comprehensive skill taxonomy across modern tech, data, software, and AI roles
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "c", "javascript", "typescript", "r",
        "sql", "go", "rust", "scala", "ruby", "php", "swift", "kotlin",
        "bash", "shell", "powershell", "perl", "dart", "html", "css", "matlab"
    ],
    "Machine Learning & Data Science": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "llm", "llms", "large language models", "generative ai",
        "genai", "transformers", "bert", "gpt", "rag", "retrieval augmented generation",
        "fine-tuning", "reinforcement learning", "recommendation systems",
        "time series", "classification", "clustering", "regression",
        "feature engineering", "a/b testing", "statistics", "statistical modeling"
    ],
    "Frameworks & Libraries": [
        "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas",
        "numpy", "scipy", "xgboost", "lightgbm", "catboost", "hugging face",
        "langchain", "llamaindex", "spacy", "nltk", "opencv", "fastapi",
        "flask", "django", "react", "next.js", "vue", "node.js", "express",
        "spring boot", "tailwind"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
        "kubernetes", "k8s", "ci/cd", "git", "github", "gitlab", "terraform",
        "ansible", "jenkins", "linux", "mlflow", "kubeflow", "wandb", "airflow"
    ],
    "Data Engineering & Databases": [
        "sql", "postgresql", "postgres", "mysql", "mongodb", "redis",
        "cassandra", "elasticsearch", "spark", "pyspark", "kafka", "hadoop",
        "hive", "snowflake", "bigquery", "redshift", "dbt", "databricks",
        "etl", "data pipelines", "pinecone", "chromadb", "milvus", "weaviate", "qdrant"
    ],
    "BI & Analytics Tools": [
        "power bi", "tableau", "excel", "looker", "metabase", "qlik",
        "google analytics", "data visualization", "matplotlib", "seaborn", "plotly"
    ]
}

# Inverted index: normalized term -> canonical display name
CANONICAL_SKILLS: Dict[str, str] = {
    "c++": "C++",
    "c#": "C#",
    "ci/cd": "CI/CD",
    "k8s": "Kubernetes",
    "llm": "LLMs",
    "llms": "LLMs",
    "nlp": "NLP",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "rag": "RAG",
    "bert": "BERT",
    "gpt": "GPT",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "sql": "SQL",
    "etl": "ETL",
    "power bi": "Power BI",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "hugging face": "Hugging Face",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "next.js": "Next.js",
    "node.js": "Node.js",
    "vue": "Vue.js",
    "react": "React",
    "fastapi": "FastAPI",
    "bigquery": "BigQuery",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "spark": "Apache Spark",
    "pyspark": "PySpark",
    "kafka": "Apache Kafka",
    "airflow": "Apache Airflow"
}


def _compile_skill_patterns() -> List[Tuple[re.Pattern, str, str]]:
    """Compile regex patterns for robust matching with word boundaries."""
    patterns = []
    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            escaped = re.escape(skill)
            # Handle special characters like c++, c#, .js without breaking \b boundary
            if skill.endswith("+") or skill.endswith("#"):
                pattern = rf"(?i)(?:\b|(?<=\s)){escaped}(?=\s|[.,;!?]|$)"
            elif skill.startswith("."):
                pattern = rf"(?i)(?<=\s){escaped}\b"
            else:
                pattern = rf"(?i)\b{escaped}\b"
            
            canonical = CANONICAL_SKILLS.get(skill.lower(), skill.title())
            patterns.append((re.compile(pattern), canonical, category))
    return patterns


_COMPILED_PATTERNS = _compile_skill_patterns()


def extract_skills(text: str) -> Dict[str, Set[str]]:
    """Extract recognized skills from text, categorized.
    
    Returns
    -------
    dict: {category: set_of_canonical_skill_names}
    """
    if not text:
        return {}

    extracted_by_cat: Dict[str, Set[str]] = {}
    found_skills: Set[str] = set()

    for pattern, canonical, category in _COMPILED_PATTERNS:
        if canonical not in found_skills and pattern.search(text):
            found_skills.add(canonical)
            extracted_by_cat.setdefault(category, set()).add(canonical)

    return extracted_by_cat


def extract_skills_flat(text: str) -> Set[str]:
    """Return all unique canonical skills found in text as a flat set."""
    by_cat = extract_skills(text)
    flat: Set[str] = set()
    for s_set in by_cat.values():
        flat.update(s_set)
    return flat


def analyze_skill_gap(jd_text: str, resume_text: str) -> Dict[str, any]:
    """Perform a skill gap analysis between a job description and a resume.
    
    Returns
    -------
    dict with:
      - jd_skills: list of skills required by JD
      - resume_skills: list of skills found in Resume
      - matched_skills: list of skills present in both
      - missing_skills: list of required skills absent from Resume
      - extra_skills: list of candidate skills not requested in JD
      - match_percentage: float (0.0 to 100.0)
    """
    jd_skills = extract_skills_flat(jd_text)
    resume_skills = extract_skills_flat(resume_text)

    matched = sorted(list(jd_skills & resume_skills))
    missing = sorted(list(jd_skills - resume_skills))
    extra = sorted(list(resume_skills - jd_skills))

    total_jd = len(jd_skills)
    match_pct = round((len(matched) / total_jd * 100), 1) if total_jd > 0 else 100.0

    return {
        "jd_skills": sorted(list(jd_skills)),
        "resume_skills": sorted(list(resume_skills)),
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "match_percentage": match_pct,
        "total_required": total_jd,
        "total_matched": len(matched)
    }
