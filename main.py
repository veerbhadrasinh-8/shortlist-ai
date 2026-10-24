"""
main.py
-------
Entry point for the AI Resume Screening System.

Usage examples
--------------
# 1. Quick demo with built-in synthetic data:
    python main.py --demo

# 2. Screen PDFs in a folder against a JD text file:
    python main.py --resume-dir data/resumes/ --jd-file data/job_description.txt

# 3. Use the Kaggle CSV dataset:
    python main.py --kaggle-csv data/UpdatedResumeDataSet.csv --jd "Data Scientist NLP Python"

# 4. Advanced: use BERT embeddings instead of TF-IDF:
    python main.py --demo --method bert
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import List, Tuple

import pandas as pd

from preprocessing      import preprocess_resume, preprocess_job_description
from feature_extraction import get_extractor
from similarity         import rank_resumes, precision_recall_f1
from visualization      import generate_all_plots


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demo data  (used when --demo flag is passed)
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_JD = """
We are looking for a Data Scientist with strong Python skills.
Requirements:
- 3+ years experience in machine learning and statistical modelling
- Proficient in Python, scikit-learn, pandas, NumPy
- Experience with NLP, TF-IDF, word embeddings, BERT
- Familiarity with SQL and data pipelines
- Good communication and presentation skills
- Experience with deep learning frameworks (TensorFlow or PyTorch) is a plus
"""

_DEMO_RESUMES: List[Tuple[str, str]] = [
    ("Alice_Chen",
     "Data Scientist with 5 years experience in Python machine learning NLP. "
     "Built TF-IDF and BERT text classifiers at scale. Expert in scikit-learn, "
     "pandas, TensorFlow. MS in Computer Science."),

    ("Bob_Kumar",
     "Software Engineer skilled in Java and JavaScript. Some Python experience. "
     "Worked on REST APIs, microservices, CI/CD. No machine learning background."),

    ("Carol_Smith",
     "Statistician with PhD. R, SAS, SPSS. Hypothesis testing, regression, "
     "time-series. Limited Python but strong SQL, data pipelines, communication."),

    ("David_Lee",
     "Machine Learning Engineer. Python pandas numpy scikit-learn XGBoost. "
     "NLP projects: sentiment analysis named entity recognition. AWS SageMaker."),

    ("Eva_Novak",
     "Junior Data Analyst. Excel Power BI Tableau. Basic Python pandas. "
     "1 year experience dashboarding and reporting. No deep learning."),

    ("Frank_Wang",
     "Deep Learning researcher. PyTorch TensorFlow BERT GPT transformers. "
     "Published NLP papers. Python expert. scikit-learn pandas."),

    ("Grace_Kim",
     "Business Analyst. PowerPoint Excel stakeholder management. "
     "No coding or data science background. Project management PMP."),

    ("Hiro_Tanaka",
     "Data Engineer. Spark Kafka Airflow SQL Python ETL pipelines. "
     "Some sklearn, mainly infrastructure focus, less modelling."),

    ("Isla_Patel",
     "NLP Scientist. Python NLTK spaCy transformers BERT fine-tuning. "
     "Text classification summarisation QA. 4 years experience."),

    ("James_Okafor",
     "Frontend Developer. React TypeScript CSS HTML animations UX design. "
     "Zero data science or Python machine learning experience."),

    ("Karen_Russo",
     "ML Researcher. Python sklearn statsmodels pandas matplotlib seaborn. "
     "Regression classification clustering. Good communicator."),

    ("Leo_Martinez",
     "Data Science Intern. Python numpy pandas matplotlib. Completed Coursera "
     "ML course. Academic projects in logistic regression and k-means."),
]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_from_folder(resume_dir: str) -> Tuple[List[str], List[str]]:
    """Load .pdf or .txt resumes from a directory (including subfolders)."""
    ids, texts = [], []

    for root, dirs, files in os.walk(resume_dir):
        for fname in files:
            fpath = os.path.join(root, fname)

            if fname.lower().endswith(".pdf"):
                print(f"Loading PDF: {fpath}")  # DEBUG LINE
                text = preprocess_resume(fpath)
                ids.append(os.path.splitext(fname)[0])
                texts.append(text)

            elif fname.lower().endswith(".txt"):
                print(f"Loading TXT: {fpath}")  # DEBUG LINE
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    text = preprocess_resume(f.read())
                ids.append(os.path.splitext(fname)[0])
                texts.append(text)

    if not ids:
        print(f"⚠ No .pdf or .txt files found in '{resume_dir}'.")

    return ids, texts


def load_from_kaggle_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load resumes from the Kaggle UpdatedResumeDataSet.csv.

    Expected columns: 'Resume_str' (text) and 'Category' (job role).
    Falls back to first two columns if names differ.
    """
    df = pd.read_csv(csv_path)
    # Detect columns
    if "Resume_str" in df.columns:
        text_col = "Resume_str"
    else:
        text_col = df.columns[1]   # assume 2nd column

    ids   = [f"resume_{i+1:04d}" for i in range(len(df))]
    texts = [preprocess_resume(str(t)) for t in df[text_col].fillna("")]
    return ids, texts


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    jd_raw: str,
    resume_ids: List[str],
    resume_raw_texts: List[str],
    method: str = "tfidf",
    out_dir: str = "report",
    threshold: float = 0.0,
    relevant_ids: List[str] | None = None,
) -> pd.DataFrame:
    """Full pipeline: preprocess → vectorise → rank → visualise."""

    print(f"\n{'='*60}")
    print("  AI Resume Screening System")
    print(f"{'='*60}")
    print(f"  Method       : {method.upper()}")
    print(f"  Resumes      : {len(resume_ids)}")
    print(f"  Output dir   : {out_dir}/")
    print(f"{'='*60}\n")

    # 1. Preprocess
    print("🧹  Preprocessing …")
    jd_clean = preprocess_job_description(jd_raw)
    resume_cleaned = [preprocess_resume(t) for t in resume_raw_texts]
    print(f"    JD tokens : {len(jd_clean.split())}")
    print(f"    Avg resume tokens : "
          f"{sum(len(t.split()) for t in resume_cleaned) // max(len(resume_cleaned),1)}")

    # 2. Feature extraction
    print(f"\n🧠  Extracting features ({method.upper()}) …")
    extractor = get_extractor(method)
    jd_vec, resume_vecs = extractor.fit_transform(jd_clean, resume_cleaned)

    # 3. Rank
    print("\n⚙️   Computing cosine similarity & ranking …")
    df = rank_resumes(jd_vec, resume_vecs, resume_ids, threshold=threshold)

    # 4. Print results
    print("\n🏆  RANKING RESULTS")
    print("-" * 55)
    print(f"{'Rank':<6} {'Resume ID':<25} {'Score':<8} {'Label'}")
    print("-" * 55)
    for _, row in df.head(20).iterrows():
        print(f"{int(row['rank']):<6} {row['resume_id']:<25} "
              f"{row['score']:<8.4f} {row['label']}")
    if len(df) > 20:
        print(f"  … and {len(df)-20} more.")
    print("-" * 55)

    # 5. Stats
    print(f"\n📊  Score statistics")
    print(f"    Mean   : {df['score'].mean():.4f}")
    print(f"    Median : {df['score'].median():.4f}")
    print(f"    Max    : {df['score'].max():.4f}")
    print(f"    Min    : {df['score'].min():.4f}")

    # 6. Evaluation (if labels provided)
    if relevant_ids:
        metrics = precision_recall_f1(df, relevant_ids)
        print(f"\n🧪  Evaluation (vs ground-truth {len(relevant_ids)} relevant)")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1-score  : {metrics['f1']:.4f}")

    # 7. Top keywords
    if method == "tfidf" and hasattr(extractor, "top_jd_keywords"):
        kw = extractor.top_jd_keywords(jd_vec, n=15)
        print(f"\n🔑  Top JD keywords : {', '.join(kw)}")

    # 8. Visualisations
    generate_all_plots(df, jd_clean, out_dir=out_dir)

    # 9. Save CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ranked_resumes.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾  Results saved → {csv_path}")
    print(f"\n✅  Done! Open '{out_dir}/' to view charts and CSV.\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        AI Resume Screening System
        --------------------------
        Ranks resumes by cosine similarity to a job description.
        """),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo",         action="store_true",
                     help="Run with built-in synthetic data")
    src.add_argument("--resume-dir",   metavar="DIR",
                     help="Folder containing .pdf or .txt resumes")
    src.add_argument("--kaggle-csv",   metavar="CSV",
                     help="Path to Kaggle UpdatedResumeDataSet.csv")

    p.add_argument("--jd-file",  metavar="FILE",
                   help="Path to plain-text job description file")
    p.add_argument("--jd",       metavar="TEXT",
                   help="Job description as an inline string")
    p.add_argument("--method",   choices=["tfidf", "bert"], default="tfidf",
                   help="Feature extraction method (default: tfidf)")
    p.add_argument("--out-dir",  default="report",
                   help="Output directory for charts and CSV (default: report/)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Minimum score threshold to include a resume (default: 0)")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Determine JD ─────────────────────────────────────────────────────────
    if args.demo:
        jd_raw       = _DEMO_JD
        resume_ids   = [r[0] for r in _DEMO_RESUMES]
        resume_texts = [r[1] for r in _DEMO_RESUMES]
        relevant_ids = ["Alice_Chen", "David_Lee", "Frank_Wang", "Isla_Patel",
                        "Karen_Russo"]
    else:
        # Load JD
        if args.jd_file:
            with open(args.jd_file, encoding="utf-8") as f:
                jd_raw = f.read()
        elif args.jd:
            jd_raw = args.jd
        else:
            parser.error(
                "Provide a job description with --jd-file, --jd, or use --demo."
            )

        # Load resumes
        if args.resume_dir:
            resume_ids, resume_texts = load_from_folder(args.resume_dir)
        elif args.kaggle_csv:
            resume_ids, resume_texts = load_from_kaggle_csv(args.kaggle_csv)
        else:
            parser.error(
                "Provide resume source with --resume-dir, --kaggle-csv, or use --demo."
            )

        relevant_ids = None

    if not resume_ids:
        print("No resumes loaded. Exiting.")
        sys.exit(1)

    run_pipeline(
        jd_raw        = jd_raw,
        resume_ids    = resume_ids,
        resume_raw_texts = resume_texts,
        method        = args.method,
        out_dir       = args.out_dir,
        threshold     = args.threshold,
        relevant_ids  = relevant_ids,
    )


if __name__ == "__main__":
    main()
