# ShortlistAI — Candidate Screening & Shortlisting Platform

An end-to-end NLP screening system that automatically matches and ranks resumes against job descriptions using a **Hybrid Engine (TF-IDF + Sentence-BERT + Skill Ontology)** with an interactive **Streamlit Web Dashboard**.

---

## 🌟 Key Features

- **Interactive Streamlit UI**: Drag-and-drop multiple PDF / TXT resumes or run with 12 built-in candidate profiles.
- **Hybrid Scoring Architecture**:
  - **Lexical Matching**: TF-IDF n-gram vectorizer with sublinear term-frequency regularisation.
  - **Dense Semantic Embeddings**: Sentence-BERT (`all-MiniLM-L6-v2`) capturing context and meaning.
  - **Skill Ontology Matching**: Domain-specific skill extraction across Languages, ML/AI, Cloud/DevOps, Databases, and BI tools.
- **Explainable Skill Gap Analysis**: Dynamically surfaces **Matched Competencies** (green badges) and **Missing Critical Skills** (red badges) per candidate.
- **Fast In-Memory Extraction**: Multi-threaded, stream-based PDF extraction via PyMuPDF.
- **Visual Analytics**: Interactive leaderboards, rating distribution donuts, candidate score comparisons, and CSV export.

---

## 📁 Project Structure

```
AI Project/
│
├── app.py                   ← Streamlit interactive web application
├── skills_extractor.py      ← Skill taxonomy extraction & gap analysis engine
├── preprocessing.py         ← PDF stream/file extraction, NLTK text normalization
├── feature_extraction.py    ← TF-IDF vectorizer + Sentence-BERT embeddings
├── similarity.py            ← Cosine similarity, hybrid fusion, evaluation metrics
├── visualization.py         ← Matplotlib, Seaborn & WordCloud chart generation
├── main.py                  ← CLI pipeline orchestrator
├── requirements.txt         ← Project dependencies
└── report/                  ← Auto-created output directory for charts & CSV
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### 3. Run CLI Pipeline (Alternative)
```bash
# Built-in demo (12 synthetic resumes)
python main.py --demo

# BERT embeddings via CLI
python main.py --demo --method bert

# Screen PDF folder against a JD file
python main.py --resume-dir data/resumes/ --jd-file data/jd.txt
```

---

## 🧠 Scoring Formula

The Hybrid Engine calculates candidate relevance using weighted multi-signal fusion:

$$\text{Final Score} = w_{\text{tfidf}} \times S_{\text{tfidf}} + w_{\text{bert}} \times S_{\text{bert}} + w_{\text{skills}} \times S_{\text{skills}}$$

| Score Range | Classification | Description |
|---|---|---|
| **≥ 0.60** | **Excellent** | Exceptional skill overlap and contextual alignment |
| **0.40 – 0.59** | **Good** | Strong candidate with minor skill gaps |
| **0.20 – 0.39** | **Fair** | Partial match; requires upskilling or role realignment |
| **< 0.20** | **Poor** | Unrelated background or missing prerequisite domains |

---

## 💼 Resume Description Template

When listing this project on your resume, use this bullet-point template:

> **ShortlistAI — Candidate Screening & Shortlisting Platform**  
> *Python, Streamlit, Scikit-Learn, Sentence-BERT, PyMuPDF, NLTK, Matplotlib*  
> • Built an end-to-end NLP candidate screening engine supporting PDF & TXT bulk extraction via PyMuPDF with sub-second per-resume processing.  
> • Implemented a hybrid ranking pipeline combining TF-IDF lexical frequency with Sentence-BERT (`all-MiniLM-L6-v2`) dense embeddings for context-aware ranking.  
> • Engineered an explainable Skill Gap Analysis engine mapped across tech stacks (languages, cloud, ML, databases) to identify critical missing competencies.  
> • Designed a reactive Streamlit web dashboard featuring real-time ranking tables, distribution analytics, and instant CSV reporting.
