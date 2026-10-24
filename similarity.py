"""
similarity.py
-------------
Computes cosine similarity between the job-description vector
and each resume vector, then returns a ranked DataFrame.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def rank_resumes(
    jd_vec: np.ndarray,
    resume_vecs: np.ndarray,
    resume_ids: List[str],
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute cosine similarity scores and return a ranked DataFrame.

    Parameters
    ----------
    jd_vec       : shape (1, n_features)  – job description vector
    resume_vecs  : shape (n, n_features)  – resume matrix
    resume_ids   : list of length n       – resume identifiers (filenames / IDs)
    threshold    : float                  – keep only scores ≥ threshold

    Returns
    -------
    pd.DataFrame with columns [rank, resume_id, score, label]
        score  : cosine similarity ∈ [0, 1]
        label  : 'Excellent' ≥ 0.6 | 'Good' ≥ 0.4 | 'Fair' ≥ 0.2 | 'Poor'
    """
    # ── Cosine similarity ─────────────────────────────────────────────────────
    # Works for both sparse (TF-IDF) and dense (BERT) matrices
    scores = cosine_similarity(jd_vec, resume_vecs).flatten()

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "resume_id": resume_ids,
        "score":     np.round(scores, 4),
    })

    df = df[df["score"] >= threshold].copy()
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1                        # 1-based rank
    df.index.name = "rank"

    # ── Label ─────────────────────────────────────────────────────────────────
    def _label(s: float) -> str:
        if s >= 0.60:
            return "Excellent"
        if s >= 0.40:
            return "Good"
        if s >= 0.20:
            return "Fair"
        return "Poor"

    df["label"] = df["score"].apply(_label)
    return df.reset_index()   # expose rank as column


def rank_resumes_hybrid(
    tfidf_scores: np.ndarray,
    semantic_scores: np.ndarray | None,
    skill_ratios: List[float],
    resume_ids: List[str],
    w_tfidf: float = 0.35,
    w_semantic: float = 0.45,
    w_skills: float = 0.20,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute weighted hybrid score combining keyword, semantic, and skill-match signals.

    Parameters
    ----------
    tfidf_scores    : 1D array of TF-IDF cosine similarities ∈ [0, 1]
    semantic_scores : 1D array of BERT cosine similarities ∈ [0, 1] (or None)
    skill_ratios    : list of skill match ratios ∈ [0, 1]
    resume_ids      : candidate identifiers
    w_tfidf         : weight for TF-IDF keyword match
    w_semantic      : weight for dense semantic embeddings
    w_skills        : weight for exact skill taxonomy match
    threshold       : minimum final score cutoff

    Returns
    -------
    pd.DataFrame with columns:
        [rank, resume_id, hybrid_score, tfidf_score, semantic_score, skill_score, label]
    """
    n = len(resume_ids)
    tf_scores = np.asarray(tfidf_scores).flatten()
    sk_scores = np.asarray(skill_ratios).flatten()

    if semantic_scores is not None:
        sem_scores = np.asarray(semantic_scores).flatten()
        total_w = w_tfidf + w_semantic + w_skills
        if total_w == 0:
            total_w = 1.0
        final_scores = (w_tfidf * tf_scores + w_semantic * sem_scores + w_skills * sk_scores) / total_w
    else:
        sem_scores = np.zeros(n)
        total_w = w_tfidf + w_skills
        if total_w == 0:
            total_w = 1.0
        final_scores = (w_tfidf * tf_scores + w_skills * sk_scores) / total_w

    df = pd.DataFrame({
        "resume_id":      resume_ids,
        "score":           np.round(final_scores, 4),
        "tfidf_score":    np.round(tf_scores, 4),
        "semantic_score": np.round(sem_scores, 4) if semantic_scores is not None else None,
        "skill_score":    np.round(sk_scores, 4),
    })

    df = df[df["score"] >= threshold].copy()
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.index.name = "rank"

    def _label(s: float) -> str:
        if s >= 0.60:
            return "Excellent"
        if s >= 0.40:
            return "Good"
        if s >= 0.20:
            return "Fair"
        return "Poor"

    df["label"] = df["score"].apply(_label)
    return df.reset_index()



def precision_recall_f1(
    df: pd.DataFrame,
    relevant_ids: List[str],
) -> dict:
    """Simple precision / recall / F1 given a ground-truth relevant set.

    Parameters
    ----------
    df           : output of rank_resumes()
    relevant_ids : list of resume_ids that are truly relevant

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    retrieved = set(df["resume_id"].tolist())
    relevant  = set(relevant_ids)

    tp = len(retrieved & relevant)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall    = tp / len(relevant)  if relevant  else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }
