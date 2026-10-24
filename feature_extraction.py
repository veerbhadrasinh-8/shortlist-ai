"""
feature_extraction.py
----------------------
Converts cleaned text into numeric vectors using TF-IDF.
Optional: sentence-transformer embeddings (BERT-like) if installed.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Optional BERT / sentence-transformers ─────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF (default, recommended)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFExtractor:
    """Fit a TF-IDF vocabulary on job description + all resumes, then transform.

    Usage
    -----
    extractor = TFIDFExtractor()
    jd_vec, resume_vecs = extractor.fit_transform(jd_clean, resume_texts)
    """

    def __init__(
        self,
        max_features: int = 5_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,   # log(1+tf) – mild regularisation
        )
        self._fitted = False

    def fit_transform(
        self,
        jd_text: str,
        resume_texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on corpus (JD + resumes) and return vectors.

        Returns
        -------
        jd_vector     : shape (1, n_features)
        resume_matrix : shape (n_resumes, n_features)
        """
        corpus = [jd_text] + resume_texts
        matrix = self.vectorizer.fit_transform(corpus)
        self._fitted = True

        jd_vec = matrix[0]          # sparse row
        resume_mat = matrix[1:]     # sparse matrix

        return jd_vec, resume_mat

    def get_feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        return self.vectorizer.get_feature_names_out().tolist()

    def top_jd_keywords(self, jd_vec: np.ndarray, n: int = 20) -> List[str]:
        """Return the top-n TF-IDF keywords from the job description vector."""
        names = self.get_feature_names()
        scores = np.asarray(jd_vec.todense()).flatten()
        top_idx = scores.argsort()[::-1][:n]
        return [names[i] for i in top_idx if scores[i] > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-Transformer (advanced, optional)
# ─────────────────────────────────────────────────────────────────────────────

class BERTExtractor:
    """Dense embeddings via sentence-transformers.

    Requires:  pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def fit_transform(
        self,
        jd_text: str,
        resume_texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode JD and resumes into dense vectors.

        Returns
        -------
        jd_vector     : shape (1, embedding_dim)
        resume_matrix : shape (n_resumes, embedding_dim)
        """
        corpus = [jd_text] + resume_texts
        embeddings = self.model.encode(corpus, show_progress_bar=False)
        jd_vec = embeddings[0:1]
        resume_mat = embeddings[1:]
        return jd_vec, resume_mat


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_extractor(method: str = "tfidf") -> TFIDFExtractor | BERTExtractor:
    """Return the requested extractor.

    Parameters
    ----------
    method : 'tfidf' | 'bert'
    """
    if method == "tfidf":
        return TFIDFExtractor()
    elif method == "bert":
        return BERTExtractor()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'tfidf' or 'bert'.")
