"""
preprocessing.py
----------------
Handles all text cleaning for resumes and job descriptions.
Supports: plain text strings and PDF file paths.
"""

import re
import string
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK data (silent after first run) ──────────────────────
for _pkg in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


# ── Optional PDF extraction ───────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except ImportError:
    _PYMUPDF = False

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract
    _PDFMINER = True
except ImportError:
    _PDFMINER = False


# ── Constants ─────────────────────────────────────────────────────────────────
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ── Public helpers ────────────────────────────────────────────────────────────

def pdf_to_text(pdf_path: str) -> str:
    """Extract raw text from a PDF file.

    Tries PyMuPDF first (faster), falls back to pdfminer.
    Raises RuntimeError if neither library is installed.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if _PYMUPDF:
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)

    if _PDFMINER:
        return _pdfminer_extract(pdf_path)

    raise RuntimeError(
        "No PDF library found. Install PyMuPDF:  pip install pymupdf\n"
        "or pdfminer.six:                         pip install pdfminer.six"
    )


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Extract raw text from in-memory PDF bytes."""
    if _PYMUPDF:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)

    if _PDFMINER:
        import io
        return _pdfminer_extract(io.BytesIO(pdf_bytes))

    raise RuntimeError("No PDF library available to parse bytes.")



def clean_text(text: str, lemmatize: bool = True) -> str:
    """Full preprocessing pipeline for a single text string.

    Steps
    -----
    1. Lowercase
    2. Remove URLs, emails, special characters, digits
    3. Tokenise
    4. Remove stopwords
    5. Lemmatize (optional, default True)

    Returns a single cleaned string.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove noise
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # e-mails
    text = re.sub(r"\d+", " ", text)                       # digits
    text = text.translate(
        str.maketrans(string.punctuation,
                      " " * len(string.punctuation))
    )                                                       # punctuation
    text = re.sub(r"\s+", " ", text).strip()               # extra whitespace

    # 3. Tokenise
    tokens = word_tokenize(text)

    # 4. Remove stopwords (keep tokens ≥ 2 chars)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 2]

    # 5. Lemmatize
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_resume(source: str, lemmatize: bool = True) -> str:
    """Preprocess a resume.

    Parameters
    ----------
    source : str
        Either a file path ending in '.pdf' or raw resume text.
    lemmatize : bool
        Whether to lemmatize tokens.

    Returns
    -------
    str  – cleaned text
    """
    if source.strip().lower().endswith(".pdf"):
        raw = pdf_to_text(source)
    else:
        raw = source
    return clean_text(raw, lemmatize=lemmatize)


def preprocess_job_description(jd_text: str, lemmatize: bool = True) -> str:
    """Preprocess a job description string."""
    return clean_text(jd_text, lemmatize=lemmatize)
