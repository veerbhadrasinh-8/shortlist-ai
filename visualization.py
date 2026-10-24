"""
visualization.py
----------------
All charts and word-cloud plots for the resume screener.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from wordcloud import WordCloud
    _WC_AVAILABLE = True
except ImportError:
    _WC_AVAILABLE = False


# ── Style ─────────────────────────────────────────────────────────────────────
_PALETTE  = sns.color_palette("viridis", as_cmap=False)
_ACCENT   = "#4f8ef7"
_GOOD     = "#2ecc71"
_WARN     = "#f39c12"
_POOR     = "#e74c3c"
_BG       = "#0f1117"
_FG       = "#e0e0e0"

_LABEL_COLOR = {
    "Excellent": _GOOD,
    "Good":      _ACCENT,
    "Fair":      _WARN,
    "Poor":      _POOR,
}

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor":   _BG,
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  _FG,
    "xtick.color":      _FG,
    "ytick.color":      _FG,
    "text.color":       _FG,
    "grid.color":       "#222",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})


def _save(fig: plt.Figure, out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"  ✓ Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Bar chart – top-N candidates
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_candidates(
    df: pd.DataFrame,
    top_n: int = 15,
    out_dir: str = "report",
    filename: str = "top_candidates.png",
) -> str:
    """Horizontal bar chart of the top-N ranked resumes."""
    data = df.head(top_n).copy()
    colors = [_LABEL_COLOR.get(lbl, _ACCENT) for lbl in data["label"]]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.45)))
    bars = ax.barh(data["resume_id"][::-1], data["score"][::-1],
                   color=colors[::-1], edgecolor="none", height=0.65)

    # Score annotations
    for bar, score in zip(bars, data["score"][::-1]):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=8, color=_FG
        )

    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Cosine Similarity Score", fontsize=10)
    ax.set_title(f"Top {top_n} Candidates by Match Score", fontsize=13,
                 fontweight="bold", pad=12)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=l)
                      for l, c in _LABEL_COLOR.items()]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.15, fontsize=8)

    fig.tight_layout()
    return _save(fig, out_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Histogram – score distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distribution(
    df: pd.DataFrame,
    bins: int = 20,
    out_dir: str = "report",
    filename: str = "score_distribution.png",
) -> str:
    """Histogram of all similarity scores."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["score"], bins=bins, color=_ACCENT, edgecolor=_BG, alpha=0.85)
    ax.axvline(df["score"].mean(), color=_WARN,  linestyle="--",
               linewidth=1.5, label=f"Mean  {df['score'].mean():.3f}")
    ax.axvline(df["score"].median(), color=_GOOD, linestyle=":",
               linewidth=1.5, label=f"Median {df['score'].median():.3f}")

    ax.set_xlabel("Cosine Similarity Score", fontsize=10)
    ax.set_ylabel("Number of Resumes",       fontsize=10)
    ax.set_title("Score Distribution Across All Resumes",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y")
    fig.tight_layout()
    return _save(fig, out_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Word cloud – job description keywords
# ─────────────────────────────────────────────────────────────────────────────

def plot_wordcloud(
    text: str,
    title: str = "Job Description Keywords",
    out_dir: str = "report",
    filename: str = "jd_wordcloud.png",
) -> str:
    """Generate a word cloud from the cleaned job-description text."""
    if not _WC_AVAILABLE:
        print("  ⚠  wordcloud not installed (pip install wordcloud). Skipping.")
        return ""

    wc = WordCloud(
        width=900, height=450,
        background_color=_BG,
        colormap="cool",
        max_words=120,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    return _save(fig, out_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Label pie / donut chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distribution(
    df: pd.DataFrame,
    out_dir: str = "report",
    filename: str = "label_distribution.png",
) -> str:
    """Donut chart showing Excellent / Good / Fair / Poor breakdown."""
    counts = df["label"].value_counts().reindex(
        ["Excellent", "Good", "Fair", "Poor"], fill_value=0
    )
    colors = [_LABEL_COLOR[l] for l in counts.index]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"width": 0.5, "edgecolor": _BG},
        textprops={"color": _FG, "fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title("Candidate Quality Distribution",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return _save(fig, out_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Combo: good vs bad comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_good_vs_bad(
    df: pd.DataFrame,
    top_n: int = 5,
    out_dir: str = "report",
    filename: str = "good_vs_bad.png",
) -> str:
    """Side-by-side bar: top-N vs bottom-N resumes."""
    top  = df.head(top_n)
    bot  = df.tail(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={"wspace": 0.35})

    for ax, subset, color, ttl in [
        (axes[0], top, _GOOD,  f"Top {top_n} Matches"),
        (axes[1], bot, _POOR,  f"Bottom {top_n} Matches"),
    ]:
        ax.barh(subset["resume_id"], subset["score"],
                color=color, edgecolor="none", height=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for i, (_, row) in enumerate(subset.iterrows()):
            ax.text(row["score"] + 0.01, i, f'{row["score"]:.3f}',
                    va="center", fontsize=8)

    fig.suptitle("Best vs Worst Candidate Comparison",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    df: pd.DataFrame,
    jd_clean: str,
    out_dir: str = "report",
) -> List[str]:
    """Run every visualization and return list of saved paths."""
    print("\n📊  Generating visualizations …")
    paths = []
    paths.append(plot_top_candidates(df,       out_dir=out_dir))
    paths.append(plot_score_distribution(df,   out_dir=out_dir))
    paths.append(plot_wordcloud(jd_clean,      out_dir=out_dir))
    paths.append(plot_label_distribution(df,   out_dir=out_dir))
    if len(df) >= 6:
        paths.append(plot_good_vs_bad(df,      out_dir=out_dir))
    return [p for p in paths if p]
