"""
app.py
------
Shortlist — Professional Candidate Screening & Qualification Analysis Platform.
Clean B2B SaaS interface designed for HR teams, recruiters, and hiring managers.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import (
    clean_text,
    pdf_bytes_to_text,
    preprocess_job_description,
    preprocess_resume,
)
from feature_extraction import TFIDFExtractor, BERTExtractor, _ST_AVAILABLE
from similarity import rank_resumes, rank_resumes_hybrid
from skills_extractor import analyze_skill_gap
from main import _DEMO_JD, _DEMO_RESUMES


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShortlistAI | Candidate Screening & Evaluation",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Clean, professional B2B SaaS styling
st.markdown("""
<style>
    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1E293B;
    }
    
    /* Header */
    .app-header {
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }

    /* Professional Badges */
    .badge-matched {
        display: inline-block;
        background-color: #F8FAFC;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px 3px 2px 0;
    }
    .badge-missing {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px 3px 2px 0;
    }
    .badge-extra {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 400;
        margin: 2px 3px 2px 0;
    }

    /* Section Cards */
    .content-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Primary Button Overrides */
    div.stButton > button:first-child {
        background-color: #0F172A;
        color: #FFFFFF;
        border: 1px solid #0F172A;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.5rem 1.25rem;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E293B;
        border-color: #1E293B;
        color: #FFFFFF;
    }

    /* Clean Metric Boxes */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache Engines ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing analysis engine...")
def load_smart_engine():
    if _ST_AVAILABLE:
        try:
            return BERTExtractor(model_name="all-MiniLM-L6-v2")
        except Exception:
            return None
    return None


@st.cache_resource
def load_quick_engine():
    return TFIDFExtractor()


# ── Sample Job Templates ─────────────────────────────────────────────────────
ROLE_TEMPLATES = {
    "Data Scientist": _DEMO_JD.strip(),
    "Full-Stack Web Developer": """
We are hiring a Full-Stack Web Developer to build scalable web applications.
Key Requirements:
- 3+ years experience with React, TypeScript, HTML, and CSS
- Solid backend development in Node.js, Express, FastAPI, or Django
- Hands-on database design with PostgreSQL, MySQL, or MongoDB
- Experience with Docker, Git, CI/CD, and REST APIs
- Familiarity with cloud platforms (AWS, Azure, or GCP)
""".strip(),
    "Cloud & DevOps Engineer": """
We are looking for a Cloud & DevOps Engineer to manage infrastructure.
Key Requirements:
- Extensive experience with AWS, Azure, or Google Cloud (GCP)
- Proficient in Docker containerization, Kubernetes, and Terraform
- Automated CI/CD pipeline creation (GitHub Actions, Jenkins)
- Scripting in Python, Bash, or Go
- Linux server administration and monitoring
""".strip(),
    "Custom Job Description (Paste Your Own)": ""
}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ShortlistAI")
    st.caption("Candidate Screening & Qualification Analysis")
    st.markdown("---")

    st.markdown("**Configuration**")
    screening_mode = st.radio(
        "Evaluation Method:",
        ["Comprehensive (Context & Skills)", "Keyword Match Only"],
        help="Comprehensive mode analyzes semantic relevance, experience context, and verified skills. Keyword mode focuses strictly on term matches."
    )

    st.markdown("---")
    min_match_score = st.slider(
        "Minimum Match Threshold:",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        format="%d%%",
        help="Only display candidates meeting or exceeding this match percentage."
    )

    st.markdown("---")
    st.caption("**Workflow Overview**")
    st.caption(
        "1. Select or paste a Job Description.\n"
        "2. Upload candidate resumes (PDF or TXT).\n"
        "3. Review ranked applicants, verified skills, and missing requirements."
    )


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">ShortlistAI</div>
    <div class="app-subtitle">Screen, evaluate, and rank candidate resumes against role requirements.</div>
</div>
""", unsafe_allow_html=True)


tab_batch, tab_single, tab_guide = st.tabs([
    "Batch Screening",
    "Single Candidate Review",
    "Documentation & Guide"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: BATCH SCREENING
# ═════════════════════════════════════════════════════════════════════════════
with tab_batch:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**1. Job Description**")
        template_selection = st.selectbox(
            "Select role template:",
            list(ROLE_TEMPLATES.keys()),
            index=0,
            label_visibility="collapsed"
        )
        
        job_description_text = st.text_area(
            "Job requirements text:",
            value=ROLE_TEMPLATES[template_selection],
            height=230,
            placeholder="Paste role responsibilities, required qualifications, and technical competencies...",
            label_visibility="collapsed"
        )

    with col_right:
        st.markdown("**2. Candidate Resumes**")
        resume_source = st.radio(
            "Select source:",
            ["Use 12 Sample Candidates (Demo)", "Upload Resumes (PDF / TXT)"],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_files = None
        if resume_source == "Upload Resumes (PDF / TXT)":
            uploaded_files = st.file_uploader(
                "Upload resumes:",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload multiple PDF or TXT resume files."
            )
            if uploaded_files:
                st.caption(f"{len(uploaded_files)} candidate resume(s) uploaded.")
        else:
            st.caption("12 sample candidate resumes loaded across different seniorities and technical backgrounds.")

    st.markdown("")
    start_screening = st.button("Screen Applicants", type="primary", use_container_width=True)

    if start_screening:
        if not job_description_text.strip():
            st.warning("Please enter or select a job description.")
        else:
            with st.spinner("Analyzing candidate profiles and evaluating qualifications..."):
                start_clock = time.time()

                candidate_names = []
                candidate_raw_texts = []
                candidate_clean_texts = []

                if resume_source == "Upload Resumes (PDF / TXT)":
                    if not uploaded_files:
                        st.error("Please upload at least one candidate resume.")
                        st.stop()
                    for f in uploaded_files:
                        fname = f.name
                        name = os.path.splitext(fname)[0].replace("_", " ").title()
                        if fname.lower().endswith(".pdf"):
                            raw = pdf_bytes_to_text(f.read())
                        else:
                            raw = f.read().decode("utf-8", errors="ignore")
                        
                        candidate_names.append(name)
                        candidate_raw_texts.append(raw)
                        candidate_clean_texts.append(clean_text(raw))
                else:
                    for raw_name, raw in _DEMO_RESUMES:
                        display_name = raw_name.replace("_", " ").title()
                        candidate_names.append(display_name)
                        candidate_raw_texts.append(raw)
                        candidate_clean_texts.append(clean_text(raw))

                # Analyze skill matches
                gaps = []
                skill_scores = []
                for raw_text in candidate_raw_texts:
                    g = analyze_skill_gap(job_description_text, raw_text)
                    gaps.append(g)
                    skill_scores.append(g["match_percentage"] / 100.0)

                # Processing
                quick_engine = load_quick_engine()
                jd_clean = preprocess_job_description(job_description_text)
                jd_vec, res_vec = quick_engine.fit_transform(jd_clean, candidate_clean_texts)
                from sklearn.metrics.pairwise import cosine_similarity
                kw_scores = cosine_similarity(jd_vec, res_vec).flatten()

                smart_scores = None
                if "Comprehensive" in screening_mode:
                    smart_engine = load_smart_engine()
                    if smart_engine is not None:
                        jd_b, res_b = smart_engine.fit_transform(job_description_text, candidate_raw_texts)
                        smart_scores = cosine_similarity(jd_b, res_b).flatten()

                threshold_normalized = min_match_score / 100.0
                if "Comprehensive" in screening_mode and smart_scores is not None:
                    ranked_df = rank_resumes_hybrid(
                        tfidf_scores=kw_scores,
                        semantic_scores=smart_scores,
                        skill_ratios=skill_scores,
                        resume_ids=candidate_names,
                        w_tfidf=0.30,
                        w_semantic=0.50,
                        w_skills=0.20,
                        threshold=threshold_normalized
                    )
                else:
                    ranked_df = rank_resumes_hybrid(
                        tfidf_scores=kw_scores,
                        semantic_scores=None,
                        skill_ratios=skill_scores,
                        resume_ids=candidate_names,
                        w_tfidf=0.60,
                        w_semantic=0.0,
                        w_skills=0.40,
                        threshold=threshold_normalized
                    )

                def rating_label(score_val):
                    pct = score_val * 100
                    if pct >= 65:
                        return "Strong Match"
                    elif pct >= 45:
                        return "Good Match"
                    elif pct >= 30:
                        return "Moderate Match"
                    return "Low Match"

                ranked_df["fit_rating"] = ranked_df["score"].apply(rating_label)
                ranked_df["match_pct_display"] = (ranked_df["score"] * 100).round(1).astype(str) + "%"

                name_to_gap = {candidate_names[i]: gaps[i] for i in range(len(candidate_names))}
                ranked_df["matched_skills"] = ranked_df["resume_id"].apply(lambda n: name_to_gap[n]["matched_skills"])
                ranked_df["missing_skills"] = ranked_df["resume_id"].apply(lambda n: name_to_gap[n]["missing_skills"])
                ranked_df["skills_matched_count"] = ranked_df["matched_skills"].apply(len)
                ranked_df["skills_missing_count"] = ranked_df["missing_skills"].apply(len)

                elapsed_seconds = time.time() - start_clock

            st.markdown("---")

            # ── Summary Metrics ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Evaluated", len(ranked_df))
            with m2:
                top_cand = ranked_df.iloc[0]["resume_id"] if not ranked_df.empty else "None"
                top_pct = f"{ranked_df.iloc[0]['score']*100:.0f}%" if not ranked_df.empty else "0%"
                st.metric("Top Candidate", top_cand, f"{top_pct} match")
            with m3:
                strong_count = (ranked_df["fit_rating"] == "Strong Match").sum()
                st.metric("Strong Matches", int(strong_count))
            with m4:
                st.metric("Evaluation Time", f"{elapsed_seconds:.2f}s")

            # ── Candidate Leaderboard Table ──────────────────────────────────
            st.markdown("#### Applicant Rankings")

            table_df = ranked_df[[
                "rank", "resume_id", "match_pct_display", "fit_rating",
                "skills_matched_count", "skills_missing_count"
            ]].copy()

            table_df.rename(columns={
                "rank": "Rank",
                "resume_id": "Candidate",
                "match_pct_display": "Overall Match",
                "fit_rating": "Assessment",
                "skills_matched_count": "Requirements Met",
                "skills_missing_count": "Missing Requirements"
            }, inplace=True)

            def style_assessment(val):
                if val == "Strong Match":
                    return "background-color: #F0FDF4; color: #166534; font-weight: 600;"
                elif val == "Good Match":
                    return "background-color: #EFF6FF; color: #1E40AF; font-weight: 600;"
                elif val == "Moderate Match":
                    return "background-color: #FFFBEB; color: #92400E;"
                return "background-color: #F8FAFC; color: #64748B;"

            st.dataframe(
                table_df.style.map(style_assessment, subset=["Assessment"]),
                use_container_width=True,
                height=300
            )

            # Export
            export_df = ranked_df[["rank", "resume_id", "match_pct_display", "fit_rating", "matched_skills", "missing_skills"]].copy()
            export_df.rename(columns={
                "rank": "Rank",
                "resume_id": "Candidate",
                "match_pct_display": "Match Score",
                "fit_rating": "Assessment",
                "matched_skills": "Verified Skills",
                "missing_skills": "Missing Requirements"
            }, inplace=True)
            
            st.download_button(
                "Export Shortlist (CSV)",
                data=export_df.to_csv(index=False),
                file_name="candidate_shortlist.csv",
                mime="text/csv"
            )

            # ── Detailed Candidate Cards ─────────────────────────────────────
            st.markdown("#### Candidate Qualification Details")
            st.caption("Select a candidate to inspect verified skills and missing prerequisites.")

            for _, row in ranked_df.iterrows():
                badge_title = f"#{row['rank']} {row['resume_id']} — {row['match_pct_display']} ({row['fit_rating']})"
                with st.expander(badge_title):
                    col_met, col_miss = st.columns(2)
                    
                    with col_met:
                        st.markdown("**Verified Qualifications & Skills**")
                        matched = row["matched_skills"]
                        if matched:
                            pills = " ".join([f'<span class="badge-matched">{s}</span>' for s in matched])
                            st.markdown(pills, unsafe_allow_html=True)
                        else:
                            st.caption("No key required skills directly identified.")

                    with col_miss:
                        st.markdown("**Missing or Unmentioned Requirements**")
                        missing = row["missing_skills"]
                        if missing:
                            pills = " ".join([f'<span class="badge-missing">{s}</span>' for s in missing])
                            st.markdown(pills, unsafe_allow_html=True)
                        else:
                            st.caption("All primary requirements satisfied.")

            # ── Analytics ────────────────────────────────────────────────────
            st.markdown("#### Evaluation Overview")
            c1, c2 = st.columns(2)

            with c1:
                fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.8))
                top_candidates = ranked_df.head(7).copy()
                top_candidates["score_pct"] = top_candidates["score"] * 100
                
                sns.barplot(
                    data=top_candidates,
                    x="score_pct",
                    y="resume_id",
                    color="#1E3A8A",
                    ax=ax_bar
                )
                ax_bar.set_title("Top Candidate Match Scores", fontsize=10, fontweight="bold", pad=10)
                ax_bar.set_xlabel("Match Score (%)", fontsize=9)
                ax_bar.set_ylabel("")
                ax_bar.set_xlim(0, 100)
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_bar)

            with c2:
                fig_pie, ax_pie = plt.subplots(figsize=(6.5, 3.8))
                counts = ranked_df["fit_rating"].value_counts()
                palette = {
                    "Strong Match": "#15803D",
                    "Good Match": "#2563EB",
                    "Moderate Match": "#D97706",
                    "Low Match": "#94A3B8"
                }
                colors = [palette.get(k, "#CBD5E1") for k in counts.index]

                ax_pie.pie(
                    counts,
                    labels=counts.index,
                    autopct="%1.0f%%",
                    colors=colors,
                    startangle=140,
                    textprops={'fontsize': 9},
                    wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.55}
                )
                ax_pie.set_title("Applicant Distribution by Assessment", fontsize=10, fontweight="bold", pad=10)
                plt.tight_layout()
                st.pyplot(fig_pie)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: SINGLE CANDIDATE REVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("**Individual Candidate Evaluation**")
    st.caption("Evaluate a single candidate's resume against job requirements to review skill coverage and generate targeted interview discussion points.")

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.caption("Job Description")
        single_jd = st.text_area(
            "Job Description",
            value=ROLE_TEMPLATES["Data Scientist"],
            height=200,
            key="single_jd_input",
            label_visibility="collapsed"
        )
    with s_col2:
        st.caption("Candidate Resume Text")
        single_res = st.text_area(
            "Candidate Resume Text",
            value="Data Analyst with 2 years experience. Skilled in Python, pandas, SQL, Power BI, Excel. Familiar with machine learning and scikit-learn. Basic cloud knowledge in AWS.",
            height=200,
            key="single_res_input",
            label_visibility="collapsed"
        )

    if st.button("Evaluate Candidate", type="primary"):
        if not single_jd.strip() or not single_res.strip():
            st.warning("Please provide both the Job Description and the Resume text.")
        else:
            with st.spinner("Evaluating candidate qualifications..."):
                gap = analyze_skill_gap(single_jd, single_res)
                
                smart_engine = load_smart_engine()
                if smart_engine is not None:
                    j_b, r_b = smart_engine.fit_transform(single_jd, [single_res])
                    from sklearn.metrics.pairwise import cosine_similarity
                    semantic_score = float(cosine_similarity(j_b, r_b)[0][0])
                else:
                    semantic_score = 0.5

                quick_engine = load_quick_engine()
                jd_clean = preprocess_job_description(single_jd)
                res_clean = clean_text(single_res)
                j_q, r_q = quick_engine.fit_transform(jd_clean, [res_clean])
                from sklearn.metrics.pairwise import cosine_similarity
                kw_score = float(cosine_similarity(j_q, r_q)[0][0])

                skill_pct = gap["match_percentage"] / 100.0
                combined_score = (0.45 * semantic_score + 0.35 * kw_score + 0.20 * skill_pct)
                overall_percentage = min(100.0, max(0.0, combined_score * 100))

                st.markdown("---")
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Overall Match", f"{overall_percentage:.0f}%")
                with k2:
                    st.metric("Requirements Met", f"{gap['total_matched']} of {gap['total_required']}")
                with k3:
                    if overall_percentage >= 65:
                        verdict = "Strong Match"
                    elif overall_percentage >= 45:
                        verdict = "Good Match"
                    elif overall_percentage >= 30:
                        verdict = "Moderate Match"
                    else:
                        verdict = "Low Match"
                    st.metric("Candidate Assessment", verdict)

                st.markdown("#### Qualification Breakdown")
                q_left, q_right = st.columns(2)
                with q_left:
                    st.markdown("**Verified Skills**")
                    if gap["matched_skills"]:
                        pills = " ".join([f'<span class="badge-matched">{s}</span>' for s in gap["matched_skills"]])
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No key required skills identified.")

                with q_right:
                    st.markdown("**Missing Requirements**")
                    if gap["missing_skills"]:
                        pills = " ".join([f'<span class="badge-missing">{s}</span>' for s in gap["missing_skills"]])
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("All primary requirements satisfied.")

                if gap["missing_skills"]:
                    st.markdown("---")
                    st.markdown("#### Recommended Interview Focus Areas")
                    st.caption("Suggested discussion topics to explore areas not documented in the resume:")
                    for skill in gap["missing_skills"][:4]:
                        st.markdown(f"- **{skill}:** *\"Could you describe any past experience or familiarity you have working with {skill}?\"*")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: USER GUIDE
# ═════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.markdown("#### Platform Documentation")
    
    st.markdown("""
    ##### 1. Scoring Methodology
    Candidate fit is evaluated across three core criteria:
    - **Contextual Experience**: Analyzes role domain and overall responsibility alignment.
    - **Terminology Coverage**: Compares industry and role-specific technical terms.
    - **Skill Requirements**: Checks for the presence of mandatory technical tools, platforms, and competencies.

    ##### 2. Assessment Tiers
    | Assessment | Match Range | Guidance |
    | :--- | :--- | :--- |
    | **Strong Match** | 65% – 100% | High-priority candidate meeting primary role requirements. |
    | **Good Match** | 45% – 64% | Solid candidate with minor skill gaps. |
    | **Moderate Match** | 30% – 44% | Partial fit; may require onboarding or training. |
    | **Low Match** | Below 30% | Significant misalignment with core prerequisites. |

    ##### 3. Best Practices
    - Ensure job descriptions clearly outline mandatory technical competencies and qualifications.
    - Standard text-based PDF and TXT formats are recommended.
    - All processing runs locally within your environment.
    """)
