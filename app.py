# import module
import os
import re
import tempfile
import streamlit as st

# import functions
try:
    from extractor_number14_TRAINING_MAIN_UI import clean_for_keyword_matching, extract_resume_texts
except ImportError:
    from extractor import clean_for_keyword_matching, extract_resume_texts

try:
    from scoring_number14_TRAINING_MAIN_UI import calculate_keyword_score, calculate_skill_score_details, calculate_ml_similarity_score, calculate_hybrid_score, get_recommendation_label, get_recommendation_reason, find_missing_jd_keywords, build_missing_skills_suggestion, find_matched_keywords, build_score_explanation, train_selection_model, predict_selection_probability
except ImportError:
    from scoring import calculate_keyword_score, calculate_skill_score_details, calculate_ml_similarity_score, calculate_hybrid_score, get_recommendation_label, get_recommendation_reason, find_missing_jd_keywords, build_missing_skills_suggestion, find_matched_keywords, build_score_explanation, train_selection_model, predict_selection_probability


# ========= Job Profile (Software Engineer) =========


JOB_PROFILES = {
    "Software Engineer": {
        "skills": ["python", "javascript", "sql", "git", "api", "rest", "html", "css", "testing", "software development"],
        "experience": ["software engineer", "software development", "backend", "frontend", "front-end", "full stack", "full-stack", "web app", "api", "apis", "restful api", "system design", "debugging", "bugs", "testing", "unit testing", "implemented", "built", "developed"],
        "education": ["computer science", "software engineering", "information technology", "information systems", "bsc", "b.s", "bs", "bachelor", "degree"],
        "default_profile": """
        Software Engineer profile requiring Python, JavaScript, SQL, Git, RESTful APIs,
        HTML, CSS, testing, unit testing, web application development, software engineering,
        debugging, backend or frontend development, Computer Science or related degree,
        and practical project or work experience.
        """,
    },
    "Data Analyst": {
        "skills": ["sql", "python", "excel", "power bi", "tableau", "pandas", "data analysis", "statistics", "dashboard", "reporting"],
        "experience": ["data analysis", "data cleaning", "data visualization", "dashboard", "reporting", "business intelligence", "kpi", "insights", "excel", "power bi", "tableau", "sql query", "analytics"],
        "education": ["data science", "statistics", "mathematics", "computer science", "business analytics", "information systems", "bachelor", "degree"],
        "default_profile": """
        Data Analyst profile requiring SQL, Python, Excel, Power BI or Tableau,
        data cleaning, data visualization, dashboards, reporting, statistics,
        KPIs, business intelligence, and analytical problem solving.
        """,
    },
    "Cybersecurity Analyst": {
        "skills": ["network security", "siem", "firewall", "linux", "splunk", "incident response", "vulnerability assessment", "risk assessment", "security monitoring", "penetration testing"],
        "experience": ["security monitoring", "threat detection", "incident response", "incident handling", "vulnerability scanning", "log analysis", "security audit", "firewall", "malware", "phishing", "risk assessment"],
        "education": ["cybersecurity", "information security", "computer science", "information technology", "network security", "bachelor", "degree"],
        "default_profile": """
        Cybersecurity Analyst profile requiring SIEM, Splunk, firewall, Linux,
        network security, incident response, vulnerability assessment, log analysis,
        threat detection, security monitoring, and information security knowledge.
        """,
    },
    "UI/UX Designer": {
        "skills": ["figma", "adobe xd", "wireframe", "prototype", "user research", "usability testing", "ui design", "ux design", "design system", "visual design"],
        "experience": ["user research", "wireframing", "prototyping", "usability testing", "user interface", "user experience", "mobile design", "web design", "design system", "persona", "journey map", "mockup"],
        "education": ["design", "graphic design", "interaction design", "multimedia", "human computer interaction", "hci", "bachelor", "degree"],
        "default_profile": """
        UI/UX Designer profile requiring Figma or Adobe XD, wireframing,
        prototyping, user research, usability testing, UI design, UX design,
        design systems, visual design, and web or mobile product design experience.
        """,
    },
    "Project Manager": {
        "skills": ["project management", "agile", "scrum", "jira", "planning", "stakeholder management", "risk management", "budgeting", "communication", "leadership"],
        "experience": ["project planning", "project delivery", "agile", "scrum", "stakeholder management", "timeline", "budget", "risk management", "team coordination", "sprint", "roadmap", "reporting"],
        "education": ["project management", "business administration", "management", "information systems", "pmp", "scrum master", "bachelor", "degree"],
        "default_profile": """
        Project Manager profile requiring project planning, Agile or Scrum,
        Jira, stakeholder management, communication, leadership, budgeting,
        risk management, timeline control, reporting, and team coordination.
        """,
    },
}

DEFAULT_JOB_PROFILE_TEXT = JOB_PROFILES["Software Engineer"]["default_profile"].strip()

# Rule-based score category weights
WEIGHTS = {"skills": 0.5, "experience": 0.3, "education": 0.2}
HYBRID_WEIGHTS = {"rule_based": 0.7, "ml_similarity": 0.3}


def get_effective_job_description(jd_text: str, use_jd: bool, selected_profile: dict):
    """
    Return the text used for ML similarity.
    If no JD is provided, use the selected job profile so scoring does not become unfairly weak.
    """
    if use_jd and jd_text and jd_text.strip():
        return jd_text.strip(), "Custom Job Description"
    return selected_profile["default_profile"].strip(), "Default " + selected_profile.get("name", "Job") + " Profile"


# ========= function to extract extra keywords from job description =========
def keywords_from_job_description(jd_text: str, max_keywords: int = 30):
    """
    Extract useful keywords and phrases from a job description.
    """
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "has", "have", "in", "is", "it", "of", "on", "or",
        "our", "that", "the", "their", "this", "to", "we", "with",
        "you", "your", "will", "role", "candidate", "responsibilities",
        "requirements", "required", "preferred", "ability", "strong",
        "good", "excellent", "knowledge", "understanding", "experience",
        "work", "working", "team", "teams"
    }

    important_phrases = [
        "machine learning", "deep learning", "data analysis",
        "data science", "software development", "software engineering",
        "web development", "mobile development", "frontend development",
        "backend development", "full stack", "full-stack",
        "rest api", "restful api", "api integration", "system design",
        "object oriented programming", "unit testing", "software testing",
        "database design", "cloud computing", "version control",
        "git workflow", "agile development", "problem solving",
        "computer science", "information technology"
    ]

    text = jd_text.lower()
    text = text.replace("/", " ").replace("-", " ")

    keyword_scores = {}

    for phrase in important_phrases:
        normalized_phrase = phrase.replace("-", " ")
        count = text.count(normalized_phrase)
        if count > 0:
            keyword_scores[normalized_phrase] = keyword_scores.get(normalized_phrase, 0) + (count * 3)

    raw_words = re.findall(r"[a-zA-Z][a-zA-Z0-9.+#]*", text)

    for raw_word in raw_words:
        word = raw_word.strip(".")

        if len(word) < 3 and word not in {"c++", "c#", "js"}:
            continue
        if word in stop_words:
            continue
        keyword_scores[word] = keyword_scores.get(word, 0) + 1

    ranked_keywords = sorted(keyword_scores, key=lambda k: (-keyword_scores[k], k))
    return ranked_keywords[:max_keywords]


# ========= function to detect resume sections =========
def detect_resume_sections(raw_text: str):
    """
    Detect common resume sections using headings.
    This function uses raw_text because line breaks are still available.
    """
    section_patterns = {
        "skills": [
            "skills", "technical skills", "key skills", "programming skills",
            "technologies", "tools", "technical expertise"
        ],
        "experience": [
            "experience", "work experience", "professional experience",
            "employment history", "career history"
        ],
        "education": [
            "education", "academic background", "qualification", "qualifications"
        ],
        "projects": [
            "projects", "project experience", "academic projects", "personal projects"
        ],
        "certifications": [
            "certifications", "certification", "certificates", "courses", "training"
        ],
    }

    heading_to_section = {}
    for section, headings in section_patterns.items():
        for heading in headings:
            heading_to_section[heading] = section

    sections = {section: "" for section in section_patterns}
    sections["other"] = ""

    current_section = "other"
    lines = raw_text.splitlines()

    for line in lines:
        cleaned_line = line.strip().lower().strip(":-|")
        cleaned_line = re.sub(r"\s+", " ", cleaned_line)

        # Treat short lines that match known headings as section headers.
        if cleaned_line in heading_to_section and len(cleaned_line.split()) <= 4:
            current_section = heading_to_section[cleaned_line]
            continue

        sections[current_section] += line + "\n"

    return {name: content.strip() for name, content in sections.items() if content.strip()}


def section_text(sections, preferred_sections, fallback_text):
    """
    Combine preferred sections. If they do not exist, use the full resume.
    """
    combined = "\n".join(sections.get(section, "") for section in preferred_sections).strip()
    return combined if combined else fallback_text


# ========= function to score one resume =========
def score_resume(file_path: str, selected_profile=None, extra_jd_keywords=None, job_description_text="", trained_model_info=None):
    # Extract both versions of text.
    resume_texts = extract_resume_texts(file_path)
    raw_text = resume_texts["raw_text"]       # keeps punctuation, line breaks, context
    clean_text = resume_texts["clean_text"]   # cleaned version for keyword matching

    # Detect sections from raw text.
    sections = detect_resume_sections(raw_text)

    selected_profile = selected_profile or JOB_PROFILES["Software Engineer"]
    skills = selected_profile["skills"].copy()
    exp = selected_profile["experience"].copy()
    edu = selected_profile["education"].copy()

    if extra_jd_keywords:
        exp = list(set(exp + extra_jd_keywords))

    # raw section text for skill context because it keeps sentence meaning
    skills_raw_text = section_text(sections, ["skills", "experience", "projects"], raw_text)

    # cleaned section text for keyword matching
    exp_raw_text = section_text(sections, ["experience", "projects"], raw_text)
    edu_raw_text = section_text(sections, ["education", "certifications"], raw_text)

    exp_clean_text = clean_for_keyword_matching(exp_raw_text)
    edu_clean_text = clean_for_keyword_matching(edu_raw_text)

    skill_details = calculate_skill_score_details(skills_raw_text, skills)
    skill_score = skill_details["score"]
    exp_score = calculate_keyword_score(exp_clean_text, exp, required_matches=5)
    edu_score = calculate_keyword_score(edu_clean_text, edu, required_matches=2)

    rule_based_score = (
        WEIGHTS["skills"] * skill_score
        + WEIGHTS["experience"] * exp_score
        + WEIGHTS["education"] * edu_score
    )

    # ML/NLP similarity between the full resume and the JD
    ml_similarity_score = calculate_ml_similarity_score(raw_text, job_description_text)

    # calibration fix:
    if job_description_text and selected_profile and job_description_text.strip() == selected_profile["default_profile"].strip():
        ml_similarity_score = max(ml_similarity_score, rule_based_score)

    #  hybrid score combines ML similarity with the explainable rule-based score.
    hybrid_score = calculate_hybrid_score(
        rule_based_score,
        ml_similarity_score,
        rule_weight=HYBRID_WEIGHTS["rule_based"],
        ml_weight=HYBRID_WEIGHTS["ml_similarity"],
    )

    recommendation_label = get_recommendation_label(hybrid_score)
    recommendation_reason = get_recommendation_reason(recommendation_label)

    # optional supervised ML score from uploaded historical training dataset.
    supervised_ml_score = predict_selection_probability(
        trained_model_info,
        raw_text,
        job_description_text,
    )

    # suggest missing skills/requirements from the job description
    jd_keywords = extra_jd_keywords or []
    missing_jd_keywords = find_missing_jd_keywords(clean_text, jd_keywords)
    missing_suggestion = build_missing_skills_suggestion(
        missing_default_skills=skill_details["missing_skills"],
        missing_jd_keywords=missing_jd_keywords,
    )

    #explainability - show the evidence behind the score
    matched_experience_keywords = find_matched_keywords(exp_clean_text, exp)
    matched_education_keywords = find_matched_keywords(edu_clean_text, edu)
    explanation = build_score_explanation(
        skill_score=skill_score,
        experience_score=exp_score,
        education_score=edu_score,
        rule_based_score=rule_based_score,
        ml_similarity_score=ml_similarity_score,
        hybrid_score=hybrid_score,
        recommendation_label=recommendation_label,
        matched_skill_aliases=skill_details["matched_aliases"],
        missing_skills=skill_details["missing_skills"],
        matched_experience_keywords=matched_experience_keywords,
        matched_education_keywords=matched_education_keywords,
        missing_jd_keywords=missing_jd_keywords,
        supervised_ml_score=supervised_ml_score,
        training_dataset_used=bool(trained_model_info and trained_model_info.get("is_trained")),
    )

    detected_sections = [section.title() for section in sections.keys() if section != "other"]

    return {
        "skill_score": skill_score,
        "experience_score": exp_score,
        "education_score": edu_score,
        "rule_based_score": rule_based_score,
        "ml_similarity_score": ml_similarity_score,
        "hybrid_score": hybrid_score,
        "recommendation_label": recommendation_label,
        "recommendation_reason": recommendation_reason,
        "supervised_ml_score": supervised_ml_score,
        "training_dataset_used": bool(trained_model_info and trained_model_info.get("is_trained")),
        "detected_sections": detected_sections,
        "matched_skill_aliases": skill_details["matched_aliases"],
        "missing_skills": skill_details["missing_skills"],
        "missing_jd_keywords": missing_jd_keywords,
        "missing_suggestion": missing_suggestion,
        "matched_experience_keywords": matched_experience_keywords,
        "matched_education_keywords": matched_education_keywords,
        "explanation_summary": explanation["summary"],
        "explanation_details": explanation["details"],
        "raw_text_length": len(raw_text),
        "clean_text_length": len(clean_text),
    }


# ========= UI + History layer (presentation only) =========
# NOTE: All NLP preprocessing, alias matching and scoring logic above this line
# is unchanged. This section only handles the user interface and the local
# SQLite screening-history database.

import sqlite3
import pandas as pd
from datetime import datetime

# Candidates scoring at or above this Hybrid Final Score are counted as "Top Tier".
TOP_TIER_THRESHOLD = 80

# Local history database, stored right next to this script so it is portable.
HISTORY_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "screening_history.db",
)


# ========= Screening history (lightweight SQLite database) =========
def init_history_db():
    """Create the screening_history table if it does not already exist."""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS screening_history (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                job_scope         TEXT    NOT NULL,
                total_scanned     INTEGER NOT NULL,
                top_tier_matches  INTEGER NOT NULL,
                top_candidate     TEXT,
                top_score         REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_screening_run(job_scope, total_scanned, top_tier_matches, top_candidate, top_score):
    """Persist a single screening run to the local history database."""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO screening_history
                (timestamp, job_scope, total_scanned, top_tier_matches, top_candidate, top_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                job_scope,
                int(total_scanned),
                int(top_tier_matches),
                top_candidate,
                float(top_score) if top_score is not None else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_screening_history():
    """Return the full screening history as a pandas DataFrame (newest first)."""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        history_df = pd.read_sql_query(
            """
            SELECT
                timestamp        AS "Timestamp",
                job_scope        AS "Job Scope",
                total_scanned    AS "Total Scanned",
                top_tier_matches AS "Top Tier Matches",
                top_candidate    AS "Top Candidate",
                top_score        AS "Top Score (%)"
            FROM screening_history
            ORDER BY id DESC
            """,
            conn,
        )
    finally:
        conn.close()
    return history_df


def clear_screening_history():
    """Delete every row from the screening history table."""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        conn.execute("DELETE FROM screening_history")
        conn.commit()
    finally:
        conn.close()


def _safe_rerun():
    """Rerun the app in a way that works across Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ========= Streamlit page setup =========
st.set_page_config(
    page_title="Smart Resume Screening | HR Suite",
    page_icon="🧭",
    layout="wide",
)

init_history_db()


# ========= Custom CSS (premium enterprise look) =========
def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .stApp { background: #f5f7fb; }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        /* ---------- Hero banner ---------- */
        .hr-hero {
            background: linear-gradient(120deg, #0f172a 0%, #1e293b 45%, #1d4ed8 130%);
            border-radius: 18px;
            padding: 30px 34px;
            color: #f8fafc;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
            margin-bottom: 22px;
        }
        .hr-hero-badge {
            display: inline-block;
            font-size: 0.70rem;
            letter-spacing: 0.16em;
            font-weight: 700;
            text-transform: uppercase;
            color: #bfdbfe;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(191, 219, 254, 0.35);
            padding: 5px 12px;
            border-radius: 999px;
            margin-bottom: 14px;
        }
        .hr-hero h1 {
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0 0 6px 0;
            line-height: 1.15;
            color: #ffffff;
        }
        .hr-hero p {
            margin: 0;
            font-size: 0.98rem;
            color: #cbd5e1;
            font-weight: 400;
        }

        /* ---------- Config chips ---------- */
        .hr-chips {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin: 6px 0 10px 0;
        }
        .hr-chip {
            display: flex;
            flex-direction: column;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 18px;
            min-width: 150px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .hr-chip-k {
            font-size: 0.70rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 3px;
        }
        .hr-chip-v {
            font-size: 1.0rem;
            color: #0f172a;
            font-weight: 700;
        }

        /* ---------- Metric cards ---------- */
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10);
        }
        [data-testid="stMetricLabel"] p {
            color: #64748b;
            font-weight: 600;
            font-size: 0.82rem;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
            font-size: 1.9rem;
        }

        /* ---------- Section headings ---------- */
        .hr-section {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin: 4px 0 2px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .hr-section:before {
            content: "";
            width: 4px;
            height: 18px;
            border-radius: 4px;
            background: linear-gradient(#2563eb, #1d4ed8);
            display: inline-block;
        }

        /* ---------- Expander cards ---------- */
        [data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            background: #ffffff !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            margin-bottom: 8px;
        }
        [data-testid="stExpander"] summary { font-weight: 600; color: #1e293b; }

        /* ---------- Buttons ---------- */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
            border: 1px solid transparent;
            transition: all 0.12s ease;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(120deg, #2563eb, #1d4ed8);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
        }
        .stButton > button[kind="primary"]:hover { filter: brightness(1.05); }
        .stDownloadButton > button { border-radius: 10px; font-weight: 600; }

        /* ---------- Tabs ---------- */
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 6px; }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-weight: 600;
            padding: 10px 18px;
            border-radius: 10px 10px 0 0;
        }

        /* ---------- Dataframe ---------- */
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        /* ---------- Sidebar brand ---------- */
        .hr-brand {
            background: linear-gradient(135deg, #1e293b, #1d4ed8);
            color: #fff;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
        }
        .hr-brand .hr-brand-title { font-weight: 800; font-size: 1.05rem; }
        .hr-brand .hr-brand-sub { font-size: 0.78rem; color: #cbd5e1; margin-top: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="hr-section">{text}</div>', unsafe_allow_html=True)


inject_theme()

# ========= Hero banner =========
st.markdown(
    """
    <div class="hr-hero">
        <span class="hr-hero-badge">Enterprise HR Suite</span>
        <h1>Smart Resume Screening System</h1>
        <p>AI-assisted candidate evaluation &nbsp;•&nbsp; Hybrid NLP + Machine Learning scoring across multiple job scopes</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========= Sidebar: Job Setup =========
with st.sidebar:
    st.markdown(
        """
        <div class="hr-brand">
            <div class="hr-brand-title">🧭 Screening Console</div>
            <div class="hr-brand-sub">Configure the role, then run screening</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Job Setup")

    position = st.selectbox("Job Position", list(JOB_PROFILES.keys()), index=0)
    selected_profile = JOB_PROFILES[position].copy()
    selected_profile["name"] = position

    jd = st.text_area(
        "Job Description (optional)",
        placeholder="Paste job description here (skills/requirements).",
        height=200,
    )

    use_jd = st.checkbox("Use Job Description to enrich scoring", value=True)

    if use_jd and jd.strip():
        preview_keywords = keywords_from_job_description(jd)
        st.markdown("**Extracted JD Keywords/Phrases:**")
        st.write(", ".join(preview_keywords[:15]))
    else:
        st.info(
            f"No Job Description provided. The app will use the default {position} "
            f"profile for ML similarity scoring."
        )

    st.divider()

    # Quick history snapshot in the sidebar.
    _hist_snapshot = load_screening_history()
    st.caption("History snapshot")
    if _hist_snapshot.empty:
        st.caption("No runs logged yet.")
    else:
        st.metric("Runs logged", len(_hist_snapshot))


# ========= Tabs =========
tab_screen, tab_history, tab_about = st.tabs(["🔍  Screening", "🗂️  History", "ℹ️  About"])


# ---------- Results renderer (reads from session state so results survive tab switches) ----------
def render_results():
    results = st.session_state.get("last_results")
    if not results:
        st.info("Upload at least one resume and click **Run Screening** to see ranked results.")
        return

    result_position = st.session_state.get("last_position", "")
    scoring_mode = st.session_state.get("last_scoring_mode", "")

    total_scanned = len(results)
    top_tier = sum(1 for r in results if r["Hybrid Final Score (%)"] >= TOP_TIER_THRESHOLD)
    top = results[0]
    avg_score = round(sum(r["Hybrid Final Score (%)"] for r in results) / total_scanned, 1)

    # ----- Dashboard metrics -----
    section_title("Screening Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Resumes Scanned", total_scanned)
    m2.metric(f"Top Tier Matches  (≥ {TOP_TIER_THRESHOLD}%)", top_tier)
    m3.metric("Highest Score", f"{top['Hybrid Final Score (%)']}%")
    m4.metric("Average Score", f"{avg_score}%")
    st.caption(f"Scope: **{result_position}**  •  Scoring mode: **{scoring_mode}**")

    st.divider()

    # ----- Ranked table with progress bars (presentation only) -----
    section_title("Ranked Candidates")
    results_df = pd.DataFrame(results)
    score_progress = st.column_config.ProgressColumn(
        format="%.1f%%", min_value=0, max_value=100
    )
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hybrid Final Score (%)": score_progress,
            "Rule-Based Final Score (%)": score_progress,
            "ML Similarity Score (%)": score_progress,
            "Skill Score (%)": score_progress,
            "Experience Score (%)": score_progress,
            "Education Score (%)": score_progress,
        },
    )

    st.divider()

    # ----- Candidate explanation cards -----
    section_title("Candidate Explanations")
    for index, row in enumerate(results, start=1):
        with st.expander(
            f"#{index}  {row['Candidate File']}  —  {row['Recommendation']}  —  "
            f"{row['Hybrid Final Score (%)']}%"
        ):
            st.text(row["Detailed Explanation"])

    # ----- Top candidate highlight -----
    st.success(
        f"🏆 Top Candidate: {top['Candidate File']}  —  Hybrid Final Score: "
        f"{top['Hybrid Final Score (%)']}%  —  {top['Recommendation']}  —  {top['Scoring Mode']}"
    )


# ========= Screening tab =========
with tab_screen:
    # Current configuration summary chips.
    scoring_source = "Custom Job Description" if (use_jd and jd.strip()) else f"Default {position} Profile"
    st.markdown(
        f"""
        <div class="hr-chips">
            <div class="hr-chip"><span class="hr-chip-k">Job Scope</span><span class="hr-chip-v">{position}</span></div>
            <div class="hr-chip"><span class="hr-chip-k">Scoring Source</span><span class="hr-chip-v">{scoring_source}</span></div>
            <div class="hr-chip"><span class="hr-chip-k">Top Tier Cutoff</span><span class="hr-chip-v">≥ {TOP_TIER_THRESHOLD}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ----- Optional training dataset (logic unchanged) -----
    section_title("Optional Training Dataset")

    use_training_section = st.checkbox(
        "Use optional training dataset",
        value=False,
        help="Turn this on only if you want to upload historical resume/job decision data.",
    )

    trained_model_info = {
        "is_trained": False,
        "message": "No training data provided.",
        "model": None,
        "rows_used": 0,
    }
    use_training_model_for_reference = False

    if use_training_section:
        st.caption(
            "Upload historical examples if you want the app to show an additional supervised ML score. "
            "This is optional and does not replace the hybrid score."
        )

        training_csv = st.file_uploader(
            "Upload training CSV (optional)",
            type=["csv"],
            key="training_dataset_csv",
            help="Expected columns: resume_text, job_description, label. Labels can be selected/rejected or 1/0.",
        )

        use_training_model_for_reference = st.checkbox(
            "Show supervised ML score from training dataset",
            value=True,
            help="This does not replace the hybrid score. It adds an extra score learned from your historical data.",
        )

        if training_csv is not None:
            try:
                training_df = pd.read_csv(training_csv)
                trained_model_info = train_selection_model(training_df)
                if trained_model_info.get("is_trained"):
                    st.success(trained_model_info.get("message"))
                else:
                    st.warning(trained_model_info.get("message"))
            except Exception as exc:
                st.error(f"Could not read training CSV: {exc}")

    st.divider()

    # ----- Upload resumes -----
    section_title("Upload Resumes (PDF / DOCX)")

    uploaded_files = st.file_uploader(
        "Upload multiple files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    run_clicked = st.button(
        "Run Screening",
        type="primary",
        disabled=(len(uploaded_files) == 0),
    )

    # ----- Run screening (scoring loop is UNCHANGED) -----
    if run_clicked:
        results = []
        extra_keywords = []
        selected_profile = JOB_PROFILES[position].copy()
        selected_profile["name"] = position
        effective_jd_text, scoring_mode = get_effective_job_description(jd, use_jd, selected_profile)

        # Only enrich keyword scoring with JD keywords when the user actually provides a JD.
        # If no JD is provided, default skill/experience/education lists are already used.
        if use_jd and jd.strip():
            extra_keywords = keywords_from_job_description(jd)

        with st.spinner("Scoring resumes..."):
            for uf in uploaded_files:
                suffix = os.path.splitext(uf.name)[1].lower()

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name

                try:
                    scores = score_resume(
                        tmp_path,
                        selected_profile=selected_profile,
                        extra_jd_keywords=extra_keywords,
                        job_description_text=effective_jd_text,
                        trained_model_info=trained_model_info if use_training_model_for_reference else None,
                    )

                    results.append({
                        "Candidate File": uf.name,
                        "Job Position": position,
                        "Scoring Mode": scoring_mode,
                        "Skill Score (%)": round(scores["skill_score"], 2),
                        "Experience Score (%)": round(scores["experience_score"], 2),
                        "Education Score (%)": round(scores["education_score"], 2),
                        "Rule-Based Final Score (%)": round(scores["rule_based_score"], 2),
                        "ML Similarity Score (%)": round(scores["ml_similarity_score"], 2),
                        "Hybrid Final Score (%)": round(scores["hybrid_score"], 2),
                        "Supervised ML Score (%)": round(scores["supervised_ml_score"], 2),
                        "Training Dataset Used": "Yes" if scores["training_dataset_used"] else "No",
                        "Recommendation": scores["recommendation_label"],
                        "Recommendation Reason": scores["recommendation_reason"],
                        "Detected Sections": ", ".join(scores["detected_sections"]) or "Not detected",
                        "Matched Skill Aliases": ", ".join(scores["matched_skill_aliases"]) or "None",
                        "Missing Skills": ", ".join(scores["missing_skills"]) or "None",
                        "Missing JD Keywords": ", ".join(scores["missing_jd_keywords"]) or "None",
                        "Improvement Suggestion": scores["missing_suggestion"],
                        "Matched Experience/JD Keywords": ", ".join(scores["matched_experience_keywords"]) or "None",
                        "Matched Education Keywords": ", ".join(scores["matched_education_keywords"]) or "None",
                        "Score Explanation Summary": scores["explanation_summary"],
                        "Detailed Explanation": scores["explanation_details"],
                        "Raw Text Chars": scores["raw_text_length"],
                        "Clean Text Chars": scores["clean_text_length"],
                        "Preprocessing Used": "raw for sections/context, clean for keyword scores",
                    })

                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        results.sort(key=lambda x: x["Hybrid Final Score (%)"], reverse=True)

        # Persist to session so results survive tab switches / reruns.
        st.session_state["last_results"] = results
        st.session_state["last_position"] = position
        st.session_state["last_scoring_mode"] = scoring_mode

        # Log this run to the SQLite screening history.
        if results:
            top_row = results[0]
            top_tier_count = sum(
                1 for r in results if r["Hybrid Final Score (%)"] >= TOP_TIER_THRESHOLD
            )
            save_screening_run(
                job_scope=position,
                total_scanned=len(results),
                top_tier_matches=top_tier_count,
                top_candidate=top_row["Candidate File"],
                top_score=top_row["Hybrid Final Score (%)"],
            )

    st.divider()
    render_results()


# ========= History tab =========
with tab_history:
    section_title("Screening History")
    st.caption("Every screening run is logged locally with SQLite. Newest first.")

    history_df = load_screening_history()

    if history_df.empty:
        st.info("No screening runs recorded yet. Run a screening to start building history.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Runs Logged", len(history_df))
        h2.metric("Resumes Scanned (all runs)", int(history_df["Total Scanned"].sum()))
        h3.metric("Top Tier Matches (all runs)", int(history_df["Top Tier Matches"].sum()))

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Top Score (%)": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )

        col_dl, col_clear = st.columns([1, 1])
        with col_dl:
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download History (CSV)",
                data=csv_bytes,
                file_name="screening_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️  Clear History", use_container_width=True):
                clear_screening_history()
                _safe_rerun()


# ========= About tab =========
with tab_about:
    section_title("How This System Works")
    with st.expander("System Explanation", expanded=True):
        st.markdown(
            """
This system uses a hybrid resume screening approach. First, it applies NLP preprocessing to extract and clean resume text from PDF and DOCX files. Then, it performs rule-based scoring using job-specific skills, experience keywords, education keywords, and skill alias matching.

The system also applies machine learning using TF-IDF vectorization and cosine similarity to compare the resume with the job description or selected job profile. If a training dataset is uploaded, the system can train a supervised logistic regression model to estimate candidate suitability.

The final candidate ranking is generated using a hybrid score: 70% rule-based scoring and 30% ML similarity. This approach keeps the system explainable while still using machine learning to improve matching accuracy.
            """
        )

    with st.expander("NLP used in"):
        st.markdown(
            """
            1. Text preprocessing
            2. Resume section detection
            3. Job description keyword and phrase extraction
            4. Skill alias matching
            """
        )

    with st.expander("Machine learning is used in"):
        st.markdown(
            """
            1. TF-IDF vectorization
            2. Cosine similarity
            3. Optional supervised Logistic Regression if training CSV is uploaded
            """
        )
