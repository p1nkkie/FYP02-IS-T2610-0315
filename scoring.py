import re


# ---------------- calculate_keyword_score ----------------------------
def calculate_keyword_score(resume_text, keywords, required_matches=None):
    """
    Calibrated keyword percentage score.

    Old behavior divided matched keywords by the full keyword list, which made
    scores too low when the list contained many alternatives.

    New behavior:
    - Count matched keywords.
    - If required_matches is provided, use that as the denominator.
    - Cap the score at 100%.
    """
    if not keywords:
        return 0.0

    matched = 0
    seen = set()

    for kw in keywords:
        kw = str(kw).lower().strip()
        if not kw or kw in seen:
            continue
        seen.add(kw)

        if kw in resume_text:
            matched += 1

    denominator = required_matches if required_matches else len(seen)
    if denominator <= 0:
        return 0.0

    return min(100.0, (matched / denominator) * 100)


# ---------------- skill alias matching ----------------------------
# frameworks, or closely related tool names.
SKILL_ALIASES = {
    "python": ["python", "py", "pandas", "numpy", "django", "flask", "fastapi"],
    "java": ["java", "spring", "spring boot", "j2ee"],
    "javascript": [
        "javascript", "java script", "js", "node.js", "node js",
        "react", "react.js", "react js", "vue", "angular", "typescript", "ts"
    ],
    "c++": ["c++", "cpp", "c plus plus"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "mssql", "sql server"],
    "git": ["git", "github", "gitlab", "bitbucket", "version control"],
    "api": ["api", "apis", "rest api", "restful api", "graphql", "endpoint", "endpoints"],
    "rest": ["rest", "rest api", "restful", "restful api"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "bootstrap", "tailwind", "sass", "scss"],
    "testing": ["testing", "unit testing", "qa", "quality assurance", "selenium", "test"],
    "software development": ["software development", "software engineer", "developed", "built", "implemented", "web app", "application"],

    "excel": ["excel", "microsoft excel", "spreadsheet", "pivot table", "vlookup"],
    "power bi": ["power bi", "powerbi", "power-bi"],
    "tableau": ["tableau"],
    "pandas": ["pandas"],
    "data analysis": ["data analysis", "analytics", "data analyst", "analyzed data"],
    "statistics": ["statistics", "statistical analysis", "stats"],
    "dashboard": ["dashboard", "dashboards", "dashboarding"],
    "reporting": ["reporting", "reports", "report"],
    "network security": ["network security", "networking security"],
    "siem": ["siem", "security information and event management"],
    "firewall": ["firewall", "firewalls"],
    "linux": ["linux", "unix"],
    "splunk": ["splunk"],
    "incident response": ["incident response", "incident handling"],
    "vulnerability assessment": ["vulnerability assessment", "vulnerability scanning", "vapt"],
    "risk assessment": ["risk assessment", "risk management"],
    "security monitoring": ["security monitoring", "monitoring"],
    "penetration testing": ["penetration testing", "pentest", "pen test"],
    "figma": ["figma"],
    "adobe xd": ["adobe xd", "xd"],
    "wireframe": ["wireframe", "wireframes", "wireframing"],
    "prototype": ["prototype", "prototypes", "prototyping"],
    "user research": ["user research", "ux research"],
    "usability testing": ["usability testing", "user testing"],
    "ui design": ["ui design", "user interface"],
    "ux design": ["ux design", "user experience"],
    "design system": ["design system", "design systems"],
    "visual design": ["visual design"],
    "project management": ["project management", "managed project", "project manager"],
    "agile": ["agile", "agile development"],
    "scrum": ["scrum", "scrum master"],
    "jira": ["jira"],
    "planning": ["planning", "project planning"],
    "stakeholder management": ["stakeholder management", "stakeholders"],
    "budgeting": ["budgeting", "budget"],
    "communication": ["communication", "communication skills"],
    "leadership": ["leadership", "led team", "team lead"],

}


# positive context
POSITIVE_CONTEXT = [
    "developed", "built", "created", "implemented", "designed",
    "used", "worked with", "experience in", "experienced in",
    "proficient in", "skilled in", "hands-on", "hands on", "using"
]

# weak context 
WEAK_CONTEXT = [
    "basic understanding", "familiar with", "knowledge of",
    "exposure to", "learned", "studied"
]

# negative context
NEGATIVE_CONTEXT = [
    "no experience", "without experience", "not experienced",
    "never used", "lack of experience", "do not know",
    "not familiar with"
]


def split_sentences(text):
    """
    Split resume text into simple sentences.
    """
    return re.split(r"[.!?\n]", text.lower())


def normalize_for_alias_matching(text):
    """
    Normalize text for matching aliases while keeping useful symbols like +, # and .
    """
    text = text.lower()
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_term(text, term):
    """
    Check whether a skill/alias appears as a real term, not only as part of another word.
    Example: 'java' should not match 'javascript'.
    """
    text = normalize_for_alias_matching(text)
    term = normalize_for_alias_matching(term)

    pattern = r"(?<![a-zA-Z0-9+#.])" + re.escape(term) + r"(?![a-zA-Z0-9+#.])"
    return re.search(pattern, text) is not None


def get_aliases_for_skill(skill):
    """
    Return aliases for a skill. If no alias list exists, use the skill itself.
    """
    skill = skill.lower().strip()
    aliases = SKILL_ALIASES.get(skill, [skill])

    # Remove duplicates while keeping order.
    seen = set()
    unique_aliases = []
    for alias in aliases:
        alias = alias.lower().strip()
        if alias and alias not in seen:
            unique_aliases.append(alias)
            seen.add(alias)
    return unique_aliases


def find_matching_alias(sentence, skill):
    """
    Return the alias found in a sentence for a main skill.
    """
    for alias in get_aliases_for_skill(skill):
        if contains_term(sentence, alias):
            return alias
    return None


def get_skill_context_score(sentence, skill):
    """
    Check the surrounding words around a skill or its aliases.
    Return:
    1.0 = strong positive match
    0.5 = weak match
    0.0 = negative/no match
    1.0 = skill exists in resume
    """
    matching_alias = find_matching_alias(sentence, skill)
    if not matching_alias:
        return 0.0

    # check negative first
    for phrase in NEGATIVE_CONTEXT:
        if phrase in sentence:
            return 0.0

    # check positive context
    for phrase in POSITIVE_CONTEXT:
        if phrase in sentence:
            return 1.0

    # check weak context
    for phrase in WEAK_CONTEXT:
        if phrase in sentence:
            return 0.5

    # default: skill exists in resume, so count it as a skill match
    return 1.0


def calculate_skill_score_details(resume_text, skills):
    """
    Calculate skill score and return explainable alias details.
    """
    if not skills:
        return {
            "score": 0,
            "matched_skills": [],
            "matched_aliases": [],
            "missing_skills": [],
        }

    total_score = 0
    sentences = split_sentences(resume_text)
    matched_skills = []
    matched_aliases = []
    missing_skills = []

    for skill in skills:
        best_score_for_skill = 0
        best_alias_for_skill = None

        for sentence in sentences:
            context_score = get_skill_context_score(sentence, skill)
            matching_alias = find_matching_alias(sentence, skill)

            if context_score > best_score_for_skill:
                best_score_for_skill = context_score
                best_alias_for_skill = matching_alias

        total_score += best_score_for_skill

        if best_score_for_skill > 0:
            matched_skills.append(skill)
            matched_aliases.append(f"{skill}={best_alias_for_skill}")
        else:
            missing_skills.append(skill)

    return {
        "score": (total_score / len(skills)) * 100,
        "matched_skills": matched_skills,
        "matched_aliases": matched_aliases,
        "missing_skills": missing_skills,
    }


def calculate_skill_score(resume_text, skills):
    """
    Backward-compatible function.
    Returns only the percentage score.
    """
    return calculate_skill_score_details(resume_text, skills)["score"]


# ---------------- ML resume-job similarity scoring ----------------------------
def calculate_ml_similarity_score(resume_text, job_description_text):
    """
    Calculate resume-to-job-description similarity using TF-IDF + cosine similarity.

    This is a lightweight machine learning/NLP method:
    - TF-IDF learns which words/phrases are important in the resume and JD.
    - Cosine similarity compares both vectors.

    Returns a score from 0 to 100.
    """
    resume_text = (resume_text or "").strip()
    job_description_text = (job_description_text or "").strip()

    if not resume_text or not job_description_text:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        vectors = vectorizer.fit_transform([resume_text, job_description_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity * 100)

    except Exception:
        # Safe fallback if scikit-learn is not installed.
        resume_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9.+#]*", resume_text.lower()))
        jd_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9.+#]*", job_description_text.lower()))

        if not resume_words or not jd_words:
            return 0.0

        overlap = resume_words.intersection(jd_words)
        return float((len(overlap) / len(jd_words)) * 100)


# ---------------- hybrid scoring ----------------------------
def calculate_hybrid_score(rule_based_score, ml_similarity_score, rule_weight=0.7, ml_weight=0.3):
    """
        Hybrid Score = 70% rule-based score + 30% ML similarity score
    """
    try:
        rule_based_score = float(rule_based_score or 0)
        ml_similarity_score = float(ml_similarity_score or 0)
        rule_weight = float(rule_weight)
        ml_weight = float(ml_weight)
    except (TypeError, ValueError):
        return 0.0

    total_weight = rule_weight + ml_weight
    if total_weight <= 0:
        return 0.0

    normalized_rule_weight = rule_weight / total_weight
    normalized_ml_weight = ml_weight / total_weight

    score = (normalized_rule_weight * rule_based_score) + (normalized_ml_weight * ml_similarity_score)
    return max(0.0, min(100.0, score))


# ---------------- recommendation labels ----------------------------
def get_recommendation_label(hybrid_score):
    """
        80 - 100 : Highly Recommended
        65 - 79  : Recommended
        50 - 64  : Needs Review
        0 - 49   : Not Suitable
    """
    try:
        score = float(hybrid_score or 0)
    except (TypeError, ValueError):
        score = 0.0

    if score >= 80:
        return "Highly Recommended"
    if score >= 65:
        return "Recommended"
    if score >= 50:
        return "Needs Review"
    return "Not Suitable"


def get_recommendation_reason(label):
    reasons = {
        "Highly Recommended": "Strong match based on rule-based scoring and ML similarity.",
        "Recommended": "Good match, but some skills or requirements may still need review.",
        "Needs Review": "Partial match. Recruiter should manually check experience and missing skills.",
        "Not Suitable": "Low match for the current job description and required profile.",
    }
    return reasons.get(label, "Manual review recommended.")

# ---------------- missing skills suggestions ----------------------------
def find_missing_jd_keywords(resume_text, jd_keywords, max_missing=12):
    resume_text = normalize_for_alias_matching(resume_text or "")
    missing = []
    seen = set()

    for keyword in jd_keywords or []:
        keyword = str(keyword).lower().strip()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)

        # Very generic words are not useful as suggestions.
        if keyword in {"software", "engineer", "developer", "candidate", "role", "team"}:
            continue

        if not contains_term(resume_text, keyword):
            missing.append(keyword)

        if len(missing) >= max_missing:
            break

    return missing


def build_missing_skills_suggestion(missing_default_skills=None, missing_jd_keywords=None):
    missing_default_skills = missing_default_skills or []
    missing_jd_keywords = missing_jd_keywords or []

    combined = []
    for item in list(missing_default_skills) + list(missing_jd_keywords):
        item = str(item).strip()
        if item and item not in combined:
            combined.append(item)

    if not combined:
        return "No major missing skills detected from the current job profile and job description."

    top_items = combined[:8]
    return "Candidate may improve or clarify these areas: " + ", ".join(top_items) + "."



# ---------------- explainability helpers ----------------------------
def find_matched_keywords(text, keywords, max_items=20):
    text = normalize_for_alias_matching(text or "")
    matched = []
    seen = set()

    for keyword in keywords or []:
        keyword = str(keyword).lower().strip()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)

        if contains_term(text, keyword) or keyword in text:
            matched.append(keyword)

        if len(matched) >= max_items:
            break

    return matched


def build_score_explanation(
    skill_score,
    experience_score,
    education_score,
    rule_based_score,
    ml_similarity_score,
    hybrid_score,
    recommendation_label,
    matched_skill_aliases=None,
    missing_skills=None,
    matched_experience_keywords=None,
    matched_education_keywords=None,
    missing_jd_keywords=None,
    supervised_ml_score=0.0,
    training_dataset_used=False,
):
    matched_skill_aliases = matched_skill_aliases or []
    missing_skills = missing_skills or []
    matched_experience_keywords = matched_experience_keywords or []
    matched_education_keywords = matched_education_keywords or []
    missing_jd_keywords = missing_jd_keywords or []

    summary_parts = []
    if matched_skill_aliases:
        summary_parts.append("Matched skills: " + ", ".join(matched_skill_aliases[:6]))
    else:
        summary_parts.append("No default skill matches detected")

    if missing_skills:
        summary_parts.append("Missing default skills: " + ", ".join(missing_skills[:6]))

    if matched_experience_keywords:
        summary_parts.append("Matched experience/JD keywords: " + ", ".join(matched_experience_keywords[:6]))

    if missing_jd_keywords:
        summary_parts.append("Missing JD keywords: " + ", ".join(missing_jd_keywords[:6]))

    short_summary = " | ".join(summary_parts)

    detailed_lines = [
        "Score breakdown:",
        f"- Skill Score: {skill_score:.2f}%",
        f"- Experience Score: {experience_score:.2f}%",
        f"- Education Score: {education_score:.2f}%",
        f"- Rule-Based Final Score: {rule_based_score:.2f}%",
        f"- ML Similarity Score: {ml_similarity_score:.2f}%",
        f"- Hybrid Final Score: {hybrid_score:.2f}%",
        f"- Recommendation: {recommendation_label}",
        "",
        "Evidence found:",
        "- Matched skill aliases: " + (", ".join(matched_skill_aliases) if matched_skill_aliases else "None"),
        "- Matched experience/JD keywords: " + (", ".join(matched_experience_keywords) if matched_experience_keywords else "None"),
        "- Matched education keywords: " + (", ".join(matched_education_keywords) if matched_education_keywords else "None"),
        "",
        "Gaps found:",
        "- Missing default skills: " + (", ".join(missing_skills) if missing_skills else "None"),
        "- Missing JD keywords: " + (", ".join(missing_jd_keywords) if missing_jd_keywords else "None"),
    ]

    if training_dataset_used:
        detailed_lines.extend([
            "",
            "Optional supervised ML:",
            f"- Supervised ML Score: {supervised_ml_score:.2f}%",
        ])

    detailed_lines.extend([
        "",
        "Formula used:",
        "- Rule-Based Score = 50% skills + 30% experience + 20% education",
        "- Hybrid Final Score = 70% rule-based score + 30% ML similarity score",
    ])

    return {
        "summary": short_summary,
        "details": "\n".join(detailed_lines),
    }


# ---------------- supervised training dataset option ----------------------------
def normalize_training_label(label):
    """
    Convert common label formats into 1 = selected/suitable and 0 = rejected/not suitable.
    Accepted positive examples: selected, hire, hired, suitable, shortlisted, recommended, 1, yes, true.
    Accepted negative examples: rejected, not selected, not suitable, unsuitable, 0, no, false.
    """
    value = str(label).strip().lower()

    positive_values = {
        "1", "yes", "true", "selected", "select", "hire", "hired",
        "suitable", "shortlisted", "recommended", "pass", "accepted"
    }
    negative_values = {
        "0", "no", "false", "rejected", "reject", "not selected",
        "not suitable", "unsuitable", "not shortlisted", "fail", "declined"
    }

    if value in positive_values:
        return 1
    if value in negative_values:
        return 0
    return None


def _get_case_insensitive_column(dataframe, possible_names):
    lookup = {str(col).strip().lower(): col for col in dataframe.columns}
    for name in possible_names:
        key = name.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def train_selection_model(training_dataframe):
    """
    Train a simple supervised ML model from a CSV/DataFrame.

    Required columns, case-insensitive:
        resume_text, job_description, label

    The model learns from historical examples where label means selected/not selected.
    Returns a dictionary containing training status and the trained pipeline.
    """
    if training_dataframe is None or getattr(training_dataframe, "empty", True):
        return {
            "is_trained": False,
            "message": "No training data provided.",
            "model": None,
            "rows_used": 0,
        }

    resume_col = _get_case_insensitive_column(training_dataframe, ["resume_text", "resume", "cv_text", "candidate_resume"])
    jd_col = _get_case_insensitive_column(training_dataframe, ["job_description", "jd", "job_desc", "description"])
    label_col = _get_case_insensitive_column(training_dataframe, ["label", "selected", "status", "decision", "outcome"])

    missing_columns = []
    if resume_col is None:
        missing_columns.append("resume_text")
    if jd_col is None:
        missing_columns.append("job_description")
    if label_col is None:
        missing_columns.append("label")

    if missing_columns:
        return {
            "is_trained": False,
            "message": "Training CSV missing required column(s): " + ", ".join(missing_columns),
            "model": None,
            "rows_used": 0,
        }

    texts = []
    labels = []

    for _, row in training_dataframe.iterrows():
        resume_text = str(row.get(resume_col, "") or "").strip()
        jd_text = str(row.get(jd_col, "") or "").strip()
        label = normalize_training_label(row.get(label_col, ""))

        if not resume_text or not jd_text or label is None:
            continue

        texts.append("Resume:\n" + resume_text + "\n\nJob Description:\n" + jd_text)
        labels.append(label)

    if len(texts) < 4:
        return {
            "is_trained": False,
            "message": "Need at least 4 valid training rows with resume_text, job_description, and label.",
            "model": None,
            "rows_used": len(texts),
        }

    if len(set(labels)) < 2:
        return {
            "is_trained": False,
            "message": "Training data must contain both positive and negative labels.",
            "model": None,
            "rows_used": len(texts),
        }

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        model = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        model.fit(texts, labels)

        return {
            "is_trained": True,
            "message": f"Training model ready. Rows used: {len(texts)}.",
            "model": model,
            "rows_used": len(texts),
        }

    except Exception as exc:
        return {
            "is_trained": False,
            "message": f"Could not train model: {exc}",
            "model": None,
            "rows_used": len(texts),
        }


def predict_selection_probability(model_info, resume_text, job_description_text):
    """
    Predict candidate suitability using the optional supervised training dataset model.
    Returns a percentage from 0 to 100.
    """
    if not model_info or not model_info.get("is_trained") or model_info.get("model") is None:
        return 0.0

    resume_text = (resume_text or "").strip()
    job_description_text = (job_description_text or "").strip()
    if not resume_text or not job_description_text:
        return 0.0

    text = "Resume:\n" + resume_text + "\n\nJob Description:\n" + job_description_text

    try:
        model = model_info["model"]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([text])[0][1]
            return float(probability * 100)

        prediction = model.predict([text])[0]
        return 100.0 if int(prediction) == 1 else 0.0

    except Exception:
        return 0.0
