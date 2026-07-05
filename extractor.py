import re
import string

import PyPDF2
import docx

try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None


# ---------------- function preprocess_text ----------------------------
def preprocess_text(text):
    """
    Clean text for keyword matching.
    Use this for keyword scoring only, not for sentence/context analysis.
    """
    fallback_stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "with", "you", "your"
    }

    try:
        stop_words = set(stopwords.words("english")) if stopwords else fallback_stop_words
    except LookupError:
        stop_words = fallback_stop_words

    # remove punctuation for keyword matching
    text = text.translate(str.maketrans("", "", string.punctuation))

    # tokenize and remove stopwords
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]

    return " ".join(cleaned_words)


# ---------------- function normalize_raw_text ----------------------------
def normalize_raw_text(text):
    """
    Keep useful structure for context/section detection:
    - preserves line breaks
    - preserves punctuation
    - fixes weird PDF spacing
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # fix cases like P y t h o n -> Python
    text = re.sub(r"(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)", "", text)

    # normalize spaces but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------- function normalize_text ----------------------------
def normalize_text(text):
    """
    Flatten text for keyword matching.
    """
    text = re.sub(r"(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- function extract_raw_text ----------------------------
def extract_raw_text(file_path):
    """
    Extract resume text while preserving line breaks and punctuation.
    Use this for section detection, sentence context, and future ML/embedding work.
    """
    text = ""

    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return normalize_raw_text(text).lower()


# ---------------- function clean_for_keyword_matching ----------------------------
def clean_for_keyword_matching(raw_text):
    """
    Convert raw resume text into a cleaned version for keyword matching.
    """
    text = raw_text.lower()
    text = normalize_text(text)
    text = preprocess_text(text)
    return text


# ---------------- function extract_resume_texts ----------------------------
def extract_resume_texts(file_path):
    """
    Return both versions of resume text:
    raw_text   = keeps sentence/section meaning
    clean_text = optimized for keyword matching
    """
    raw_text = extract_raw_text(file_path)
    clean_text = clean_for_keyword_matching(raw_text)

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
    }


# ---------------- function extract_text ----------------------------
def extract_text(file_path):
    """
    Backward-compatible function for older code.
    Returns cleaned text for keyword matching.
    """
    return extract_resume_texts(file_path)["clean_text"]
