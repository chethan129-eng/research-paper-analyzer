import streamlit as st
import numpy as np
import re

# Try PDF reading
try:
    from PyPDF2 import PdfReader
except:
    PdfReader = None

# ==============================
# TEXT EXTRACTION
# ==============================

def extract_text(file):
    text = ""

    if PdfReader:
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except:
            text = ""

    return text


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(text):
    words = text.split()

    abstract_length = len(words)

    title_length = len(words[:15])  # first line approx

    # fake venue length (based on keywords)
    venue_length = len(re.findall(r'conference|journal|ieee|springer', text.lower()))

    # citation proxy (count references)
    citation_count = len(re.findall(r'\[\d+\]', text))

    log_citation = np.log1p(citation_count)

    recency = 2025 - 2000  # assume modern paper

    return np.array([[abstract_length, title_length, venue_length, log_citation, recency]])


# ==============================
# SMART SCORING (ML-LIKE LOGIC)
# ==============================
def predict_rating(features):
    abstract_length, title_length, venue_length, log_citation, recency = features[0]

    # Better balanced scoring
    score = (
        min(abstract_length / 1000, 5) +   # cap long papers
        title_length * 0.1 +
        venue_length * 1.0 +
        log_citation * 2.0 +
        recency * 0.05
    )

    # Normalize to 1–10 properly
    rating = np.clip(score, 1, 10)

    return round(rating, 2)
def verdict(rating):
    if rating >= 8:
        return "Highly Recommended ✅"
    elif rating >= 5:
        return "Average ⚠️"
    else:
        return "Not Recommended ❌"


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Research Paper Analyzer", layout="centered")

st.title("📄 Research Paper Analyzer")
st.write("Upload a research paper PDF to get rating")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    st.info("Processing PDF...")

    text = extract_text(uploaded_file)

    if len(text.strip()) < 50:
        st.error("⚠️ Unable to extract text properly from PDF")
    else:
        features = extract_features(text)

        rating = predict_rating(features)

        result = verdict(rating)

        # ==============================
        # OUTPUT
        # ==============================

        st.subheader("📊 Analysis Result")

        st.write("### ⭐ Rating:", rating, "/ 10")

        st.write("### 📈 Verdict:", result)

        st.write("### 📄 Extracted Info")

        st.write("Text Length:", len(text.split()), "words")

        st.success("Analysis Complete ✅")
