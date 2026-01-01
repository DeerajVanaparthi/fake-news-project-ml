import streamlit as st
import re
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Fake News Detection System")

news = st.text_area("Enter News Text")

if st.button("Check"):
    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        cleaned = clean_text(news)
        if len(cleaned.split()) < 20:
            st.warning("Please enter a longer news article (minimum 20 words).")
            st.stop()

        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100


        if pred == 0:
            st.error(f"FAKE NEWS ❌ (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"REAL NEWS ✅ (Confidence: {confidence:.2f}%)")

