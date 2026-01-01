import re
import joblib

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

news = input("Enter news text: ")
news_clean = clean_text(news)
if len(news_clean.split()) < 20:
    print("⚠ Please enter a longer news article (at least 20 words).")
    exit()


news_vec = vectorizer.transform([news_clean])
pred = model.predict(news_vec)[0]
proba = model.predict_proba(news_vec)[0]
confidence = max(proba) * 100


print("\nResult:")
print("FAKE" if pred == 0 else "REAL")
print(f"Confidence: {confidence:.2f}%")


