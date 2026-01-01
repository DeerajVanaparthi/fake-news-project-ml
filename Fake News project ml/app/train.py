import numpy as np
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths to dataset
fake_path = "../data/Fake.csv"
true_path = "../data/True.csv"

# Load files
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine both
data = pd.concat([fake, true], ignore_index=True)



# Create a safe text column
if "text" in data.columns and "title" in data.columns:
    data["full_text"] = data["title"].fillna("") + " " + data["text"].fillna("")
elif "text" in data.columns:
    data["full_text"] = data["text"].fillna("")
elif "title" in data.columns:
    data["full_text"] = data["title"].fillna("")
else:
    raise ValueError("Dataset must contain 'text' or 'title' column.")

# =============================
# DATASET STATISTICS
# =============================

print("\nDataset Statistics:")

# Count fake vs real news
print("Number of Fake news articles:", (data["label"] == 0).sum())
print("Number of Real news articles:", (data["label"] == 1).sum())

# Average article length (in words)
data["length"] = data["full_text"].apply(lambda x: len(x.split()))
print("Average article length:", int(data["length"].mean()), "words")


# Cleaning function
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["full_text"] = data["full_text"].apply(clean_text)

X = data["full_text"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)   # ✅ ONLY transform



# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
from sklearn.metrics import accuracy_score, classification_report

train_acc = model.score(X_train_vec, y_train)
test_acc = model.score(X_test_vec, y_test)

print("Training Accuracy:", round(train_acc, 4))
print("Testing Accuracy:", round(test_acc, 4))


# Predict on test data
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_fake = np.argsort(coefficients)[:10]
top_real = np.argsort(coefficients)[-10:]

print("\nTop Fake News Indicators:")
for i in top_fake:
    print(feature_names[i])

print("\nTop Real News Indicators:")
for i in top_real:
    print(feature_names[i])


# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Training complete!")
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Saved in app folder:")
print("fake_news_model.pkl")
print("tfidf_vectorizer.pkl")

import joblib

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
