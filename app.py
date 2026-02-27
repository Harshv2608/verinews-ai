from flask import Flask, render_template, request
import joblib
import numpy as np
from scipy.sparse import hstack
import spacy
import re
import os
import joblib
from train_model import train

MODEL_PATH = "models/fake_news_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Training automatically...")
    train()

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
vectorizer = model_data["vectorizer"]
app = Flask(__name__)

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Linguistic Feature Extraction
# -----------------------------
def extract_linguistic_features(text):
    doc = nlp(text)

    words = [token.text for token in doc]
    total_words = len(words)

    superlatives = sum(1 for token in doc if token.tag_ == "JJS")
    proper_nouns = sum(1 for token in doc if token.pos_ == "PROPN")
    first_person = sum(1 for token in doc if token.text.lower() in ["i", "we", "us", "our"])
    exclamations = text.count("!")
    questions = text.count("?")
    all_caps = sum(1 for word in words if word.isupper())

    sentences = list(doc.sents)
    avg_sentence_length = total_words / len(sentences) if sentences else 0

    return np.array([[
        superlatives / total_words if total_words else 0,
        proper_nouns / total_words if total_words else 0,
        first_person / total_words if total_words else 0,
        exclamations / total_words if total_words else 0,
        all_caps / total_words if total_words else 0,
        questions / total_words if total_words else 0,
        avg_sentence_length
    ]])


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    word_count = 0
    all_caps_ratio = 0
    question_ratio = 0
    avg_word_length = 0

    if request.method == "POST":

        try:
            text = request.form.get("news_text", "").strip()

            if not text:
                error = "Please enter news content before analyzing."
                return render_template("index.html", error=error)

            # ---------- Basic UI Features ----------
            words = re.findall(r'\b\w+\b', text)
            word_count = len(words)

            all_caps_words = [w for w in words if w.isupper()]
            all_caps_ratio = len(all_caps_words) / word_count if word_count else 0

            question_marks = text.count("?")
            question_ratio = question_marks / word_count if word_count else 0

            avg_word_length = np.mean([len(w) for w in words]) if word_count else 0

            # ---------- Model Prediction ----------
            tfidf_features = vectorizer.transform([text])
            linguistic_features = extract_linguistic_features(text)

            combined_features = hstack([tfidf_features, linguistic_features])

            result = model.predict(combined_features)[0]
            probability = model.predict_proba(combined_features)[0]

            confidence = round(max(probability) * 100, 2)
            prediction = "Fake News" if result == 0 else "Real News"

        except Exception as e:
            error = "Something went wrong during analysis. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        word_count=word_count,
        all_caps_ratio=round(all_caps_ratio, 4),
        question_ratio=round(question_ratio, 4),
        avg_word_length=round(avg_word_length, 2),
        error=error
    )

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)