import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

MODEL_PATH = "models/fake_news_model.pkl"

def train():

    print("Loading dataset...")

    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data.sample(frac=1, random_state=42)

    X = data["text"]
    y = data["label"]

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    print("Training SVM...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_vec, y)

    print("Saving model...")
    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": model,
        "vectorizer": vectorizer
    }, MODEL_PATH)

    print("Training complete.")

if __name__ == "__main__":
    train()