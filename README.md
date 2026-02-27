# 📰 VeriNews AI

![Python](https://img.shields.io/badge/Backend-Python-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-black)
![Machine Learning](https://img.shields.io/badge/Model-SVM-success)
![NLP](https://img.shields.io/badge/NLP-TFIDF-orange)
![Status](https://img.shields.io/badge/Project-Production--Ready-brightgreen)

A production-ready machine learning web application that detects fake news using advanced linguistic signal analysis and a Linear Support Vector Machine (SVM) classifier.

---

## 🔍 Overview

**VeriNews AI** analyzes news articles and evaluates their authenticity using a combination of statistical Natural Language Processing (NLP) techniques and interpretable linguistic features.

The system provides:

- Automated fake/real classification  
- Confidence percentage scoring  
- Transparent linguistic insights  
- Real-time processing visualization  

This project combines predictive modeling with explainability to make AI decisions more interpretable.

---

## 🧠 How It Works

1. Input news text is preprocessed and cleaned.
2. TF-IDF vectors are generated from the text.
3. Linguistic features are extracted:
   - Word count
   - All-caps ratio
   - Question mark frequency
   - Average word length
4. TF-IDF features and engineered linguistic features are combined.
5. A trained Linear SVM model performs classification.
6. A confidence score is calculated and displayed to the user.

---

## ✨ Features

- Real-time news analysis  
- Confidence percentage scoring  
- Linguistic transparency panel  
- Terminal-style processing visualization  
- Minimalist professional UI  
- Production deployment ready  

---

## 🛠 Tech Stack

### Backend
- Python  
- Flask  
- Scikit-learn  
- spaCy  
- NumPy  
- SciPy  

### Frontend
- HTML  
- Tailwind CSS  
- Vanilla JavaScript  

### Deployment
- Gunicorn  
- Render  

---

## 📦 Installation (Local Development)

### 1️⃣ Clone the Repository

    git clone https://github.com/YOUR_USERNAME/verinews-ai.git
    cd verinews-ai

### 2️⃣ Create Virtual Environment

    python -m venv venv

### 3️⃣ Activate Virtual Environment

Mac/Linux:

    source venv/bin/activate

Windows:

    venv\Scripts\activate

### 4️⃣ Install Dependencies

    pip install -r requirements.txt

### 5️⃣ Download spaCy Model

    python -m spacy download en_core_web_sm

### 6️⃣ Run the Application

    python app.py

Visit in browser:

    http://127.0.0.1:5000

---

## 🚀 Deployment

This project is configured for production deployment using:

    gunicorn app:app

Recommended platform: Render

---

## 📁 Project Structure

    verinews-ai/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── models/
    │   └── fake_news_model.pkl
    ├── templates/
    │   └── index.html

---

## 📊 Model Details

- Model Type: Linear Support Vector Machine (SVM)  
- Vectorization: TF-IDF  
- Feature Engineering: Linguistic signal metrics  
- Output: Binary Classification (Fake / Real)  
- Confidence: Decision function based scoring  

---

## 🎯 Key Concepts Demonstrated

- Machine Learning pipeline design  
- Feature engineering  
- Text vectorization  
- NLP preprocessing  
- Model explainability  
- Web application integration  
- Production deployment  

---

## 🔮 Future Improvements

- Transformer-based model (BERT / RoBERTa)  
- Multi-language support  
- API endpoint versioning  
- Model retraining pipeline  
- User authentication and logging  
- Analytics dashboard  

---

## 👨‍💻 Author

Harsh Vardhan  
VeriNews AI – Fake News Detection System  

---

## 📄 License

This project is intended for educational and demonstration purposes.
