<div align="center">

<img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.56-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-75.04%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/F1%20Score-74.93%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/AUC--ROC-0.9473-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>

---

# 🧠 MindBridge
### Mental Health Text Classification — 7-Class NLP Pipeline

*Classifying mental health status from social media text using TF-IDF and classical ML*

[Live Demo](#streamlit-app) · [Paper](#paper) · [Results](#results) · [Setup](#setup)

</div>

---

## Overview

**MindBridge** is an end-to-end NLP pipeline that classifies user-generated text into 7 mental health categories using TF-IDF vectorization and classical machine learning models. The project demonstrates that lightweight, interpretable ML approaches can achieve strong performance on fine-grained mental health text classification without requiring deep learning infrastructure.

**Target Classes:** Normal · Depression · Anxiety · Stress · Suicidal · Bipolar · Personality Disorder

---

## Results

### Model Comparison

| Model | Accuracy | Weighted F1 | AUC-ROC |
|-------|----------|-------------|---------|
| Naive Bayes | 69.51% | 68.98% | 0.9264 |
| Random Forest | 70.25% | 68.26% | 0.9332 |
| Linear SVM | 74.06% | 73.71% | 0.9335 |
| **Logistic Regression (ours)** | **75.04%** | **74.93%** | **0.9473** |

### Ablation Study — TF-IDF Configuration

| Configuration | Weighted F1 |
|--------------|-------------|
| Unigrams only | 0.7381 |
| Unigrams + Bigrams | 0.7433 |
| + Sublinear TF scaling | 0.7471 |
| + Vocabulary 20k | **0.7471** |

### Key Findings
- Logistic Regression achieves the best performance across all metrics
- Mean AUC-ROC of **0.9473** indicates excellent multi-class discrimination
- Biggest confusion: Depression ↔ Suicidal (clinically meaningful overlap)
- Personality Disorder has lowest recall due to severe class imbalance (2.0%)

---

## Project Structure

```
MIND_BRIDGE/
│
├── data/
│   └── Combined Data.csv          # Kaggle dataset (53k samples)
│
├── notebooks/
│   ├── 00_Problem_Understanding.ipynb
│   ├── 01_EDA.ipynb                # Class distribution, word clouds, top words
│   ├── 02_Preprocessing.ipynb      # Text cleaning, TF-IDF, SMOTE
│   ├── 03_Modeling.ipynb           # Train 4 models, compare results
│   └── 04_Evaluation.ipynb         # Tuning, ablation, ROC-AUC, interpretability
│
├── models/
│   ├── best_model.pkl              # Logistic Regression (C=5.0)
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
│   ├── label_encoder.pkl           # Class label encoder
│   ├── svm_model.pkl
│   ├── naive_bayes_model.pkl
│   └── random_forest_model.pkl
│
├── results/
│   └── plots/                      # All saved visualizations
│       ├── class_distribution.png
│       ├── wordclouds.png
│       ├── top_words_per_class.png
│       ├── model_comparison.png
│       ├── confusion_matrix_logistic_regression.png
│       ├── roc_auc_curves.png
│       ├── ablation_study.png
│       └── top_features_per_class.png
│
├── app.py                          # Streamlit web application
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) |
| Total samples | ~53,000 |
| After cleaning | ~51,000 |
| Features | `statement` (text), `status` (label) |
| Classes | 7 |
| Split | 70% train / 15% val / 15% test (stratified) |

### Class Distribution

| Class | Samples | % of Data |
|-------|---------|-----------|
| Normal | 16,343 | 31.0% |
| Depression | 15,404 | 29.2% |
| Suicidal | 10,652 | 20.2% |
| Anxiety | 3,841 | 7.3% |
| Bipolar | 2,777 | 5.3% |
| Stress | 2,587 | 4.9% |
| Personality disorder | 1,077 | 2.0% |

---

## Pipeline

```
Raw Text
   ↓
Text Cleaning (lowercase, URLs, punctuation, stopwords, lemmatization)
   ↓
TF-IDF Vectorization (20k features, unigrams + bigrams, sublinear TF)
   ↓
Train/Val/Test Split (70/15/15, stratified)
   ↓
Model Training (LR, NB, RF, SVM)
   ↓
Evaluation (Accuracy, F1, AUC-ROC, Confusion Matrix)
   ↓
Best Model: Logistic Regression (C=5.0, class_weight='balanced')
```

---

## Streamlit App

A live web application for real-time mental health text classification.

**Features:**
- Real-time 7-class prediction
- Confidence scores with visual bars
- Color-coded results per mental health category
- Crisis resources for high-risk predictions
- Text statistics panel

**Run locally:**
```bash
streamlit run app.py
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Vaibhav2040/MindBridge.git
cd MindBridge

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# 5. Run notebooks in order
jupyter notebook notebooks/01_EDA.ipynb

# 6. Launch web app
streamlit run app.py
```

---

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| ML | scikit-learn, imbalanced-learn |
| NLP | NLTK, TF-IDF |
| Visualization | matplotlib, seaborn, wordcloud |
| Web App | Streamlit |
| Data | pandas, numpy, scipy |
| Version Control | Git, GitHub |

---

## Paper

This project is accompanied by a research paper:

**MindBridge: Mental Health Text Classification Using TF-IDF and Machine Learning**
Vaibhav Hasmukh Bhai Patel · Siddharth Sunil Jadhav

---

## Authors

| Name | Student ID | Contributions |
|------|-----------|---------------|
| Vaibhav Hasmukh Bhai Patel | U01130755 | Conceptualization, Methodology, Software, Visualization, Writing |
| Siddharth Sunil Jadhav | U01108649 | Data Curation, Preprocessing, Validation, Writing — Review |

---

## Disclaimer

This project is for research and educational purposes only. MindBridge is not a medical diagnostic tool and should not be used as a substitute for professional mental health evaluation or treatment.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>MindBridge · github.com/Vaibhav2040/MindBridge</sub>
</div>
