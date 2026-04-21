<div align="center">

<img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.56-red?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-73.55%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/F1%20Score-73.45%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/AUC--ROC-0.9365-brightgreen?style=for-the-badge"/>
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

| Model | Accuracy | Weighted F1 | Macro F1 |
|-------|----------|-------------|----------|
| Naive Bayes | 67.78% | 67.14% | 55.67% |
| Random Forest | 69.08% | 66.91% | 52.99% |
| Linear SVM | 72.61% | 72.25% | 66.66% |
| **Logistic Regression (ours)** | **73.55%** | **73.45%** | **68.04%** |

### Per-Class Performance (Tuned Logistic Regression)

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.88 | 0.91 | 0.89 | 2,406 |
| Anxiety | 0.74 | 0.83 | 0.78 | 544 |
| Bipolar | 0.70 | 0.77 | 0.73 | 375 |
| Suicidal | 0.65 | 0.70 | 0.68 | 1,596 |
| Depression | 0.75 | 0.57 | 0.65 | 2,264 |
| Personality Disorder | 0.46 | 0.57 | 0.51 | 134 |
| Stress | 0.40 | 0.62 | 0.49 | 345 |

### Ablation Study — TF-IDF Configuration

| Configuration | Weighted F1 | Macro F1 |
|--------------|-------------|----------|
| Unigrams only | 0.7147 | 0.6439 |
| Unigrams + Bigrams | 0.7292 | 0.6664 |
| + Sublinear TF scaling | 0.7327 | 0.6714 |
| Vocab 10k | 0.7252 | 0.6603 |
| **Final config (20k)** | **0.7327** | **0.6714** |

### Key Findings
- Logistic Regression achieves the best performance across all metrics
- Mean AUC-ROC of **0.9365** indicates strong multi-class discrimination
- Biggest confusion: Depression ↔ Suicidal (548 + 368 misclassifications — clinically meaningful overlap)
- Personality Disorder and Stress have lowest F1 due to severe class imbalance and ambiguous vocabulary
- Every TF-IDF design choice (bigrams, sublinear TF, 20k vocab) is justified by ablation results

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
│       ├── cleaning_quality.png
│       ├── roc_auc_curves.png
│       ├── ablation_study.png
│       ├── per_class_f1.png
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
| After cleaning | 51,093 |
| Features | `statement` (text), `status` (label) |
| Classes | 7 |
| Split | 70% train / 15% val / 15% test (stratified) |

### Class Distribution

| Class | Samples | % of Data | Avg Words |
|-------|---------|-----------|-----------|
| Normal | 16,040 | 31.4% | 17 |
| Depression | 15,094 | 29.5% | 168 |
| Suicidal | 10,644 | 20.8% | 147 |
| Anxiety | 3,623 | 7.1% | 143 |
| Bipolar | 2,501 | 4.9% | 178 |
| Stress | 2,296 | 4.5% | 112 |
| Personality Disorder | 895 | 1.8% | 178 |

---

## Pipeline

```
Raw Text
   ↓
Text Cleaning (lowercase, URLs, punctuation, stopwords, lemmatization)
   ↓  Average word reduction: 52% (113 → 50 words)
TF-IDF Vectorization (20k features, unigrams + bigrams, sublinear TF)
   ↓  Matrix sparsity: 99.75%
SMOTE Oversampling (training set only → 11,228 samples per class)
   ↓
Train/Val/Test Split (70/15/15, stratified, seed=42)
   ↓
Model Training (LR, NB, RF, SVM — all with balanced class weights)
   ↓
Hyperparameter Tuning (3-fold GridSearchCV on C parameter)
   ↓
Evaluation (Accuracy, Weighted F1, Macro F1, AUC-ROC, Confusion Matrix)
   ↓
Best Model: Logistic Regression (C=5.0, class_weight='balanced')
```

---

## Streamlit App

A live web application for real-time mental health text classification.

**Features:**
- Real-time 7-class prediction with confidence scores
- Visual confidence bars for all classes
- Color-coded results per mental health category
- Crisis resources for high-risk predictions
- Prediction history with session tracking
- Text statistics panel (word count, character count, clean tokens)
- Export prediction history as CSV

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
jupyter notebook notebooks/00_Problem_Understanding.ipynb

# 6. Launch web app
streamlit run app.py
```

---

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| ML | scikit-learn, imbalanced-learn (SMOTE) |
| NLP | NLTK, TF-IDF |
| Visualization | matplotlib, seaborn, wordcloud |
| Web App | Streamlit |
| Data | pandas, numpy, scipy |
| Serialization | joblib |
| Version Control | Git, GitHub |

---

## Paper

This project is accompanied by a research paper:

**MindBridge: Multi-Class Mental Health Text Classification Using Classical Machine Learning**
Vaibhav Patel · Siddharth Jadhav

---

## Authors

| Name | Student ID | Contributions |
|------|-----------|---------------|
| Vaibhav Patel | U01130755 | Conceptualization, Methodology, Software, Visualization, Writing |
| Siddharth Jadhav | U01108649 | Data Curation, Preprocessing, Validation, Writing — Review |

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
