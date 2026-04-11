import streamlit as st
import joblib
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Page config
st.set_page_config(
    page_title="MindBridge",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
    .title-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-left: 5px solid #6C63FF;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .result-card {
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .confidence-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        height: 10px;
        margin: 4px 0 12px 0;
        overflow: hidden;
    }
    .stTextArea textarea {
        background-color: #16213e !important;
        color: white !important;
        border: 1px solid #6C63FF !important;
        border-radius: 8px !important;
        font-size: 15px !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #6C63FF, #4cc9f0);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }
    .stButton button:hover {
        opacity: 0.9;
    }
    h1, h2, h3, p, label { color: white !important; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    le = joblib.load('models/label_encoder.pkl')
    model = joblib.load('models/best_model.pkl')
    return vectorizer, le, model


def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


CLASS_INFO = {
    'Normal': {
        'color': '#06d6a0',
        'icon': '✅',
        'desc': 'The text shows no significant signs of mental health distress.',
        'advice': 'Keep maintaining healthy habits and social connections.'
    },
    'Depression': {
        'color': '#4cc9f0',
        'icon': '💙',
        'desc': 'The text contains indicators associated with depression.',
        'advice': 'Consider speaking with a mental health professional. You are not alone.'
    },
    'Anxiety': {
        'color': '#f77f00',
        'icon': '🟡',
        'desc': 'The text shows signs associated with anxiety.',
        'advice': 'Breathing exercises and mindfulness can help. Consider professional support.'
    },
    'Suicidal': {
        'color': '#ff6b6b',
        'icon': '🔴',
        'desc': 'The text contains language associated with suicidal ideation.',
        'advice': 'Please reach out immediately. Crisis line: 988 (US) | International: findahelpline.com'
    },
    'Bipolar': {
        'color': '#f72585',
        'icon': '🟣',
        'desc': 'The text shows patterns associated with bipolar disorder.',
        'advice': 'Professional diagnosis is important. Please consult a psychiatrist.'
    },
    'Stress': {
        'color': '#48CAE4',
        'icon': '🔵',
        'desc': 'The text reflects high levels of stress.',
        'advice': 'Try to identify stressors and consider stress management techniques.'
    },
    'Personality disorder': {
        'color': '#DDA0DD',
        'icon': '🟤',
        'desc': 'The text shows patterns associated with personality disorders.',
        'advice': 'Professional evaluation is recommended for accurate diagnosis.'
    }
}


# Header
st.markdown("""
<div class="title-card">
    <h1 style="margin:0; font-size:32px; color:#6C63FF !important;">🧠 MindBridge</h1>
    <p style="margin:8px 0 0; color:#aaa !important; font-size:15px;">
        Mental Health Text Classification — 7-Class NLP Pipeline
    </p>
    <p style="margin:4px 0 0; color:#666 !important; font-size:12px;">
        Model: Logistic Regression + TF-IDF &nbsp;|&nbsp; 
        Accuracy: 75.04% &nbsp;|&nbsp; 
        F1: 74.93%
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Enter text to analyze")
st.caption("This tool is for research purposes only. Not a substitute for professional medical advice.")

# Input
user_input = st.text_area(
    label="Text input",
    placeholder="Type or paste any text here — a journal entry, social media post, or message...",
    height=160,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("Analyze Text")

# Disclaimer
st.markdown("""
<div style="background:rgba(255,100,100,0.1); border:1px solid rgba(255,100,100,0.3); 
            border-radius:8px; padding:10px 14px; margin-top:10px;">
    <p style="margin:0; font-size:12px; color:#ff9999 !important;">
        Disclaimer: This tool uses machine learning for research purposes only. 
        It is not a medical diagnostic tool. Always consult a qualified professional.
    </p>
</div>
""", unsafe_allow_html=True)

# Analysis
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif len(user_input.split()) < 3:
        st.warning("Please enter at least a few words for accurate analysis.")
    else:
        with st.spinner("Analyzing..."):
            vectorizer, le, model = load_models()
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            predicted_class = le.inverse_transform([prediction])[0]

            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(vectorized)[0]
            else:
                # For LinearSVC use decision function
                decision = model.decision_function(vectorized)[0]
                probs = np.exp(decision) / np.exp(decision).sum()

            info = CLASS_INFO.get(predicted_class, {
                'color': '#6C63FF',
                'icon': 'ℹ️',
                'desc': '',
                'advice': ''
            })

        # Result
        st.markdown(f"""
        <div class="result-card" style="border-color:{info['color']}; 
             background: linear-gradient(135deg, rgba(26,26,46,0.9), rgba(22,33,62,0.9));">
            <h2 style="color:{info['color']} !important; margin:0 0 8px;">
                {info['icon']} {predicted_class}
            </h2>
            <p style="color:#ccc !important; margin:0 0 16px; font-size:14px;">
                {info['desc']}
            </p>
            <div style="background:rgba(255,255,255,0.05); border-radius:8px; 
                        padding:12px 16px; border-left:3px solid {info['color']};">
                <p style="margin:0; font-size:13px; color:#aaa !important;">
                    {info['advice']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence scores
        st.markdown("### Confidence Scores")
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            class_name = le.inverse_transform([idx])[0]
            prob = probs[idx]
            color = CLASS_INFO.get(class_name, {}).get('color', '#6C63FF')
            is_predicted = class_name == predicted_class
            weight = "bold" if is_predicted else "normal"
            st.markdown(f"""
            <div style="margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="font-size:13px; font-weight:{weight}; color:{'white' if is_predicted else '#aaa'} !important;">
                        {class_name}
                    </span>
                    <span style="font-size:13px; color:{color} !important; font-weight:{weight};">
                        {prob*100:.1f}%
                    </span>
                </div>
                <div class="confidence-bar">
                    <div style="width:{prob*100}%; height:100%; background:{color}; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Stats
        st.markdown("### Text Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", len(user_input.split()))
        with col2:
            st.metric("Characters", len(user_input))
        with col3:
            st.metric("Clean tokens", len(cleaned.split()))

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#444 !important; font-size:12px;">
    MindBridge — NLP Mental Health Classification &nbsp;|&nbsp; 
    Vaibhav Patel · Siddharth Jadhav &nbsp;|&nbsp;
    github.com/Vaibhav2040/MindBridge
</p>
""", unsafe_allow_html=True)