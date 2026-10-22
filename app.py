
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page title and theme layout
st.set_page_config(page_title="Emotion Analyzer", page_icon="✨", layout="centered")

# Emotion to Emoji mapping for better visual representation
EMOJI_DICT = {
    "admiration": "❤️", "amusement": "ん", "anger": "😡", "annoyance": "😒", 
    "approval": "👍", "caring": "🤗", "confusion": "😕", "curiosity": "🤔", 
    "desire": "😏", "disappointment": "😞", "disapproval": "👎", "disgust": "🤢", 
    "embarrassment": "😳", "excitement": "🤩", "fear": "😨", "gratitude": "🙏", 
    "grief": "😢", "joy": "😊", "love": "❤️", "nervousness": "😰", 
    "neutral": "😐", "optimism": "🌅", "pride": "🥇", "realization": "💡", 
    "relief": "😌", "remorse": "🙏", "sadness": "😢", "surprise": "😲"
}

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("emotion_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None

model, vectorizer = load_assets()

st.title("✨ Text-Based Emotion Classification ✨")
st.write("Enter a sentence below to analyze its underlying emotion using our trained Logistic Regression classifier.")

user_input = st.text_area("Input Sentence", placeholder="Type something like 'I am so grateful for your help!'...")

if st.button("Analyze Emotion", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model or Vectorizer is missing. Check your project directory!")
    else:
        # Preprocess and predict
        features = vectorizer.transform([user_input])
        prediction = model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_
            prob_df = pd.DataFrame({
                "Emotion": classes,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
        else:
            prob_df = None

        emoji = EMOJI_DICT.get(prediction.lower(), "📝")
        
        # Show major prediction
        st.success(f"**Predicted Emotion:** {prediction.title()} {emoji}")
        
        # Display confidence distribution
        if prob_df is not None:
            st.subheader("📊 Prediction Confidence Distribution")
            # Show top 5 classifications
            for _, row in prob_df.head(5).iterrows():
                st.write(f"{row['Emotion'].title()} {EMOJI_DICT.get(row['Emotion'].lower(), '')}")
                st.progress(float(row['Probability']))
