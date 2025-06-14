import streamlit as st
from main import FraudJobDetector
import pandas as pd

# Initialize detector and load model/vectorizer
detector = FraudJobDetector()
model_loaded = detector.load_models(filepath="models", deploy_only=True)

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️", layout="centered")
st.title("🕵️‍♂️ Fake Job Detection")
st.markdown("Check if a job description is **fraudulent** or **real** using an AI model.")

# Check if model is available
if not model_loaded:
    st.error("❌ Model not loaded. Please run training first to generate XGBoost model.")
    st.stop()

# Text input for prediction
job_text = st.text_area("✍️ Enter job description text below:", height=300)

if st.button("🚀 Predict"):
    if not job_text.strip():
        st.warning("Please enter a valid job description.")
    else:
        # Create DataFrame and preprocess
        input_df = pd.DataFrame({'text': [job_text.lower()]})
        input_df['text'] = input_df['text'].apply(
            lambda x: ' '.join([w for w in x.split() if w not in detector.stop_words])
        )

        # Predict
        prediction, prob = detector.predict_deployment(input_df['text'])

        # Apply custom threshold: invert logic
        fraud_prob = prob[0]
        if fraud_prob > 0.75:
            label = "✅ Real"
        else:
            label = "🚨 Fraudulent"

        # Display result
        st.subheader("🔍 Prediction Result")
        st.markdown(f"**Result:** {label}")
        st.markdown(f"**Probability of Fraud:** `{fraud_prob*100:.2f}%`")

# Footer
st.markdown("---")
st.markdown("<center><small>Fraud Job Detector | XGBoost Model | Streamlit App</small></center>", unsafe_allow_html=True)
