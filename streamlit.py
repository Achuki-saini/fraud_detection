import streamlit as st
import joblib
import numpy as np

# Set up Streamlit page config
st.set_page_config(page_title="Fake Job Detection", layout="centered")

# Title
st.title("🕵️‍♂️ Fake Job Detection")
st.write("Check if a job description is **fraudulent or real** using an AI model.")

# Try loading the model and vectorizer from the 'models/' directory
try:
    model = joblib.load("models/xgboost_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    st.success("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found. Please run `main.py` first to generate them.")
    st.stop()

# Text input
job_description = st.text_area("✍️ Enter the job description here:")

# Predict button
if st.button("🚀 Predict"):
    if not job_description.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        # Vectorize and predict
        try:
            input_vec = vectorizer.transform([job_description])
            prediction = model.predict(input_vec)
            probability = model.predict_proba(input_vec)[0][1]

            if prediction[0] == 1:
                st.error(f"❌ This job is likely **Fake** with {probability*100:.2f}% probability.")
            else:
                st.success(f"✅ This job appears to be **Real** with {100 - probability*100:.2f}% confidence.")
        except Exception as e:
            st.exception(f"Error during prediction: {e}")
