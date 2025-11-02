import streamlit as st
import joblib
import os

# ---- Load Model ----
MODEL_PATH = "models/fake_job_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"

st.set_page_config(page_title="Fake Job Detection", page_icon="🕵️", layout="centered")

st.title("🕵️ Fake Job Posting Detection App")
st.write("Paste a job description below and find out if it's **Real** or **Fake**!")

# Check model files
if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    st.error("❌ Model files not found! Please train the model first.")
    st.stop()

# Load saved model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# ---- User Input ----
job_text = st.text_area("📝 Job Description", height=200, placeholder="Enter or paste job description here...")

if st.button("🔍 Check Job"):
    if not job_text.strip():
        st.warning("⚠️ Please enter a job description.")
    else:
        # Transform and predict
        input_vec = vectorizer.transform([job_text])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚫 This looks like a **Fake Job Posting!**")
        else:
            st.success("✅ This appears to be a **Real Job Posting.**")
