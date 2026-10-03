import streamlit as st
import pickle
import os

st.set_page_config(page_title="Spam Message Detector", page_icon="📩", layout="centered")

MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

st.title("📩 Spam Message Detector")
st.write("Simple Machine Learning project using Python, TF-IDF and Logistic Regression.")
st.info("Enter an SMS/message below and click **Check Message**.")

if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.warning("Model files are missing. First run: python train_model.py")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize. Call now!"
)

if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.error("Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        probability = model.predict_proba(message_vector).max()

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
            st.write(f"Confidence: **{probability:.2%}**")
        else:
            st.success("✅ HAM (NOT SPAM)")
            st.write(f"Confidence: **{probability:.2%}**")

st.markdown("---")
st.caption("MCA Minor Project | Spam Detection using Machine Learning")
