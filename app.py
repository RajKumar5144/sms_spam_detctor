import streamlit as st
import pickle
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="SMS Spam Detection | ML App",
    page_icon="📩",
    layout="centered"
)

# ---------------- Debug (TEMPORARY BUT SAFE) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("📁 App directory:", BASE_DIR)
st.write("📄 Files found:", os.listdir(BASE_DIR))

# ---------------- Load Model ----------------
@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

    if not os.path.exists(model_path):
        st.error("❌ model.pkl not found")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error("❌ vectorizer.pkl not found")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

# ---------------- Sidebar ----------------
st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
**SMS Spam Detection App**

**Tech Stack**
- Python
- Scikit-learn
- NLP (TF-IDF)
- Streamlit

**Model**
- Logistic Regression

**Use Case**
- Spam filtering
- Fraud prevention
""")

# ---------------- Main UI ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>📩 SMS Spam Detection</h1>
    <p style='text-align: center; color: gray;'>
    Enter an SMS message to classify it as <b>Spam</b> or <b>Not Spam</b>
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- Input ----------------
message = st.text_area(
    "✍️ Enter SMS Text",
    placeholder="Congratulations! You won a free coupon. Call now...",
    height=150
)

# ---------------- Prediction ----------------
if st.button("🔍 Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        vectorized_msg = vectorizer.transform([message])
        spam_prob = model.predict_proba(vectorized_msg)[0][1]

        st.markdown("---")
        st.subheader("📈 Prediction Result")

        st.progress(spam_prob)

        col1, col2 = st.columns(2)
        col1.metric("Spam Probability", f"{spam_prob:.2%}")
        col2.metric("Not Spam Probability", f"{(1 - spam_prob):.2%}")

        if spam_prob >= 0.3:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **NOT SPAM**")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built using ML & NLP | Streamlit Cloud</p>",
    unsafe_allow_html=True
)
