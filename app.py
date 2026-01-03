import streamlit as st
import pickle
import os

# ---------------- Page Config (SEO + UI) ----------------
st.set_page_config(
    page_title="SMS Spam Detection | ML App",
    page_icon="📩",
    layout="centered"
)

# ---------------- Load Model (CLOUD SAFE) ----------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

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
- Trained on labeled SMS data  

**Use Case**
- Detect spam messages  
- Fraud prevention  
- SMS filtering systems
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 *Built for ML & Data Science Portfolio*")

# ---------------- Main UI ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>📩 SMS Spam Detection</h1>
    <p style='text-align: center; color: gray;'>
    Enter an SMS message and check whether it is <b>Spam</b> or <b>Not Spam</b> using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- Input Section ----------------
st.markdown("### ✍️ Enter SMS Text")
message = st.text_area(
    "",
    placeholder="Example: Congratulations! You won a free coupon. Call now...",
    height=150
)

# ---------------- Prediction ----------------
if st.button("🔍 Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        vectorized_msg = vectorizer.transform([message])
        spam_prob = model.predict_proba(vectorized_msg)[0][1]

        st.markdown("---")
        st.markdown("### 📈 Prediction Result")

        st.progress(spam_prob)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spam Probability", f"{spam_prob:.2%}")
        with col2:
            st.metric("Not Spam Probability", f"{(1 - spam_prob):.2%}")

        if spam_prob >= 0.3:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **NOT SPAM**")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built using ML & NLP | Streamlit Web App</p>",
    unsafe_allow_html=True
)
