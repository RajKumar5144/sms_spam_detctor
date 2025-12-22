import streamlit as st
import pickle

st.write("RUNNING: SMS SPAM DETECTION PROJECT")

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 SMS Spam Detection App")

message = st.text_area("Enter the SMS text")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        vectorized_msg = vectorizer.transform([message]).toarray()

        # 🔑 Use probability instead of hard class
        spam_prob = model.predict_proba(vectorized_msg)[0][1]

        if spam_prob >= 0.3:   # threshold for imbalanced data
            st.error(f"🚨 This message is SPAM (confidence: {spam_prob:.2f})")
        else:
            st.success(f"✅ This message is NOT Spam (confidence: {1 - spam_prob:.2f})")
