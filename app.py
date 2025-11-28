import streamlit as st
import pickle

# Load model
with open("news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

st.title("📰 Fake vs Real News Classifier")

headline = st.text_input("Enter a news headline:")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline!")
    else:
        vec = vectorizer.transform([headline])
        prediction = model.predict(vec)[0]
        st.success(f"This news is likely: {prediction.upper()}")
