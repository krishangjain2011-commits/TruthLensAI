import streamlit as st
from datetime import datetime
import random
import pickle

# ---------- LOAD MODEL ----------
try:
    with open("news_model.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="TruthLensAI",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* Background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e1e2f, #2c2c3c);
    color: #ffffff;
}
/* Card container */
.card {
    background: rgba(40, 40, 55, 0.85);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    max-width: 700px;
    margin: auto;
    color: #ffffff;
}
/* Title row */
.title-row {
    display: flex;
    align-items: center;
    gap: 12px;
}
/* Heading styles */
h1 {
    font-family: 'Arial', sans-serif;
    font-size: 3rem;
    background: linear-gradient(to right, #ff8c94, #a18cd1, #fbc2eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h3 {
    font-family: 'Arial', sans-serif;
    font-weight: normal;
    color: #ffffff;
}
/* Buttons */
.stButton>button {
    background-color: #a18cd1;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1.5rem;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #ff8c94;
    transform: scale(1.05);
}
.stTextInput > div, .stSelectbox > div {
    background-color: rgba(60,60,75,0.9);
    border-radius: 10px;
    padding: 0.5rem;
    color: #ffffff;
}
input::placeholder {
    color: #e0e0e0;
}
label {
    color: #ffffff !important;
}

/* RADIO BUTTON TEXT WHITE FIX */
.stRadio label {
    color: white !important;
}

.tips-box {
    background-color: rgba(70,70,90,0.9);
    padding: 1rem;
    border-radius: 10px;
    margin-top: 1rem;
    border-left: 5px solid #a18cd1;
}
.faq-box {
    background-color: rgba(70,70,90,0.9);
    padding: 1rem;
    border-radius: 10px;
    margin-top: 1rem;
    border-left: 5px solid #ff8c94;
}
.white-box {
    background: rgba(255, 255, 255, 0.15);
    padding: 1.2rem;
    border-radius: 12px;
    margin-top: 1rem;
    backdrop-filter: blur(5px);
}
.feedback-box {
    background-color: rgba(100,100,120,0.85);
    padding: 1rem;
    border-radius: 10px;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# ---------- SIDEBAR NAV ----------
st.sidebar.title("Navigation")
st.session_state.current_page = st.sidebar.selectbox(
    "Go to:",
    ["Home", "Analyze Headline", "History & Insights"],
    index=["Home", "Analyze Headline", "History & Insights"].index(st.session_state.current_page)
)

# ---------- TIPS ----------
tips_list = [
    "💡 Did you know? Over 50% of news shared on social media is never actually read.",
    "📌 Tip: Check multiple sources before trusting a news headline.",
    "⚠️ Beware: Sensational headlines are more likely to be fake.",
    "📰 Fact: Images and videos can be manipulated to spread misinformation.",
    "🔍 Always verify the publication date and source of the news."
]
current_tip = random.choice(tips_list)

# ---------- FAQ ----------
faq = {
    "What is TruthLensAI?": "TruthLensAI analyzes headlines to detect potential fake news.",
    "How do I use the app?": "Enter a headline, select gender & platform, click Analyze News.",
    "Any tips for spotting fake news?": "Check multiple sources, verify images, and watch for sensational language.",
    "Can I see previous headlines?": "Yes! Navigate to the 'History & Insights' page to see past analyses.",
    "Why is gender and platform asked?": "These inputs help show patterns and insights in how news spreads across demographics."
}

# ---------- HOME ----------
if st.session_state.current_page == "Home":
    st.markdown("""
    <div class="title-row">
        <img src="logo.png.png" width="70">
        <h1>TruthLensAI</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Detect fake news and explore insights!</h3>", unsafe_allow_html=True)

    if st.button("Go to Analyze Headline"):
        st.session_state.current_page = "Analyze Headline"

    st.markdown("---")
    st.markdown("""
    Welcome to **TruthLensAI**! This app allows you to:
    - Enter a news headline  
    - Analyze it for possible fake news indicators  
    - See platform & gender-based patterns  
    - Track your past analyses  
    """)

    st.markdown(f'<div class="tips-box">{current_tip}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="white-box">
        <h3>How to Use</h3>
        <p>
        1. Go to <b>Analyze Headline</b>.<br>
        2. Enter the headline.<br>
        3. Select gender & platform.<br>
        4. Click Analyze.<br>
        5. Check history anytime.<br>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Frequently Asked Questions")
    question_list = ["Select a question"] + list(faq.keys())
    selected = st.selectbox("Click for answer:", question_list)
    if selected != "Select a question":
        st.markdown(f'<div class="faq-box">{faq[selected]}</div>', unsafe_allow_html=True)

# ---------- ANALYZE HEADLINE ----------
elif st.session_state.current_page == "Analyze Headline":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="title-row">
            <img src="logo.png.png" width="70">
            <h1>TruthLensAI</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3>Detect fake news and explore insights</h3>", unsafe_allow_html=True)
        st.write("---")

        headline = st.text_input("Enter the news headline here:")

        # ----- Complexity Score -----
        if headline:
            wc = len(headline.split())
            if wc <= 5:
                complexity = 25
                label = "Low"
            elif wc <= 10:
                complexity = 50
                label = "Medium"
            else:
                complexity = 90
                label = "High"

            st.markdown(f"**Complexity Score:** {label} ({wc} words)")
            st.progress(complexity / 100)

        gender = st.radio("Select your gender:", ["Male", "Female", "Other"])
        platform = st.selectbox("Where did you see this news?", ["Instagram", "YouTube", "Facebook", "Twitter"])

        st.write("---")
        st.markdown(f"**Date:** {datetime.today().strftime('%d %B %Y')}")

        # ---------- Prediction ----------
        if st.button("Analyze News"):
            if headline.strip() == "":
                st.warning("Please enter a headline!")
            else:
                try:
                    vec = vectorizer.transform([headline])
                    prediction = model.predict(vec)[0]

                    st.success(f"Prediction: **{prediction.upper()}** 📰🔥")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

                st.session_state.history.append({
                    "headline": headline,
                    "gender": gender,
                    "platform": platform,
                    "date": datetime.today().strftime("%d %B %Y")
                })

            # Feedback
            st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
            st.markdown("### Feedback")
            feedback = st.text_area("Your thoughts:")
            if st.button("Submit Feedback"):
                st.success("Thanks for your feedback! ❤️")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- HISTORY ----------
elif st.session_state.current_page == "History & Insights":
    st.header("Analysis History")

    if st.session_state.history:
        for i, r in enumerate(st.session_state.history, 1):
            st.markdown(f"**{i}. {r['headline']}**")
            st.markdown(f"Platform: {r['platform']} | Gender: {r['gender']} | Date: {r['date']}")
            st.markdown("---")
    else:
        st.info("No headlines analyzed yet!")
