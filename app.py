import streamlit as st
import requests
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Personalized Learning System",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# OPENROUTER CONFIG
# --------------------------------------------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

# --------------------------------------------------
# OPENROUTER API FUNCTION (AS YOU REQUESTED)
# --------------------------------------------------
def query_llm(prompt_text):

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Personalized Learning System",
        },
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI tutor. Explain concepts clearly and suggest learning resources."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
        }),
        timeout=30
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error("LLM API Error")
        st.code(response.text)
        return None

# --------------------------------------------------
# DATASET (SIMULATED LEARNER DATA)
# --------------------------------------------------
@st.cache_data
def load_learner_data():
    np.random.seed(42)
    data = {
        "quiz_score": np.random.randint(30, 100, 200),
        "time_spent": np.random.randint(5, 90, 200),
        "attempts": np.random.randint(1, 5, 200),
        "level": np.random.choice(
            ["Beginner", "Intermediate", "Advanced"],
            200,
            p=[0.4, 0.35, 0.25]
        )
    }
    return pd.DataFrame(data)

df = load_learner_data()

# --------------------------------------------------
# ML MODEL TRAINING
# --------------------------------------------------
encoder = LabelEncoder()
df["level_encoded"] = encoder.fit_transform(df["level"])

X = df[["quiz_score", "time_spent", "attempts"]]
y = df["level_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# --------------------------------------------------
# SIDEBAR – USER INPUT
# --------------------------------------------------
with st.sidebar:
    st.title("⚙ Learner Inputs")

    topic = st.text_input(
        "Topic you want to learn",
        placeholder="e.g., Linear Algebra, Neural Networks"
    )

    quiz_score = st.slider(
        "Quiz Score (%)",
        min_value=0,
        max_value=100,
        value=65
    )

    time_spent = st.slider(
        "Time Spent (minutes)",
        min_value=1,
        max_value=120,
        value=30
    )

    attempts = st.slider(
        "Number of Attempts",
        min_value=1,
        max_value=5,
        value=2
    )

    generate_btn = st.button("🚀 Generate Personalized Plan")

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("📚 AI-Driven Personalized Learning System")

st.markdown("""
This system demonstrates **AI in Personalized Learning** by combining:

- 🧠 **Machine Learning** (Learner Level Prediction)
- 🗣️ **Natural Language Processing** (Content Generation)
""")

st.success(f"📊 Learner Model Accuracy: **{accuracy:.2f}**")

# --------------------------------------------------
# PERSONALIZATION PIPELINE
# --------------------------------------------------
if generate_btn:

    if not topic:
        st.warning("Please enter a topic.")
    else:
        learner_input = np.array([[quiz_score, time_spent, attempts]])

        predicted_level_encoded = model.predict(learner_input)[0]
        predicted_level = encoder.inverse_transform(
            [predicted_level_encoded]
        )[0]

        st.subheader("🧠 Predicted Learner Profile")
        st.write(f"**Predicted Level:** `{predicted_level}`")

        prompt = f"""
The learner wants to study **{topic}**.
Their predicted knowledge level is **{predicted_level}**.

Tasks:
1. Explain the topic clearly for this level.
2. Recommend 3 learning resources.
3. Give 2 study tips personalized for this learner.
"""

        with st.spinner("Generating personalized learning content..."):
            content = query_llm(prompt)

        if content:
            st.subheader("📘 Personalized Learning Content")
            st.markdown(content)

# --------------------------------------------------
# DATA VISIBILITY (FOR EXAMINER)
# --------------------------------------------------
with st.expander("📂 View Learner Dataset (AI Evidence)"):
    st.dataframe(df.head(25))

with st.expander("🧠 Model Explanation"):
    st.markdown("""
- **Model:** Logistic Regression  
- **Inputs:** Quiz score, Time spent, Attempts  
- **Output:** Learner level (Beginner / Intermediate / Advanced)  
- **Purpose:** Adapt learning content dynamically  
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "🛠️ Developed by **Vignesh S** | Minor Project – Artificial Intelligence",
    unsafe_allow_html=True
)
