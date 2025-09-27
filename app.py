import streamlit as st
from transformers import pipeline

# --- Config ---
MODEL_REPO = "stefanusbenhard/student_model"  # <-- change if you used a different HF repo

LABEL_RESPONSES = {
    "postpartum": (
        "It sounds like you may be experiencing **postpartum-related struggles**. "
        "You're not alone 💙. Many new mothers feel this way. "
        "If symptoms persist, consider reaching out to a professional. "
        "Would you like me to suggest coping resources?"
    ),
    "no": (
        "I don’t detect signs of postpartum depression in what you shared. "
        "Still, sharing your feelings can be very healthy. "
        "How are you coping these days?"
    ),
    "major depressive": (
        "Your message seems closer to **major depressive symptoms**. "
        "If these feelings last more than two weeks, it's important to seek professional help."
    ),
    "bipolar": (
        "This message reflects **bipolar-like patterns** (mood swings or shifts). "
        "Please consider reaching out to a clinician if these symptoms interfere with daily life."
    ),
    "psychotic": (
        "Your text indicates **psychotic-like language**. "
        "This can be very serious, and I strongly recommend speaking to a professional urgently."
    ),
    "atypical": (
        "This message aligns more with **atypical depression** patterns. "
        "It's still important to care for yourself and consider consulting a professional."
    ),
}

@st.cache_resource(show_spinner=True)
def load_pipeline():
    # If your HF model is private, add: token=st.secrets["HF_TOKEN"]
    return pipeline("text-classification", model=MODEL_REPO, tokenizer=MODEL_REPO)

pipe = load_pipeline()

st.set_page_config(page_title="Postpartum Screening Assistant", page_icon="🤖")
st.title("🤖 ChatGPT-like Postpartum Screening Assistant")
st.markdown(
    "This is a **prototype assistant** trained to detect potential signs of postpartum "
    "depression and related conditions. ⚠️ It is **not a medical device**."
)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your message here...")

if user_input:
    result = pipe(user_input, truncation=True)[0]
    label, score = result["label"], float(result["score"])
    response = LABEL_RESPONSES.get(
        label, f"I detect signals related to **{label}**. Please consider professional advice."
    )
    response = f"{response}\n\n_(confidence: {score:.2f})_"
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)
