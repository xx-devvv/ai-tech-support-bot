import streamlit as st
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- SETUP & LOAD ARTIFACTS ---
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()


@st.cache_resource
def load_artifacts():
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('responses.pkl', 'rb') as f:
        responses = pickle.load(f)
    return model, vectorizer, responses


model, vectorizer, responses = load_artifacts()


def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])


def keyword_scan(text):
    text = text.lower()

    # --- LEVEL 1: CONTEXT AWARENESS ---
    if any(phrase in text for phrase in
           ["internet is good", "wifi is good", "net is fine", "internet is working", "no lag"]):
        if "delay" in text or "lag" in text or "stutter" in text:
            return "game_lag"
        return None

        # --- LEVEL 2: SPECIFIC PHRASE MATCHING ---
    if "input delay" in text or "input lag" in text or "mouse delay" in text:
        return "game_lag"
    if "blue screen" in text or "bsod" in text:
        return "pc_freeze_crash"
    if "black screen" in text:
        return "no_display"

    # --- LEVEL 3: KEYWORD SCAN ---
    keywords = {
        "made you": "creator",
        "created you": "creator",
        "developer": "creator",
        "who are you": "about",
        "what are you": "about",
        "bot": "about",
        "wifi": "wifi_no_internet",
        "internet": "wifi_no_internet",
        "slow": "slow_pc",
        "lag": "slow_pc",
        "freeze": "pc_freeze_crash",
        "crash": "pc_freeze_crash",
        "stutter": "game_lag",
        "fps": "game_lag",
        "heat": "laptop_overheat",
        "hot": "laptop_overheat",
        "fan": "laptop_overheat",
        "battery": "battery_drain",
        "drain": "battery_drain",
        "display": "no_display",
        "monitor": "no_display",
        "turn on": "no_display",
        "turning": "no_display",
        "won't start": "no_display",
        "wont start": "no_display",
        "boot": "boot_loop",
        "startup": "boot_loop",
        "sound": "no_sound",
        "audio": "no_sound",
        "mic": "mic_issue",
        "microphone": "mic_issue",
        "virus": "virus_malware",
        "hack": "virus_malware",
        "update": "update_fail",
        "storage": "storage_full",
        "space": "storage_full",
        "usb": "usb_issues",
        "mouse": "usb_issues"
    }

    for word, tag in keywords.items():
        if word in text:
            return tag

    return None


def get_response(user_input):
    processed_input = preprocess(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    prediction = model.predict(vectorized_input)[0]
    probs = model.predict_proba(vectorized_input)
    confidence = max(probs[0])

    if confidence < 0.5:
        manual_tag = keyword_scan(user_input)
        if manual_tag:
            return manual_tag, 0.95

    return prediction, confidence


# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="AI Tech Support", page_icon="🤖", layout="centered")

# Custom CSS for a "Tech" Vibe
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    /* Header Styling */
    h1 {
        color: #00e5ff !important;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    /* Subtitle Styling */
    .subtitle {
        color: #b0bec5;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
        font-family: 'Arial', sans-serif;
    }
    /* Chat Bubble Styling */
    .stChatMessage {
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("🤖 AI Tech Support Bot")
st.markdown(
    '<p class="subtitle">Powered by <b>NLP (TF-IDF)</b> & <b>Logistic Regression</b><br>Designed for efficient troubleshooting and instant support.</p>',
    unsafe_allow_html=True)

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("🛠️ Capabilities")
    st.markdown("""
    This bot can help with:
    - 📶 **Internet Issues** (WiFi, Slow Speed)
    - 💻 **Hardware** (Crashes, Black Screen)
    - 🔋 **Power** (Battery, Won't Turn On)
    - 🎮 **Gaming** (Lag, FPS Drops)
    - 🦠 **Security** (Virus, Malware)
    """)
    st.divider()
    st.markdown("*Built by Dev Pandey*")

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = (
        "👋 **Hello! I am your Automated Tech Support Assistant.**\n\n"
        "I am trained using Natural Language Processing to diagnose computer issues.\n"
        "Try saying: *'My wifi is slow'* or *'My laptop is overheating'*."
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your issue (e.g., 'No Internet')..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    intent, confidence = get_response(prompt)

    if confidence > 0.4:
        response_text = random.choice(responses[intent])
        final_response = f"{response_text} \n\n :grey_exclamation: *Confidence: {confidence:.2f}*"
    else:
        final_response = (
            "⚠️ I am not sure exactly what the issue is. \n\n"
            "Could you try using keywords like **'Wifi'**, **'Screen'**, or **'Battery'**?\n"
            "Otherwise, please contact human support at **support@example.com**."
        )

    with st.chat_message("assistant"):
        st.markdown(final_response)
    st.session_state.messages.append({"role": "assistant", "content": final_response})