


# 🤖 AI Tech Support Chatbot (NLP)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-ML-orange)

An intelligent chatbot designed to automate Level 1 technical support. Built using **Natural Language Processing (NLP)** techniques and a **Hybrid Classification Approach** (TF-IDF + Keyword Safety Net) to accurately diagnose common computer issues.

---

## 📖 Project Overview
This project was developed to assist users with common technical problems such as internet connectivity issues, system crashes, and hardware malfunctions. It utilizes a **Logistic Regression** model trained on TF-IDF vectors for general intent recognition, reinforced by a **Keyword Safety Net** to handle specific technical terms with high precision.

### Key Capabilities:
* **Diagnose Issues:** Identifies 20+ common PC problems (e.g., Blue Screen, Slow WiFi, No Audio).
* **Smart Fallback:** Detects when it doesn't know an answer and guides the user to human support.
* **Context Awareness:** Distinguishes between "My internet is **bad**" (Internet Issue) and "My internet is **good**" (Game Lag).
* **Interactive UI:** A modern, dark-themed interface built with Streamlit.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit (Web Interface)
* **NLP & ML:**
    * **NLTK:** Tokenization & Lemmatization
    * **Scikit-Learn:** TF-IDF Vectorization & Logistic Regression
* **Data Handling:** JSON (Intent dataset), Pickle (Model serialization)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/ai-tech-support-bot.git](https://github.com/yourusername/ai-tech-support-bot.git)
cd ai-tech-support-bot

```

### 2. Install Dependencies

Ensure you have Python installed. Then run:

```bash
pip install -r requirements.txt

```

### 3. Train the Model

Before running the bot, you must generate the model files (`chatbot_model.pkl`, etc.).

```bash
python train_model.py

```

*You should see a message: "Training complete! Model is now smarter and more confident."*

### 4. Run the Application

Launch the chatbot interface:

```bash
streamlit run app.py

```

---

## 📂 Project Structure

```text
ai-tech-support-bot/
│
├── app.py                # Main application file (Streamlit UI & Logic)
├── train_model.py        # Script to train the ML model
├── intents.json          # Dataset containing patterns and responses
├── requirements.txt      # List of Python dependencies
├── README.md             # Project documentation
│
└── (Generated Artifacts)
    ├── chatbot_model.pkl # Trained Logistic Regression model
    ├── vectorizer.pkl    # TF-IDF Vectorizer
    └── responses.pkl     # Response dictionary

```

---

## 💡 How It Works

1. **Preprocessing:** User input is tokenized and lemmatized (e.g., "running" -> "run") using NLTK.
2. **Vectorization:** The text is converted into numbers using **TF-IDF** (Term Frequency-Inverse Document Frequency).
3. **Prediction (Layer 1):** The **Logistic Regression** model predicts the intent (e.g., `wifi_issue`) and assigns a confidence score.
4. **Keyword Safety Net (Layer 2):** If the model's confidence is low (< 0.5), the system scans for specific keywords (e.g., "blue screen", "HDMI"). If a match is found, it overrides the model to ensure accurate support.
5. **Context Check:** The bot checks for "Anti-Keywords" (e.g., if a user says "Internet is good", it ignores the `internet_issue` tag).

---

## 🧪 Example Queries

Try asking the bot:

* *"My computer is not turning on"*
* *"I have a blue screen error"*
* *"My wifi is connected but no internet"*
* *"Who made you?"*
* *"My FPS is dropping in games"*

---

## 👨‍💻 Author

**Dev Pandey**

* **Role:** Developer & NLP Engineer
* **Project Type:** Internship Project (SPARKIIT / Wipro DICE ID)

---

## 📝 License

This project is open-source and available for educational purposes.


---
