import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize preprocessing
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load Dataset
print("Loading dataset...")
with open('intents.json', 'r') as file:
    intents = json.load(file)

tags = []
patterns = []
responses = {}

# Preprocessing Loop
for intent in intents['intents']:
    tag = intent['tag']
    responses[tag] = intent['responses']
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(tag)

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

cleaned_patterns = [preprocess(pattern) for pattern in patterns]

# Vectorization (TF-IDF)
print("Vectorizing data...")
# ngram_range=(1,2) helps it understand phrases like "not working" better than just "not" and "working"
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(cleaned_patterns)
y = tags

# Model Training
print("Training model...")
# C=10 reduces regularization, making the model more confident in its predictions
model = LogisticRegression(random_state=0, max_iter=200, C=10)
model.fit(X, y)

# Save Artifacts
print("Saving model artifacts...")
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

print("Training complete! Model is now smarter and more confident.")