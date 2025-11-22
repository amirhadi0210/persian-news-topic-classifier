import sys
import os
import joblib
import string
from pathlib import Path
from parsivar import Normalizer, Tokenizer  # Required dependency

# ------------------------------------------------------
# 1. Setup Paths
# ------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR
MODEL_PATH = BASE_DIR / "models" / "persian_classifier_v1.pkl"
STOPWORDS_PATH = BASE_DIR / "data" / "stopwords.json" # Adjust if your file is named differently

# Add project root to path
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# ------------------------------------------------------
# 2. Define Missing Functions (CRITICAL FOR PICKLE)
# ------------------------------------------------------
# These objects must be initialized before loading the model
normalizer = Normalizer()
my_tokenizer = Tokenizer()

# Robust stopwords loading
try:
    if STOPWORDS_PATH.exists():
        persian_stopwords = joblib.load(STOPWORDS_PATH)
    else:
        # Fallback for demo purposes if file is missing
        print(f"⚠️ Warning: Stopwords file not found at {STOPWORDS_PATH}. Preprocessing might be degraded.")
        persian_stopwords = []
except Exception as e:
    print(f"⚠️ Warning: Error loading stopwords: {e}")
    persian_stopwords = []

# The exact function used in training MUST be defined here
def preprocessor(input_text):
    # Safety check for non-string inputs
    if not isinstance(input_text, str):
        return str(input_text)
        
    punc_removed = input_text.translate(str.maketrans('', '', string.punctuation))
    normalized = normalizer.normalize(punc_removed)
    tokens = my_tokenizer.tokenize_sentences(normalized)
    filtered = []
    for token in tokens:
        token = str(token)
        token = token.lower()
        if not token in persian_stopwords and not token.isdigit():
            filtered.append(token)
    output = ' '.join(filtered)
    return output

def tokenizer(text):
    return my_tokenizer.tokenize_words(text)

# ------------------------------------------------------
# 3. Prediction Logic
# ------------------------------------------------------
def load_and_predict(text_input):
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return None, None

    # Load the artifact
    # Now that 'preprocessor' is defined above, this will work!
    artifact = joblib.load(MODEL_PATH)

    model = artifact['pipeline']
    encoder = artifact['encoder']

    # Predict
    prediction_idx = model.predict([text_input])[0]
    
    # Calculate Confidence
    try:
        scores = model.decision_function([text_input])[0]
        confidence = max(scores)
    except:
        # Fallback for classifiers without decision_function
        try:
            scores = model.predict_proba([text_input])[0]
            confidence = max(scores)
        except:
            confidence = 0.0

    label = encoder.inverse_transform([prediction_idx])[0]
    return label, confidence

# ------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------
if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        sample_text = sys.argv[1]
    else:
        sample_text = "تیم ملی فوتبال ایران در جام جهانی عملکرد خوبی داشت"
        print(f"No input provided. Using sample: '{sample_text}'")

    label, confidence = load_and_predict(sample_text)
    
    if label:
        print("-" * 30)
        print(f"Input: {sample_text[:50]}...")
        print(f"Prediction: {label}")
        if confidence:
            print(f"Confidence: {confidence:.4f}")
        print("-" * 30)