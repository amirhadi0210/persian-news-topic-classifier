from flask import Flask, request, jsonify
import joblib
import sys
import string
from pathlib import Path
from parsivar import Normalizer, Tokenizer

app = Flask(__name__)

# ------------------------------------------------------
# 1. Setup Paths & Imports
# ------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR
MODEL_PATH = BASE_DIR / "models" / "persian_classifier_v1.pkl"
STOPWORDS_PATH = BASE_DIR / "data" / "stopwords.json"

# Ensure src can be imported if needed
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# ------------------------------------------------------
# 2. Define Helper Functions (Required for Pickle)
# ------------------------------------------------------
normalizer = Normalizer()
my_tokenizer = Tokenizer()

try:
    persian_stopwords = joblib.load(STOPWORDS_PATH)
except:
    persian_stopwords = []

def preprocessor(input_text):
    if not isinstance(input_text, str): return str(input_text)
    punc_removed = input_text.translate(str.maketrans('', '', string.punctuation))
    normalized = normalizer.normalize(punc_removed)
    tokens = my_tokenizer.tokenize_sentences(normalized)
    filtered = [str(t).lower() for t in tokens if str(t).lower() not in persian_stopwords and not str(t).isdigit()]
    return ' '.join(filtered)

def tokenizer(text):
    return my_tokenizer.tokenize_words(text)

# ------------------------------------------------------
# 3. Load Model
# ------------------------------------------------------
print("⏳ Loading model...")
if MODEL_PATH.exists():
    artifact = joblib.load(MODEL_PATH)
    model = artifact['pipeline']
    encoder = artifact['encoder']
    features = artifact.get('features', ['title', 'description']) # fallback
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found!")
    model = None

# ------------------------------------------------------
# 4. API Endpoints
# ------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        
        # Combine features dynamically based on what the model expects
        content_parts = [str(data.get(f, '')) for f in features]
        content = " ".join(content_parts)
        
        # Predict
        pred_idx = model.predict([content])[0]
        category = encoder.inverse_transform([pred_idx])[0]
        
        # Confidence (Optional)
        try:
            conf = max(model.decision_function([content])[0])
        except:
            conf = 0.0

        return jsonify({
            'status': 'success',
            'category': category,
            'confidence': float(conf),
            'model_version': artifact.get('deployment_config', {}).get('model_version', 'unknown')
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)