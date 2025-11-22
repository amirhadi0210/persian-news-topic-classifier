# Persian Content Intelligence System 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-green)

**A high-throughput automated content tagging engine designed to reduce manual moderation costs by ~80% while maintaining high data quality via Human-in-the-Loop (HITL) fallbacks.**

---

## 🎯 Business Value & Impact

This system replaces manual categorization of news articles with a machine learning pipeline. It is optimized for speed and reliability in a production environment.

| Metric | Performance | Business Implication |
| :--- | :--- | :--- |
| **Automation Rate** | **~80%** | Only 20% of articles require manual review. |
| **Accuracy (F1)** | **0.80** | High reliability on 17 distinct content categories. |
| **Throughput** | **10k+ / day** | Lightweight architecture allows massive scaling on CPU. |
| **Risk Control** | **Confidence Scoring** | Predictions below **65% confidence** are flagged for human review. |

---

## 📂 Repository Structure

The project follows a modular structure separating experimentation (notebooks) from production logic (scripts).

```text
.
├── data/                    # Raw training data and stopwords
├── models/                  # Serialized artifacts (Pipeline + Encoders)
├── src/                     # Configuration and utility modules
│   └── config.py
├── notebooks/
    └──test.ipynb            # Experimentation, EDA, and Model Training
├── predict.py               # CLI tool for single-instance inference
├── api.py                   # Flask REST API for production deployment
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
---
2. Install dependencies:

Bash

pip install -r requirements.txt
3. Verify Artifacts: Ensure models/persian_classifier_v1.pkl exists. If not, run the training notebook to generate it.

🚀 Usage
1. Command Line Inference (CLI)
For testing individual sentences or debugging:

Bash

python predict.py "تیم ملی فوتبال ایران در جام جهانی عملکرد خوبی داشت"
Output:

Plaintext

Input: تیم ملی فوتبال ایران در جام جهانی...
Prediction: ورزش
Confidence: 0.9214
2. REST API (Production Mode)
Start the Flask server (or deploy via Gunicorn):

Bash

python api.py
Make a Request (using cURL):

Bash

curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "نرخ تورم کاهش یافت", "description": "گزارش بانک مرکزی نشان میدهد..."}'
JSON Response:

JSON

{
    "category": "تجارت و اقتصاد",
    "confidence": 0.88,
    "model_version": "v1.2_prod",
    "status": "success"
}
📊 Model Performance
The model utilizes a LinearSVC architecture with TF-IDF vectorization, chosen for its superior performance on high-dimensional sparse text data compared to heavier Deep Learning models.

Cross-Validation F1: 0.81 (±0.016)

Held-Out Test F1: 0.80

Confusion Matrix
The matrix below highlights the model's ability to distinguish between closely related categories (e.g., Economy vs. Social).

(Note: Ensure https://www.google.com/search?q=confusion_matrix.png is saved in your root folder)

🔧 Technical Details
Preprocessing Pipeline
Normalization: Character standardization (Arabic to Persian mappings) using parsivar.

Tokenization: Sentence and word-level splitting.

Cleaning: Removal of punctuation, digits, and domain-specific stopwords.

Configuration
Settings can be adjusted in src/config.py:

min_confidence_threshold: 0.65 (Predictions below this trigger review).

fallback_category: اجتماعی (Social).

ngram_range: (1, 3).

📈 Roadmap
[ ] Dockerization: Containerize the API for Kubernetes deployment.

[ ] Monitoring: Integrate Prometheus for drift detection.

[ ] Transformer Upgrade: Test ParsBERT for complex, ambiguous categories.
