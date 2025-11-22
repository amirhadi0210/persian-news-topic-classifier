# Persian News Category Classifier 🚀

A production-ready machine learning pipeline for classifying Persian news articles using LinearSVC + TF-IDF. Includes CLI tools, REST API, preprocessing utilities, and deployment configuration.

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

---

## 📘 Usage Guide

### 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Verify that the model exists:**

```
models/persian_classifier_v1.pkl
```

If missing, run the training notebook to generate it.

---

## 🖥️ Command Line Usage (CLI)

Test a single prediction:

```bash
python predict.py "تیم ملی فوتبال ایران در جام جهانی عملکرد خوبی داشت"
```

Example Output:

```
Input: تیم ملی فوتبال ایران در جام جهانی...
Prediction: ورزش
Confidence: 0.9214
```

---

## 🌐 REST API Usage (Production Mode)

Start the API server:

```bash
python api.py
```

Send a request with `curl`:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "نرخ تورم کاهش یافت", "description": "گزارش بانک مرکزی نشان میدهد..."}'
```

Example Response:

```json
{
    "category": "تجارت و اقتصاد",
    "confidence": 0.88,
    "model_version": "v1.2_prod",
    "status": "success"
}
```

---

## 📊 Model Performance

The classifier uses **LinearSVC + TF-IDF**, effective for sparse Persian text.

* **Cross-Validation F1:** 0.81 (±0.016)
* **Held-Out Test F1:** 0.80

### Confusion Matrix

Ensure the file exists in the project root:

```
./figures/confusion_matrix.png
```

---

## 🔧 Preprocessing Pipeline

* Character normalization using Parsivar
* Sentence and word tokenization
* Punctuation and digit removal
* Domain-specific stopword filtering

---

## ⚙️ Configuration (`src/config.py`)

|                  Parameter | Description                                     |
| -------------------------: | :---------------------------------------------- |
| `min_confidence_threshold` | Predictions below this threshold trigger review |
|        `fallback_category` | Output class when confidence is too low         |
|              `ngram_range` | TF-IDF n-gram window, default `(1, 3)`          |

---

## 📈 Roadmap

* [ ] Dockerize API for Kubernetes deployment
* [ ] Add Prometheus monitoring for drift detection
* [ ] Upgrade model using ParsBERT for ambiguous cases

---

## 👤 Author

**Amirhadi Souratian**
Data Scientist / ML Engineer


