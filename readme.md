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
├── reports/                 # Generated performance visualizations
├── test.ipynb               # Experimentation, EDA, and Model Training
├── predict.py               # CLI tool for single-instance inference
├── api.py                   # Flask REST API for production deployment
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
