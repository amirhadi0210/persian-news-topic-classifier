from pathlib import Path

# Always resolve project root regardless of where script/notebook runs
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIGURE_DIR = BASE_DIR / 'figures'


FIGURE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CONFIG = {
    'random_state': 42,
    'test_size': 0.10,
    'n_folds': 5,
    'ngram_range': (1, 3),
    'min_df': 3,
    'max_df': 0.9,
    'model_C': 1.5,

    # paths
    'data_path': DATA_DIR / 'yektanet_train.csv',
    'stopwords_path': DATA_DIR / 'stopwords.json',
    'save_path': MODEL_DIR / 'persian_classifier_v1.pkl',
    'save_path': MODEL_DIR / 'persian_classifier_v1.pkl',
    'figure_path': FIGURE_DIR / 'confusion_matrix.png'
}

DEPLOYMENT_CONFIG = {
    'model_version': 'v1.2_prod',
    'min_confidence_threshold': 0.65,  # Reject predictions below this
    'fallback_category': 'اجتماعی',  # Default for low confidence
    'monitoring': {
        'log_predictions': True,
        'alert_on_drift': True,
        'retrain_threshold': 0.75  # Retrain if F1 drops below
    },
    'business_metrics': {
        'target_f1': 0.80,
        'annual_cost_savings': 2_000,  # USD
        'processing_capacity': 10000  # articles/day
    }
}