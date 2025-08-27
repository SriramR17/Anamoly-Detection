"""
Simple Configuration for Network Anomaly Detection
==================================================
Clean, minimal configuration with only essential settings.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent  # Current directory (project root)
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files
TRAIN_FILE = DATA_DIR / "ML-MATT-CompetitionQT1920_train.csv"
TEST_FILE = DATA_DIR / "ML-MATT-CompetitionQT1920_test.csv"

# Column definitions
TARGET_COL = 'Unusual'
TIME_COL = 'Time'
CELL_COL = 'CellName'

# Numeric features
NUMERIC_COLS = [
    'PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL',
    'maxThr_DL', 'maxThr_UL', 'meanUE_DL', 'meanUE_UL',
    'maxUE_DL', 'maxUE_UL', 'maxUE_UL+DL'
]

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Comprehensive model parameters for algorithm testing
MODEL_PARAMS = {
    # Existing algorithms
    'RandomForest': {'n_estimators': 100, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'LogisticRegression': {'random_state': RANDOM_STATE, 'max_iter': 1000},
    'GradientBoosting': {'n_estimators': 100, 'random_state': RANDOM_STATE},
    
    # Additional Tree-based algorithms
    'DecisionTree': {'random_state': RANDOM_STATE, 'max_depth': 10},
    'ExtraTrees': {'n_estimators': 100, 'random_state': RANDOM_STATE, 'n_jobs': -1},
    'AdaBoost': {'n_estimators': 50, 'random_state': RANDOM_STATE},
    
    # Support Vector Machine
    'SVM': {'kernel': 'rbf', 'random_state': RANDOM_STATE, 'probability': True},
    'LinearSVM': {'random_state': RANDOM_STATE, 'max_iter': 1000},
    
    # Instance-based algorithms
    'KNN': {'n_neighbors': 5, 'n_jobs': -1},
    
    # Naive Bayes
    'GaussianNB': {},
    
    # Neural Network
    'MLP': {'hidden_layer_sizes': (100, 50), 'random_state': RANDOM_STATE, 'max_iter': 500},
    
    # XGBoost (if available)
    'XGBoost': {'n_estimators': 100, 'random_state': RANDOM_STATE, 'eval_metric': 'logloss'},
    
    # LightGBM (if available)
    'LightGBM': {'n_estimators': 100, 'random_state': RANDOM_STATE, 'verbose': -1},
    
    # CatBoost (if available)
    'CatBoost': {'iterations': 100, 'random_state': RANDOM_STATE, 'verbose': False}
}

# Output files
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"
RESULTS_PLOT = OUTPUT_DIR / "results.png"
