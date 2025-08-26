"""
Simple Data Loader for Network Anomaly Detection
================================================
Clean, straightforward data loading with basic validation.
"""

import pandas as pd
from config import TRAIN_FILE, TEST_FILE, TARGET_COL, NUMERIC_COLS


def load_data(train_path=None, test_path=None):
    """
    Load training and test datasets.
    
    Returns:
        tuple: (train_data, test_data)
    """
    print("Loading data...")
    
    # Use default paths if not provided
    train_path = train_path or TRAIN_FILE
    test_path = test_path or TEST_FILE
    
    # Load datasets
    train_data = pd.read_csv(train_path, encoding='latin-1')
    test_data = pd.read_csv(test_path, encoding='latin-1')
    
    print(f"✓ Training data: {train_data.shape}")
    print(f"✓ Test data: {test_data.shape}")
    
    # Basic validation
    _validate_data(train_data, test_data)
    
    return train_data, test_data


def _validate_data(train_data, test_data):
    """Simple validation of the datasets."""
    
    # Check for empty datasets
    if train_data.empty or test_data.empty:
        raise ValueError("Empty dataset found")
    
    # Check required columns
    required_cols = NUMERIC_COLS + ['Time', 'CellName']
    
    missing_train = set(required_cols) - set(train_data.columns)
    if missing_train:
        raise ValueError(f"Missing columns in training data: {missing_train}")
    
    missing_test = set(required_cols) - set(test_data.columns)
    if missing_test:
        raise ValueError(f"Missing columns in test data: {missing_test}")
    
    # Check target column exists in training data
    if TARGET_COL not in train_data.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training data")
    
    # Check target values
    unique_targets = train_data[TARGET_COL].unique()
    if not all(t in [0, 1] for t in unique_targets):
        print(f"Warning: Unexpected target values: {unique_targets}")
    
    # Report missing values
    train_missing = train_data.isnull().sum().sum()
    test_missing = test_data.isnull().sum().sum()
    
    if train_missing > 0:
        print(f"Warning: {train_missing} missing values in training data")
    if test_missing > 0:
        print(f"Warning: {test_missing} missing values in test data")
    
    print("✓ Data validation passed")


if __name__ == "__main__":
    # Test the data loader
    try:
        train_data, test_data = load_data()
        print("✓ Data loader test successful")
    except Exception as e:
        print(f"❌ Error: {e}")
