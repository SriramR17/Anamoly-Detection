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
    

    return train_data, test_data


if __name__ == "__main__":
    # Test the data loader
    try:
        train_data, test_data = load_data()
        print("✓ Data loader test successful")
    except Exception as e:
        print(f"❌ Error: {e}")
