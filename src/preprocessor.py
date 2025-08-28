"""
Simple Preprocessor for Network Anomaly Detection
=================================================
Clean, essential preprocessing with minimal complexity.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import TARGET_COL, TIME_COL, CELL_COL, NUMERIC_COLS


def preprocess_data(train_data, test_data):
    """
    Preprocess training and test data with essential features.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        
    Returns:
        tuple: (X_train, y_train, X_test, feature_names)
    """
    print("Preprocessing data...")
    
    # Copy data to avoid modifying originals
    train = train_data.copy()
    test = test_data.copy()
    
    train['total_users'] = train['meanUE_UL'] + train['meanUE_DL']
    test['total_users'] = test['meanUE_UL'] + test['meanUE_DL']


    # 4. Prepare feature matrices
    X_train, y_train = _prepare_features(train, include_target=True)
    X_test, _ = _prepare_features(test, include_target=False)

    X_train=train[['meanUE_UL','meanUE_DL','PRBUsageUL','PRBUsageDL','total_users']]
    X_test=test[['meanUE_UL','meanUE_DL','PRBUsageUL','PRBUsageDL','total_users']]
    
    
    # 5. Scale features
    scaler = StandardScaler()
    
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    
    print(f"✓ Preprocessing complete:")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return X_train, y_train, X_test, list(X_train.columns)


def _prepare_features(data, include_target=True):
    """Prepare feature matrix."""
    
    # Select feature columns (exclude original categorical and time columns)
    exclude_cols = [TIME_COL, CELL_COL]
    if include_target:
        exclude_cols.append(TARGET_COL)
    
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Prepare features
    X = data[feature_cols].copy()
    
    # Clean numeric columns - replace invalid values with NaN
    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values with median imputation
    if X.isnull().any().any():
        print("  Filling missing values with median")
        X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = X.fillna(X.median())
    
    # Prepare target if needed
    y = None
    if include_target and TARGET_COL in data.columns:
        y = data[TARGET_COL]

    return X, y


if __name__ == "__main__":
    # Test the preprocessor
    try:
        from src.data_loader import load_data
        train_data, test_data = load_data()
        X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
        

        print("✓ Preprocessing test successful")
        print(f"Features created: {features[:5]}...")
        print(X_train.columns)  
    except Exception as e:
        print(f"❌ Error: {e}")
