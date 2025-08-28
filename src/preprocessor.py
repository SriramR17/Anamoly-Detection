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
    
    # 1. Basic time features
    train = _add_time_features(train)
    test = _add_time_features(test)
    
    # 2. Derived features
    train = _add_derived_features(train)
    test = _add_derived_features(test)
    
    # 3. Encode categorical features
    train, test = _encode_categorical(train, test)
    
    # 4. Prepare feature matrices
    X_train, y_train = _prepare_features(train, include_target=True)
    X_test, _ = _prepare_features(test, include_target=False)
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_train=X_train[['meanUE_UL','meanUE_DL','PRBUsageUL','PRBUsageDL','total_users']]
    X_test=X_test[['meanUE_UL','meanUE_DL','PRBUsageUL','PRBUsageDL','total_users']]
    
    print(f"✓ Preprocessing complete:")
    print(f"  Features: {len(X_train.columns)}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return X_train, y_train, X_test, list(X_train.columns)


def _add_time_features(data):
    """Add simple time-based features."""
    
    # Extract hour from time
    data['Hour'] = pd.to_datetime(data[TIME_COL], format='%H:%M').dt.hour
    
    # Business hours indicator (8 AM - 6 PM)
    data['is_business_hours'] = ((data['Hour'] >= 8) & (data['Hour'] < 18)).astype(int)
    
    # Peak hours indicator (8-10 AM, 5-7 PM)
    data['is_peak_hours'] = (((data['Hour'] >= 8) & (data['Hour'] < 10)) | 
                            ((data['Hour'] >= 17) & (data['Hour'] < 19))).astype(int)
    
    # Night time indicator (10 PM - 6 AM)
    data['is_night'] = ((data['Hour'] >= 22) | (data['Hour'] < 6)).astype(int)
    
    return data


def _add_derived_features(data):
    """Add derived features."""
    
    # Total traffic indicators
    data['total_prb'] = data['PRBUsageUL'] + data['PRBUsageDL']
    data['total_throughput'] = data['meanThr_UL'] + data['meanThr_DL']
    data['total_users'] = data['meanUE_UL'] + data['meanUE_DL']
    
    # Simple ratios (with small constant to avoid division by zero)
    epsilon = 1e-8
    data['ul_dl_prb_ratio'] = data['PRBUsageUL'] / (data['PRBUsageDL'] + epsilon)
    data['ul_dl_thr_ratio'] = data['meanThr_UL'] / (data['meanThr_DL'] + epsilon)
    
    # Efficiency: throughput per resource block
    data['efficiency_ul'] = data['meanThr_UL'] / (data['PRBUsageUL'] + epsilon)
    data['efficiency_dl'] = data['meanThr_DL'] / (data['PRBUsageDL'] + epsilon)
    
    return data


def _encode_categorical(train_data, test_data):
    """Encode categorical features."""
    
    # Encode cell names
    encoder = LabelEncoder()
    
    # Fit on training data
    train_data['cell_encoded'] = encoder.fit_transform(train_data[CELL_COL])
    
    # Handle unseen categories in test data
    test_cells = test_data[CELL_COL].unique()
    train_cells = train_data[CELL_COL].unique()
    unseen_cells = set(test_cells) - set(train_cells)
    
    if unseen_cells:
        print(f"Warning: {len(unseen_cells)} unseen cells in test data")
        # Replace with most common cell from training
        most_common_cell = train_data[CELL_COL].mode()[0]
        test_data_copy = test_data.copy()
        for cell in unseen_cells:
            test_data_copy.loc[test_data_copy[CELL_COL] == cell, CELL_COL] = most_common_cell
        test_data = test_data_copy
    
    test_data['cell_encoded'] = encoder.transform(test_data[CELL_COL])
    
    return train_data, test_data


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
        print(X_train.columns)  # Show first 5 feature names
    except Exception as e:
        print(f"❌ Error: {e}")
