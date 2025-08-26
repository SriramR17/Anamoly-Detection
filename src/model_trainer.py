"""
Simple Model Trainer for Network Anomaly Detection
==================================================
Clean, straightforward model training with essential algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_PARAMS, RANDOM_STATE


def train_models(X_train, y_train):
    """
    Train multiple models and select the best one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Results including trained models and scores
    """
    print("Training models...")
    
    models = {
        'RandomForest': RandomForestClassifier(**MODEL_PARAMS['RandomForest']),
        'LogisticRegression': LogisticRegression(**MODEL_PARAMS['LogisticRegression']),
        'GradientBoosting': GradientBoostingClassifier(**MODEL_PARAMS['GradientBoosting'])
    }
    
    results = {}
    scores = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
        
        # Store model and scores
        results[name] = model
        scores[name] = {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
        
        print(f"    CV F1 Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Select best model based on F1 score
    best_model_name = max(scores.keys(), key=lambda x: scores[x]['cv_f1_mean'])
    best_model = results[best_model_name]
    
    print(f"\n✓ Best model: {best_model_name}")
    print(f"  F1 Score: {scores[best_model_name]['cv_f1_mean']:.3f}")
    
    return {
        'models': results,
        'scores': scores,
        'best_model_name': best_model_name,
        'best_model': best_model
    }


def make_predictions(model, X_test):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    print("Making predictions...")
    
    # Generate predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of anomaly
    
    anomaly_count = np.sum(predictions)
    anomaly_rate = np.mean(predictions) * 100
    
    print(f"✓ Predictions complete:")
    print(f"  Total samples: {len(predictions):,}")
    print(f"  Predicted anomalies: {anomaly_count:,} ({anomaly_rate:.1f}%)")
    
    return predictions, probabilities


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance (if you have true labels for test data).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test labels
        
    Returns:
        dict: Evaluation metrics
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    
    print("Model Evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.3f}")
    
    return metrics


if __name__ == "__main__":
    # Test the model trainer
    try:
        from data_loader import load_data
        from preprocessor import preprocess_data
        
        # Load and preprocess data
        train_data, test_data = load_data()
        X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
        
        # Train models
        results = train_models(X_train, y_train)
        
        # Make predictions
        predictions, probabilities = make_predictions(results['best_model'], X_test)
        
        print("✓ Model training test successful")
        
    except Exception as e:
        print(f"❌ Error: {e}")
