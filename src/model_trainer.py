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
import os
import joblib


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


def get_or_train_model(X_train, y_train, model_path=None, force_retrain=False):
    """
    Load existing model or train new one if none exists.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        model_path: Path to saved model file (defaults to project root)
        force_retrain: If True, force retraining even if model exists
        
    Returns:
        dict: Results including best model and metadata
    """
    import os
    import joblib
    from pathlib import Path
    
    # Default model path - look in project root
    if model_path is None:
        # Get the project root (parent of src directory)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        model_path = project_root / 'models/best_model.joblib'
    else:
        model_path = Path(model_path)
    
    # Force retraining if requested
    if force_retrain:
        print("🔄 Force retrain requested. Training new models...")
    elif model_path.exists():
        print(f"✓ Found existing model at {model_path}. Loading...")
        try:
            best_model = joblib.load(model_path)
            
            # Create a result structure similar to train_models output
            results = {
                'models': {'LoadedModel': best_model},
                'scores': {'LoadedModel': {'cv_f1_mean': 'N/A (pre-trained)', 'cv_f1_std': 'N/A'}},
                'best_model_name': 'LoadedModel',
                'best_model': best_model
            }
            
            print("✓ Model loaded successfully. Skipping training.")
            return results
            
        except Exception as e:
            print(f"⚠ Warning: Could not load existing model ({e}). Training new model...")
    else:
        print(f"✗ No existing model found at {model_path}. Training new models...")
    
    # Train new models if no existing model, loading failed, or force retrain
    results = train_models(X_train, y_train)
    
    # Save the newly trained model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results['best_model'], model_path)
    print(f"✓ Best model saved to {model_path}")
    
    return results


def force_retrain_model(X_train, y_train, model_path=None):
    """
    Force retrain and save a new model, overwriting any existing one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_path: Path to save the model (defaults to project root)
        
    Returns:
        dict: Results including trained models and scores
    """
    return get_or_train_model(X_train, y_train, model_path, force_retrain=True)


if __name__ == "__main__":
    MODEL_PATH = 'models/best_model.joblib'
    
    try:
        # Import necessary modules
        from data_loader import load_data
        from preprocessor import preprocess_data
        
        best_model = None

        # Check if a model file exists
        if os.path.exists(MODEL_PATH):
            print("✓ Found existing model. Loading...")
            best_model = joblib.load(MODEL_PATH)
            
        else:
            # If no model is found, proceed with training
            print("✗ No existing model found. Training new models...")
            
            # Load and preprocess data
            train_data, test_data = load_data()
            X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
            
            # Train models
            results = train_models(X_train, y_train)
            best_model = results['best_model']
            
            # Save the best model for future use
            joblib.dump(best_model, MODEL_PATH)
            print(f"✓ Best model saved to {MODEL_PATH}")
        
        # Now that we have a model (either loaded or newly trained), make predictions
        predictions, probabilities = make_predictions(best_model, X_test)
        
        # This part of the code is for testing purposes.
        # It assumes `y_test` is available, which isn't the case in your original `make_predictions` call.
        # If `y_test` is available, you would uncomment and use this:
        # evaluate_model(best_model, X_test, y_test)
        
        print("✓ Model training and prediction process complete.")
        
    except Exception as e:
        print(f"❌ Error: {e}")