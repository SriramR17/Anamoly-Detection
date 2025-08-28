"""
Simplified Model Trainer for Network Anomaly Detection
======================================================
Clean, streamlined model training with essential algorithms.
"""

import pandas as pd
import numpy as np
import warnings
import time
import joblib
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import MODEL_PARAMS

warnings.filterwarnings('ignore')


def get_model_instance(model_name, params):
    """Create model instance with optional dependency handling."""
    model_map = {
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'GradientBoosting': GradientBoostingClassifier,
        'DecisionTree': DecisionTreeClassifier,
        'ExtraTrees': ExtraTreesClassifier,
        'KNN': KNeighborsClassifier,
        'GaussianNB': GaussianNB,
    }
    
    # Handle optional libraries
    if model_name == 'XGBoost':
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        except ImportError:
            print(f"⚠ XGBoost not available, skipping...")
            return None
    
    elif model_name == 'LightGBM':
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        except ImportError:
            print(f"⚠ LightGBM not available, skipping...")
            return None
        
    elif model_name == 'CatBoost':
        try:
            import catboost as cb
            return cb.CatBoostClassifier(**params)
        except ImportError:
            print(f"  ⚠ CatBoost not available, skipping...")
            return None
    
    elif model_name in model_map:
        return model_map[model_name](**params)
    
    return None


def train_models(X_train, y_train):
    """Train multiple models and return the best one."""
    print(f"🚀 Training models on dataset shape: {X_train.shape}")
    print(f"🎯 Target distribution: {np.bincount(y_train)}\n")
    
    models = {}
    scores = {}
    
    # Initialize available models
    for model_name, params in MODEL_PARAMS.items():
        model = get_model_instance(model_name, params)
        if model is not None:
            models[model_name] = model
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"🎯 Training {name}...")
        
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Get cross-validation scores
            cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            
            scores[name] = {
                'model': model,
                'accuracy': cv_accuracy.mean(),
                'accuracy_std': cv_accuracy.std(),
                'f1': cv_f1.mean(),
                'f1_std': cv_f1.std(),
                'time': training_time
            }
            
            print(f"  ✅ Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
            print(f"  📏 F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
            print(f"  ⏱️ Time: {training_time:.2f}s\n")
            
        except Exception as e:
            print(f"  ❌ Error training {name}: {e}\n")
    
    if not scores:
        raise Exception("No models were successfully trained!")
    
    # Select best model by accuracy
    best_name = max(scores.keys(), key=lambda x: scores[x]['accuracy'])
    best_model = scores[best_name]['model']
    
    # Print results summary
    print("="*60)
    print("🏆 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':<10} {'F1':<10} {'Time(s)':<8}")
    print("-" * 45)
    
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, score_dict in sorted_models:
        print(f"{name:<15} {score_dict['accuracy']:.4f}     {score_dict['f1']:.4f}     {score_dict['time']:.2f}")
    
    print(f"\n🥇 BEST MODEL: {best_name}")
    print(f"🎯 Accuracy: {scores[best_name]['accuracy']:.4f}")
    print("="*60 + "\n")
    
    return {
        'best_model': best_model,
        'best_name': best_name,
        'all_scores': scores
    }


def make_predictions(model, X_test):
    """Make predictions using the trained model."""
    print("Making predictions...")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    anomaly_count = np.sum(predictions)
    anomaly_rate = np.mean(predictions) * 100
    
    print(f"✓ Total samples: {len(predictions):,}")
    print(f"✓ Predicted anomalies: {anomaly_count:,} ({anomaly_rate:.1f}%)")
    
    return predictions, probabilities



def get_or_train_model(X_train, y_train, model_path=None, force_retrain=False):
    """Load existing model or train new one."""
    # Set default path
    if model_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        model_path = project_root / 'models/best_model.joblib'
    else:
        model_path = Path(model_path)
    
    # Check for existing model
    if not force_retrain and model_path.exists():
        print(f"✓ Loading existing model from {model_path}")
        try:
            model = joblib.load(model_path)
            return {
                'best_model': model,
                'best_name': model.__class__.__name__,
                'loaded': True
            }
        except Exception as e:
            print(f"⚠ Could not load model ({e}). Training new one...")
    
    # Train new model
    if force_retrain:
        print("🔄 Force retrain requested. Training new models...")
    else:
        print(f"✗ No existing model found. Training new models...")
    
    results = train_models(X_train, y_train)
    
    # Save new model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results['best_model'], model_path)
    print(f"✓ Best model saved to {model_path}")
    
    return results



if __name__ == "__main__":
    MODEL_PATH = 'models/best_model.joblib'
    
    try:
        from data_loader import load_data
        from preprocessor import preprocess_data
        
        # Check for existing model
        if os.path.exists(MODEL_PATH):
            print("✓ Loading existing model...")
            best_model = joblib.load(MODEL_PATH)
        else:
            print("✗ Training new model...")
            
            # Load and preprocess data
            train_data, test_data = load_data()
            X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
            
            # Train and save model
            results = train_models(X_train, y_train)
            best_model = results['best_model']
            joblib.dump(best_model, MODEL_PATH)
            print(f"✓ Model saved to {MODEL_PATH}")
        
        # Make predictions
        predictions, probabilities = make_predictions(best_model, X_test)
        print("✓ Process complete.")
        
    except Exception as e:
        print(f"❌ Error: {e}")