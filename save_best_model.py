#!/usr/bin/env python3
"""
Script to train and save the best performing model (LightGBM) to models directory.
Based on our comprehensive algorithm testing, LightGBM achieved superior performance.
"""

import sys
import os
from pathlib import Path
import joblib

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """
    Train the best model (LightGBM) and save it to models directory.
    """
    print("🚀 Training and saving the best model (LightGBM)...")
    print("=" * 60)
    
    try:
        # Import required modules
        from src.data_loader import load_data
        from src.preprocessor import preprocess_data
        from src.model_trainer import train_models
        from config import MODEL_PARAMS
        
        # Import LightGBM
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        print("📊 Loading and preprocessing data...")
        # Load and preprocess data
        train_data, test_data = load_data()
        X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]:,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Test samples: {X_test.shape[0]:,}")
        
        # Train the best model (LightGBM) with optimal parameters
        print("\n🌟 Training LightGBM (best performing algorithm)...")
        
        # Use optimized parameters from config or fall back to defaults
        lgbm_params = MODEL_PARAMS.get('LightGBM', {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        })
        
        best_model = LGBMClassifier(**lgbm_params)
        
        # Train the model
        print("   Training on full dataset...")
        best_model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        print("   Evaluating performance with cross-validation...")
        cv_accuracy = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        cv_f1 = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
        cv_precision = cross_val_score(best_model, X_train, y_train, cv=5, scoring='precision')
        cv_recall = cross_val_score(best_model, X_train, y_train, cv=5, scoring='recall')
        
        print(f"\n📊 Model Performance (5-fold CV):")
        print(f"   🎯 Accuracy:  {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
        print(f"   📏 F1-Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"   🎪 Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
        print(f"   🔍 Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
        
        # Save the model
        model_path = models_dir / "best_model.joblib"
        print(f"\n💾 Saving model to: {model_path.absolute()}")
        
        joblib.dump(best_model, model_path)
        
        print("✅ Model saved successfully!")
        
        # Test loading the model to ensure it works
        print("\n🧪 Testing model loading...")
        loaded_model = joblib.load(model_path)
        
        # Make a quick prediction to verify
        sample_prediction = loaded_model.predict(X_test[:5])
        sample_probabilities = loaded_model.predict_proba(X_test[:5])
        
        print("✅ Model loading test successful!")
        print(f"   Sample predictions: {sample_prediction}")
        print(f"   Sample probabilities shape: {sample_probabilities.shape}")
        
        # Show model information
        print(f"\n📋 Model Information:")
        print(f"   Algorithm: LightGBM")
        print(f"   Parameters: {best_model.get_params()}")
        print(f"   Features used: {X_train.shape[1]}")
        print(f"   Classes: {best_model.classes_}")
        
        # Feature importance (top 10)
        if hasattr(best_model, 'feature_importances_'):
            print(f"\n🔝 Top 10 Most Important Features:")
            feature_importance = list(zip(features, best_model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(feature_importance[:10], 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Best model saved and ready for use! 🌟🔥")
        print("=" * 60)
        print(f"📁 Model location: {model_path.absolute()}")
        print(f"🏆 Algorithm: LightGBM")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    saved_path = main()
    if saved_path:
        print(f"\n✅ Model successfully saved to: {saved_path}")
    else:
        print("\n❌ Failed to save model")
