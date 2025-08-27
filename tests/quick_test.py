#!/usr/bin/env python3
"""
Quick test script to verify enhanced model trainer functionality.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("🧪 Running quick test of enhanced model trainer...")
    
    try:
        # Import modules
        from src.model_trainer import train_models
        from src.data_loader import load_data
        from src.preprocessor import preprocess_data
        
        # Load data (subset for quick test)
        print("📊 Loading data...")
        train_data, test_data = load_data()
        X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
        
        # Use subset for quick test
        subset_size = 2000
        X_train_small = X_train[:subset_size]
        y_train_small = y_train[:subset_size]
        
        print(f"✅ Using subset of {subset_size} samples for quick test")
        
        # Test model training with focus on accuracy
        print("🚀 Training models (focus on accuracy)...")
        results = train_models(X_train_small, y_train_small, focus_on_accuracy=True)
        
        # Show results
        best_model = results['best_model_name']
        best_accuracy = results['scores'][best_model]['cv_accuracy_mean']
        
        print(f"\n✅ QUICK TEST SUCCESSFUL!")
        print(f"🏆 Best model: {best_model}")
        print(f"🎯 Best accuracy: {best_accuracy:.4f}")
        
        # Test making predictions
        print("\n🔮 Testing prediction functionality...")
        from src.model_trainer import make_predictions
        predictions, probabilities = make_predictions(results['best_model'], X_test[:100])
        
        print("✅ Predictions successful!")
        print(f"   Predicted {len(predictions)} samples")
        print(f"   Anomaly rate: {(predictions.sum() / len(predictions) * 100):.1f}%")
        
        print("\n🎉 ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
