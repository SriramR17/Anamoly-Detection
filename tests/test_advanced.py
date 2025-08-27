#!/usr/bin/env python3
"""
Test script for advanced algorithms (XGBoost, LightGBM).
"""
import sys
import os
from pathlib import Path

# Set working directory and add src to path
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("🧪 Testing advanced algorithms (XGBoost, LightGBM)...")
    
    try:
        # Import modules
        from src.model_trainer import train_models
        from src.data_loader import load_data
        from src.preprocessor import preprocess_data
        
        # Load data
        print("📊 Loading data...")
        train_data, test_data = load_data()
        X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
        
        # Use smaller subset for quick test
        subset_size = 1500
        X_train_small = X_train[:subset_size]
        y_train_small = y_train[:subset_size]
        
        print(f"✅ Using subset of {subset_size} samples for advanced algorithm test")
        
        # Train models (should now include XGBoost and LightGBM)
        print("🚀 Training models with advanced algorithms...")
        results = train_models(X_train_small, y_train_small, focus_on_accuracy=True)
        
        # Show results
        best_model = results['best_model_name']
        best_accuracy = results['scores'][best_model]['cv_accuracy_mean']
        
        print(f"\n✅ ADVANCED ALGORITHM TEST SUCCESSFUL!")
        print(f"🏆 Best model: {best_model}")
        print(f"🎯 Best accuracy: {best_accuracy:.4f}")
        
        # Show all trained models
        print(f"\n📊 Trained {len(results['models'])} algorithms:")
        for name in results['models'].keys():
            acc = results['scores'][name]['cv_accuracy_mean']
            print(f"   • {name}: {acc:.4f}")
        
        print("\n🎉 ADVANCED ALGORITHM TESTING COMPLETE!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
