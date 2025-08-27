#!/usr/bin/env python3
"""
Algorithm Testing Script for Network Anomaly Detection
======================================================
This script tests all available ML algorithms to find the one with highest accuracy.

Usage:
    python test_algorithms.py

Features:
- Tests 12+ ML algorithms
- Focuses on accuracy as primary metric
- Generates performance visualizations
- Saves detailed results to CSV
- Provides comprehensive performance comparison
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

def main():
    """
    Main function to test all ML algorithms and find the best one.
    """
    print("🚀 STARTING COMPREHENSIVE ML ALGORITHM TESTING")
    print("=" * 60)
    
    try:
        # Import required modules
        from data_loader import load_data
        from preprocessor import preprocess_data
        from model_trainer import test_all_algorithms
        
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        train_data, test_data = load_data()
        X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]:,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Test samples: {X_test.shape[0]:,}")
        
        # Test all algorithms
        print("\n🧪 Testing all available ML algorithms...")
        results = test_all_algorithms(X_train, y_train, save_results=True)
        
        # Show final recommendations
        print("\n" + "="*60)
        print("🏆 FINAL RECOMMENDATIONS")
        print("="*60)
        
        best_model_name = results['best_model_name']
        best_accuracy = results['scores'][best_model_name]['cv_accuracy_mean']
        best_f1 = results['scores'][best_model_name]['cv_f1_mean']
        training_time = results['scores'][best_model_name]['training_time']
        
        print(f"🥇 BEST ALGORITHM: {best_model_name}")
        print(f"🎯 Accuracy: {best_accuracy:.4f}")
        print(f"📏 F1-Score: {best_f1:.4f}")
        print(f"⏱️  Training Time: {training_time:.2f}s")
        
        # Show top 3 algorithms
        sorted_models = sorted(results['scores'].items(), 
                             key=lambda x: x[1]['cv_accuracy_mean'], 
                             reverse=True)
        
        print("\n🏅 TOP 3 ALGORITHMS:")
        for i, (name, scores) in enumerate(sorted_models[:3], 1):
            print(f"  {i}. {name}: {scores['cv_accuracy_mean']:.4f} accuracy")
        
        # Usage recommendation
        print("\n💡 USAGE RECOMMENDATION:")
        print(f"   Update your main model training to use: {best_model_name}")
        print(f"   This algorithm achieved the highest accuracy of {best_accuracy:.4f}")
        
        print("\n📊 Check the 'results' folder for:")
        print("   - algorithm_comparison.png (performance charts)")
        print("   - algorithm_comparison_results.csv (detailed metrics)")
        
        print("\n✅ ALGORITHM TESTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all data files are in the 'data' directory")
        print("2. Check if all required packages are installed:")
        print("   pip install scikit-learn pandas numpy matplotlib")
        print("3. Optional: Install additional packages for more algorithms:")
        print("   pip install xgboost lightgbm catboost")
        return None

if __name__ == "__main__":
    results = main()
