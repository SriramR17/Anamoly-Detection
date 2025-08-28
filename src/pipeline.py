"""
Simple Network Anomaly Detection System
=======================================
Clean, easy-to-understand main script that runs the complete pipeline.
"""

import pandas as pd
import numpy as np
from data_loader import load_data
from explorer import explore_data
from preprocessor import preprocess_data
from model_trainer import train_models, make_predictions, get_or_train_model
from config import OUTPUT_DIR, PREDICTIONS_FILE


def run_anomaly_detection():
    """
    Run the complete anomaly detection pipeline.
    
    This function does everything in a simple, step-by-step manner:
    1. Load data
    2. Explore data (optional)
    3. Preprocess data
    4. Train models
    5. Make predictions
    6. Save results
    """
    print("="*60)
    print("NETWORK ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    try:
        # Step 1: Load data
        print("\n1. Loading data...")
        train_data, test_data = load_data()
        
        # Step 2: Explore data (optional - comment out to skip)
        print("\n2. Exploring data...")
        exploration_results = explore_data(train_data)
        
        # Step 3: Preprocess data
        print("\n3. Preprocessing data...")
        X_train, y_train, X_test, feature_names = preprocess_data(train_data, test_data)
        
        # Step 4: Train models (or load existing)
        print("\n4. Loading/Training models...")
        model_results = get_or_train_model(X_train, y_train)
        
        # Step 5: Make predictions
        print("\n5. Making predictions...")
        predictions, probabilities = make_predictions(model_results['best_model'], X_test)
        
        # Step 6: Save results
        print("\n6. Saving results...")
        save_results(test_data, predictions, probabilities)
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {model_results['best_model']}")
        print(f"Predicted anomalies: {np.sum(predictions):,} out of {len(predictions):,}")
        print(f"Results saved to: {OUTPUT_DIR}")
        
        return {
            'exploration': exploration_results,
            'model_results': model_results,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return None


def save_results(test_data, predictions, probabilities):
    """Save prediction results to CSV file."""
    
    # Create results dataframe
    results_df = test_data.copy()
    results_df['Predicted_Anomaly'] = predictions
    results_df['Anomaly_Probability'] = probabilities
    
    # Save detailed results
    results_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"✓ Results saved to: {PREDICTIONS_FILE}")
    
    # Also save simple submission format
    simple_submission = pd.DataFrame({
        'Index': range(len(predictions)),
        'Predicted_Unusual': predictions
    })
    
    simple_file = OUTPUT_DIR / "simple_predictions.csv"
    simple_submission.to_csv(simple_file, index=False)
    print(f"✓ Simple predictions saved to: {simple_file}")


if __name__ == "__main__":
    # Run the complete system
    results = run_anomaly_detection()
    
    if results:
        print("\n🎉 All done! Check the output folder for your results.")
