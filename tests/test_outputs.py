#!/usr/bin/env python3
"""
Test script to validate the output files from the anomaly detection system.
"""

import pandas as pd
import os

def test_output_files():
    """Test all generated output files."""
    
    print("=== OUTPUT FILE VALIDATION ===")
    
    # Test predictions.csv
    print("\n1. Testing predictions.csv...")
    if os.path.exists('results\predictions.csv'):
        df = pd.read_csv('results\predictions.csv')
        print(f"   ✓ Shape: {df.shape}")
        print(f"   ✓ Has prediction columns: {'Predicted_Anomaly' in df.columns and 'Anomaly_Probability' in df.columns}")
        print(f"   ✓ Anomaly predictions: {df['Predicted_Anomaly'].sum()} out of {len(df)} ({df['Predicted_Anomaly'].mean()*100:.1f}%)")
        print(f"   ✓ Probability range: {df['Anomaly_Probability'].min():.3f} to {df['Anomaly_Probability'].max():.3f}")
    else:
        print("   ❌ predictions.csv not found")
    
    # Test simple_predictions.csv
    print("\n2. Testing simple_predictions.csv...")
    if os.path.exists('results\simple_predictions.csv'):
        simple = pd.read_csv('results\simple_predictions.csv')
        print(f"   ✓ Shape: {simple.shape}")
        print(f"   ✓ Columns: {list(simple.columns)}")
        print(f"   ✓ Sample data:\n{simple.head()}")
    else:
        print("   ❌ simple_predictions.csv not found")
    
    # Test exploration.png
    print("\n3. Testing exploration.png...")
    if os.path.exists('results\exploration.png'):
        size = os.path.getsize('results\exploration.png')
        print(f"   ✓ File exists, size: {size:,} bytes")
    else:
        print("   ❌ exploration.png not found")
    
    # Check all output files
    print("\n4. All results files:")
    if os.path.exists('results'):
        output_files = os.listdir('results')
        for f in sorted(output_files):
            size = os.path.getsize(f'results\{f}')
            print(f"   - {f} ({size:,} bytes)")
    else:
        print("   ❌ results directory not found")
    
    print("\n=== VALIDATION COMPLETE ===")

if __name__ == "__main__":
    test_output_files()
