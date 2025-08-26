#!/usr/bin/env python3
"""
Network Anomaly Detection System - Main Launcher
================================================
Professional entry point for the anomaly detection pipeline.

This script launches the complete anomaly detection system from the project root.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main pipeline
from main import run_anomaly_detection

def main():
    """Main entry point for the anomaly detection system."""
    
    print("🚀 Network Anomaly Detection System")
    print("=" * 50)
    print("Professional ML Pipeline for Network Security")
    print("=" * 50)
    
    # Run the complete pipeline
    results = run_anomaly_detection()
    
    if results:
        print("\n" + "=" * 50)
        print("✅ SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print("\n📊 Quick Summary:")
        print(f"   • Best Model: {results['model_results']['best_model_name']}")
        print(f"   • Anomalies Detected: {results['predictions'].sum():,}")
        print(f"   • Total Samples: {len(results['predictions']):,}")
        print(f"   • Anomaly Rate: {results['predictions'].mean()*100:.1f}%")
        
        print("\n📁 Output Files:")
        print("   • results/predictions.csv - Detailed predictions")
        print("   • results/simple_predictions.csv - Competition format")
        print("   • results/exploration.png - Data visualizations")
        
        print("\n🎯 Next Steps:")
        print("   • Review results in results/ directory")
        print("   • Check docs/documentation.txt for details")
        print("   • Run tests/test_outputs.py for validation")
        
    else:
        print("\n" + "=" * 50)
        print("❌ SYSTEM EXECUTION FAILED")
        print("=" * 50)
        print("Please check the error messages above and:")
        print("• Ensure all dependencies are installed")
        print("• Verify data files exist in data/ directory")
        print("• Check docs/documentation.txt for troubleshooting")

if __name__ == "__main__":
    main()
