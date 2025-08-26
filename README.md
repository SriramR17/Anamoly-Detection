# Network Anomaly Detection System

A professional machine learning system for detecting network anomalies using traffic metrics and behavioral patterns.

## 🎯 Overview

This system analyzes network traffic data to identify unusual patterns that may indicate security threats, performance issues, or anomalous behavior. It uses ensemble machine learning methods to achieve high accuracy in anomaly detection.

## 🏗️ Project Structure

```
CTS/
├── src/                    # Source code
│   ├── main.py            # Main pipeline orchestrator
│   ├── config.py          # Configuration and parameters
│   ├── data_loader.py     # Data loading and validation
│   ├── explorer.py        # Exploratory data analysis
│   ├── preprocessor.py    # Data preprocessing and feature engineering
│   └── model_trainer.py   # Model training and evaluation
├── data/                   # Raw data files
│   ├── ML-MATT-CompetitionQT1920_train.csv
│   └── ML-MATT-CompetitionQT1920_test.csv
├── results/                # Generated outputs
│   ├── predictions.csv     # Detailed predictions with probabilities
│   ├── simple_predictions.csv     # Competition submission format
│   └── exploration.png     # Data exploration visualizations
├── docs/                   # Documentation
│   ├── documentation.txt   # Complete project documentation
│   ├── SIMPLE_README.md    # Technical details
│   └── TEST_SUMMARY.md     # Testing and validation report
├── tests/                  # Test scripts and validation
│   └── test_outputs.py     # Output validation tests
├── config/                 # Configuration files
│   └── requirements.txt    # Python dependencies
├── models/                 # Trained model artifacts
│   ├── cellname_encoder.joblib
│   └── feature_scaler.joblib
└── scripts/               # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (Anaconda recommended)
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation
1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r config/requirements.txt
   ```

### Usage
Run the complete pipeline:
```bash
python main.py
```

Individual components can be tested:
```bash
python src/data_loader.py
python src/preprocessor.py
python src/model_trainer.py
python tests/test_outputs.py
```

## 📊 Results

- **Best Model**: RandomForest
- **Performance**: F1 Score = 0.812
- **Test Predictions**: 1,847 anomalies detected out of 9,158 samples (20.2%)
- **Key Insights**:
  - Peak anomaly hours: 22:00, 18:00, 20:00
  - Top anomaly cells: 5BLTE, 10CLTE, 7ALTE, 5CLTE, 6VLTE

## 🔧 Features

### Data Processing
- Handles encoding issues (latin-1)
- Cleans invalid numeric values
- Median imputation for missing data
- Feature scaling and normalization

### Feature Engineering
- Time-based features (hour, business hours, peak hours, night time)
- Traffic aggregations (total PRB, throughput, users)
- Efficiency ratios (UL/DL ratios, throughput per resource block)
- Cell name encoding

### Machine Learning
- Multiple model comparison (RandomForest, Logistic Regression, Gradient Boosting)
- Cross-validation evaluation
- Automated best model selection
- Probability-based predictions

## 📈 Output Files

- `results/predictions.csv`: Complete predictions with all features and probabilities
- `results/simple_predictions.csv`: Clean submission format (Index, Predicted_Unusual)
- `results/exploration.png`: Data exploration visualizations

## 📚 Documentation

Complete documentation is available in the `docs/` directory:
- `documentation.txt`: Comprehensive project documentation
- `SIMPLE_README.md`: Technical implementation details
- `TEST_SUMMARY.md`: Testing and validation report

## 🧪 Testing

The system includes comprehensive testing:
- Individual component validation
- Complete pipeline testing
- Output file validation
- Performance benchmarking

Run tests:
```bash
python tests/test_outputs.py
```

## 🔮 Future Enhancements

- Model persistence and loading
- Hyperparameter optimization
- Real-time prediction API
- Advanced feature selection
- Class imbalance handling
- Logging system

## 📜 License

This project is for educational and research purposes.

## 👥 Contributors

Developed and tested using professional ML engineering practices.

---

For detailed technical documentation, see `docs/documentation.txt`
