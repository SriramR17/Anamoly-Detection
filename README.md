# Network Anomaly Detection System

A high-performance machine learning system for detecting network anomalies in telecom traffic data using advanced gradient boosting algorithms.

## 🎯 Project Overview

This project implements a comprehensive anomaly detection pipeline for network traffic analysis. The system processes telecom network metrics to identify unusual patterns that may indicate security threats, performance issues, or network failures.

**Key Achievement**: Achieved **98.93% accuracy** and **98.03% F1-score** using CatBoost algorithm, outperforming 10 other machine learning algorithms in comprehensive testing.

## 📊 Performance Results

### Best Model Performance (CatBoost)
- **Accuracy**: 98.93% ± 0.15%
- **F1-Score**: 98.03% ± 0.29%
- **Precision**: 99.51% ± 0.13%
- **Recall**: 96.59% ± 0.53%
- **Training Time**: 0.94 seconds
- **Anomalies Detected**: 2,372 out of 9,158 test samples (25.9%)

### Algorithm Comparison Results
| Rank | Algorithm | Accuracy | F1-Score | Training Time |
|------|-----------|----------|----------|---------------|
| 🥇 1 | **CatBoost** | **0.9893** | **0.9803** | 0.94s |
| 🥈 2 | LightGBM | 0.9889 | 0.9795 | 0.15s |
| 🥉 3 | XGBoost | 0.9825 | 0.9675 | 0.25s |
| 4 | RandomForest | 0.9158 | 0.8246 | 0.78s |
| 5 | GradientBoosting | 0.9091 | 0.8030 | 9.63s |

*Full results with 11 algorithms tested available in `results/algorithm_comparison_results.csv`*

## 🏗️ Project Structure

```
CTS/
├── src/                           # Core source code
│   ├── main.py                   # Main pipeline orchestrator
│   ├── config.py                 # Configuration parameters
│   ├── data_loader.py            # Data loading and validation
│   ├── explorer.py               # Data exploration and visualization
│   ├── preprocessor.py           # Feature engineering and preprocessing
│   └── model_trainer.py          # ML model training and evaluation
├── data/                          # Raw dataset
│   ├── ML-MATT-CompetitionQT1920_train.csv  (36,904 samples)
│   └── ML-MATT-CompetitionQT1920_test.csv   (9,158 samples)
├── models/                        # Trained model artifacts
│   ├── best_model.joblib         # CatBoost model (98.93% accuracy)
│   ├── cellname_encoder.joblib   # Cell name label encoder
│   └── feature_scaler.joblib     # Feature scaling transformer
├── results/                       # Generated outputs and analysis
│   ├── algorithm_comparison_results.csv    # Complete algorithm benchmarks
│   ├── algorithm_comparison.png            # Performance visualizations
│   ├── predictions.csv                     # Detailed predictions with probabilities
│   ├── simple_predictions.csv              # Competition submission format
│   └── exploration.png                     # Data exploration plots
├── tests/                         # Testing and validation scripts
│   ├── test_algorithms.py        # Comprehensive algorithm testing
│   ├── test_advanced.py          # Advanced testing scenarios
│   └── quick_test.py             # Quick validation tests
├── real_time_testing/             # Real-time detection capabilities
│   ├── network_monitor.py        # Network monitoring utilities
│   └── realtime_detector.py      # Real-time anomaly detection
├── main.py                        # Main application entry point
├── save_best_model.py            # Model persistence script
└── config.py                     # Global configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- Conda environment recommended
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, catboost, xgboost, lightgbm

### Installation
1. **Clone/download the project**:
   ```bash
   git clone [repository] CTS
   cd CTS
   ```

2. **Set up conda environment**:
   ```bash
   conda create -n anomaly-detection python=3.8
   conda activate anomaly-detection
   ```

3. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   pip install catboost xgboost lightgbm  # For advanced algorithms
   ```

### Usage

#### Run Complete Pipeline
```bash
python main.py
```

#### Test All Algorithms (Find Best Model)
```bash
python tests/test_algorithms.py
```

#### Train and Save Best Model
```bash
python save_best_model.py
```

## 📈 Data Analysis Insights

### Dataset Characteristics
- **Training Data**: 36,904 samples with 14 features
- **Test Data**: 9,158 samples with 13 features
- **Target Distribution**: 72.4% Normal (26,721), 27.6% Anomaly (10,183)
- **Missing Values**: 183 in training data, 6 in test data (handled via median imputation)

### Key Patterns Discovered
- **Peak Anomaly Hours**: 22:00 (10 PM), 18:00 (6 PM), 20:00 (8 PM)
- **High-Risk Cells**: 5BLTE, 10CLTE, 7ALTE, 5CLTE, 6VLTE
- **Most Important Features** (CatBoost):
  1. meanUE_UL (38.77%) - Mean uplink user equipment
  2. meanUE_DL (23.43%) - Mean downlink user equipment  
  3. PRBUsageDL (14.77%) - Physical Resource Block usage downlink
  4. total_users (12.41%) - Total active users
  5. PRBUsageUL (5.73%) - Physical Resource Block usage uplink

### Feature Engineering
- **Time-based features**: Hour extraction, business hours flag, peak hours detection
- **Traffic aggregations**: Total PRB usage, combined throughput, user counts
- **Efficiency ratios**: UL/DL ratios, throughput per resource block
- **Categorical encoding**: Cell name label encoding
- **Scaling**: StandardScaler for numerical features

## 🔬 Machine Learning Pipeline

### 1. Data Loading & Validation
- Automatic encoding detection (latin-1 for compatibility)
- Missing value detection and reporting
- Data type validation and cleaning

### 2. Exploratory Data Analysis
- Statistical summaries and distributions
- Correlation analysis and feature relationships
- Time-pattern analysis for anomaly detection
- Cell-level anomaly distribution analysis

### 3. Preprocessing & Feature Engineering
- Missing value imputation using median strategy
- Feature scaling with StandardScaler
- Categorical variable encoding
- New feature creation (time-based, aggregated, ratios)

### 4. Model Training & Selection
- **11 algorithms tested**: CatBoost, LightGBM, XGBoost, RandomForest, GradientBoosting, ExtraTrees, DecisionTree, AdaBoost, KNN, LogisticRegression, GaussianNB
- **5-fold cross-validation** for robust evaluation
- **Multiple metrics**: Accuracy, F1-Score, Precision, Recall
- **Automatic best model selection** based on accuracy

### 5. Model Persistence & Deployment
- Best model saved using joblib for fast loading
- Feature transformers persisted for consistent preprocessing
- Model validation with sample predictions

## 📁 Output Files

### Algorithm Testing Results
- `results/algorithm_comparison_results.csv`: Complete performance metrics for all 11 algorithms
- `results/algorithm_comparison.png`: Comparative performance visualizations

### Predictions
- `results/predictions.csv`: Detailed predictions with confidence scores and all features
- `results/simple_predictions.csv`: Clean submission format (Index, Predicted_Unusual)

### Data Analysis
- `results/exploration.png`: Data exploration visualizations including distributions, patterns, and correlations

### Model Artifacts
- `models/best_model.joblib`: Trained CatBoost model (98.93% accuracy)
- `models/cellname_encoder.joblib`: Cell name encoder for categorical features
- `models/feature_scaler.joblib`: Feature scaler for numerical normalization

## 🧪 Testing & Validation

### Comprehensive Algorithm Testing
The system includes extensive testing capabilities:
- **Algorithm benchmarking**: Tests 11 different ML algorithms
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Multiple metrics**: Accuracy, F1, Precision, Recall evaluation
- **Performance visualization**: Automated chart generation
- **Statistical analysis**: Mean and standard deviation reporting

### Model Validation
- **Loading verification**: Tests model persistence and loading
- **Prediction validation**: Verifies model inference capabilities
- **Feature importance**: Analyzes and reports key predictive features

## 💡 Technical Highlights

### Advanced Features
- **Gradient Boosting Excellence**: Top 3 models are all gradient boosting variants
- **Feature Importance Analysis**: Identifies most predictive network metrics
- **Time Pattern Recognition**: Detects peak anomaly time periods
- **Cell-Level Analysis**: Identifies high-risk network cells
- **Robust Preprocessing**: Handles missing data and encoding issues
- **Cross-Validation**: Ensures model generalization

### Performance Optimization
- **Fast Training**: Best model trains in under 1 second
- **High Accuracy**: 98.93% accuracy with 98.03% F1-score
- **Low False Positives**: 99.51% precision minimizes false alarms
- **Good Recall**: 96.59% recall ensures anomaly detection coverage

## 🔮 Use Cases & Applications

### Network Operations
- **Real-time monitoring**: Detect network anomalies as they occur
- **Predictive maintenance**: Identify potential network issues before failures
- **Security monitoring**: Detect unusual traffic patterns indicating attacks
- **Performance optimization**: Identify underperforming network segments

### Business Intelligence
- **Traffic analysis**: Understand network usage patterns
- **Resource planning**: Optimize network resource allocation
- **Quality assurance**: Monitor service quality metrics
- **Incident response**: Rapid identification of network issues

## 🎯 Future Enhancements

### Technical Improvements
- **Hyperparameter optimization**: Automated parameter tuning using Optuna/GridSearch
- **Ensemble methods**: Combine top-performing models for even better accuracy
- **Feature selection**: Automated feature selection to reduce dimensionality
- **Real-time API**: REST API for real-time anomaly detection
- **Streaming processing**: Apache Kafka integration for continuous monitoring

### Operational Features
- **Alerting system**: Automated notifications for detected anomalies
- **Dashboard**: Web-based monitoring dashboard
- **Model retraining**: Automated model updates with new data
- **Explainability**: SHAP values for prediction explanations
- **A/B testing**: Framework for model comparison in production

## 📊 Benchmarking Results

### Algorithm Performance Summary
```
CatBoost      : 98.93% accuracy (🏆 WINNER)
LightGBM      : 98.89% accuracy
XGBoost       : 98.25% accuracy
RandomForest  : 91.58% accuracy
GradientBoost : 90.91% accuracy
ExtraTrees    : 87.83% accuracy
DecisionTree  : 87.66% accuracy
AdaBoost      : 81.20% accuracy
KNN           : 74.86% accuracy
LogisticReg   : 71.96% accuracy
GaussianNB    : 32.77% accuracy
```

### Key Insights
1. **Gradient boosting algorithms dominate**: Top 3 are all gradient boosting variants
2. **CatBoost provides best balance**: Highest accuracy with reasonable training time
3. **Tree-based methods excel**: All top performers use tree-based approaches
4. **Linear methods struggle**: Logistic regression shows poor performance on this dataset
5. **Speed vs accuracy tradeoff**: LightGBM offers fastest training with excellent accuracy

## 📜 Technical Specifications

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, CatBoost, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Cross-Validation**: 5-fold stratified CV
- **Evaluation Metrics**: Accuracy, F1-Score, Precision, Recall
- **Feature Engineering**: 23 engineered features from 14 raw features



