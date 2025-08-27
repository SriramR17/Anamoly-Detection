# Technical Documentation - Network Anomaly Detection System

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Pipeline](#data-pipeline)
3. [Model Implementation](#model-implementation)
4. [API Reference](#api-reference)
5. [Performance Analysis](#performance-analysis)
6. [Deployment Guide](#deployment-guide)

## System Architecture

### Core Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Data Loader       │    │   Preprocessor      │    │   Model Trainer    │
│  (data_loader.py)   │───▶│ (preprocessor.py)   │───▶│ (model_trainer.py)  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│     Explorer        │    │   Feature Store     │    │   Model Registry    │
│   (explorer.py)     │    │   (transformers)    │    │   (models/*.joblib) │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Technology Stack

- **Core ML Framework**: scikit-learn 1.3+
- **Advanced Models**: CatBoost, XGBoost, LightGBM
- **Data Processing**: pandas 2.0+, numpy 1.24+
- **Visualization**: matplotlib 3.7+, seaborn 0.12+
- **Model Persistence**: joblib 1.3+
- **Configuration**: Python configparser

## Data Pipeline

### 1. Data Loading (`src/data_loader.py`)

```python
def load_data():
    """
    Load training and test datasets with automatic encoding detection.
    
    Returns:
        tuple: (train_data, test_data) pandas DataFrames
    """
```

**Features:**
- Automatic encoding detection (handles latin-1, utf-8)
- Missing value detection and reporting
- Data shape validation
- Type inference and cleaning

**Data Schema:**
```
Training Data: (36,904 × 14)
Test Data: (9,158 × 13)

Columns:
- CellName: Network cell identifier
- PRBUsageUL/DL: Physical Resource Block usage
- meanThr_UL/DL: Mean throughput metrics
- maxThr_UL/DL: Maximum throughput values
- meanUE_UL/DL: Mean user equipment counts
- maxUE_UL/DL: Maximum user equipment counts
- Time: Timestamp information
- Unusual: Target variable (training only)
```

### 2. Data Exploration (`src/explorer.py`)

```python
def explore_data(data):
    """
    Comprehensive exploratory data analysis.
    
    Args:
        data: pandas DataFrame with network metrics
        
    Returns:
        dict: Analysis results including patterns and insights
    """
```

**Analysis Components:**
- Statistical summaries and distributions
- Missing value patterns
- Target variable distribution
- Time-based anomaly patterns
- Cell-level anomaly analysis
- Feature correlation matrices
- Outlier detection

### 3. Data Preprocessing (`src/preprocessor.py`)

```python
def preprocess_data(train_data, test_data):
    """
    Complete preprocessing pipeline with feature engineering.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        
    Returns:
        tuple: (X_train, y_train, X_test, feature_names)
    """
```

**Preprocessing Steps:**
1. **Missing Value Imputation**: Median strategy for numerical features
2. **Feature Engineering**:
   - Time extraction (hour from timestamp)
   - Business hours flag (9 AM - 5 PM)
   - Peak hours detection
   - Night time indicator
   - Traffic aggregations
   - Efficiency ratios
3. **Categorical Encoding**: Label encoding for cell names
4. **Feature Scaling**: StandardScaler for numerical features
5. **Feature Selection**: 23 engineered features from 14 raw features

**Engineered Features:**
```python
# Time-based features
'Hour', 'is_business_hours', 'is_peak_hours', 'is_night'

# Aggregated features  
'total_prb', 'total_throughput_ul', 'total_throughput_dl', 'total_users'

# Efficiency ratios
'ul_dl_thr_ratio', 'ul_dl_ue_ratio', 'thr_per_prb_ul', 'thr_per_prb_dl'

# Original features (scaled)
'PRBUsageUL', 'PRBUsageDL', 'meanThr_UL', 'meanThr_DL', 
'maxThr_UL', 'maxThr_DL', 'meanUE_UL', 'meanUE_DL', 
'maxUE_UL', 'maxUE_DL', 'cell_encoded'
```

## Model Implementation

### 1. Algorithm Configuration

```python
MODEL_PARAMS = {
    'CatBoost': {
        'iterations': 100,
        'verbose': False,
        'random_state': 42
    },
    'LightGBM': {
        'n_estimators': 100,
        'random_state': 42,
        'verbose': -1
    },
    'XGBoost': {
        'n_estimators': 100,
        'random_state': 42,
        'verbosity': 0
    },
    # ... additional models
}
```

### 2. Training Pipeline (`src/model_trainer.py`)

```python
def train_models(X_train, y_train, focus_on_accuracy=True):
    """
    Train multiple models with comprehensive evaluation.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        focus_on_accuracy: Whether to prioritize accuracy over F1
        
    Returns:
        dict: Training results with models, scores, and best model
    """
```

**Training Process:**
1. **Model Initialization**: Create instances of all available algorithms
2. **Cross-Validation**: 5-fold stratified CV for robust evaluation
3. **Metric Calculation**: Accuracy, F1-Score, Precision, Recall
4. **Model Selection**: Automatic selection of best performing model
5. **Performance Reporting**: Comprehensive results table

### 3. Model Evaluation Metrics

```python
# Evaluation metrics calculated
metrics = {
    'cv_accuracy_mean': float,      # Mean cross-validation accuracy
    'cv_accuracy_std': float,       # Standard deviation of accuracy
    'cv_f1_mean': float,           # Mean F1-score
    'cv_f1_std': float,            # Standard deviation of F1-score
    'cv_precision_mean': float,     # Mean precision
    'cv_precision_std': float,      # Standard deviation of precision
    'cv_recall_mean': float,        # Mean recall
    'cv_recall_std': float,         # Standard deviation of recall
    'training_time': float          # Training time in seconds
}
```

## API Reference

### Main Pipeline (`main.py`)

```python
def run_anomaly_detection():
    """
    Execute complete anomaly detection pipeline.
    
    Returns:
        dict: Results including model performance and predictions
    """
```

**Pipeline Stages:**
1. Data loading and validation
2. Exploratory data analysis
3. Data preprocessing and feature engineering
4. Model training/loading
5. Prediction generation
6. Results saving

### Algorithm Testing (`tests/test_algorithms.py`)

```python
def test_all_algorithms(X_train, y_train, save_results=True):
    """
    Test all available ML algorithms and find the best performer.
    
    Args:
        X_train: Training features
        y_train: Training labels
        save_results: Whether to save results to files
        
    Returns:
        dict: Comprehensive testing results
    """
```

### Model Persistence (`save_best_model.py`)

```python
def save_best_model():
    """
    Train and save the best performing model (CatBoost).
    
    Returns:
        str: Path to saved model file
    """
```

## Performance Analysis

### Benchmark Results Summary

| Algorithm | Accuracy (μ±σ) | F1-Score (μ±σ) | Training Time | Memory Usage |
|-----------|----------------|----------------|---------------|--------------|
| **CatBoost** | **0.9893±0.0015** | **0.9803±0.0029** | **0.94s** | Low |
| LightGBM | 0.9889±0.0018 | 0.9795±0.0034 | 0.15s | Very Low |
| XGBoost | 0.9825±0.0015 | 0.9675±0.0029 | 0.25s | Low |
| RandomForest | 0.9158±0.0019 | 0.8246±0.0046 | 0.78s | Medium |
| GradientBoosting | 0.9091±0.0022 | 0.8030±0.0057 | 9.63s | High |

### Feature Importance Analysis

**CatBoost Feature Importance (Top 10):**
```
1. meanUE_UL         : 38.77% (Mean uplink user equipment)
2. meanUE_DL         : 23.43% (Mean downlink user equipment)
3. PRBUsageDL        : 14.77% (Physical Resource Block usage DL)
4. total_users       : 12.41% (Total active users)
5. PRBUsageUL        : 5.73%  (Physical Resource Block usage UL)
6. maxUE_UL          : 1.67%  (Maximum uplink user equipment)
7. cell_encoded      : 0.85%  (Encoded cell identifier)
8. Hour              : 0.51%  (Hour of day)
9. is_business_hours : 0.29%  (Business hours flag)
10. total_prb        : 0.21%  (Total PRB usage)
```

### Performance Characteristics

**Model Complexity:**
- **CatBoost**: Medium complexity, gradient boosting with categorical features
- **LightGBM**: Low complexity, optimized gradient boosting
- **XGBoost**: Medium complexity, regularized gradient boosting

**Inference Speed:**
- All models: < 1ms per prediction
- Batch processing: ~1000 predictions/second
- Memory footprint: < 50MB loaded model

## Deployment Guide

### Production Deployment

#### 1. Model Loading
```python
import joblib
from pathlib import Path

# Load trained model and transformers
model_path = Path("models/best_model.joblib")
scaler_path = Path("models/feature_scaler.joblib") 
encoder_path = Path("models/cellname_encoder.joblib")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
```

#### 2. Real-time Prediction
```python
def predict_anomaly(raw_data):
    """
    Real-time anomaly prediction for single sample.
    
    Args:
        raw_data: dict with network metrics
        
    Returns:
        tuple: (prediction, probability, confidence)
    """
    # Preprocess input data
    processed_data = preprocess_sample(raw_data)
    
    # Make prediction
    prediction = model.predict([processed_data])[0]
    probability = model.predict_proba([processed_data])[0][1]
    
    # Calculate confidence
    confidence = max(probability, 1 - probability)
    
    return prediction, probability, confidence
```

#### 3. Batch Processing
```python
def batch_predict(data_batch):
    """
    Efficient batch prediction for multiple samples.
    
    Args:
        data_batch: pandas DataFrame with network metrics
        
    Returns:
        pandas DataFrame with predictions and probabilities
    """
    # Preprocess batch
    X_batch = preprocess_data_batch(data_batch)
    
    # Batch prediction
    predictions = model.predict(X_batch)
    probabilities = model.predict_proba(X_batch)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'anomaly_probability': probabilities,
        'confidence': np.maximum(probabilities, 1 - probabilities)
    })
    
    return results
```

### Monitoring and Maintenance

#### Model Performance Monitoring
```python
def monitor_model_performance(predictions, actual_labels):
    """
    Monitor model performance over time.
    
    Args:
        predictions: Model predictions
        actual_labels: Ground truth labels
        
    Returns:
        dict: Performance metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics = {
        'accuracy': accuracy_score(actual_labels, predictions),
        'f1_score': f1_score(actual_labels, predictions),
        'precision': precision_score(actual_labels, predictions),
        'recall': recall_score(actual_labels, predictions)
    }
    
    return metrics
```

#### Model Retraining Trigger
```python
def check_model_drift(current_performance, baseline_performance, threshold=0.05):
    """
    Check if model performance has degraded significantly.
    
    Args:
        current_performance: Current model metrics
        baseline_performance: Baseline metrics from training
        threshold: Performance degradation threshold
        
    Returns:
        bool: True if retraining is needed
    """
    accuracy_drop = baseline_performance['accuracy'] - current_performance['accuracy']
    f1_drop = baseline_performance['f1_score'] - current_performance['f1_score']
    
    return accuracy_drop > threshold or f1_drop > threshold
```

### Configuration Management

#### Environment Configuration
```python
# config/production.py
PRODUCTION_CONFIG = {
    'model_path': 'models/best_model.joblib',
    'batch_size': 1000,
    'prediction_threshold': 0.5,
    'monitoring_interval': 3600,  # seconds
    'retraining_threshold': 0.05,
    'alert_threshold': 0.8,  # anomaly probability
    'log_level': 'INFO'
}
```

#### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Security Considerations

1. **Input Validation**: Validate all input data types and ranges
2. **Model Security**: Secure model files and prevent unauthorized access
3. **Data Privacy**: Implement data anonymization if required
4. **Access Control**: Restrict API access with authentication
5. **Audit Logging**: Log all predictions and system activities

### Scalability

#### Horizontal Scaling
- **Load Balancing**: Distribute prediction requests across multiple instances
- **Containerization**: Docker containers for consistent deployment
- **Microservices**: Separate preprocessing, prediction, and postprocessing services

#### Vertical Scaling
- **GPU Acceleration**: Use GPU-optimized versions of algorithms if available
- **Memory Optimization**: Efficient data structures and batch processing
- **CPU Optimization**: Multi-threading for parallel processing

---

**Technical Specifications:**
- **Python Version**: 3.8+
- **Memory Requirements**: 2GB RAM minimum, 4GB recommended
- **CPU Requirements**: 2+ cores recommended for production
- **Storage**: 1GB for models and artifacts
- **Network**: HTTP/HTTPS for API endpoints

This technical documentation provides comprehensive implementation details for deploying and maintaining the network anomaly detection system in production environments.
