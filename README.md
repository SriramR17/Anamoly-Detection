# Network Anomaly Detection System

A professional-grade, full-stack anomaly detection system for network traffic analysis. This project combines advanced machine learning algorithms with a modern web dashboard to provide comprehensive anomaly detection, monitoring, and visualization capabilities.

## 🎯 Project Overview

This system implements a complete anomaly detection pipeline for telecom network analysis, featuring:

- **🧠 Advanced ML Pipeline**: 11 algorithms tested with 98.93% accuracy achieved using CatBoost
- **🌐 Full-Stack Web Application**: FastAPI backend + React TypeScript frontend
- **📊 Real-time Dashboard**: Interactive data visualization and monitoring
- **⚡ High Performance**: Sub-second training with enterprise-grade accuracy
- **🔧 Production Ready**: Comprehensive testing, logging, and error handling

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
├── src/                           # Core ML pipeline source code
│   ├── pipeline.py               # Main pipeline orchestrator
│   ├── config.py                 # Configuration parameters
│   ├── data_loader.py            # Data loading and validation
│   ├── explorer.py               # Data exploration and visualization
│   ├── preprocessor.py           # Feature engineering and preprocessing
│   └── model_trainer.py          # ML model training and evaluation
├── api/                           # FastAPI backend
│   └── app.py                    # FastAPI server and REST endpoints
├── frontend/                      # React TypeScript frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── App.tsx              # Main React application
│   │   ├── api.ts               # API client for backend communication
│   │   └── types.d.ts           # TypeScript type definitions
│   ├── package.json             # Frontend dependencies and scripts
│   └── vite.config.ts           # Vite build configuration
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
├── main.py                        # ML training entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation file
```

## 🚀 Quick Start

### Prerequisites

**Backend Requirements:**
- Python 3.8+ (tested with Python 3.9 and 3.13)
- Conda/Virtual environment recommended
- FastAPI and ML dependencies

**Frontend Requirements:**
- Node.js 16+ (tested with Node.js 22.11.0)
- npm 8+ (tested with npm 10.9.0)

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone [repository] CTS
cd CTS
```

#### 2. Backend Setup

**Option A: Using Conda Environment**
```bash
conda create -n anomaly-detection python=3.9
conda activate anomaly-detection
```

**Option B: Using Existing Environment (Windows)**
```powershell
# If you have an existing environment like d:/tfnew
d:\tfnew\python.exe -m pip install -r requirements.txt
```

**Install Python Dependencies:**
```bash
# Core dependencies
pip install -r requirements.txt

# Or install manually:
pip install pandas numpy scikit-learn matplotlib seaborn joblib
pip install catboost xgboost lightgbm fastapi uvicorn
```

#### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

### Usage

#### 🧠 1. Train the ML Model (First Time)
```bash
# Using system Python
python main.py

# Using specific environment (Windows)
d:\tfnew\python.exe main.py
```

**Output:**
- Trains and evaluates 11 ML algorithms
- Saves best model to `models/best_model.joblib`
- Generates predictions in `results/predictions.csv`
- Creates visualizations in `results/exploration.png`

#### 🌐 2. Run the Web Application

**Terminal 1 - Start FastAPI Backend:**
```bash
# Using system Python
python api/app.py

# Using specific environment (Windows)
d:\tfnew\python.exe api\app.py
```
Backend runs on: http://localhost:8000

**Terminal 2 - Start React Frontend:**
```bash
cd frontend
npm run dev
```
Frontend runs on: http://localhost:5173

#### 📊 3. Access the Dashboard
Open your browser and navigate to: **http://localhost:5173**

The dashboard provides:
- Real-time anomaly detection metrics
- Model performance statistics
- Interactive data visualizations
- Algorithm comparison results

### Additional Commands

#### Test All Algorithms (Advanced)
```bash
python tests/test_algorithms.py
```

#### Train and Save Specific Model
```bash
python save_best_model.py
```

#### Build Frontend for Production
```bash
cd frontend
npm run build
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

## 🌐 Web Dashboard Features

The React TypeScript frontend provides a comprehensive anomaly detection dashboard:

### Dashboard Components
- **📊 Overview Cards**: Total samples, anomalies detected, accuracy rates
- **🕰 Real-time Metrics**: Live updates from ML pipeline results
- **📈 Interactive Charts**: Anomaly trends, hourly patterns, model performance
- **🛡 Alert System**: Visual indicators for high anomaly rates
- **🔍 Data Table**: Detailed predictions with filtering and sorting
- **🧠 Model Analytics**: Algorithm comparison and performance metrics

### Technology Stack
- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS for modern UI components
- **Charts**: Recharts for interactive data visualization
- **Icons**: Lucide React for consistent iconography
- **HTTP Client**: Axios for API communication

### API Endpoints

The FastAPI backend provides RESTful endpoints for data access:

#### Dashboard Data
```http
GET /api/dashboard
```
Returns overview metrics including total samples, anomalies, and rates.

#### Model Performance
```http
GET /api/get_evaluation_metrics
```
Returns best model accuracy, F1-score, precision, and recall.

#### Algorithm Comparison
```http
GET /api/best_models
```
Returns top 11 models with performance metrics.

#### Predictions Data
```http
GET /api/predictions
```
Returns all predictions with detailed probability scores.

#### Statistics
```http
GET /api/stats
```
Returns detailed statistical analysis of predictions.

### CORS Configuration
The FastAPI backend is configured to accept requests from the React development server (`http://localhost:5173`) with full CORS support.

## 🔧 Production Deployment

### Backend Deployment
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment
```bash
# Build for production
cd frontend
npm run build

# Serve static files (example with serve)
npx serve -s dist -l 3000
```

### Docker Deployment (Future)
The project structure supports containerization with Docker:
```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim as backend
# ... backend setup

FROM node:18-alpine as frontend
# ... frontend build

FROM nginx:alpine as production
# ... serve static files and proxy API
```

## 🛠 Troubleshooting

### Common Issues

**Backend Issues:**
- **Import Error**: Ensure `src` directory is in Python path
- **Missing Models**: Run `python main.py` to train models first
- **Port Conflicts**: Change port in `api/app.py` if 8000 is occupied

**Frontend Issues:**
- **API Connection**: Verify backend is running on http://localhost:8000
- **Build Errors**: Delete `node_modules` and run `npm install`
- **Port Issues**: Vite automatically finds available ports

**Data Issues:**
- **Missing Data**: Ensure CSV files exist in `data/` directory
- **Encoding Problems**: Data loader handles latin-1 encoding automatically
- **Memory Issues**: Large datasets may require increased system memory

### Logs and Debugging
- **Backend Logs**: FastAPI provides detailed error messages
- **Frontend Logs**: Check browser developer console
- **ML Pipeline**: Verbose output shows training progress

## 📜 Technical Specifications

### Backend Stack
- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, CatBoost, XGBoost, LightGBM
- **Web Framework**: FastAPI 0.95.1, Uvicorn 0.22.0
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Persistence**: joblib
- **Cross-Validation**: 5-fold stratified CV
- **Evaluation Metrics**: Accuracy, F1-Score, Precision, Recall
- **Feature Engineering**: 23 engineered features from 14 raw features

### Frontend Stack
- **Language**: TypeScript 5.5+
- **Framework**: React 18.3+ with Vite 5.4+
- **Styling**: Tailwind CSS 3.4+
- **Charts**: Recharts 3.1+ for data visualization
- **HTTP Client**: Axios 1.6+ for API calls
- **Icons**: Lucide React 0.344+ for UI icons
- **Build Tool**: Vite with ESLint and TypeScript support

### System Requirements
- **Minimum RAM**: 4GB (8GB recommended for large datasets)
- **Storage**: 2GB free space for models and results
- **CPU**: Multi-core processor recommended for ML training
- **Network**: Internet connection for package downloads



