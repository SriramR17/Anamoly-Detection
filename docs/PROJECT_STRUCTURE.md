# Project Structure - Network Anomaly Detection System

This document describes the professional organization of the Network Anomaly Detection System codebase.

## 📁 Directory Structure

```
CTS/  (Project Root)
├── main.py                 # 🚀 Main entry point - run this to execute the system
├── README.md              # 📖 Project overview and quick start guide
├── requirements.txt       # 📦 Python dependencies
├── .gitignore            # 🚫 Git ignore patterns
│
├── src/                   # 💻 Source Code
│   ├── main.py               # Pipeline orchestrator
│   ├── config.py             # Configuration and parameters
│   ├── data_loader.py        # Data loading and validation
│   ├── explorer.py           # Exploratory data analysis
│   ├── preprocessor.py       # Data preprocessing and feature engineering
│   └── model_trainer.py      # Model training and evaluation
│
├── data/                  # 📊 Raw Data Files
│   ├── ML-MATT-CompetitionQT1920_train.csv
│   └── ML-MATT-CompetitionQT1920_test.csv
│
├── results/               # 📈 Generated Outputs
│   ├── predictions.csv           # Detailed predictions with probabilities
│   ├── simple_predictions.csv    # Competition submission format
│   ├── exploration.png           # Data exploration visualizations
│   └── [other generated files]   # Various analysis outputs
│
├── docs/                  # 📚 Documentation
│   ├── documentation.txt         # Complete project documentation
│   ├── PROJECT_STRUCTURE.md     # This file - project organization
│   ├── SIMPLE_README.md          # Technical implementation details
│   ├── SIMPLIFICATION_SUMMARY.md # Development history
│   └── TEST_SUMMARY.md           # Testing and validation report
│
├── tests/                 # 🧪 Testing and Validation
│   └── test_outputs.py           # Output validation tests
│
├── config/                # ⚙️ Configuration Files
│   └── requirements.txt      # Python package requirements
│
├── models/                # 🤖 Model Artifacts
│   ├── cellname_encoder.joblib   # Trained categorical encoder
│   └── feature_scaler.joblib     # Trained feature scaler
│
└── scripts/               # 🛠️ Utility Scripts
    └── [future utility scripts]
```

## 🎯 Key Components

### Entry Points
- **`main.py`**: Professional entry point that can be run from project root
- **`src/main.py`**: Core pipeline implementation

### Core Modules
- **`config.py`**: Centralized configuration management
- **`data_loader.py`**: Robust data loading with encoding handling
- **`explorer.py`**: Data analysis and visualization
- **`preprocessor.py`**: Feature engineering and data cleaning
- **`model_trainer.py`**: Machine learning pipeline

### Data Flow
```
data/ → src/data_loader.py → src/preprocessor.py → src/model_trainer.py → results/
```

## 🚀 Usage Patterns

### Standard Usage
```bash
# From project root
python main.py
```

### Development/Testing
```bash
# Individual component testing
python src/data_loader.py
python src/preprocessor.py
python src/model_trainer.py

# Validation
python tests/test_outputs.py
```

### Configuration
- Modify `src/config.py` for parameters
- Update `requirements.txt` for dependencies
- Check `config/` for additional settings

## 🔄 Data Flow

1. **Data Loading**: `data/` → `src/data_loader.py`
2. **Exploration**: Training data → `src/explorer.py` → visualizations
3. **Preprocessing**: Raw data → `src/preprocessor.py` → engineered features
4. **Training**: Features → `src/model_trainer.py` → trained models
5. **Prediction**: Test data → trained model → `results/predictions.csv`
6. **Validation**: `tests/test_outputs.py` validates all outputs

## 📊 Output Organization

### Primary Results
- `results/predictions.csv`: Complete predictions with probabilities and original data
- `results/simple_predictions.csv`: Clean competition/submission format

### Analysis Outputs
- `results/exploration.png`: Data visualization plots
- `results/[various].txt`: Analysis reports and summaries

### Model Artifacts
- `models/`: Serialized encoders and scalers for reproducibility

## 🧪 Testing Strategy

- **Component Tests**: Each module can be run independently
- **Integration Tests**: `main.py` runs the complete pipeline
- **Output Validation**: `tests/test_outputs.py` verifies all generated files
- **Documentation**: All processes documented in `docs/documentation.txt`

## 🔧 Development Workflow

1. **Setup**: Install requirements from `requirements.txt`
2. **Data**: Ensure CSV files are in `data/` directory
3. **Development**: Modify modules in `src/`
4. **Testing**: Run individual modules and `tests/test_outputs.py`
5. **Execution**: Run `python main.py` for complete pipeline
6. **Results**: Review outputs in `results/` directory
7. **Documentation**: Update `docs/documentation.txt` as needed

## 📈 Professional Features

- **Separation of Concerns**: Clear separation between data, code, results, and documentation
- **Reproducibility**: All paths and configurations centralized
- **Testing**: Comprehensive validation of outputs
- **Documentation**: Complete project documentation
- **Version Control**: Professional `.gitignore` for Python projects
- **Entry Points**: Multiple ways to run the system (development vs production)

## 🚀 Future Enhancements

The structure supports:
- Additional model implementations in `src/`
- More comprehensive tests in `tests/`
- Utility scripts in `scripts/`
- Extended documentation in `docs/`
- Multiple configuration files in `config/`
- Model versioning in `models/`

This professional structure ensures maintainability, scalability, and collaboration readiness.
