# Simple Network Anomaly Detection

A clean, easy-to-understand system for detecting network anomalies.

## What it does

This system analyzes network traffic data and predicts whether the activity is:
- **0 (Normal)**: Typical network behavior
- **1 (Anomaly)**: Unusual activity that needs attention

## Quick Start

1. **Install requirements:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. **Run the system:**
```bash
python simple_main.py
```

That's it! The system will automatically:
- Load your data
- Explore patterns
- Train models
- Make predictions
- Save results to the `output/` folder

## Files Overview

### Core Files (Simple Version)
- `simple_main.py` - Main script to run everything
- `simple_config.py` - Basic settings
- `simple_data_loader.py` - Loads and validates data
- `simple_explorer.py` - Basic data exploration
- `simple_preprocessor.py` - Feature engineering
- `simple_model_trainer.py` - Model training

### Data Files (Put these in `dataset/` folder)
- `ML-MATT-CompetitionQT1920_train.csv` - Training data
- `ML-MATT-CompetitionQT1920_test.csv` - Test data

### Output Files (Generated in `output/` folder)
- `predictions.csv` - Detailed predictions
- `simple_predictions.csv` - Simple submission format
- `exploration.png` - Data exploration plots

## Features

The system creates these simple features from your network data:

**Time Features:**
- Hour of day
- Business hours indicator
- Peak hours indicator 
- Night time indicator

**Traffic Features:**
- Total resource usage
- Total throughput
- Total users
- Upload/download ratios
- Efficiency metrics

## Models

Trains three simple but effective models:
1. **Random Forest** - Good for capturing complex patterns
2. **Logistic Regression** - Simple baseline model
3. **Gradient Boosting** - Sequential learning model

The system automatically picks the best model based on F1 score.

## Understanding the Results

After running, check these files:

1. **`exploration.png`** - Shows data patterns and anomaly distributions
2. **`predictions.csv`** - Your test data with predictions and confidence scores
3. **`simple_predictions.csv`** - Just the predictions in submission format

## Customization

Want to modify something? The code is designed to be simple:

- **Change models**: Edit `MODEL_PARAMS` in `simple_config.py`
- **Add features**: Modify `_add_simple_features()` in `simple_preprocessor.py`
- **Skip exploration**: Comment out step 2 in `simple_main.py`

## Data Requirements

Your CSV files should have these columns:
- `Time` (HH:MM format)
- `CellName` (cell identifier)
- `PRBUsageUL`, `PRBUsageDL` (resource usage)
- `meanThr_UL`, `meanThr_DL` (throughput)
- `maxThr_UL`, `maxThr_DL` (max throughput)
- `meanUE_UL`, `meanUE_DL` (user equipment)
- `maxUE_UL`, `maxUE_DL` (max user equipment)
- `maxUE_UL+DL` (combined max users)
- `Unusual` (target - only in training data)

## Troubleshooting

**Common issues:**
- **File not found**: Put your CSV files in the `dataset/` folder
- **Import errors**: Install required packages with pip
- **Memory issues**: The simplified system uses much less memory than the original

## Why This is Better

The simplified version:
- ✅ **Easy to read** - Clear, simple code structure
- ✅ **Easy to modify** - Each step is separate and straightforward  
- ✅ **Fast execution** - Removed unnecessary complexity
- ✅ **Same accuracy** - Keeps the essential features that matter
- ✅ **Better debugging** - Clear error messages and logging

---

**Need the original complex version?** It's still available in the other files, but this simplified version should be much easier to understand and maintain!
