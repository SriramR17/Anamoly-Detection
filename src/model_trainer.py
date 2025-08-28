"""
Simple Model Trainer for Network Anomaly Detection
==================================================
Clean, straightforward model training with essential algorithms.
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import MODEL_PARAMS, RANDOM_STATE
import os
import joblib


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def get_model_instance(model_name, params):
    """
    Get model instance with error handling for optional dependencies.
    """
    try:
        model_map = {
            'RandomForest': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
            'GradientBoosting': GradientBoostingClassifier,
            'DecisionTree': DecisionTreeClassifier,
            'ExtraTrees': ExtraTreesClassifier,
            'AdaBoost': AdaBoostClassifier,
            'KNN': KNeighborsClassifier,
            'GaussianNB': GaussianNB,
        }
        
        # Try XGBoost
        if model_name == 'XGBoost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(**params)
            except ImportError:
                print(f"  ⚠ XGBoost not available, skipping...")
                return None
        
        # Try LightGBM
        elif model_name == 'LightGBM':
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**params)
            except ImportError:
                print(f"  ⚠ LightGBM not available, skipping...")
                return None
        
        # Try CatBoost
        elif model_name == 'CatBoost':
            try:
                import catboost as cb
                return cb.CatBoostClassifier(**params)
            except ImportError:
                print(f"  ⚠ CatBoost not available, skipping...")
                return None
        
        # Standard sklearn models
        elif model_name in model_map:
            return model_map[model_name](**params)
        else:
            print(f"  ⚠ Unknown model: {model_name}, skipping...")
            return None
            
    except Exception as e:
        print(f"  ⚠ Error creating {model_name}: {e}, skipping...")
        return None


def train_models(X_train, y_train, focus_on_accuracy=True):
    """
    Train multiple models and select the best one with comprehensive algorithm testing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        focus_on_accuracy: If True, prioritize accuracy over F1 score
        
    Returns:
        dict: Results including trained models and scores
    """
    print("🚀 Training comprehensive set of ML algorithms...")
    print(f"📊 Dataset shape: {X_train.shape}")
    print(f"🎯 Target distribution: {np.bincount(y_train)}\n")
    
    models = {}
    results = {}
    scores = {}
    training_times = {}
    
    # Initialize all available models
    for model_name, params in MODEL_PARAMS.items():
        model = get_model_instance(model_name, params)
        if model is not None:
            models[model_name] = model
    
    print(f"🔧 Training {len(models)} algorithms...\n")
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"  🎯 Training {name}...")
        
        try:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Record training time
            training_time = time.time() - start_time
            training_times[name] = training_time
            
            # Cross-validation scores for both accuracy and F1
            cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            cv_precision = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')
            cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
            
            # Store model and comprehensive scores
            results[name] = model
            scores[name] = {
                'cv_accuracy_mean': cv_accuracy.mean(),
                'cv_accuracy_std': cv_accuracy.std(),
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std(),
                'cv_precision_mean': cv_precision.mean(),
                'cv_precision_std': cv_precision.std(),
                'cv_recall_mean': cv_recall.mean(),
                'cv_recall_std': cv_recall.std(),
                'training_time': training_time
            }
            
            print(f"    ✅ Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
            print(f"    📏 F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
            print(f"    ⏱️  Time: {training_time:.2f}s\n")
            
        except Exception as e:
            print(f"    ❌ Error training {name}: {e}\n")
            continue
    
    if not scores:
        raise Exception("No models were successfully trained!")
    
    # Select best model based on chosen metric
    metric_key = 'cv_accuracy_mean' if focus_on_accuracy else 'cv_f1_mean'
    metric_name = 'Accuracy' if focus_on_accuracy else 'F1 Score'
    
    best_model_name = max(scores.keys(), key=lambda x: scores[x][metric_key])
    best_model = results[best_model_name]
    best_score = scores[best_model_name][metric_key]
    
    print("\n" + "="*70)
    print("🏆 MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    # Sort models by chosen metric
    sorted_models = sorted(scores.items(), key=lambda x: x[1][metric_key], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<15} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12} {'Time(s)':<8}")
    print("-" * 80)
    
    for i, (name, score_dict) in enumerate(sorted_models, 1):
        print(f"{i:<4} {name:<15} "
              f"{score_dict['cv_accuracy_mean']:.4f}       "
              f"{score_dict['cv_f1_mean']:.4f}       "
              f"{score_dict['cv_precision_mean']:.4f}       "
              f"{score_dict['cv_recall_mean']:.4f}       "
              f"{score_dict['training_time']:.2f}")
    
    print("-" * 80)
    print(f"🥇 BEST MODEL: {best_model_name}")
    print(f"🎯 Best {metric_name}: {best_score:.4f}")
    print(f"⏱️  Training Time: {scores[best_model_name]['training_time']:.2f}s")
    print("="*70 + "\n")
    
    return {
        'models': results,
        'scores': scores,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'training_times': training_times,
        'focus_metric': metric_name.lower()
    }


def make_predictions(model, X_test):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    print("Making predictions...")
    
    # Generate predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of anomaly
    
    anomaly_count = np.sum(predictions)
    anomaly_rate = np.mean(predictions) * 100
    
    print(f"✓ Predictions complete:")
    print(f"  Total samples: {len(predictions):,}")
    print(f"  Predicted anomalies: {anomaly_count:,} ({anomaly_rate:.1f}%)")
    
    return predictions, probabilities


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance (if you have true labels for test data).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test labels
        
    Returns:
        dict: Evaluation metrics
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions)
    }
    
    print("Model Evaluation:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.3f}")
    
    return metrics


def get_or_train_model(X_train, y_train, model_path=None, force_retrain=False):
    """
    Load existing model or train new one if none exists.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        model_path: Path to saved model file (defaults to project root)
        force_retrain: If True, force retraining even if model exists
        
    Returns:
        dict: Results including best model and metadata
    """
    import os
    import joblib
    from pathlib import Path
    
    # Default model path - look in project root
    if model_path is None:
        # Get the project root (parent of src directory)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        model_path = project_root / 'models/best_model.joblib'
    else:
        model_path = Path(model_path)
    
    # Force retraining if requested
    if force_retrain:
        print("🔄 Force retrain requested. Training new models...")
    elif model_path.exists():
        print(f"✓ Found existing model at {model_path}. Loading...")
        try:
            best_model = joblib.load(model_path)
            
            # Create a result structure similar to train_models output
            results = {
                'models': {'LoadedModel': best_model},
                'scores': {'LoadedModel': {'cv_f1_mean': 'N/A (pre-trained)', 'cv_f1_std': 'N/A'}},
                'best_model_name': best_model.__class__.__name__,
                'best_model': best_model
            }
            
            print("✓ Model loaded successfully. Skipping training.")
            return results
            
        except Exception as e:
            print(f"⚠ Warning: Could not load existing model ({e}). Training new model...")
    else:
        print(f"✗ No existing model found at {model_path}. Training new models...")
    
    # Train new models if no existing model, loading failed, or force retrain
    results = train_models(X_train, y_train)
    
    # Save the newly trained model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results['best_model'], model_path)
    print(f"✓ Best model saved to {model_path}")
    
    return results


def plot_model_comparison(scores, focus_metric='accuracy', save_path=None):
    """
    Create visualization comparing model performances.
    
    Args:
        scores: Dictionary of model scores from train_models
        focus_metric: Primary metric to highlight ('accuracy' or 'f1')
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        models = list(scores.keys())
        accuracies = [scores[m]['cv_accuracy_mean'] for m in models]
        f1_scores = [scores[m]['cv_f1_mean'] for m in models]
        precisions = [scores[m]['cv_precision_mean'] for m in models]
        recalls = [scores[m]['cv_recall_mean'] for m in models]
        times = [scores[m]['training_time'] for m in models]
        
        # Sort by the focus metric
        if focus_metric.lower() == 'accuracy':
            sorted_data = sorted(zip(models, accuracies, f1_scores, precisions, recalls, times), 
                               key=lambda x: x[1], reverse=True)
        else:
            sorted_data = sorted(zip(models, accuracies, f1_scores, precisions, recalls, times), 
                               key=lambda x: x[2], reverse=True)
        
        models, accuracies, f1_scores, precisions, recalls, times = zip(*sorted_data)
        
        # Colors for better visualization
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        # 1. Accuracy comparison
        bars1 = ax1.bar(range(len(models)), accuracies, color=colors)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. F1 Score comparison
        bars2 = ax2.bar(range(len(models)), f1_scores, color=colors)
        ax2.set_title('Model F1-Score Comparison', fontweight='bold')
        ax2.set_ylabel('F1-Score')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Precision vs Recall scatter
        scatter = ax3.scatter(recalls, precisions, c=accuracies, s=[t*10 for t in times], 
                            cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall (Color=Accuracy, Size=Time)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add model labels to scatter points
        for i, model in enumerate(models):
            ax3.annotate(model, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter, ax=ax3, label='Accuracy')
        
        # 4. Training time comparison
        bars4 = ax4.bar(range(len(models)), times, color=colors)
        ax4.set_title('Training Time Comparison', fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        for i, (bar, time_val) in enumerate(zip(bars4, times)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance visualization saved to: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("⚠ Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"⚠ Error creating visualization: {e}")


def test_all_algorithms(X_train, y_train, X_test=None, y_test=None, save_results=True):
    """
    Comprehensive testing of all available ML algorithms for finding the best one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional, for final evaluation)
        y_test: Test labels (optional, for final evaluation)
        save_results: Whether to save results and visualizations
        
    Returns:
        dict: Complete results including all models and performance metrics
    """
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE ML ALGORITHM TESTING")
    print("🎯 Goal: Find the algorithm with highest accuracy")
    print("="*80)
    
    # Train all models with focus on accuracy
    results = train_models(X_train, y_train, focus_on_accuracy=True)
    
    # Create performance visualization
    if save_results:
        from config import OUTPUT_DIR
        plot_path = OUTPUT_DIR / "algorithm_comparison.png"
        plot_model_comparison(results['scores'], focus_metric='accuracy', save_path=plot_path)
    
    # If test data is provided, evaluate the best model
    if X_test is not None and y_test is not None:
        print("\n🔍 FINAL EVALUATION ON TEST SET")
        print("-" * 40)
        best_model = results['best_model']
        test_metrics = evaluate_model(best_model, X_test, y_test)
        results['test_metrics'] = test_metrics
    
    # Save detailed results to CSV if requested
    if save_results:
        save_algorithm_results(results)
    
    print("\n" + "="*80)
    print("✅ ALGORITHM TESTING COMPLETE")
    print(f"🏆 RECOMMENDED ALGORITHM: {results['best_model_name']}")
    print("="*80)
    
    return results


def save_algorithm_results(results):
    """
    Save algorithm comparison results to CSV file.
    
    Args:
        results: Results dictionary from train_models or test_all_algorithms
    """
    try:
        from config import OUTPUT_DIR
        
        # Create results DataFrame
        data = []
        for model_name, scores in results['scores'].items():
            data.append({
                'Algorithm': model_name,
                'Accuracy_Mean': scores['cv_accuracy_mean'],
                'Accuracy_Std': scores['cv_accuracy_std'],
                'F1_Mean': scores['cv_f1_mean'],
                'F1_Std': scores['cv_f1_std'],
                'Precision_Mean': scores['cv_precision_mean'],
                'Precision_Std': scores['cv_precision_std'],
                'Recall_Mean': scores['cv_recall_mean'],
                'Recall_Std': scores['cv_recall_std'],
                'Training_Time': scores['training_time']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy_Mean', ascending=False)
        
        # Save to CSV
        csv_path = OUTPUT_DIR / "algorithm_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"📁 Detailed results saved to: {csv_path}")
        
        # Print summary table
        print("\n📋 SUMMARY TABLE:")
        print(df.to_string(index=False, float_format='%.4f'))
        
    except Exception as e:
        print(f"⚠ Error saving results: {e}")


def force_retrain_model(X_train, y_train, model_path=None):
    """
    Force retrain and save a new model, overwriting any existing one.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_path: Path to save the model (defaults to project root)
        
    Returns:
        dict: Results including trained models and scores
    """
    return get_or_train_model(X_train, y_train, model_path, force_retrain=True)


if __name__ == "__main__":
    MODEL_PATH = 'models/best_model.joblib'
    
    try:
        # Import necessary modules
        from data_loader import load_data
        from preprocessor import preprocess_data
        
        best_model = None

        # Check if a model file exists
        if os.path.exists(MODEL_PATH):
            print("✓ Found existing model. Loading...")
            best_model = joblib.load(MODEL_PATH)
            
        else:
            # If no model is found, proceed with training
            print("✗ No existing model found. Training new models...")
            
            # Load and preprocess data
            train_data, test_data = load_data()
            X_train, y_train, X_test, features = preprocess_data(train_data, test_data)
            
            # Train models
            results = train_models(X_train, y_train)
            best_model = results['best_model']
            
            # Save the best model for future use
            joblib.dump(best_model, MODEL_PATH)
            print(f"✓ Best model saved to {MODEL_PATH}")
        
        # Now that we have a model (either loaded or newly trained), make predictions
        predictions, probabilities = make_predictions(best_model, X_test)
        
        # This part of the code is for testing purposes.
        # It assumes `y_test` is available, which isn't the case in your original `make_predictions` call.
        # If `y_test` is available, you would uncomment and use this:
        # evaluate_model(best_model, X_test, y_test)
        
        print("✓ Model training and prediction process complete.")
        
    except Exception as e:
        print(f"❌ Error: {e}")