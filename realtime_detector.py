"""
Real-Time Anomaly Detection System
==================================
Uses live network monitoring data to detect anomalies in real-time.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import threading
import queue
import joblib
from pathlib import Path

from network_monitor import NetworkMonitor
from src.data_loader import load_data
from src.preprocessor import preprocess_data,_add_derived_features
from src.model_trainer import train_models

from src.config import TRAIN_FILE, TEST_FILE, TARGET_COL, NUMERIC_COLS

class RealTimeAnomalyDetector:
    """
    Real-time anomaly detection system that monitors network data live
    and detects anomalies using the trained model.
    """
    
    def __init__(self, model_path=None, retrain_hours=24):
        """
        Initialize the real-time detector.
        
        Args:
            model_path: Path to pre-trained model (optional)
            retrain_hours: Hours between automatic model retraining
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.retrain_hours = retrain_hours
        self.last_train_time = None
        
        # Real-time data management
        self.network_monitor = NetworkMonitor(collection_interval=10)  # Every 10 seconds
        self.prediction_queue = queue.Queue(maxsize=100)
        self.is_detecting = False
        self.detection_thread = None
        
        # Alert settings
        self.anomaly_threshold = 0.5  # Probability threshold for anomaly alerts
        self.alert_callback = None
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'anomalies_detected': 0,
            'last_anomaly': None,
            'uptime_start': datetime.now()
        }
        
        # Load or train initial model
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("No pre-trained model found. Will train on first batch of data.")
    
    def train_initial_model(self):
        """Train the initial model using existing data."""
        print("🤖 Training initial anomaly detection model...")
        
        try:
            # Load training data
            train_data, test_data = load_data("data\ML-MATT-CompetitionQT1920_train.csv","data\ML-MATT-CompetitionQT1920_test.csv")
            
            # Preprocess data and train model
            X_train, y_train, X_test, feature_names = preprocess_data(train_data, test_data)
            
            # Train models and get the best one
            model_results = train_models(X_train, y_train)
            
            # Store the trained components
            self.model = model_results['best_model']
            self.feature_names = feature_names
            self.last_train_time = datetime.now()
            
            print(f"✅ Model trained successfully: {model_results['best_model_name']}")
            print(f"   F1 Score: {model_results['scores'][model_results['best_model_name']]['cv_f1_mean']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training initial model: {e}")
            return False
    
    def preprocess_realtime_data(self, network_data):
        """
        Preprocess real-time network data for prediction.
        
        Args:
            network_data: Dict containing network metrics
            
        Returns:
            numpy.array: Preprocessed features ready for prediction
        """
        try:
            # Convert single data point to DataFrame
            df = pd.DataFrame([network_data])
            
            # Apply the same preprocessing steps as in training
            from src.preprocessor import _add_time_features, _add_derived_features
            
            # Add time features
            df = _add_time_features(df)
            
            # Add derived features  
            df = _add_derived_features(df)
            
            # Encode cell name (simple label encoding for real-time)
            cell_mapping = {
                '1ALTE': 0, '2BLTE': 1, '3CLTE': 2, '4ALTE': 3, '5BLTE': 4,
                '6ULTE': 5, '7ALTE': 6, '8BLTE': 7, '9ALTE': 8, '10CLTE': 9
            }
            df['cell_encoded'] = df['CellName'].map(cell_mapping).fillna(0)
            
            # Select the same features used in training
            feature_cols = [col for col in df.columns if col not in ['Time', 'CellName', 'system_cpu', 'system_memory', 'raw_throughput_kbps']]
            
            # Ensure we have all expected features
            X = df[feature_cols].fillna(0)
            
            # Handle any missing features by padding with zeros
            if len(X.columns) < len(self.feature_names):
                for col in self.feature_names:
                    if col not in X.columns:
                        X[col] = 0
            
            # Reorder columns to match training data
            X = X.reindex(columns=self.feature_names, fill_value=0)
            
            return X.values
            
        except Exception as e:
            print(f"Error preprocessing real-time data: {e}")
            return None
    
    def predict_anomaly(self, network_data):
        """
        Predict if the current network data represents an anomaly.
        
        Args:
            network_data: Dict containing network metrics
            
        Returns:
            tuple: (is_anomaly, probability, prediction_details)
        """
        if not self.model:
            return None, None, "Model not trained"
        
        try:
            # Preprocess the data
            X = self.preprocess_realtime_data(network_data)
            
            if X is None:
                return None, None, "Preprocessing failed"
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get anomaly probability (class 1)
            anomaly_prob = probability[1] if len(probability) > 1 else 0.0
            
            is_anomaly = prediction == 1
            
            prediction_details = {
                'timestamp': datetime.now(),
                'cell_name': network_data.get('CellName', 'Unknown'),
                'prediction': int(prediction),
                'anomaly_probability': float(anomaly_prob),
                'is_anomaly': is_anomaly,
                'confidence': 'High' if anomaly_prob > 0.8 else 'Medium' if anomaly_prob > 0.5 else 'Low'
            }
            
            # Update statistics
            self.stats['total_predictions'] += 1
            if is_anomaly:
                self.stats['anomalies_detected'] += 1
                self.stats['last_anomaly'] = datetime.now()
            
            return is_anomaly, anomaly_prob, prediction_details
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, f"Prediction error: {e}"
    
    def _detection_loop(self):
        """Main detection loop (runs in separate thread)."""
        print("🔍 Starting real-time anomaly detection...")
        
        while self.is_detecting:
            try:
                # Get recent network data
                recent_data = self.network_monitor.get_recent_data(minutes=1)
                
                if not recent_data.empty:
                    # Get the most recent data point
                    latest_data = recent_data.iloc[-1].to_dict()
                    
                    # Make prediction
                    is_anomaly, probability, details = self.predict_anomaly(latest_data)
                    
                    if details:
                        # Add to prediction queue
                        self.prediction_queue.put(details)
                        
                        # Print real-time status
                        status = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
                        print(f"{status} | {details['timestamp'].strftime('%H:%M:%S')} | "
                              f"{details['cell_name']} | Prob: {probability:.3f} | "
                              f"Conf: {details['confidence']}")
                        
                        # Trigger alert if anomaly detected above threshold
                        if is_anomaly and probability > self.anomaly_threshold:
                            self._trigger_alert(details)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(5)
    
    def _trigger_alert(self, details):
        """
        Trigger an alert for detected anomaly.
        
        Args:
            details: Prediction details dict
        """
        alert_msg = (f"🚨 ANOMALY ALERT 🚨\n"
                    f"Time: {details['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Cell: {details['cell_name']}\n"
                    f"Probability: {details['anomaly_probability']:.1%}\n"
                    f"Confidence: {details['confidence']}")
        
        print("\n" + "="*50)
        print(alert_msg)
        print("="*50 + "\n")
        
        # Call custom alert callback if provided
        if self.alert_callback:
            self.alert_callback(details)
    
    def set_alert_callback(self, callback_func):
        """
        Set a custom callback function for anomaly alerts.
        
        Args:
            callback_func: Function that takes prediction details as argument
        """
        self.alert_callback = callback_func
    
    def start_detection(self):
        """Start real-time anomaly detection."""
        if not self.model:
            print("Training initial model...")
            if not self.train_initial_model():
                print("❌ Cannot start detection without a trained model")
                return False
        
        if self.is_detecting:
            print("⚠️ Detection is already running")
            return False
        
        # Start network monitoring
        self.network_monitor.start_monitoring()
        
        # Wait a moment for initial data collection
        time.sleep(3)
        
        # Start detection thread
        self.is_detecting = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        print("✅ Real-time anomaly detection started")
        return True
    
    def stop_detection(self):
        """Stop real-time anomaly detection."""
        self.is_detecting = False
        self.network_monitor.stop_monitoring()
        
        if self.detection_thread:
            self.detection_thread.join(timeout=10)
        
        print("🛑 Real-time detection stopped")
    
    def get_recent_predictions(self, count=10):
        """
        Get recent predictions from the queue.
        
        Args:
            count: Number of recent predictions to return
            
        Returns:
            list: Recent prediction details
        """
        predictions = []
        temp_queue = []
        
        # Extract predictions from queue
        while not self.prediction_queue.empty() and len(predictions) < count:
            pred = self.prediction_queue.get()
            predictions.append(pred)
            temp_queue.append(pred)
        
        # Put them back in the queue
        for pred in reversed(temp_queue):
            self.prediction_queue.put(pred)
        
        return list(reversed(predictions))  # Most recent first
    
    def get_statistics(self):
        """Get detection statistics."""
        uptime = datetime.now() - self.stats['uptime_start']
        
        stats = self.stats.copy()
        stats['uptime_hours'] = uptime.total_seconds() / 3600
        stats['anomaly_rate'] = (self.stats['anomalies_detected'] / max(1, self.stats['total_predictions'])) * 100
        stats['predictions_per_hour'] = self.stats['total_predictions'] / max(0.1, stats['uptime_hours'])
        
        return stats
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model:
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'train_time': self.last_train_time
            }, filepath)
            print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.last_train_time = data.get('train_time', datetime.now())
            print(f"📂 Model loaded from: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


def main():
    """Demo of real-time anomaly detection."""
    detector = RealTimeAnomalyDetector()
    
    # Custom alert callback
    def my_alert_handler(details):
        """Custom alert handler - you can add email, SMS, etc. here."""
        print(f"🔔 Custom Alert: Anomaly detected in cell {details['cell_name']} "
              f"with {details['anomaly_probability']:.1%} probability")
        
        # Here you could add:
        # - Send email notification
        # - Write to log file
        # - Send to monitoring system
        # - Trigger automated response
    
    detector.set_alert_callback(my_alert_handler)
    
    try:
        print("="*60)
        print("REAL-TIME NETWORK ANOMALY DETECTION SYSTEM")
        print("="*60)
        print("This system will:")
        print("  1. Monitor your network usage in real-time")
        print("  2. Convert it to cellular network format")
        print("  3. Detect anomalies using your trained model")
        print("  4. Alert you when anomalies are detected")
        print()
        
        # Start detection
        if detector.start_detection():
            print("🎯 Detection active! System is monitoring for anomalies...")
            print("💡 Try generating network load to see anomaly detection in action:")
            print("   - Stream videos, download large files, run speed tests")
            print("   - Open many browser tabs, update software")
            print()
            print("📊 Live Detection Feed:")
            print("-" * 60)
            
            # Keep running and show periodic statistics
            while True:
                time.sleep(30)  # Show stats every 30 seconds
                
                stats = detector.get_statistics()
                recent_preds = detector.get_recent_predictions(count=5)
                
                print(f"\n📈 Detection Stats (Uptime: {stats['uptime_hours']:.1f}h):")
                print(f"   Total predictions: {stats['total_predictions']}")
                print(f"   Anomalies detected: {stats['anomalies_detected']} ({stats['anomaly_rate']:.1f}%)")
                print(f"   Predictions/hour: {stats['predictions_per_hour']:.1f}")
                
                if recent_preds:
                    print(f"\n🔍 Recent Predictions:")
                    for pred in recent_preds[-3:]:  # Show last 3
                        status = "🔴" if pred['is_anomaly'] else "🟢"
                        print(f"   {status} {pred['timestamp'].strftime('%H:%M:%S')} | "
                              f"{pred['cell_name']} | {pred['anomaly_probability']:.3f}")
        
        else:
            print("❌ Failed to start detection")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping detection...")
        detector.stop_detection()
        
        # Show final statistics
        stats = detector.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Runtime: {stats['uptime_hours']:.1f} hours")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Anomalies detected: {stats['anomalies_detected']}")
        print(f"   Detection rate: {stats['anomaly_rate']:.1f}%")
        
        # Save model
        model_path = "models/realtime_model.joblib"
        detector.save_model(model_path)
        
        print(f"\n✅ Real-time detection session complete!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        detector.stop_detection()


if __name__ == "__main__":
    main()
