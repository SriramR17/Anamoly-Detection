"""
Simple FastAPI backend for Anomaly Detection Dashboard
====================================================
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

app = FastAPI(title="Anomaly Detection API", version="1.0.0")

# Enable CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store data
current_data = None
current_results = None

@app.get("/")
async def root():
    return {"message": "Anomaly Detection API is running!"}

@app.get("/api/dashboard")
async def get_dashboard_data():
    """Get dashboard overview data"""
    try:
        # Try to load existing results
        results_path = Path(__file__).parent.parent / "results" / "predictions.csv"
        
        if results_path.exists():
            df = pd.read_csv(results_path)
            total_samples = len(df)
            anomalies = df['Predicted_Anomaly'].sum() if 'Predicted_Anomaly' in df.columns else 0
            anomaly_rate = (anomalies / total_samples * 100) if total_samples > 0 else 0
            
            # Get recent anomalies (last 10)
            recent_anomalies = df[df['Predicted_Anomaly'] == 1].tail(10) if 'Predicted_Anomaly' in df.columns else pd.DataFrame()
            
            return {
                "status": "success",
                "data": {
                    "total_samples": int(total_samples),
                    "total_anomalies": int(anomalies),
                    "anomaly_rate": round(float(anomaly_rate), 2),
                    "recent_anomalies": recent_anomalies.to_dict('records') if not recent_anomalies.empty else [],
                    "last_updated": "Just now"
                }
            }
        else:
            return {
                "status": "no_data",
                "message": "No predictions found. Run the anomaly detection first.",
                "data": {
                    "total_samples": 0,
                    "total_anomalies": 0,
                    "anomaly_rate": 0,
                    "recent_anomalies": [],
                    "last_updated": "Never"
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading best models: {str(e)}")

@app.get("/api/get_evaluation_metrics")
async def get_evaluation_metrics():
    try:
        csv_file_path = Path(__file__).parent.parent / 'results' / 'algorithm_comparison_results.csv'
        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found.")

        df = pd.read_csv(csv_file_path)

        # --- FIX ---
        # Find the index of the row with the highest 'Accuracy_Mean'
        max_acc_index = df['Accuracy_Mean'].idxmax()

        metrics = df.loc[max_acc_index]

        # Now the columns match
        return {
            "status": "success",
            "data": {
                "Accuracy": metrics['Accuracy_Mean'],
                "F1_Score": metrics['F1_Mean'],
                "Precision": metrics['Precision_Mean'],
                "Recall": metrics['Recall_Mean']
            }
        }

    except KeyError as e:
        # Specific error for missing columns
        raise HTTPException(status_code=500, detail=f"A required column is missing from the CSV: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Metrics: {str(e)}")


@app.get("/api/best_models")
async def get_best_models():
    """
    Retrieves the top 5 best-performing models based on 'Accuracy_Mean'
    from the 'algorithm_comparison_results.csv' file.
    
    The results are formatted into a nested list of dictionaries.
    """
    try:

        csv_file_path = Path(__file__).parent.parent / "results" / "algorithm_comparison_results.csv"
        
        if not csv_file_path.exists():
            # If the file doesn't exist, raise an HTTP 404 Not Found error.
            raise HTTPException(status_code=404, detail="Results file not found.")
        
        # Read the data from the CSV file into a pandas DataFrame.
        df = pd.read_csv(csv_file_path)

        best_models_df = df.sort_values(by="Accuracy_Mean", ascending=False).head(11)
                
        formatted_data = []
        for _, row in best_models_df.iterrows():

            model_name = row['Algorithm']
                        
            metrics = row[['Accuracy_Mean', 'F1_Mean', 'Precision_Mean', 'Recall_Mean']].to_dict()
                        
            model_entry = {model_name: metrics}
                        
            formatted_data.append(model_entry)            

        print(formatted_data)
            
        return {
            "status": "success",
            "data": formatted_data  
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading best models: {str(e)}")

@app.get("/api/predictions")
async def get_predictions():
    """Get all predictions with pagination"""
    try:
        results_path = Path(__file__).parent.parent / "results" / "predictions.csv"
        
        if not results_path.exists():
            return {"status": "no_data", "data": [], "total": 0}
            
        df = pd.read_csv(results_path)
        
        # Convert to records and handle NaN values
        records = df.fillna(0).to_dict('records')
        
        return {
            "status": "success",
            "data": records,  # Limit to first 100 for simplicity
            "total": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {str(e)}")

@app.get("/api/stats")
async def get_statistics():
    """Get detailed statistics"""
    try:
        results_path = Path(__file__).parent.parent / "results" / "predictions.csv"
        
        if not results_path.exists():
            return {"status": "no_data", "data": {}}
            
        df = pd.read_csv(results_path)
        
        # Calculate basic stats
        stats = {
            "total_records": len(df),
            "anomalies": int(df['Predicted_Anomaly'].sum()) if 'Predicted_Anomaly' in df.columns else 0,
            "normal": int(len(df) - df['Predicted_Anomaly'].sum()) if 'Predicted_Anomaly' in df.columns else len(df),
        }
        
        # Add probability distribution if available
        if 'Anomaly_Probability' in df.columns:
            stats["avg_anomaly_prob"] = float(df['Anomaly_Probability'].mean())
            stats["max_anomaly_prob"] = float(df['Anomaly_Probability'].max())
            stats["min_anomaly_prob"] = float(df['Anomaly_Probability'].min())
        
        return {"status": "success", "data": stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading statistics: {str(e)}")

@app.post("/api/run-detection")
async def run_detection():
    """Trigger anomaly detection pipeline"""
    try:
        # Import and run the detection
        from main import run_anomaly_detection
        
        results = run_anomaly_detection()
        
        if results:
            return {
                "status": "success",
                "message": "Anomaly detection completed successfully",
                "data": {
                    "best_model": results['model_results']['best_model_name'],
                    "anomalies_detected": int(results['predictions'].sum()),
                    "total_samples": len(results['predictions'])
                }
            }
        else:
            return {"status": "error", "message": "Anomaly detection failed"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running detection: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
