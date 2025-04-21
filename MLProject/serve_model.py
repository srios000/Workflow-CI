#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wine Quality Model Serving Script
---------------------------------
Script untuk serving model Wine Quality sebagai REST API menggunakan FastAPI
dan monitoring dengan Prometheus
"""

import os
import argparse
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import time
import joblib
import dagshub
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("serving.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_quality_serving")

dagshub_token = os.environ.get("DAGSHUB_TOKEN")
dagshub_username = os.environ.get("DAGSHUB_USERNAME")
repo_name = "wine_quality_mlops"

# Inisialisasi FastAPI
app = FastAPI(title="Wine Quality Prediction API", 
             description="API untuk memprediksi kualitas wine berdasarkan properti fisikokimia",
             version="1.0.0")

# Definisi metrik Prometheus
PREDICTION_COUNT = Counter('wine_prediction_total', 'Total number of wine quality predictions')
PREDICTION_LATENCY = Histogram('wine_prediction_latency_seconds', 'Latency of wine quality predictions (seconds)')
PREDICTION_VALUE = Gauge('wine_prediction_value', 'Predicted wine quality value')
PREDICTION_FEATURE_GAUGE = {}  # akan diisi nanti setelah tahu nama fitur
MODEL_INFO = Gauge('wine_model_info', 'Wine quality model information', ['model_type', 'run_id'])
MODEL_LOAD_TIME = Gauge('wine_model_load_time_seconds', 'Time taken to load the model (seconds)')
API_REQUEST_SUMMARY = Summary('wine_api_request_duration_seconds', 'API request duration in seconds')

# Variabel global untuk model dan scaler
model = None
scaler = None
feature_names = []
model_type = ""
run_id = ""

# Schema untuk input API
class WineInput(BaseModel):
    """
    Schema untuk input API: properti fisikokimia wine
    """
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class WineBatchInput(BaseModel):
    """
    Schema untuk input batch API
    """
    wines: List[WineInput]

def setup_mlflow():
    """
    Mengatur koneksi ke DagsHub untuk MLflow tracking
    """
    logger.info("Mengatur koneksi MLflow dengan DagsHub...")
    
    try:
        if dagshub_username and dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
            
            mlflow_tracking_uri = f"https://dagshub.com/{dagshub_username}/{repo_name}.mlflow"
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        else:
            raise ValueError("DagsHub credentials tidak ditemukan pada dokumen .env")
        
        # Set experiment
        mlflow.set_experiment("Wine Quality Prediction")
        
        logger.info("Koneksi MLflow dengan DagsHub berhasil")
    except Exception as e:
        logger.error(f"Gagal mengatur koneksi MLflow: {str(e)}")
        # Fallback ke local tracking
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Wine Quality Prediction")
        logger.info("Menggunakan MLflow tracking lokal")

def load_model(model_run_id=None):
    """
    Memuat model dari MLflow berdasarkan run_id atau mengambil yang terbaru
    """
    global model, scaler, feature_names, model_type, run_id
    
    start_time = time.time()
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        if model_run_id is None:
            # Jika run_id tidak diberikan, baca dari file
            if os.path.exists("best_tuned_model_run_id.txt"):
                with open("best_tuned_model_run_id.txt", "r") as f:
                    model_run_id = f.read().strip()
                logger.info(f"Menggunakan run_id dari file: {model_run_id}")
            else:
                # Ambil run terbaru
                runs = mlflow.search_runs(order_by=["attribute.start_time DESC"])
                if len(runs) > 0:
                    model_run_id = runs.iloc[0]["run_id"]
                    logger.info(f"Menggunakan run_id terbaru: {model_run_id}")
                else:
                    raise ValueError("Tidak ditemukan runs di MLflow")
        
        # Assign to global run_id
        run_id = model_run_id
        
        # Muat model dari MLflow
        logger.info(f"Memuat model dari MLflow dengan run_id: {run_id}")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        # Mendapatkan tipe model
        model_type = model.__class__.__name__
        logger.info(f"Model loaded: {model_type}")
        
        # Set metrik Prometheus untuk info model
        MODEL_INFO.labels(model_type=model_type, run_id=run_id).set(1)
        
        # Muat scaler
        if os.path.exists("winequality_preprocessing/scaler.pkl"):
            scaler = joblib.load("winequality_preprocessing/scaler.pkl")
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("Scaler not found, predictions may be inaccurate")
        
        # Muat feature names
        if os.path.exists("winequality_preprocessing/feature_names.txt"):
            with open("winequality_preprocessing/feature_names.txt", "r") as f:
                feature_names = f.read().split(',')
            logger.info(f"Feature names loaded: {feature_names}")
            
            # Inisialisasi gauge untuk setiap fitur
            for feature in feature_names:
                PREDICTION_FEATURE_GAUGE[feature] = Gauge(
                    f'wine_feature_{feature}', f'Value of {feature} feature in prediction'
                )
        else:
            feature_names = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid', 
                'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
            logger.warning("Feature names file not found, using default names")
            
            # Inisialisasi gauge untuk setiap fitur
            for feature in feature_names:
                PREDICTION_FEATURE_GAUGE[feature] = Gauge(
                    f'wine_feature_{feature}', f'Value of {feature} feature in prediction'
                )
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """
    Menjalankan aksi saat aplikasi startup
    """
    logger.info("Starting Wine Quality Prediction API...")
    
    # Memulai server Prometheus di port 8000
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    
    # Muat model
    if not load_model(run_id):
        logger.error("Failed to load model at startup")

@app.get("/")
def read_root():
    """
    Root endpoint
    """
    return {
        "message": "Wine Quality Prediction API",
        "model_type": model_type,
        "run_id": run_id,
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_type": model_type}

@app.post("/predict")
@API_REQUEST_SUMMARY.time()
def predict_quality(wine: WineInput):
    """
    Prediksi kualitas wine berdasarkan properti fisikokimia
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Konversi input ke DataFrame
        input_dict = wine.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Persiapkan data untuk prediksi
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Catat nilai fitur untuk monitoring
        for i, feature in enumerate(feature_names):
            if i < len(input_scaled[0]):
                PREDICTION_FEATURE_GAUGE[feature].set(input_scaled[0][i])
        
        # Prediksi dengan pengukuran latensi
        start_time = time.time()
        with PREDICTION_LATENCY.time():
            prediction = model.predict(input_scaled)[0]
        latency = time.time() - start_time
        
        # Catat metrik
        PREDICTION_COUNT.inc()
        PREDICTION_VALUE.set(prediction)
        
        # Return hasil prediksi
        return {
            "quality_prediction": prediction,
            "wine_properties": input_dict,
            "model_type": model_type,
            "latency_seconds": latency
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
@API_REQUEST_SUMMARY.time()
def predict_batch_quality(wines: WineBatchInput):
    """
    Prediksi kualitas untuk batch wine
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Konversi input ke DataFrame
        input_dicts = [wine.dict() for wine in wines.wines]
        input_df = pd.DataFrame(input_dicts)
        
        # Persiapkan data untuk prediksi
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Prediksi dengan pengukuran latensi
        start_time = time.time()
        with PREDICTION_LATENCY.time():
            predictions = model.predict(input_scaled).tolist()
        latency = time.time() - start_time
        
        # Catat metrik
        PREDICTION_COUNT.inc(len(predictions))
        PREDICTION_VALUE.set(np.mean(predictions))
        
        # Return hasil prediksi
        return {
            "quality_predictions": predictions,
            "count": len(predictions),
            "model_type": model_type,
            "latency_seconds": latency
        }
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
def get_model_info():
    """
    Mendapatkan informasi model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Feature importance jika tersedia
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        for i, feature in enumerate(feature_names):
            if i < len(model.feature_importances_):
                feature_importance[feature] = float(model.feature_importances_[i])
    
    return {
        "model_type": model_type,
        "run_id": run_id,
        "features": feature_names,
        "feature_importance": feature_importance
    }

@app.post("/model/reload")
def reload_model(new_run_id: Optional[str] = None):
    """
    Memuat ulang model, opsional dengan run_id baru
    """
    global run_id
    
    if new_run_id is not None:
        run_id = new_run_id
    
    success = load_model(run_id)
    
    if success:
        return {"status": "success", "message": "Model reloaded successfully", "model_type": model_type, "run_id": run_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

def main(args):
    """
    Fungsi utama untuk menjalankan server
    """
    global run_id
    
    # Set run_id from arguments if provided
    if args.run_id:
        run_id = args.run_id
    
    # Start Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Serve a wine quality prediction model")
    parser.add_argument("--run_id", type=str, default=None,
                        help="MLflow run ID for the model to serve")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the server on")
    
    args = parser.parse_args()
    
    main(args)