#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wine Quality Model Training Script
----------------------------------
Script untuk melatih model machine learning dengan dataset Wine Quality
dan melakukan logging menggunakan MLflow
"""

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import logging
import joblib
import dagshub
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_quality_training")

dagshub_token = os.environ.get("DAGSHUB_TOKEN")
dagshub_username = os.environ.get("DAGSHUB_USERNAME")
repo_name = "wine_quality_mlops"

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

def load_data(data_path='winequality_preprocessing'):
    """
    Memuat dataset yang telah diproses
    """
    logger.info(f"Memuat data dari '{data_path}'...")
    
    train_data = pd.read_csv(f'{data_path}/train_data.csv')
    test_data = pd.read_csv(f'{data_path}/test_data.csv')
    
    # Memisahkan fitur dan target
    X_train = train_data.drop('quality', axis=1)
    y_train = train_data['quality']
    
    X_test = test_data.drop('quality', axis=1)
    y_test = test_data['quality']
    
    logger.info(f"Data berhasil dimuat. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi model dengan berbagai metrik
    """
    y_pred = model.predict(X_test)
    
    # Menghitung metrik evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Menghitung metrik tambahan: median absolute error dan max error
    median_ae = np.median(np.abs(y_test - y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    
    # Mengembalikan metrik evaluasi
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "median_absolute_error": median_ae,
        "max_error": max_error
    }

def train_model(X_train, y_train, X_test, y_test, model_type="random_forest", params=None):
    """
    Melatih model berdasarkan tipe dan parameter yang diberikan
    """
    logger.info(f"Melatih model '{model_type}' dengan parameter: {params}")
    
    if params is None:
        params = {}
    
    # Memilih model berdasarkan tipe
    if model_type == "random_forest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(**params, random_state=42)
    elif model_type == "linear_regression":
        model = LinearRegression(**params)
    elif model_type == "elastic_net":
        model = ElasticNet(**params, random_state=42)
    else:
        raise ValueError(f"Tipe model '{model_type}' tidak didukung")
    
    # Melatih model
    model.fit(X_train, y_train)
    
    # Mengevaluasi model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-np.mean(cv_scores))
    
    logger.info(f"Model '{model_type}' telah dilatih dengan RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    
    return model, metrics, cv_rmse

def log_model_with_mlflow(model, model_type, params, metrics, cv_rmse, feature_names):
    """
    Melakukan logging model dan metrik dengan MLflow
    """
    logger.info(f"Logging model dan metrik dengan MLflow...")
    
    with mlflow.start_run(run_name=f"{model_type}_run"):
        # Log parameter model
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrik evaluasi
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log metrik tambahan
        mlflow.log_metric("cv_rmse", cv_rmse)
        
        # Menyimpan feature importance jika model mendukung
        if hasattr(model, 'feature_importances_'):
            # Membuat dataframe feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Menyimpan ke CSV dan log sebagai artifact
            importance_path = "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            # Log feature importance sebagai param
            for feature, importance in zip(feature_names, model.feature_importances_):
                mlflow.log_param(f"importance_{feature}", importance)
        
        # Menyimpan model
        mlflow.sklearn.log_model(model, "model")
        
        # Menyimpan model dengan joblib (backup)
        model_path = f"models/{model_type}_model.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        logger.info(f"Model telah berhasil di-log dengan MLflow")
        
        return mlflow.active_run().info.run_id

def main(args):
    """
    Fungsi utama untuk menjalankan alur pelatihan model
    """
    logger.info("Memulai proses pelatihan model Wine Quality...")
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Memuat data
        X_train, y_train, X_test, y_test = load_data(args.data_path)
        
        # Default parameters untuk model
        model_params = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5
            },
            "elastic_net": {
                "alpha": 0.5,
                "l1_ratio": 0.5
            },
            "linear_regression": {}
        }
        
        # Jika tuning diaktifkan, import dan jalankan tuning script
        if args.tuning:
            logger.info("Menjalankan hyperparameter tuning...")
            # Import tuning module
            import modelling_tuning
            tuning_results = modelling_tuning.main()
            
            # Mengambil run ID terbaik dari hasil tuning
            best_model_name = min(tuning_results.keys(), key=lambda k: tuning_results[k]["metrics"]["rmse"])
            best_run_id = tuning_results[best_model_name]["run_id"]
            
            logger.info(f"Model terbaik dari tuning: {best_model_name} dengan run_id {best_run_id}")
            return best_run_id
        else:
            # Jika tidak tuning, latih model dengan parameter default
            model_type = args.model_type
            params = model_params.get(model_type, {})
            
            # Melatih model
            model, metrics, cv_rmse = train_model(
                X_train, y_train, X_test, y_test, model_type, params
            )
            
            # Logging model ke MLflow
            run_id = log_model_with_mlflow(
                model, model_type, params, metrics, cv_rmse, X_train.columns
            )
            
            logger.info(f"Model {model_type} telah dilatih dengan RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            logger.info(f"Run ID: {run_id}")
            
            # Menyimpan run ID
            with open("best_model_run_id.txt", "w") as f:
                f.write(run_id)
            
            logger.info("Proses pelatihan model selesai!")
            return run_id
    
    except Exception as e:
        logger.error(f"Terjadi kesalahan dalam pelatihan model: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a wine quality prediction model")
    parser.add_argument("--data_path", type=str, default="winequality_preprocessing",
                        help="Path to the preprocessed data directory")
    parser.add_argument("--model_type", type=str, default="random_forest",
                        choices=["random_forest", "gradient_boosting", "elastic_net", "linear_regression"],
                        help="Type of model to train")
    parser.add_argument("--tuning", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True,
                        help="Whether to perform hyperparameter tuning")
    
    args = parser.parse_args()
    
    main(args)