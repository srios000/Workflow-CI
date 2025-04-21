#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wine Quality Model Hyperparameter Tuning Script
-----------------------------------------------
Script untuk melakukan hyperparameter tuning pada model machine learning
untuk dataset Wine Quality dan logging hasil dengan MLflow
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
import mlflow
import mlflow.sklearn
import logging
import joblib
import dagshub
from scipy.stats import randint, uniform
from dotenv import load_dotenv
load_dotenv()

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tuning.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wine_quality_tuning")

# Konfigurasi DagsHub dan MLflow
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
            raise ValueError("DagsHub credentials tidak ditemukan pada dokumen .env.")
        
        mlflow.set_experiment("Wine Quality Tuning")
        
        logger.info("Koneksi MLflow dengan DagsHub berhasil")
    except Exception as e:
        logger.error(f"Gagal mengatur koneksi MLflow: {str(e)}")
        # Fallback ke local tracking
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Wine Quality Tuning")
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
    Mengevaluasi model dengan berbagai metrics
    """
    y_pred = model.predict(X_test)
    
    # Menghitung standard evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Menghitung additional metrics
    # Median absolute error dan max error
    median_ae = np.median(np.abs(y_test - y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    explained_variance = 1 - (np.var(y_test - y_pred) / np.var(y_test))
    
    # Menghitung custom metrics: percentage of predictions within 0.5 units
    within_half_unit = np.mean(np.abs(y_test - y_pred) <= 0.5) * 100
    within_one_unit = np.mean(np.abs(y_test - y_pred) <= 1.0) * 100
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "median_absolute_error": median_ae,
        "max_error": max_error,
        "explained_variance": explained_variance,
        "within_half_unit_percent": within_half_unit,
        "within_one_unit_percent": within_one_unit
    }

def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Melakukan tuning hyperparameter untuk model Random Forest
    """
    logger.info("Memulai tuning Random Forest...")
    
    # Parameter search space
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Melakukan tuning dengan RandomizedSearchCV
    rf = RandomForestRegressor(random_state=42)
    
    # Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Menjalankan tuning
    random_search.fit(X_train, y_train)
    
    # Menentukan parameter terbaik
    best_params = random_search.best_params_
    logger.info(f"Parameter terbaik untuk Random Forest: {best_params}")
    
    # Melatih model dengan parameter terbaik
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Evaluasi model
    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Metrics untuk Random Forest tuned: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Log ke MLflow
    with mlflow.start_run(run_name="random_forest_tuned"):
        # Log parameter
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Simpan importance
        importance_path = "rf_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        # Log feature importance sebagai params
        for feature, importance in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f"importance_{feature}", importance)
        
        # Save run_id
        run_id = mlflow.active_run().info.run_id
    
    return best_model, metrics, run_id

def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Melakukan tuning hyperparameter untuk model Gradient Boosting
    """
    logger.info("Memulai tuning Gradient Boosting...")
    
    # Parameter search space
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'subsample': uniform(0.5, 0.5),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Melakukan tuning dengan RandomizedSearchCV
    gb = GradientBoostingRegressor(random_state=42)
    
    # Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_dist,
        n_iter=20,
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Menjalankan tuning
    random_search.fit(X_train, y_train)
    
    # Menentukan parameter terbaik
    best_params = random_search.best_params_
    logger.info(f"Parameter terbaik untuk Gradient Boosting: {best_params}")
    
    # Melatih model dengan parameter terbaik
    best_model = GradientBoostingRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Evaluasi model
    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Metrics untuk Gradient Boosting tuned: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Log ke MLflow
    with mlflow.start_run(run_name="gradient_boosting_tuned"):
        # Log parameter
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Simpan importance
        importance_path = "gb_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "gradient_boosting_model")
        
        # Log feature importance sebagai params
        for feature, importance in zip(X_train.columns, best_model.feature_importances_):
            mlflow.log_param(f"importance_{feature}", importance)
        
        # Save run_id
        run_id = mlflow.active_run().info.run_id
    
    return best_model, metrics, run_id

def tune_elastic_net(X_train, y_train, X_test, y_test):
    """
    Melakukan tuning hyperparameter untuk model Elastic Net
    """
    logger.info("Memulai tuning Elastic Net...")
    
    # Parameter search space
    param_dist = {
        'alpha': uniform(0, 1),
        'l1_ratio': uniform(0, 1),
        'max_iter': randint(1000, 5000),
        'tol': uniform(1e-5, 1e-3)
    }
    
    # Melakukan tuning dengan RandomizedSearchCV
    en = ElasticNet(random_state=42)
    
    # Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=en,
        param_distributions=param_dist,
        n_iter=20,
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Menjalankan tuning
    random_search.fit(X_train, y_train)
    
    # Menentukan parameter terbaik
    best_params = random_search.best_params_
    logger.info(f"Parameter terbaik untuk Elastic Net: {best_params}")
    
    # Melatih model dengan parameter terbaik
    best_model = ElasticNet(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Evaluasi model
    metrics = evaluate_model(best_model, X_test, y_test)
    logger.info(f"Metrics untuk Elastic Net tuned: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Log ke MLflow
    with mlflow.start_run(run_name="elastic_net_tuned"):
        # Log parameter
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model coefficients
        coef_df = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': best_model.coef_
        }).sort_values('coefficient', key=lambda x: abs(x), ascending=False)
        
        # Simpan coefficients
        coef_path = "elastic_net_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)
        mlflow.log_artifact(coef_path)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "elastic_net_model")
        
        # Log coefficients sebagai params
        for feature, coef in zip(X_train.columns, best_model.coef_):
            mlflow.log_param(f"coef_{feature}", coef)
        
        # Save run_id
        run_id = mlflow.active_run().info.run_id
    
    return best_model, metrics, run_id

def main():
    """
    Fungsi utama untuk menjalankan hyperparameter tuning
    """
    logger.info("Memulai proses hyperparameter tuning untuk Wine Quality...")
    
    try:
        # Setup MLflow dengan DagsHub
        setup_mlflow()
        
        # Memuat data
        X_train, y_train, X_test, y_test = load_data()
        
        # Dictionary untuk menyimpan hasil tuning
        tuning_results = {}
        
        # Tuning Random Forest
        rf_model, rf_metrics, rf_run_id = tune_random_forest(X_train, y_train, X_test, y_test)
        tuning_results["random_forest"] = {
            "model": rf_model,
            "metrics": rf_metrics,
            "run_id": rf_run_id
        }
        
        # Tuning Gradient Boosting
        gb_model, gb_metrics, gb_run_id = tune_gradient_boosting(X_train, y_train, X_test, y_test)
        tuning_results["gradient_boosting"] = {
            "model": gb_model,
            "metrics": gb_metrics,
            "run_id": gb_run_id
        }
        
        # Tuning Elastic Net
        en_model, en_metrics, en_run_id = tune_elastic_net(X_train, y_train, X_test, y_test)
        tuning_results["elastic_net"] = {
            "model": en_model,
            "metrics": en_metrics,
            "run_id": en_run_id
        }
        
        # Menentukan model terbaik berdasarkan RMSE
        best_model_name = min(tuning_results.keys(), key=lambda k: tuning_results[k]["metrics"]["rmse"])
        best_model_info = tuning_results[best_model_name]
        
        logger.info(f"Model terbaik hasil tuning: {best_model_name}")
        logger.info(f"RMSE: {best_model_info['metrics']['rmse']:.4f}, R²: {best_model_info['metrics']['r2']:.4f}")
        logger.info(f"Run ID model terbaik: {best_model_info['run_id']}")
        
        # Menyimpan run ID model terbaik
        with open("best_tuned_model_run_id.txt", "w") as f:
            f.write(best_model_info['run_id'])
        
        # Menyimpan model terbaik dengan joblib
        best_model_path = f"models/{best_model_name}_tuned_model.pkl"
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        joblib.dump(best_model_info['model'], best_model_path)
        
        logger.info(f"Model terbaik disimpan ke {best_model_path}")
        logger.info("Proses hyperparameter tuning selesai!")
        
        return tuning_results
    
    except Exception as e:
        logger.error(f"Terjadi kesalahan dalam tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()