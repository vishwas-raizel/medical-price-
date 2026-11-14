import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from backend.app.ml.datasets.loader import DatasetLoader
from backend.app.ml.preprocessing.pipeline import ProductionPreprocessor, ConformalPredictionEngine
from backend.app.ml.evaluation.metrics import compute_regression_metrics
from backend.app.ml.explainability.shap_explainer import ProductionShapEngine

SAVED_MODELS_DIR = "backend/app/ml/saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def get_candidate_models() -> Dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=3000),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    }

def train_and_benchmark() -> Dict[str, Any]:
    print("=" * 80)
    print("🚀 TRAINING ENTERPRISE MEDICAL INSURANCE PRICE PREDICTION MODELS")
    print("=" * 80)

    # 1. Load authentic dataset
    loader = DatasetLoader()
    train_df, test_df = loader.get_train_test_split(test_size=0.2, random_state=42)
    print(f"Dataset Loaded: {len(train_df)} train instances, {len(test_df)} test instances")

    X_train_raw = train_df.drop(columns=['charges'])
    y_train = train_df['charges'].values
    X_test_raw = test_df.drop(columns=['charges'])
    y_test = test_df['charges'].values

    # 2. Fit Preprocessor
    preprocessor = ProductionPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    p_features = X_train.shape[1]

    preprocessor_path = os.path.join(SAVED_MODELS_DIR, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    print(f"Fitted Preprocessor saved with {p_features} features.")

    # 3. Benchmark all algorithms
    candidate_models = get_candidate_models()
    benchmark_results = {}
    best_score = -float("inf")
    best_model_name = ""
    best_model = None

    for name, model in candidate_models.items():
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = float(np.mean(cv_scores))
        
        # Fit on full training data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = compute_regression_metrics(y_test, y_pred, p_features)
        metrics["cv_r2_mean"] = round(cv_mean, 4)
        benchmark_results[name] = metrics
        
        print(f"[{name:20}] -> Test R²: {metrics['r2']:.4f} | RMSE: ${metrics['rmse']:,.2f} | MAE: ${metrics['mae']:,.2f} | 5-Fold CV: {cv_mean:.4f}")
        
        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_model_name = name
            best_model = model

    # 4. Hyperparameter Optimization for Champion Model
    print(f"\nOptimizing Champion Model ({best_model_name}) via GridSearchCV...")
    if "Gradient Boosting" in best_model_name:
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.03, 0.05, 0.08],
            'max_depth': [3, 4],
            'subsample': [0.8, 1.0]
        }
        grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1)
    elif "XGBoost" in best_model_name:
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.03, 0.05, 0.08],
            'max_depth': [3, 4],
            'subsample': [0.8, 1.0]
        }
        grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid, cv=5, scoring='r2', n_jobs=-1)
    else:
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 8, 12, None],
            'min_samples_split': [2, 5]
        }
        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1)

    grid.fit(X_train, y_train)
    tuned_model = grid.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_metrics = compute_regression_metrics(y_test, y_pred_tuned, p_features)
    tuned_cv = float(cross_val_score(tuned_model, X_train, y_train, cv=5, scoring='r2').mean())
    tuned_metrics["cv_r2_mean"] = round(tuned_cv, 4)

    champion_label = f"Tuned {best_model_name}"
    benchmark_results[champion_label] = tuned_metrics
    benchmark_results[champion_label]["is_champion"] = True

    print(f"\nChampion Selected: {champion_label}")
    print(f" - Best Parameters: {grid.best_params_}")
    print(f" - Test R²:        {tuned_metrics['r2']:.4f}")
    print(f" - Adjusted R²:    {tuned_metrics['adjusted_r2']:.4f}")
    print(f" - Test RMSE:      ${tuned_metrics['rmse']:,.2f}")
    print(f" - Test MAE:       ${tuned_metrics['mae']:,.2f}")

    # 5. Calibrate Conformal Prediction Engine
    conformal_engine = ConformalPredictionEngine(confidence_level=0.90)
    conformal_engine.calibrate(y_test, y_pred_tuned)
    conformal_engine.save(os.path.join(SAVED_MODELS_DIR, "conformal_engine.pkl"))
    print(f"Conformal Prediction Engine calibrated: 90% Error Quantile = ${conformal_engine.quantile_residual:,.2f}")

    # 6. Fit Production SHAP Engine
    print("Fitting SHAP TreeExplainer on Champion Model...")
    shap_engine = ProductionShapEngine()
    shap_engine.fit(tuned_model, X_train[:100], preprocessor.feature_names)
    shap_engine.save(os.path.join(SAVED_MODELS_DIR, "shap_engine.pkl"))

    # Compute global SHAP importances
    global_shap = shap_engine.compute_global_importance(X_test)
    with open(os.path.join(SAVED_MODELS_DIR, "global_shap_importance.json"), "w") as f:
        json.dump(global_shap, f, indent=4)

    # 7. Dataset distribution statistics
    dataset_stats = {
        "min_charges": float(np.min(y_train)),
        "max_charges": float(np.max(y_train)),
        "mean_charges": float(np.mean(y_train)),
        "median_charges": float(np.median(y_train)),
        "p25": float(np.percentile(y_train, 25)),
        "p75": float(np.percentile(y_train, 75)),
        "p90": float(np.percentile(y_train, 90)),
        "charges_sample": [round(float(c), 2) for c in y_train[:300]]
    }
    with open(os.path.join(SAVED_MODELS_DIR, "dataset_stats.json"), "w") as f:
        json.dump(dataset_stats, f, indent=4)

    # 8. Save Champion Model & Metrics
    champion_path = os.path.join(SAVED_MODELS_DIR, "best_model.pkl")
    joblib.dump(tuned_model, champion_path)

    metrics_path = os.path.join(SAVED_MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)

    # Save metadata
    meta = {
        "champion_name": champion_label,
        "algorithm": best_model_name,
        "parameters": grid.best_params_,
        "test_r2": tuned_metrics["r2"],
        "test_rmse": tuned_metrics["rmse"],
        "test_mae": tuned_metrics["mae"],
        "cv_r2": tuned_metrics["cv_r2_mean"],
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features": preprocessor.feature_names,
        "conformal_quantile_90": conformal_engine.quantile_residual,
        "dataset_metadata": loader.metadata
    }
    with open(os.path.join(SAVED_MODELS_DIR, "champion_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("\n✅ All production ML assets, SHAP explainers & conformal engines saved successfully!")
    return benchmark_results

if __name__ == "__main__":
    train_and_benchmark()
