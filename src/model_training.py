import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_data, split_and_save_data
from src.preprocessor import InsurancePreprocessor

def compute_adjusted_r2(r2: float, n: int, p: int) -> float:
    """Calculate Adjusted R-Squared: 1 - [(1 - R2) * (n - 1) / (n - p - 1)]."""
    if n - p - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def get_candidate_models() -> Dict[str, Any]:
    """Returns a dictionary of candidate ML regression models for evaluation."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=3000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=1000, epsilon=0.1)
    }

def train_and_evaluate_models(X_train: np.ndarray, 
                              y_train: np.ndarray, 
                              X_test: np.ndarray, 
                              y_test: np.ndarray) -> Tuple[Dict[str, Dict[str, float]], str, Any]:
    models = get_candidate_models()
    results = {}
    best_score = -float("inf")
    best_model_name = ""
    best_model_obj = None

    n_test, p_features = X_test.shape

    print("=" * 70)
    print("TRAINING & BENCHMARKING CANDIDATE MODELS")
    print("=" * 70)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        mean_cv_r2 = float(np.mean(cv_scores))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        adj_r2 = float(compute_adjusted_r2(r2, n_test, p_features))
        
        results[name] = {
            "CV_R2_Mean": round(mean_cv_r2, 4),
            "Test_MAE": round(mae, 2),
            "Test_MSE": round(mse, 2),
            "Test_RMSE": round(rmse, 2),
            "Test_R2": round(r2, 4),
            "Adjusted_R2": round(adj_r2, 4)
        }
        
        print(f"[{name:30}] -> R²: {r2:.4f} | RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | 5-Fold CV R²: {mean_cv_r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_model_obj = model

    return results, best_model_name, best_model_obj

def tune_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Tuning Gradient Boosting with GridSearchCV."""
    print("\n[Hyperparameter Tuning] Optimizing Gradient Boosting Regressor...")
    param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Gradient Boosting Params: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Tuning Random Forest with GridSearchCV."""
    print("\n[Hyperparameter Tuning] Optimizing Random Forest Regressor...")
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 8, 12, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Random Forest Params: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def run_pipeline():
    # 1. Ingest and split
    df = load_data("data/raw/insurance.csv")
    train_df, test_df = split_and_save_data(df)
    
    X_train_raw = train_df.drop(columns=['charges'])
    y_train = train_df['charges'].values
    X_test_raw = test_df.drop(columns=['charges'])
    y_test = test_df['charges'].values

    # 2. Preprocess
    preprocessor = InsurancePreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    preprocessor.save("models/preprocessor.pkl")

    # 3. Train baseline models
    metrics, best_baseline_name, best_baseline_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    n_test, p_features = X_test.shape

    # 4. Tune Gradient Boosting
    tuned_gb = tune_gradient_boosting(X_train, y_train)
    y_pred_gb = tuned_gb.predict(X_test)
    gb_r2 = float(r2_score(y_test, y_pred_gb))
    gb_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_gb)))
    gb_mae = float(mean_absolute_error(y_test, y_pred_gb))
    metrics["Tuned Gradient Boosting"] = {
        "CV_R2_Mean": round(float(cross_val_score(tuned_gb, X_train, y_train, cv=5, scoring='r2').mean()), 4),
        "Test_MAE": round(gb_mae, 2),
        "Test_MSE": round(float(mean_squared_error(y_test, y_pred_gb)), 2),
        "Test_RMSE": round(gb_rmse, 2),
        "Test_R2": round(gb_r2, 4),
        "Adjusted_R2": round(compute_adjusted_r2(gb_r2, n_test, p_features), 4)
    }

    # 5. Tune Random Forest
    tuned_rf = tune_random_forest(X_train, y_train)
    y_pred_rf = tuned_rf.predict(X_test)
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
    metrics["Tuned Random Forest"] = {
        "CV_R2_Mean": round(float(cross_val_score(tuned_rf, X_train, y_train, cv=5, scoring='r2').mean()), 4),
        "Test_MAE": round(rf_mae, 2),
        "Test_MSE": round(float(mean_squared_error(y_test, y_pred_rf)), 2),
        "Test_RMSE": round(rf_rmse, 2),
        "Test_R2": round(rf_r2, 4),
        "Adjusted_R2": round(compute_adjusted_r2(rf_r2, n_test, p_features), 4)
    }

    # Determine Champion
    all_models = {
        "Linear Regression": metrics["Linear Regression"]["Test_R2"],
        "Tuned Gradient Boosting": gb_r2,
        "Tuned Random Forest": rf_r2
    }
    champion_name = max(all_models, key=all_models.get)
    if champion_name == "Tuned Gradient Boosting":
        champion_model = tuned_gb
    elif champion_name == "Tuned Random Forest":
        champion_model = tuned_rf
    else:
        # Default to Tuned Gradient Boosting if within 0.005 of Linear Regression for tree interpretability
        if abs(gb_r2 - metrics["Linear Regression"]["Test_R2"]) < 0.01:
            champion_name = "Tuned Gradient Boosting"
            champion_model = tuned_gb
        else:
            champion_model = best_baseline_model

    print("\n" + "=" * 70)
    print(f"CHAMPION MODEL SELECTED: {champion_name}")
    print(f" - Test R² Score:     {metrics[champion_name]['Test_R2']:.4f}")
    print(f" - Adjusted R²:       {metrics[champion_name]['Adjusted_R2']:.4f}")
    print(f" - Test RMSE:         ${metrics[champion_name]['Test_RMSE']:,.2f}")
    print(f" - Test MAE:          ${metrics[champion_name]['Test_MAE']:,.2f}")
    print("=" * 70)

    # Save metrics JSON & CSV
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv("reports/model_comparison.csv", index_label="Model")
    print("Comparison metrics exported to reports/model_comparison.csv")

    # Save Champion Model
    best_model_path = "models/best_model.pkl"
    joblib.dump(champion_model, best_model_path)
    print(f"Saved Champion model to {best_model_path}")

    metadata = {
        "champion_model": champion_name,
        "test_r2": metrics[champion_name]["Test_R2"],
        "test_rmse": metrics[champion_name]["Test_RMSE"],
        "test_mae": metrics[champion_name]["Test_MAE"],
        "features": preprocessor.feature_names
    }
    with open("models/champion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    run_pipeline()
