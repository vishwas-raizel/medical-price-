import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_features: int) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics."""
    n = len(y_true)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Adjusted R2
    if n - p_features - 1 > 0:
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p_features - 1))
    else:
        adj_r2 = r2
        
    # Residual diagnostics
    residuals = y_true - y_pred
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    res_skew = float(stats.skew(residuals))
    
    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4),
        "residual_mean": round(res_mean, 2),
        "residual_std": round(res_std, 2),
        "residual_skew": round(res_skew, 3)
    }
