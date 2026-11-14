import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-Specific Actuarial Feature Engineering:
    - is_obese: Clinical threshold (BMI >= 30.0).
    - obese_smoker: Multiplicative interaction term.
      In underwriting science, the compounding morbidity of tobacco use and obesity
      exhibits exponential healthcare claim probability.
    """
    def __init__(self, add_interactions: bool = True):
        self.add_interactions = add_interactions

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Enforce types
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['children'] = pd.to_numeric(df['children'], errors='coerce')
        
        # Obesity binary flag
        df['is_obese'] = (df['bmi'] >= 30.0).astype(int)
        
        # Smoker binary flag
        smoker_flag = (df['smoker'].astype(str).str.lower() == 'yes').astype(int)
        
        # Domain interaction
        if self.add_interactions:
            df['obese_smoker'] = df['is_obese'] * smoker_flag
            
        return df

class ProductionPreprocessor:
    """
    Full preprocessing and feature encoding pipeline.
    """
    def __init__(self):
        self.engineer = MedicalFeatureEngineer()
        self.num_cols = ['age', 'bmi', 'children', 'is_obese', 'obese_smoker']
        self.cat_cols = ['sex', 'smoker', 'region']
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), self.cat_cols)
            ],
            remainder='drop'
        )
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        X_eng = self.engineer.transform(X)
        self.column_transformer.fit(X_eng)
        
        cat_encoder = self.column_transformer.named_transformers_['cat']
        encoded_cat = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + encoded_cat
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor pipeline is not fitted.")
        X_eng = self.engineer.transform(X)
        return self.column_transformer.transform(X_eng)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProductionPreprocessor":
        return joblib.load(filepath)

class ConformalPredictionEngine:
    """
    Split-Conformal Prediction Engine.
    Provides statistically valid, finite-sample $(1 - \alpha)$ prediction intervals
    based on empirical calibration residuals, avoiding arbitrary confidence percentages.
    """
    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.quantile_residual: float = 0.0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        # Calculate non-conformity scores: absolute residual errors |y - y_hat|
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        # Empirical quantile at level ceil((n + 1) * (1 - alpha)) / n
        q_level = np.ceil((n + 1) * self.confidence_level) / n
        q_level = min(1.0, max(0.0, q_level))
        self.quantile_residual = float(np.quantile(residuals, q_level))

    def predict_interval(self, point_prediction: float) -> Tuple[float, float]:
        lower = max(0.0, point_prediction - self.quantile_residual)
        upper = point_prediction + self.quantile_residual
        return round(lower, 2), round(upper, 2)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ConformalPredictionEngine":
        return joblib.load(filepath)
