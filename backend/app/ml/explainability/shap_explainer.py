import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Tuple

class ProductionShapEngine:
    """
    Production SHAP (SHapley Additive exPlanations) Engine.
    Provides mathematical attribution of feature contributions to insurance price predictions.
    """
    def __init__(self):
        self.explainer = None
        self.base_value: float = 0.0
        self.feature_names: List[str] = []

    def fit(self, model: Any, X_background: np.ndarray, feature_names: List[str]):
        self.feature_names = feature_names
        
        # Check model type for optimal SHAP explainer
        try:
            # TreeExplainer is fast & exact for tree models
            self.explainer = shap.TreeExplainer(model)
            # Get expected base value
            exp_val = self.explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                self.base_value = float(exp_val[0])
            else:
                self.base_value = float(exp_val)
        except Exception:
            # Fallback to general Explainer
            self.explainer = shap.Explainer(model, X_background)
            self.base_value = float(np.mean(model.predict(X_background)))

    def explain_instance(self, X_transformed: np.ndarray, raw_features: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Computes exact local SHAP values for an individual prediction instance.
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer is not fitted.")

        shap_values = self.explainer(X_transformed)
        
        # Extract values for the single row
        if hasattr(shap_values, 'values'):
            row_shap = shap_values.values[0]
        else:
            row_shap = np.array(shap_values)[0]

        total_abs = np.sum(np.abs(row_shap)) + 1e-6
        contributions = []

        for name, val in zip(self.feature_names, row_shap):
            raw_val = raw_features.get(name, "N/A")
            contrib_amt = float(val)
            pct = float((abs(contrib_amt) / total_abs) * 100)
            
            contributions.append({
                "feature": name,
                "feature_value": raw_val,
                "contribution_amount": round(contrib_amt, 2),
                "impact_direction": "increases_cost" if contrib_amt >= 0 else "decreases_cost",
                "percentage_influence": round(pct, 1)
            })

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["contribution_amount"]), reverse=True)
        return round(self.base_value, 2), contributions

    def compute_global_importance(self, X_sample: np.ndarray) -> List[Dict[str, Any]]:
        """Computes mean absolute SHAP values across a validation sample."""
        shap_values = self.explainer(X_sample)
        if hasattr(shap_values, 'values'):
            vals = shap_values.values
        else:
            vals = np.array(shap_values)
            
        mean_abs = np.mean(np.abs(vals), axis=0)
        total = np.sum(mean_abs) + 1e-6
        
        importance_list = []
        for name, score in zip(self.feature_names, mean_abs):
            pct = float((score / total) * 100)
            importance_list.append({
                "feature": name,
                "importance": round(float(score), 2),
                "percentage": round(pct, 1)
            })
            
        importance_list.sort(key=lambda x: x["importance"], reverse=True)
        return importance_list

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProductionShapEngine":
        return joblib.load(filepath)
