import os
import sys

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from src.preprocessor import InsurancePreprocessor, MedicalFeatureEngineer

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

class InsurancePredictor:
    """
    Inference engine for Medical Insurance Price Prediction.
    Supports both single instance and batch CSV predictions.
    """
    def __init__(self, model_path: str = MODEL_PATH, preprocessor_path: str = PREPROCESSOR_PATH):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Trained model or preprocessor not found. Please train models first.")
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def assess_risk_tier(self, charges: float) -> Dict[str, str]:
        """Categorize predicted insurance cost into risk bands with actionable insights."""
        if charges < 5000:
            return {
                "tier": "Low Risk",
                "badge_color": "green",
                "description": "Favorable actuarial profile. Standard low-cost coverage recommended."
            }
        elif charges < 15000:
            return {
                "tier": "Moderate Risk",
                "badge_color": "blue",
                "description": "Average expected claims. Standard comprehensive policy recommended."
            }
        elif charges < 30000:
            return {
                "tier": "High Risk",
                "badge_color": "orange",
                "description": "Elevated healthcare risk factors detected. Premium deductible policy advisable."
            }
        else:
            return {
                "tier": "Very High / Critical Risk",
                "badge_color": "red",
                "description": "Significant compounding risk factors (e.g. smoking + high BMI). High premium policy tier."
            }

    def generate_recommendations(self, data: Dict[str, Any], predicted_charge: float) -> list:
        """Generates dynamic preventative recommendations to optimize insurance cost."""
        tips = []
        is_smoker = str(data.get("smoker", "")).lower() == "yes"
        bmi = float(data.get("bmi", 0))
        age = int(data.get("age", 0))

        if is_smoker:
            tips.append("Smoking is the #1 premium inflation factor. Enrolling in a smoking cessation program can reduce premiums by up to 50% - 60%.")
        
        if bmi >= 30:
            tips.append(f"Current BMI ({bmi:.1f}) is in the obese range (>=30). Lowering BMI into the normal range (18.5 - 24.9) significantly reduces annual underwriting charges.")
        elif bmi >= 25:
            tips.append(f"Current BMI ({bmi:.1f}) is slightly overweight. Maintaining a healthy balanced diet can keep health premiums low.")

        if age >= 50:
            tips.append("Annual preventative wellness checkups and comprehensive screening can catch conditions early and lower deductible costs.")

        if not tips:
            tips.append("Excellent health profile! Maintaining current fitness and annual check-ups will keep insurance premiums at the lowest rate.")

        return tips

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict insurance charges for an individual.
        Input dictionary keys: age, sex, bmi, children, smoker, region.
        """
        input_df = pd.DataFrame([data])
        X_proc = self.preprocessor.transform(input_df)
        prediction = float(self.model.predict(X_proc)[0])
        prediction = max(0.0, prediction) # Ensure non-negative

        risk = self.assess_risk_tier(prediction)
        recommendations = self.generate_recommendations(data, prediction)

        return {
            "predicted_charges": round(prediction, 2),
            "formatted_charges": f"${prediction:,.2f}",
            "risk_tier": risk["tier"],
            "risk_badge": risk["badge_color"],
            "risk_description": risk["description"],
            "recommendations": recommendations,
            "input_data": data
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict insurance charges for a batch of records.
        """
        required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe missing columns: {missing}")
            
        X_proc = self.preprocessor.transform(df)
        preds = self.model.predict(X_proc)
        
        result_df = df.copy()
        result_df['Predicted_Charges'] = np.round(np.maximum(0, preds), 2)
        result_df['Risk_Tier'] = [self.assess_risk_tier(p)['tier'] for p in result_df['Predicted_Charges']]
        
        return result_df

if __name__ == "__main__":
    predictor = InsurancePredictor()
    sample = {
        "age": 35,
        "sex": "male",
        "bmi": 28.5,
        "children": 2,
        "smoker": "no",
        "region": "southwest"
    }
    res = predictor.predict_single(sample)
    print("Sample Prediction Result:", res)
