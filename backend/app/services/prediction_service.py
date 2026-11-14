import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.prediction import PredictionRecord
from backend.app.schemas.prediction import PredictRequest, PredictResponse, ShapContribution, ConformalInterval
from backend.app.ml.preprocessing.pipeline import ProductionPreprocessor, ConformalPredictionEngine
from backend.app.ml.explainability.shap_explainer import ProductionShapEngine

class ProductionPredictionService:
    def __init__(self):
        models_dir = settings.SAVED_MODELS_DIR
        self.model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        self.preprocessor = ProductionPreprocessor.load(os.path.join(models_dir, "preprocessor.pkl"))
        self.conformal_engine = ConformalPredictionEngine.load(os.path.join(models_dir, "conformal_engine.pkl"))
        self.shap_engine = ProductionShapEngine.load(os.path.join(models_dir, "shap_engine.pkl"))
        
        # Load metadata & dataset stats
        with open(os.path.join(models_dir, "champion_metadata.json"), "r") as f:
            self.metadata = json.load(f)
        with open(os.path.join(models_dir, "dataset_stats.json"), "r") as f:
            self.dataset_stats = json.load(f)

    def calculate_percentile(self, cost: float) -> float:
        """Computes empirical percentile rank compared to benchmark dataset."""
        sample = np.array(self.dataset_stats.get("charges_sample", [cost]))
        pct = (sample < cost).mean() * 100
        return round(float(pct), 1)

    def assess_risk(self, charges: float) -> Tuple[str, str]:
        if charges < 6000:
            return "Low Risk", "emerald"
        elif charges < 15000:
            return "Moderate Risk", "blue"
        elif charges < 30000:
            return "High Risk", "amber"
        else:
            return "Critical Risk", "rose"

    def generate_recommendations(self, request: PredictRequest, cost: float) -> list:
        tips = []
        is_smoker = request.smoker.lower() == "yes"
        bmi = request.bmi
        age = request.age

        if is_smoker:
            # Estimate savings if non-smoker
            mock_nosmoke = pd.DataFrame([{
                "age": age, "sex": request.sex.lower(), "bmi": bmi,
                "children": request.children, "smoker": "no", "region": request.region.lower()
            }])
            X_nosmoke = self.preprocessor.transform(mock_nosmoke)
            pred_nosmoke = max(0.0, float(self.model.predict(X_nosmoke)[0]))
            savings = max(0.0, cost - pred_nosmoke)
            tips.append(
                f"Smoking is the primary actuarial cost driver. Transitioning to tobacco cessation "
                f"could lower expected annual charges by approximately ${savings:,.2f}."
            )

        if bmi >= 30.0:
            # Estimate savings if normal BMI
            mock_norm = pd.DataFrame([{
                "age": age, "sex": request.sex.lower(), "bmi": 24.5,
                "children": request.children, "smoker": request.smoker.lower(), "region": request.region.lower()
            }])
            X_norm = self.preprocessor.transform(mock_norm)
            pred_norm = max(0.0, float(self.model.predict(X_norm)[0]))
            bmi_savings = max(0.0, cost - pred_norm)
            tips.append(
                f"Current BMI ({bmi:.1f}) is in the obese range (>=30). Achieving a normal BMI (~24.5) "
                f"could yield estimated annual savings of ${bmi_savings:,.2f}."
            )
        elif bmi >= 25.0:
            tips.append(
                f"Current BMI ({bmi:.1f}) indicates mild overweight status. Regular cardiovascular exercise "
                f"and nutritional optimization help maintain preventive health pricing tiers."
            )

        if age >= 50:
            tips.append(
                "Age-associated healthcare risk can be mitigated through scheduled preventative wellness screenings, "
                "which detect early-stage indications before inpatient claims materialize."
            )

        if not tips:
            tips.append(
                "Optimal health profile. Maintaining active physical wellness and annual health evaluations "
                "ensures eligibility for lowest-tier preferred underwriting."
            )

        return tips

    def predict(self, request: PredictRequest, user_id: Optional[int] = None, db: Optional[Session] = None) -> PredictResponse:
        input_dict = {
            "age": request.age,
            "sex": request.sex.lower(),
            "bmi": request.bmi,
            "children": request.children,
            "smoker": request.smoker.lower(),
            "region": request.region.lower()
        }

        input_df = pd.DataFrame([input_dict])
        X_trans = self.preprocessor.transform(input_df)

        # 1. Point prediction
        point_pred = float(self.model.predict(X_trans)[0])
        point_pred = max(500.0, point_pred)

        # 2. Conformal Prediction Interval (finite-sample coverage)
        lower_bound, upper_bound = self.conformal_engine.predict_interval(point_pred)

        # 3. Explainable AI (SHAP attributions)
        base_val, shap_items = self.shap_engine.explain_instance(X_trans, input_dict)

        # 4. Risk assessment & Percentile
        risk_tier, badge_color = self.assess_risk(point_pred)
        percentile = self.calculate_percentile(point_pred)
        recommendations = self.generate_recommendations(request, point_pred)

        # 5. Database persistence
        record_id = None
        if db is not None:
            db_record = PredictionRecord(
                user_id=user_id,
                age=request.age,
                sex=request.sex.lower(),
                bmi=request.bmi,
                children=request.children,
                smoker=request.smoker.lower(),
                region=request.region.lower(),
                predicted_charges=round(point_pred, 2),
                monthly_charges=round(point_pred / 12.0, 2),
                risk_tier=risk_tier,
                confidence_lower=lower_bound,
                confidence_upper=upper_bound,
                shap_contributions=json.dumps(shap_items),
                recommendations=json.dumps(recommendations),
                model_version=self.metadata.get("champion_name", "v2.0-Production")
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
            record_id = db_record.id

        return PredictResponse(
            prediction_id=record_id,
            estimated_annual_cost=round(point_pred, 2),
            formatted_annual_cost=f"${point_pred:,.2f}",
            estimated_monthly_cost=round(point_pred / 12.0, 2),
            formatted_monthly_cost=f"${(point_pred / 12.0):,.2f}",
            risk_tier=risk_tier,
            risk_badge_color=badge_color,
            conformal_interval=ConformalInterval(
                confidence_level=0.90,
                lower_bound=lower_bound,
                upper_bound=upper_bound
            ),
            base_expected_value=base_val,
            shap_contributions=[ShapContribution(**item) for item in shap_items],
            dataset_percentile=percentile,
            clinical_recommendations=recommendations,
            model_version=self.metadata.get("champion_name", "v2.0-Production"),
            model_algorithm=self.metadata.get("algorithm", "Ensemble Regressor")
        )

prediction_service = ProductionPredictionService()
