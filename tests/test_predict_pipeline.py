import pytest
import numpy as np
import pandas as pd
from backend.app.ml.datasets.loader import DatasetLoader
from backend.app.ml.preprocessing.pipeline import ProductionPreprocessor, ConformalPredictionEngine
from backend.app.services.prediction_service import prediction_service
from backend.app.schemas.prediction import PredictRequest

def test_dataset_loader():
    loader = DatasetLoader()
    df = loader.load_and_validate()
    assert not df.empty
    assert len(df) >= 1300
    assert loader.metadata["sha256_hash"] is not None
    assert "limitations" in loader.metadata

def test_production_preprocessor():
    loader = DatasetLoader()
    train_df, test_df = loader.get_train_test_split()
    preprocessor = ProductionPreprocessor()
    X_train = preprocessor.fit_transform(train_df.drop(columns=['charges']))
    X_test = preprocessor.transform(test_df.drop(columns=['charges']))
    
    assert isinstance(X_train, np.ndarray)
    assert X_train.shape[1] == len(preprocessor.feature_names)
    assert "obese_smoker" in preprocessor.feature_names

def test_conformal_engine():
    engine = ConformalPredictionEngine(confidence_level=0.90)
    y_true = np.array([1000, 2000, 3000, 4000, 5000])
    y_pred = np.array([1100, 1900, 3100, 3900, 5200])
    engine.calibrate(y_true, y_pred)
    assert engine.quantile_residual > 0
    lower, upper = engine.predict_interval(3000)
    assert lower < 3000 < upper

def test_full_prediction_service():
    req = PredictRequest(
        age=35,
        sex="female",
        bmi=26.4,
        children=2,
        smoker="no",
        region="southeast"
    )
    res = prediction_service.predict(req)
    assert res.estimated_annual_cost > 0
    assert res.conformal_interval.lower_bound <= res.estimated_annual_cost <= res.conformal_interval.upper_bound
    assert len(res.shap_contributions) > 0
    assert res.risk_tier in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
    assert len(res.clinical_recommendations) > 0
