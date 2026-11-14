import os
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.data_loader import load_data, validate_and_summarize
from src.preprocessor import InsurancePreprocessor, MedicalFeatureEngineer
from src.predict import InsurancePredictor
from app.api import app

def test_data_loader():
    df = load_data("data/raw/insurance.csv")
    assert not df.empty
    assert len(df) > 1000
    expected_cols = {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'}
    assert expected_cols.issubset(set(df.columns))

def test_feature_engineer():
    sample_df = pd.DataFrame([
        {"age": 25, "sex": "male", "bmi": 32.5, "children": 0, "smoker": "yes", "region": "southwest"},
        {"age": 40, "sex": "female", "bmi": 22.0, "children": 2, "smoker": "no", "region": "northeast"}
    ])
    fe = MedicalFeatureEngineer()
    res = fe.transform(sample_df)
    
    assert 'is_obese' in res.columns
    assert 'obese_smoker' in res.columns
    assert res.iloc[0]['is_obese'] == 1
    assert res.iloc[0]['obese_smoker'] == 1
    assert res.iloc[1]['is_obese'] == 0
    assert res.iloc[1]['obese_smoker'] == 0

def test_preprocessor_pipeline():
    df = load_data("data/raw/insurance.csv")
    X = df.drop(columns=['charges'])
    preprocessor = InsurancePreprocessor()
    X_proc = preprocessor.fit_transform(X)
    
    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == len(df)
    assert X_proc.shape[1] == len(preprocessor.feature_names)

def test_single_prediction():
    if os.path.exists("models/best_model.pkl") and os.path.exists("models/preprocessor.pkl"):
        predictor = InsurancePredictor()
        patient = {
            "age": 28,
            "sex": "female",
            "bmi": 24.0,
            "children": 1,
            "smoker": "no",
            "region": "northwest"
        }
        res = predictor.predict_single(patient)
        assert "predicted_charges" in res
        assert res["predicted_charges"] > 0
        assert "risk_tier" in res
        assert "recommendations" in res

def test_batch_50_applicants_prediction():
    if os.path.exists("models/best_model.pkl") and os.path.exists("data/sample_50_applicants.csv"):
        predictor = InsurancePredictor()
        batch_df = pd.read_csv("data/sample_50_applicants.csv")
        assert len(batch_df) == 50
        result_df = predictor.predict_batch(batch_df)
        assert len(result_df) == 50
        assert "Predicted_Charges" in result_df.columns
        assert "Risk_Tier" in result_df.columns
        assert (result_df["Predicted_Charges"] > 0).all()

def test_fastapi_endpoints():
    client = TestClient(app)
    res_root = client.get("/")
    assert res_root.status_code == 200
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    res_pred = client.post("/predict", json={
        "age": 45,
        "sex": "male",
        "bmi": 29.0,
        "children": 2,
        "smoker": "no",
        "region": "southeast"
    })
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "predicted_charges" in data
    assert data["predicted_charges"] > 0
    assert "risk_tier" in data
