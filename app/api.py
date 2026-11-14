import os
import sys

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import json
from src.predict import InsurancePredictor

app = FastAPI(
    title="Medical Insurance Price Prediction API",
    description="Machine Learning REST API for predicting individualized healthcare insurance charges based on demographic and lifestyle indicators.",
    version="1.0.0"
)

# Initialize predictor
try:
    predictor = InsurancePredictor()
except Exception as e:
    predictor = None

class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of primary beneficiary (18-100)", examples=[32])
    sex: str = Field(..., description="Gender ('male' or 'female')", examples=["male"])
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index (BMI)", examples=[28.4])
    children: int = Field(0, ge=0, le=10, description="Number of children/dependents covered", examples=[1])
    smoker: str = Field(..., description="Smoking status ('yes' or 'no')", examples=["no"])
    region: str = Field(..., description="Residential area ('northeast', 'northwest', 'southeast', 'southwest')", examples=["northwest"])

class PredictionResponse(BaseModel):
    predicted_charges: float
    formatted_charges: str
    risk_tier: str
    risk_badge: str
    risk_description: str
    recommendations: List[str]
    input_data: Dict[str, Any]

@app.get("/", tags=["General"])
def root():
    return {
        "project": "Medical Insurance Price Prediction System",
        "author": "College Main Project",
        "docs_url": "/docs",
        "status": "online"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

@app.get("/model/metrics", tags=["Model Analytics"])
def get_metrics():
    metrics_path = "models/model_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics not found. Train the model first.")
    with open(metrics_path, "r") as f:
        return json.load(f)

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_insurance(patient: PatientData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor engine is not initialized. Please train the model.")
    
    data = patient.model_dump()
    # Validate categorical values
    if data["sex"].lower() not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'")
    if data["smoker"].lower() not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Smoker must be 'yes' or 'no'")
    if data["region"].lower() not in ["northeast", "northwest", "southeast", "southwest"]:
        raise HTTPException(status_code=400, detail="Region must be one of: 'northeast', 'northwest', 'southeast', 'southwest'")

    result = predictor.predict_single(data)
    return result

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
