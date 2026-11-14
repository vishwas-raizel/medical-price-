from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import io
import pandas as pd

from backend.app.database.session import get_db
from backend.app.schemas.prediction import PredictRequest, PredictResponse
from backend.app.services.prediction_service import prediction_service
from backend.app.services.auth_service import get_current_user_optional
from backend.app.services.audit_service import log_audit_event
from backend.app.models.user import User

router = APIRouter(prefix="/predict", tags=["Inference & Actuarial Prediction"])

@router.post("", response_model=PredictResponse)
def predict_insurance_cost(
    request_data: PredictRequest,
    req: Request,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    # Validate categorical constraints
    sex_clean = request_data.sex.lower().strip()
    if sex_clean not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'")
    smoker_clean = request_data.smoker.lower().strip()
    if smoker_clean not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Smoker must be 'yes' or 'no'")
    region_clean = request_data.region.lower().strip()
    valid_regions = ["northeast", "northwest", "southeast", "southwest"]
    if region_clean not in valid_regions:
        raise HTTPException(status_code=400, detail=f"Region must be one of: {valid_regions}")

    user_id = current_user.id if current_user else None
    result = prediction_service.predict(request_data, user_id=user_id, db=db)

    log_audit_event(
        db,
        action="PREDICTION",
        resource="prediction_engine",
        user_id=user_id,
        ip_address=req.client.host if req.client else None,
        details={
            "predicted_charges": result.estimated_annual_cost,
            "risk_tier": result.risk_tier,
            "smoker": smoker_clean,
            "age": request_data.age
        }
    )

    return result

@router.post("/batch")
def predict_batch_csv(
    file: UploadFile = File(...),
    req: Request = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(('.csv', '.CSV')):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    required_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV missing mandatory columns: {missing}")

    results = []
    user_id = current_user.id if current_user else None

    for idx, row in df.iterrows():
        try:
            req_item = PredictRequest(
                age=int(row['age']),
                sex=str(row['sex']),
                bmi=float(row['bmi']),
                children=int(row['children']),
                smoker=str(row['smoker']),
                region=str(row['region'])
            )
            pred = prediction_service.predict(req_item, user_id=user_id, db=db)
            results.append({
                "row_index": idx + 1,
                "age": req_item.age,
                "sex": req_item.sex,
                "bmi": req_item.bmi,
                "children": req_item.children,
                "smoker": req_item.smoker,
                "region": req_item.region,
                "estimated_annual_cost": pred.estimated_annual_cost,
                "risk_tier": pred.risk_tier,
                "confidence_lower": pred.conformal_interval.lower_bound,
                "confidence_upper": pred.conformal_interval.upper_bound
            })
        except Exception as e:
            results.append({
                "row_index": idx + 1,
                "error": str(e)
            })

    return {
        "filename": file.filename,
        "total_records_processed": len(df),
        "successful_predictions": len([r for r in results if "error" not in r]),
        "predictions": results
    }

class WhatIfSimulationRequest(BaseModel):
    baseline: PredictRequest
    modified: PredictRequest

@router.post("/simulate")
def simulate_what_if_scenario(
    req_data: WhatIfSimulationRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Run real ML models on both baseline and modified scenarios and compute actuarial delta."""
    # We do not persist scenario simulations to DB to avoid polluting user history
    base_pred = prediction_service.predict(req_data.baseline, user_id=None, db=None)
    mod_pred = prediction_service.predict(req_data.modified, user_id=None, db=None)

    delta_cost = mod_pred.estimated_annual_cost - base_pred.estimated_annual_cost
    pct_change = (delta_cost / base_pred.estimated_annual_cost) * 100 if base_pred.estimated_annual_cost > 0 else 0
    annual_savings = max(0.0, -delta_cost)
    monthly_delta = delta_cost / 12.0

    drivers = []
    if req_data.baseline.smoker != req_data.modified.smoker:
        if req_data.modified.smoker == "no":
            drivers.append("Smoking cessation eliminates the severe non-linear tobacco surcharge.")
        else:
            drivers.append("Adopting tobacco usage adds a major exponential actuarial penalty.")

    bmi_diff = req_data.modified.bmi - req_data.baseline.bmi
    if abs(bmi_diff) >= 1.0:
        if bmi_diff < 0:
            drivers.append(f"Reducing BMI by {abs(bmi_diff):.1f} points lowers cardiovascular risk profile.")
        else:
            drivers.append(f"Increasing BMI by {bmi_diff:.1f} points moves applicant into higher risk classification.")

    if req_data.modified.children != req_data.baseline.children:
        drivers.append(f"Dependent coverage adjustment changed by {req_data.modified.children - req_data.baseline.children:+d} dependent(s).")

    return {
        "baseline": {
            "inputs": req_data.baseline.model_dump() if hasattr(req_data.baseline, "model_dump") else req_data.baseline.dict(),
            "estimated_annual_cost": base_pred.estimated_annual_cost,
            "monthly_cost": base_pred.estimated_monthly_cost,
            "risk_tier": base_pred.risk_tier,
            "conformal_lower": base_pred.conformal_interval.lower_bound,
            "conformal_upper": base_pred.conformal_interval.upper_bound,
            "shap_contributions": base_pred.shap_contributions
        },
        "modified": {
            "inputs": req_data.modified.model_dump() if hasattr(req_data.modified, "model_dump") else req_data.modified.dict(),
            "estimated_annual_cost": mod_pred.estimated_annual_cost,
            "monthly_cost": mod_pred.estimated_monthly_cost,
            "risk_tier": mod_pred.risk_tier,
            "conformal_lower": mod_pred.conformal_interval.lower_bound,
            "conformal_upper": mod_pred.conformal_interval.upper_bound,
            "shap_contributions": mod_pred.shap_contributions
        },
        "comparison": {
            "delta_annual_cost": round(delta_cost, 2),
            "delta_monthly_cost": round(monthly_delta, 2),
            "percentage_change": round(pct_change, 2),
            "annual_savings": round(annual_savings, 2),
            "is_savings": delta_cost < 0,
            "key_drivers": drivers
        }
    }

