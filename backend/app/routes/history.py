from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from backend.app.database.session import get_db
from backend.app.models.prediction import PredictionRecord
from backend.app.schemas.prediction import PredictionRecordResponse
from backend.app.services.auth_service import get_current_user_optional, get_current_user
from backend.app.services.pdf_service import generate_prediction_pdf
from backend.app.models.user import User

router = APIRouter(prefix="/history", tags=["Prediction History & Audit"])

@router.get("", response_model=List[PredictionRecordResponse])
def get_prediction_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    risk_tier: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    query = db.query(PredictionRecord)
    
    # If authenticated non-admin, filter by user_id; if admin or unauthenticated, list global records
    if current_user and current_user.role != "admin":
        query = query.filter((PredictionRecord.user_id == current_user.id) | (PredictionRecord.user_id == None))
        
    if risk_tier:
        query = query.filter(PredictionRecord.risk_tier.ilike(f"%{risk_tier}%"))
        
    records = query.order_by(PredictionRecord.created_at.desc()).offset(skip).limit(limit).all()
    return records

@router.get("/export/csv")
def export_prediction_history_csv(
    risk_tier: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Export prediction audit records to CSV format."""
    import csv
    import io

    query = db.query(PredictionRecord)
    if current_user and current_user.role != "admin":
        query = query.filter((PredictionRecord.user_id == current_user.id) | (PredictionRecord.user_id == None))
    if risk_tier:
        query = query.filter(PredictionRecord.risk_tier.ilike(f"%{risk_tier}%"))

    records = query.order_by(PredictionRecord.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Age", "Sex", "BMI", "Children", "Smoker", "Region",
        "Predicted Annual Charges ($)", "Monthly Charges ($)", "Risk Tier",
        "Conformal 90% Lower ($)", "Conformal 90% Upper ($)", "Model Version", "Timestamp"
    ])

    for r in records:
        writer.writerow([
            r.id, r.age, r.sex, r.bmi, r.children, r.smoker, r.region,
            f"{r.predicted_charges:.2f}", f"{r.monthly_charges:.2f}", r.risk_tier,
            f"{r.confidence_lower:.2f}", f"{r.confidence_upper:.2f}",
            r.model_version or "v1.0.0-champion",
            r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=medinsight_prediction_history.csv"}
    )

@router.get("/{record_id}")
def get_prediction_detail(
    record_id: int,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
        
    # Parse stored JSON fields
    shap_data = json.loads(record.shap_contributions) if record.shap_contributions else []
    recs_data = json.loads(record.recommendations) if record.recommendations else []
    
    return {
        "id": record.id,
        "age": record.age,
        "sex": record.sex,
        "bmi": record.bmi,
        "children": record.children,
        "smoker": record.smoker,
        "region": record.region,
        "predicted_charges": record.predicted_charges,
        "monthly_charges": record.monthly_charges,
        "risk_tier": record.risk_tier,
        "confidence_lower": record.confidence_lower,
        "confidence_upper": record.confidence_upper,
        "shap_contributions": shap_data,
        "recommendations": recs_data,
        "created_at": record.created_at
    }

@router.get("/{record_id}/pdf")
def download_prediction_pdf(
    record_id: int,
    db: Session = Depends(get_db)
):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")

    record_dict = {
        "predicted_charges": record.predicted_charges,
        "monthly_charges": record.monthly_charges,
        "risk_tier": record.risk_tier,
        "confidence_lower": record.confidence_lower,
        "confidence_upper": record.confidence_upper,
        "age": record.age,
        "sex": record.sex,
        "bmi": record.bmi,
        "children": record.children,
        "smoker": record.smoker,
        "region": record.region,
        "shap_contributions": record.shap_contributions,
        "recommendations": record.recommendations,
        "created_at": record.created_at.strftime("%Y-%m-%d %H:%M:%S")
    }

    pdf_buffer = generate_prediction_pdf(record_dict)
    
    filename = f"Medical_Insurance_Risk_Report_{record.id}.pdf"
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction_record(
    record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
        
    if current_user.role != "admin" and record.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this record.")
        
    db.delete(record)
    db.commit()
    return None
