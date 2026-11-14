import os
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from backend.app.database.session import get_db
from backend.app.models.user import User
from backend.app.models.prediction import PredictionRecord
from backend.app.models.audit import AuditLog
from backend.app.services.auth_service import get_current_admin
from backend.app.config import settings

router = APIRouter(prefix="/admin", tags=["Administrative Control & Monitoring"])

@router.get("/system-stats")
def get_system_stats(db: Session = Depends(get_db)):
    user_count = db.query(func.count(User.id)).scalar() or 0
    pred_count = db.query(func.count(PredictionRecord.id)).scalar() or 0
    audit_count = db.query(func.count(AuditLog.id)).scalar() or 0

    return {
        "status": "operational",
        "total_registered_users": user_count,
        "total_persisted_predictions": pred_count,
        "total_audit_events": audit_count,
        "database_engine": "SQLite / SQLAlchemy ORM",
        "api_framework": "FastAPI Async ASGI"
    }

@router.get("/dataset")
def get_dataset_metadata():
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, "champion_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
            return data.get("dataset_metadata", {})
    return {
        "name": "Medical Cost Personal Dataset",
        "source": "Kaggle Benchmark",
        "total_records": 1338,
        "total_features": 6
    }

@router.get("/audit-logs")
def get_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "user_id": log.user_id,
            "action": log.action,
            "resource": log.resource,
            "ip_address": log.ip_address,
            "details": json.loads(log.details) if log.details else None,
            "timestamp": log.timestamp
        }
        for log in logs
    ]

@router.get("/models")
def get_registered_models():
    metrics_path = os.path.join(settings.SAVED_MODELS_DIR, "model_metrics.json")
    meta_path = os.path.join(settings.SAVED_MODELS_DIR, "champion_metadata.json")
    
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return {
        "active_champion": meta.get("champion_name", "Tuned Linear Regression"),
        "features": meta.get("features", []),
        "conformal_quantile_90": meta.get("conformal_quantile_90", 3347.92),
        "available_models": list(metrics.keys())
    }

class UserRoleUpdate(BaseModel):
    role: str

class UserStatusUpdate(BaseModel):
    is_active: bool

@router.get("/users")
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.id.asc()).all()
    results = []
    for u in users:
        pred_count = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.user_id == u.id).scalar() or 0
        results.append({
            "id": u.id,
            "email": u.email,
            "full_name": u.full_name or "N/A",
            "role": u.role,
            "is_active": u.is_active,
            "total_predictions": pred_count,
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    return results

@router.patch("/users/{user_id}/role")
def update_user_role(user_id: int, update: UserRoleUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = update.role.lower().strip()
    db.commit()
    return {"message": f"User {user.email} role updated to {user.role}"}

@router.patch("/users/{user_id}/status")
def update_user_status(user_id: int, update: UserStatusUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = update.is_active
    db.commit()
    return {"message": f"User {user.email} status updated to {'Active' if user.is_active else 'Deactivated'}"}

