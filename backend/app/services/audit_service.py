import json
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.audit import AuditLog

def log_audit_event(
    db: Session,
    action: str,
    resource: str,
    user_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    try:
        log_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            details=json.dumps(details) if details else None
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Failed to record audit log: {e}")
