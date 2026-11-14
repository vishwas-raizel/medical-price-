from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.database.session import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False)  # "LOGIN", "PREDICTION", "MODEL_TRAIN", "EXPORT_PDF"
    resource = Column(String(100), nullable=False)
    ip_address = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)  # JSON details
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="audit_logs")
