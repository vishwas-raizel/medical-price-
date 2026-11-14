from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.database.session import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Input Features
    age = Column(Integer, nullable=False)
    sex = Column(String(20), nullable=False)
    bmi = Column(Float, nullable=False)
    children = Column(Integer, nullable=False, default=0)
    smoker = Column(String(10), nullable=False)
    region = Column(String(50), nullable=False)
    
    # Prediction Outputs
    predicted_charges = Column(Float, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    risk_tier = Column(String(50), nullable=False)  # "Low Risk", "Moderate Risk", "High Risk", "Critical Risk"
    confidence_lower = Column(Float, nullable=False)  # Conformal prediction interval 90% lower bound
    confidence_upper = Column(Float, nullable=False)  # Conformal prediction interval 90% upper bound
    
    # Explainable AI & Advice (stored as JSON string)
    shap_contributions = Column(Text, nullable=True)  # JSON dictionary of {feature: shap_value}
    recommendations = Column(Text, nullable=True)     # JSON array of clinical guidance strings
    model_version = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="predictions")
