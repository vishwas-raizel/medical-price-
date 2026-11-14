from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.app.database.session import Base

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False, unique=True)
    algorithm = Column(String(100), nullable=False)
    hyperparameters = Column(Text, nullable=True)  # JSON
    filepath = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    train_samples = Column(Integer, nullable=False)
    test_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    evaluations = relationship("ModelEvaluation", back_populates="model_version", cascade="all, delete-orphan")

class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    metric_mae = Column(Float, nullable=False)
    metric_mse = Column(Float, nullable=False)
    metric_rmse = Column(Float, nullable=False)
    metric_r2 = Column(Float, nullable=False)
    metric_adjusted_r2 = Column(Float, nullable=False)
    metric_cv_r2 = Column(Float, nullable=False)
    evaluation_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    model_version = relationship("ModelVersion", back_populates="evaluations")

class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    source = Column(String(255), nullable=False)
    total_records = Column(Integer, nullable=False)
    total_features = Column(Integer, nullable=False)
    sha256_hash = Column(String(64), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
