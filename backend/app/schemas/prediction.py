from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of primary beneficiary", examples=[35])
    sex: str = Field(..., description="Gender ('male' or 'female')", examples=["male"])
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index", examples=[28.5])
    children: int = Field(0, ge=0, le=10, description="Number of children/dependents covered", examples=[2])
    smoker: str = Field(..., description="Smoking status ('yes' or 'no')", examples=["no"])
    region: str = Field(..., description="Region ('northeast', 'northwest', 'southeast', 'southwest')", examples=["northwest"])
    
    # Optional lifestyle & medical context (clearly distinguished as risk profiling context)
    physical_activity: Optional[str] = Field("moderate", description="Activity level: 'sedentary', 'moderate', 'active'")
    existing_conditions: Optional[List[str]] = Field(default_factory=list, description="Self-reported conditions")
    coverage_type: Optional[str] = Field("comprehensive", description="Plan tier: 'basic', 'standard', 'comprehensive'")

class ShapContribution(BaseModel):
    feature: str
    feature_value: Any
    contribution_amount: float
    impact_direction: str  # "increases_cost" or "decreases_cost"
    percentage_influence: float

class ConformalInterval(BaseModel):
    confidence_level: float = 0.90
    lower_bound: float
    upper_bound: float
    methodology: str = "Empirical Conformal Residual Quantile"

class PredictResponse(BaseModel):
    prediction_id: Optional[int] = None
    estimated_annual_cost: float
    formatted_annual_cost: str
    estimated_monthly_cost: float
    formatted_monthly_cost: str
    risk_tier: str
    risk_badge_color: str
    conformal_interval: ConformalInterval
    base_expected_value: float
    shap_contributions: List[ShapContribution]
    dataset_percentile: float
    clinical_recommendations: List[str]
    model_version: str
    model_algorithm: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PredictionRecordResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    predicted_charges: float
    monthly_charges: float
    risk_tier: str
    confidence_lower: float
    confidence_upper: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
