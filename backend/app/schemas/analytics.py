from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class CostDistributionBucket(BaseModel):
    range_label: str
    count: int
    percentage: float

class DashboardStats(BaseModel):
    total_predictions: int
    avg_predicted_cost: float
    min_predicted_cost: float
    max_predicted_cost: float
    active_model_name: str
    active_model_r2: float
    dataset_total_records: int
    cost_distribution: List[CostDistributionBucket]

class ModelBenchmarkItem(BaseModel):
    algorithm: str
    test_r2: float
    test_rmse: float
    test_mae: float
    cv_r2_mean: float
    adjusted_r2: float
    is_champion: bool = False

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    direction: str
    description: str

class AnalyticsOverview(BaseModel):
    stats: DashboardStats
    benchmarks: List[ModelBenchmarkItem]
    feature_importance: List[FeatureImportanceItem]
