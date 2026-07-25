from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class CustomerInput(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 85.5
    TotalCharges: float = 1026.0
    selected_model: str = "xgboost"

class SoftClusterProb(BaseModel):
    cluster_id: int
    cluster_name: str
    probability: float

class ShapFeatureImpact(BaseModel):
    feature: str
    value: Any
    shap_value: float

class SurvivalPoint(BaseModel):
    month: int
    survival_probability: float

class PredictionResponse(BaseModel):
    churn_probability: float
    no_churn_probability: float
    churn_prediction: bool
    risk_tier: str # Low, Medium, High, Critical
    clv_estimate: float
    expected_remaining_tenure: int
    model_used: str
    cluster_label: str
    soft_cluster_probabilities: List[SoftClusterProb]
    shap_waterfall: List[ShapFeatureImpact]
    survival_curve: List[SurvivalPoint]

class WhatIfRequest(BaseModel):
    baseline: CustomerInput
    modified: CustomerInput
    selected_model: str = "xgboost"

class WhatIfResponse(BaseModel):
    baseline_churn_prob: float
    modified_churn_prob: float
    churn_prob_delta: float
    baseline_risk_tier: str
    modified_risk_tier: str
    baseline_clv: float
    modified_clv: float
    clv_delta: float
    shap_changes: List[Dict[str, Any]]
    recommendation: str

class AiStrategyRequest(BaseModel):
    gender: str = "Female"
    tenure: int = 12
    Contract: str = "Month-to-month"
    MonthlyCharges: float = 85.5
    churn_probability: float = 78.5
    risk_tier: str = "High"
    cluster_label: str = "High-Risk Price-Sensitive"
    top_shap_drivers: List[str] = ["Contract", "MonthlyCharges", "TechSupport"]
    clv_estimate: float = 1500.0
    custom_notes: Optional[str] = None
    gemini_api_key: Optional[str] = None

class AiStrategyResponse(BaseModel):
    strategy_title: str
    executive_summary: str
    retention_offer: str
    discount_coupon: str
    estimated_cost: float
    expected_risk_reduction: float
    estimated_roi: float
    strategy_markdown: str

class DriftCheckResponse(BaseModel):
    total_records: int
    overall_psi: float
    drift_status: str # Low Drift, Moderate Drift, High Drift
    feature_psi_scores: Dict[str, float]
