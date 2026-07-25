import numpy as np
import pandas as pd
import shap
from backend.services.model_manager import model_manager
from backend.schemas import CustomerInput, PredictionResponse, SoftClusterProb, ShapFeatureImpact, SurvivalPoint, WhatIfRequest, WhatIfResponse

def get_risk_tier(prob: float) -> str:
    if prob >= 0.85:
        return "Critical"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.30:
        return "Medium"
    return "Low"

def compute_clv_and_tenure(monthly_charges: float, tenure: int, churn_prob: float):
    """Calculates CLV = MonthlyCharges * Expected_Remaining_Tenure (weighted by churn risk)."""
    base_remaining = max(1, int(60 - tenure))
    expected_remaining = max(3, int(base_remaining * (1 - (churn_prob * 0.7))))
    clv = round(monthly_charges * expected_remaining, 2)
    return clv, expected_remaining

def predict_single_customer(input_data: CustomerInput) -> PredictionResponse:
    model_manager.load_artifacts()
    
    # Preprocess
    input_dict = input_data.model_dump()
    model_name = input_dict.get('selected_model', 'xgboost')
    if model_name not in model_manager.models:
        model_name = 'xgboost'

    df_row = model_manager.preprocess_dict(input_dict)
    model = model_manager.models[model_name]

    # Model inference
    if model_name == "logistic_regression":
        X_scaled = model_manager.scaler.transform(df_row)
        y_prob = model.predict_proba(X_scaled)[0, 1]
    else:
        y_prob = model.predict_proba(df_row)[0, 1]

    churn_prob = round(float(y_prob) * 100, 2)
    no_churn_prob = round(100 - churn_prob, 2)
    is_churn = churn_prob >= 50.0
    risk_tier = get_risk_tier(churn_prob / 100.0)

    # CLV & Remaining Tenure
    clv_val, expected_tenure = compute_clv_and_tenure(
        input_data.MonthlyCharges, input_data.tenure, churn_prob / 100.0
    )

    # Soft Clustering (GMM)
    X_scaled = model_manager.scaler.transform(df_row)
    gmm_probs = model_manager.cluster_gmm.predict_proba(X_scaled)[0]
    
    soft_clusters = []
    for c_id, p_val in enumerate(gmm_probs):
        c_label = model_manager.cluster_labels.get(str(c_id), model_manager.cluster_labels.get(c_id, f"Cluster {c_id}"))
        soft_clusters.append(SoftClusterProb(
            cluster_id=c_id,
            cluster_name=c_label,
            probability=round(float(p_val) * 100, 1)
        ))
    soft_clusters.sort(key=lambda x: x.probability, reverse=True)
    dominant_cluster_label = soft_clusters[0].cluster_name

    # Local SHAP Waterfall
    shap_impacts = []
    try:
        if model_name in ["xgboost", "random_forest", "decision_tree"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(df_row)[0]
        else:
            explainer = shap.LinearExplainer(model, model_manager.scaler.transform(df_row))
            shap_vals = explainer.shap_values(X_scaled)[0]

        for feat, val, s_val in zip(model_manager.feature_names, df_row.iloc[0].values, shap_vals):
            # Show original input value for readability
            orig_val = input_dict.get(feat, val)
            shap_impacts.append(ShapFeatureImpact(
                feature=feat,
                value=orig_val,
                shap_value=round(float(s_val), 4)
            ))
        shap_impacts.sort(key=lambda x: abs(x.shap_value), reverse=True)
        shap_impacts = shap_impacts[:8]
    except Exception as e:
        print(f"Local SHAP notice: {e}")

    # Survival Curve
    survival_pts = []
    try:
        df_surv = pd.DataFrame([{
            'MonthlyCharges': input_data.MonthlyCharges,
            'is_month_to_month': 1 if input_data.Contract == 'Month-to-month' else 0,
            'has_tech_support': 1 if input_data.TechSupport == 'Yes' else 0
        }])
        
        # Estimate customer survival over months 1..72
        base_curve = model_manager.survival_baseline.get('survival_probability', [])
        timeline = model_manager.survival_baseline.get('timeline', list(range(1, 73)))
        
        # Apply hazard multiplier based on risk tier
        hazard_mult = 1.8 if is_churn else 0.7
        for m, p in zip(timeline[::3], base_curve[::3]):
            p_adj = round(max(0.01, min(0.99, p ** hazard_mult)) * 100, 1)
            survival_pts.append(SurvivalPoint(month=m, survival_probability=p_adj))
    except Exception as e:
        print(f"Survival curve notice: {e}")

    return PredictionResponse(
        churn_probability=churn_prob,
        no_churn_probability=no_churn_prob,
        churn_prediction=is_churn,
        risk_tier=risk_tier,
        clv_estimate=clv_val,
        expected_remaining_tenure=expected_tenure,
        model_used=model_name,
        cluster_label=dominant_cluster_label,
        soft_cluster_probabilities=soft_clusters,
        shap_waterfall=shap_impacts,
        survival_curve=survival_pts
    )

def simulate_what_if(req: WhatIfRequest) -> WhatIfResponse:
    baseline_res = predict_single_customer(req.baseline)
    modified_res = predict_single_customer(req.modified)

    delta_prob = round(modified_res.churn_probability - baseline_res.churn_probability, 2)
    delta_clv = round(modified_res.clv_estimate - baseline_res.clv_estimate, 2)

    if delta_prob < -5.0:
        rec = f"Great intervention! Lowered churn risk by {abs(delta_prob)}% and preserved ${delta_clv} in customer value."
    elif delta_prob > 5.0:
        rec = f"Warning: This modification increases churn risk by {delta_prob}%."
    else:
        rec = "Minor impact on churn probability. Consider bundling TechSupport or converting to an Annual Contract."

    # Compare top SHAP changes
    shap_changes = []
    base_shap_map = {item.feature: item.shap_value for item in baseline_res.shap_waterfall}
    for item in modified_res.shap_waterfall:
        b_val = base_shap_map.get(item.feature, 0.0)
        diff = round(item.shap_value - b_val, 4)
        if abs(diff) > 0.01:
            shap_changes.append({
                "feature": item.feature,
                "baseline_value": req.baseline.model_dump().get(item.feature, "N/A"),
                "modified_value": req.modified.model_dump().get(item.feature, "N/A"),
                "shap_impact_change": diff
            })

    return WhatIfResponse(
        baseline_churn_prob=baseline_res.churn_probability,
        modified_churn_prob=modified_res.churn_probability,
        churn_prob_delta=delta_prob,
        baseline_risk_tier=baseline_res.risk_tier,
        modified_risk_tier=modified_res.risk_tier,
        baseline_clv=baseline_res.clv_estimate,
        modified_clv=modified_res.clv_estimate,
        clv_delta=delta_clv,
        shap_changes=shap_changes,
        recommendation=rec
    )
