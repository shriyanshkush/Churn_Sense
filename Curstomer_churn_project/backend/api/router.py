import os
import io
import json
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import List, Dict, Any

from backend.schemas import (
    CustomerInput, PredictionResponse, WhatIfRequest, WhatIfResponse,
    AiStrategyRequest, AiStrategyResponse, DriftCheckResponse
)
from backend.services.model_manager import model_manager
from backend.services.prediction_service import predict_single_customer, simulate_what_if, get_risk_tier
from backend.services.drift_service import check_data_drift
from backend.services.gemini_service import generate_ai_retention_strategy

router = APIRouter(prefix="/api/v1", tags=["ChurnSense API"])

@router.post("/predict", response_model=PredictionResponse)
def api_predict_customer(customer: CustomerInput):
    """Predict churn probability, risk tier, CLV, soft cluster membership, SHAP waterfall, and survival curve for a customer."""
    try:
        return predict_single_customer(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/what-if", response_model=WhatIfResponse)
def api_what_if_simulation(req: WhatIfRequest):
    """Simulate real-time impact of changing customer attributes on churn probability, CLV, and SHAP drivers."""
    try:
        return simulate_what_if(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-If simulation error: {str(e)}")

@router.post("/batch-predict")
async def api_batch_predict(file: UploadFile = File(...), selected_model: str = "xgboost"):
    """Batch predict churn probabilities, risk tiers, and cluster assignments for uploaded CSV file."""
    try:
        content = await file.read()
        df_uploaded = pd.read_csv(io.BytesIO(content))
        
        model_manager.load_artifacts()
        
        results = []
        high_risk_count = 0
        total_clv = 0.0

        for idx, row in df_uploaded.iterrows():
            row_dict = row.to_dict()
            # Default missing fields if not in CSV
            c_input = CustomerInput(
                gender=str(row_dict.get('gender', row_dict.get('Gender', 'Male'))),
                SeniorCitizen=int(row_dict.get('SeniorCitizen', row_dict.get('Age', 30) >= 60)),
                tenure=int(row_dict.get('tenure', row_dict.get('Tenure', 12))),
                MonthlyCharges=float(row_dict.get('MonthlyCharges', row_dict.get('Total Spend', 1000) / max(1, row_dict.get('Tenure', 12)))),
                TotalCharges=float(row_dict.get('TotalCharges', row_dict.get('Total Spend', 500))),
                Contract=str(row_dict.get('Contract', row_dict.get('Contract Length', 'Month-to-month'))),
                selected_model=selected_model
            )
            
            res = predict_single_customer(c_input)
            
            res_dict = {
                "id": str(row_dict.get('customerID', row_dict.get('CustomerID', f"ROW-{idx+1}"))),
                "gender": c_input.gender,
                "tenure": c_input.tenure,
                "contract": c_input.Contract,
                "monthly_charges": c_input.MonthlyCharges,
                "total_charges": c_input.TotalCharges,
                "churn_probability": res.churn_probability,
                "risk_tier": res.risk_tier,
                "clv_estimate": res.clv_estimate,
                "cluster_label": res.cluster_label,
                "top_risk_driver": res.shap_waterfall[0].feature if res.shap_waterfall else "Contract"
            }
            
            if res.churn_probability >= 60.0:
                high_risk_count += 1
            total_clv += res.clv_estimate

            results.append(res_dict)

        avg_clv = round(total_clv / max(1, len(results)), 2)
        overall_churn_rate = round((high_risk_count / max(1, len(results))) * 100, 1)

        return {
            "filename": file.filename,
            "total_rows": len(results),
            "high_risk_customers": high_risk_count,
            "overall_churn_rate": overall_churn_rate,
            "average_clv": avg_clv,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.get("/clusters")
def api_get_clusters():
    """Retrieve 2D PCA cluster coordinates, GMM soft clustering metrics, and cluster summaries."""
    try:
        model_manager.load_artifacts()
        
        # PCA Points sample
        pca = model_manager.pca_2d
        gmm = model_manager.cluster_gmm
        
        # Load precomputed PCA sample points or generate from baseline
        sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_churn_data.csv")
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path).head(300)
            X_rows = []
            for _, r in df_sample.iterrows():
                c_in = CustomerInput(**{k: r[k] for k in r.index if k in CustomerInput.model_fields})
                df_encoded = model_manager.preprocess_dict(c_in.model_dump())
                X_rows.append(df_encoded.iloc[0].values)
            
            X_arr = pd.DataFrame(X_rows, columns=model_manager.feature_names)
            X_scaled = model_manager.scaler.transform(X_arr)
            pca_pts = pca.transform(X_scaled)
            gmm_labels = gmm.predict(X_scaled)
            
            xgb = model_manager.models['xgboost']
            probs = xgb.predict_proba(X_arr)[:, 1]

            scatter_points = []
            for idx in range(len(pca_pts)):
                c_id = int(gmm_labels[idx])
                scatter_points.append({
                    "x": round(float(pca_pts[idx, 0]), 3),
                    "y": round(float(pca_pts[idx, 1]), 3),
                    "cluster_id": c_id,
                    "cluster_label": model_manager.cluster_labels.get(str(c_id), f"Cluster {c_id}"),
                    "churn_probability": round(float(probs[idx]) * 100, 1),
                    "monthly_charges": float(df_sample.iloc[idx].get('MonthlyCharges', 60.0)),
                    "tenure": int(df_sample.iloc[idx].get('tenure', 12))
                })
        else:
            scatter_points = []

        return {
            "cluster_labels": model_manager.cluster_labels,
            "elbow_metrics": model_manager.elbow_metrics,
            "pca_scatter": scatter_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clusters endpoint error: {str(e)}")

@router.get("/model-insights")
def api_get_model_insights():
    """Retrieve accuracy, precision, recall, F1, ROC-AUC metrics, confusion matrices, and global SHAP values."""
    try:
        model_manager.load_artifacts()
        return {
            "models_metrics": model_manager.metrics,
            "global_shap_importance": model_manager.shap_global
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model insights error: {str(e)}")

@router.post("/drift-check", response_model=DriftCheckResponse)
async def api_drift_check(file: UploadFile = File(...)):
    """Compute Population Stability Index (PSI) to detect data drift between batch data and training distribution."""
    try:
        content = await file.read()
        df_uploaded = pd.read_csv(io.BytesIO(content))
        return check_data_drift(df_uploaded)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check error: {str(e)}")

@router.get("/sample-csv")
def api_download_sample_csv():
    """Download pre-built demo customer churn CSV for batch upload testing."""
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_churn_data.csv")
    if os.path.exists(sample_path):
        return FileResponse(sample_path, media_type="text/csv", filename="ChurnSense_Sample_Dataset.csv")
    raise HTTPException(status_code=404, detail="Sample CSV file not found.")

@router.post("/ai-strategy", response_model=AiStrategyResponse)
def api_ai_retention_strategy(req: AiStrategyRequest):
    """Generate segment-aware Gemini AI retention strategy and promotional coupon code."""
    try:
        return generate_ai_retention_strategy(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Strategy error: {str(e)}")
