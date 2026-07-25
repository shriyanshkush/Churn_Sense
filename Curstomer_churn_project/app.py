import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import json
import requests
import google.generativeai as genai

st.set_page_config(
    page_title="ChurnSense Quick Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/api/v1"

# Styling
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stMetric { background-color: #1e293b; padding: 12px; border-radius: 12px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ ChurnSense - AI Customer Churn & Segmentation Platform")
st.caption("Streamlit Quick Demo Mode &bull; Connected to FastAPI Enterprise Backend")
st.markdown("---")

# Sidebar
st.sidebar.header("🔍 Customer Profile Inputs")

selected_model = st.sidebar.selectbox(
    "Select ML Algorithm",
    ["xgboost", "random_forest", "decision_tree", "logistic_regression"],
    format_func=lambda x: {
        "xgboost": "XGBoost Classifier (Best AUC)",
        "random_forest": "Random Forest Ensemble",
        "decision_tree": "Decision Tree Baseline",
        "logistic_regression": "Logistic Regression Baseline"
    }[x]
)

gender = st.sidebar.radio("Gender", ["Male", "Female"], index=1)
senior_citizen = st.sidebar.radio("Senior Citizen", [0, 1])
partner = st.sidebar.radio("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.radio("Has Dependents?", ["Yes", "No"])
tenure = st.sidebar.slider("📆 Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("💵 Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=0.5)
total_charges = round(monthly_charges * tenure, 2)

phone_service = st.sidebar.selectbox("📞 Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("🌐 Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.radio("🔐 Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.radio("💾 Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.radio("🛡 Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.radio("🛠 Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.radio("📺 Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.radio("🎬 Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.radio("📝 Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

customer_payload = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "selected_model": selected_model
}

# Main Layout
col1, col2 = st.columns([7, 5])

with col1:
    st.subheader("📊 Live Prediction Results")
    if st.button("🔮 Predict Churn & Generate Analysis", type="primary", use_container_width=True):
        try:
            res = requests.post(f"{API_URL}/predict", json=customer_payload)
            if res.status_code == 200:
                data = res.json()
                st.session_state['pred_data'] = data
            else:
                st.error("FastAPI backend error. Run `python ml_pipeline/train_pipeline.py` and `python backend/main.py` first.")
        except Exception as e:
            st.warning(f"Backend offline. (Start FastAPI server via `uvicorn backend.main:app`). Details: {e}")

    if 'pred_data' in st.session_state:
        d = st.session_state['pred_data']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Churn Risk", f"{d['churn_probability']}%", f"{d['risk_tier']} Tier")
        m2.metric("Estimated CLV", f"${d['clv_estimate']:,.2f}", f"{d['expected_remaining_tenure']} mos remaining")
        m3.metric("Actionable GMM Cluster", d['cluster_label'])

        st.progress(int(d['churn_probability']))

        st.markdown("---")
        st.subheader("🔍 SHAP Local Feature Driver Waterfall")
        shap_df = pd.DataFrame(d['shap_waterfall'])
        if not shap_df.empty:
            st.bar_chart(data=shap_df, x='feature', y='shap_value', color='#6366f1')

with col2:
    st.subheader("🧠 Gemini Segment-Aware AI Strategy")
    st.info(f"Connected Segment: **{st.session_state.get('pred_data', {}).get('cluster_label', 'High-Risk Price-Sensitive')}**")

    user_notes = st.text_area("Additional Representative Notes", "Customer complained about bill increase.")
    if st.button("✨ Generate AI Retention Strategy & Coupon", use_container_width=True):
        try:
            ai_req = {
                "gender": gender,
                "tenure": tenure,
                "Contract": contract,
                "MonthlyCharges": monthly_charges,
                "churn_probability": st.session_state.get('pred_data', {}).get('churn_probability', 75.0),
                "risk_tier": st.session_state.get('pred_data', {}).get('risk_tier', 'High'),
                "cluster_label": st.session_state.get('pred_data', {}).get('cluster_label', 'High-Risk Price-Sensitive'),
                "top_shap_drivers": ["Contract", "MonthlyCharges", "TechSupport"],
                "clv_estimate": st.session_state.get('pred_data', {}).get('clv_estimate', 1500.0),
                "custom_notes": user_notes
            }
            ai_res = requests.post(f"{API_URL}/ai-strategy", json=ai_req)
            if ai_res.status_code == 200:
                ai_data = ai_res.json()
                st.success(f"**{ai_data['strategy_title']}**")
                st.markdown(ai_data['strategy_markdown'])
                st.code(f"PROMO CODE: {ai_data['discount_coupon']}")
        except Exception as e:
            st.error(f"Error calling AI strategy: {e}")

st.markdown("---")
st.caption("ChurnSense AI Enterprise Platform &bull; React SPA App available at `http://localhost:3000`")
