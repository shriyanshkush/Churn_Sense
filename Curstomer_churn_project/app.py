import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Dark Mode / Glassmorphism Design System)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Reset & Styling */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Cards & Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Action Cards for Landing Page */
    .option-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .option-card:hover {
        transform: translateY(-6px);
        border-color: #6366f1;
        box-shadow: 0 20px 35px -10px rgba(99, 102, 241, 0.3);
    }
    .option-icon {
        font-size: 3.5rem;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .option-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .option-desc {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 24px;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-churn {
        background: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }
    .badge-notchurn {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-recommend {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    /* Model Performance Metrics Table */
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .perf-table th {
        background-color: #1e293b;
        color: #94a3b8;
        font-weight: 600;
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid #334155;
    }
    .perf-table td {
        padding: 12px;
        border-bottom: 1px solid #334155;
        color: #f1f5f9;
    }
    .perf-table tr:hover {
        background-color: rgba(99, 102, 241, 0.05);
    }

    /* Confusion Matrix Box */
    .cm-box {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        background: #0f172a;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }
    .cm-cell {
        padding: 12px;
        border-radius: 8px;
    }
    .cm-tn { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .cm-fp { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .cm-fn { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .cm-tp { background: rgba(99, 102, 241, 0.15); color: #818cf8; }

    /* Custom Header Subtitle */
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Models & Preprocessing Encoders
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    model_configs = {
        "Decision Tree": ["decisiontree_churn_model.pkl", "decision_tree_churn_model.pkl"],
        "Random Forest": ["randomforest_churn_model.pkl", "random_forest_churn_model.pkl"],
        "XGBoost": ["xgboost_churn_model.pkl"],
        "Logistic Regression": ["logisticregression_churn_model.pkl", "logistic_regression_churn_model.pkl"]
    }
    
    loaded_models = {}
    for name, paths in model_configs.items():
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict) and "model" in data:
                            loaded_models[name] = data["model"]
                        else:
                            loaded_models[name] = data
                    break
                except Exception as e:
                    st.error(f"Error loading {path}: {e}")

    encoders = {}
    if os.path.exists("encoders.pkl"):
        try:
            with open("encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading encoders.pkl: {e}")

    feature_names = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    return loaded_models, encoders, feature_names

models, encoders, FEATURE_NAMES = load_all_models()

# ---------------------------------------------------------
# Precomputed Model Info & Benchmark Metrics (Exact Specs)
# ---------------------------------------------------------
MODEL_METRICS = {
    "Decision Tree": {
        "accuracy": "74%",
        "precision": "51.02%",
        "recall": "60.59%",
        "f1": "55.39%",
        "recommendation": None,
        "cm": {"TN": 810, "FP": 223, "FN": 147, "TP": 229},
        "report": """              precision    recall  f1-score   support

        No       0.85      0.78      0.81      1033
       Yes       0.51      0.61      0.55       376

  accuracy                           0.74      1409
 macro avg       0.68      0.70      0.68      1409
weighted avg       0.76      0.74      0.74      1409"""
    },
    "Random Forest": {
        "accuracy": "78%",
        "precision": "56.80%",
        "recall": "62.73%",
        "f1": "59.62%",
        "recommendation": None,
        "cm": {"TN": 854, "FP": 179, "FN": 140, "TP": 236},
        "report": """              precision    recall  f1-score   support

        No       0.86      0.83      0.84      1033
       Yes       0.57      0.63      0.60       376

  accuracy                           0.78      1409
 macro avg       0.71      0.73      0.72      1409
weighted avg       0.78      0.78      0.78      1409"""
    },
    "XGBoost": {
        "accuracy": "79%",
        "precision": "59.52%",
        "recall": "59.52%",
        "f1": "59.52%",
        "recommendation": "⭐ Recommended for Highest Accuracy (79%) & Precision (59.52%)",
        "cm": {"TN": 881, "FP": 152, "FN": 152, "TP": 224},
        "report": """              precision    recall  f1-score   support

        No       0.85      0.85      0.85      1033
       Yes       0.60      0.60      0.60       376

  accuracy                           0.79      1409
 macro avg       0.72      0.72      0.72      1409
weighted avg       0.79      0.79      0.79      1409"""
    },
    "Logistic Regression": {
        "accuracy": "77%",
        "precision": "54.16%",
        "recall": "78.55%",
        "f1": "64.11%",
        "recommendation": "🏆 Best Performing (Highest Recall: 78.55% & Best F1 Score: 64.11%)",
        "cm": {"TN": 782, "FP": 251, "FN": 81, "TP": 295},
        "report": """              precision    recall  f1-score   support

        No       0.91      0.76      0.83      1033
       Yes       0.54      0.79      0.64       376

  accuracy                           0.77      1409
 macro avg       0.72      0.77      0.73      1409
weighted avg       0.81      0.77      0.78      1409"""
    }
}

# ---------------------------------------------------------
# Preprocessing Helper Function
# ---------------------------------------------------------
def preprocess_features(df_raw, encoders, feature_names):
    df = df_raw.copy()
    
    # Numeric conversions
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0).astype(float)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0).astype(float)

    # Label Encoding for categorical columns
    for col, enc in encoders.items():
        if col in df.columns:
            classes = list(enc.classes_)
            df[col] = df[col].astype(str).map(
                lambda val: enc.transform([val])[0] if val in classes else enc.transform([classes[0]])[0]
            )

    return df[feature_names]

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if 'mode' not in st.session_state:
    st.session_state['mode'] = None

if 'selected_model_name' not in st.session_state:
    st.session_state['selected_model_name'] = "Logistic Regression"

# ---------------------------------------------------------
# Mode Selection Callback Functions
# ---------------------------------------------------------
def set_mode(mode_name):
    st.session_state['mode'] = mode_name

def reset_mode():
    st.session_state['mode'] = None

# =========================================================
# LANDING PAGE (mode is None)
# =========================================================
if st.session_state['mode'] is None:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0 30px 0;">
            <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
                🔮 Customer Churn Prediction Platform
            </h1>
            <p style="font-size: 1.2rem; color: #94a3b8; max-width: 700px; margin: 0 auto;">
                Enterprise Machine Learning Analytics & Predictive Intelligence. Select your operation mode to begin.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2 Landing Page Option Cards
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div class="option-card">
                <div class="option-icon">👤</div>
                <div class="option-title">Single Customer Prediction</div>
                <div class="option-desc">
                    Evaluate individual customer risk profiles interactively. Enter demographic, service, and contract features to generate real-time churn predictions and model performance breakdowns.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Single Customer Prediction", use_container_width=True, type="primary", key="btn_single"):
            set_mode("Single Customer Prediction")
            st.rerun()

    with col2:
        st.markdown("""
            <div class="option-card">
                <div class="option-icon">📊</div>
                <div class="option-title">Batch Prediction</div>
                <div class="option-desc">
                    Upload batch CSV files for multi-customer inference. Analyze bulk churn predictions, execute K-Means customer segmentation clustering, and download exported prediction reports.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("📁 Start Batch Prediction", use_container_width=True, type="primary", key="btn_batch"):
            set_mode("Batch Prediction")
            st.rerun()

    st.markdown("""
        <div style="margin-top: 60px; text-align: center; color: #64748b; font-size: 0.85rem;">
            🔒 All inference is performed locally using pre-trained machine learning models. No data is stored or logged.
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# SINGLE CUSTOMER PREDICTION MODE
# =========================================================
elif st.session_state['mode'] == "Single Customer Prediction":
    # Top Navigation Bar
    n_col1, n_col2 = st.columns([8, 2])
    with n_col1:
        st.title("👤 Single Customer Churn Prediction")
        st.caption("Real-Time Churn Probability Scoring & Feature Analysis")
    with n_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Switch Mode", type="secondary", use_container_width=True):
            reset_mode()
            st.rerun()

    st.markdown("---")

    # Sidebar Model Selection
    st.sidebar.header("⚙️ Model Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Machine Learning Model",
        options=["Logistic Regression", "XGBoost", "Random Forest", "Decision Tree"],
        index=0,
        help="Logistic Regression and XGBoost are recommended for optimal recall and accuracy."
    )
    st.session_state['selected_model_name'] = selected_model_name

    # Display Recommendation Alert in Sidebar
    rec = MODEL_METRICS[selected_model_name]["recommendation"]
    if rec:
        st.sidebar.info(rec)

    # 3 Specific Tabs ONLY (No K-Means tab)
    tab_form, tab_result, tab_model_info = st.tabs([
        "📋 Input Form",
        "🎯 Prediction Result",
        "📈 Model Info / Performance"
    ])

    # --- TAB 1: Input Form ---
    with tab_form:
        st.subheader("📝 Enter Customer Characteristics")
        st.write("Fill out the demographic, service, and account billing features below:")

        with st.form("single_pred_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographic & Account")
                gender = st.selectbox("gender", ["Male", "Female"])
                senior_citizen = st.selectbox("SeniorCitizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12, step=1)
                paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])

            with col2:
                st.markdown("##### 📞 Phone & Internet Services")
                phone_service = st.selectbox("PhoneService", ["Yes", "No"])
                multiple_lines = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])

            with col3:
                st.markdown("##### 💳 Subscriptions & Charges")
                streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("PaymentMethod", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges = st.number_input("MonthlyCharges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
                total_charges = st.number_input("TotalCharges ($)", min_value=0.0, max_value=10000.0, value=round(monthly_charges * tenure, 2), step=1.0)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

        if submit_btn:
            input_dict = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            df_input = pd.DataFrame([input_dict])
            df_proc = preprocess_features(df_input, encoders, FEATURE_NAMES)

            selected_model = models.get(selected_model_name)
            if selected_model:
                pred_class = selected_model.predict(df_proc)[0]
                if hasattr(selected_model, "predict_proba"):
                    prob = selected_model.predict_proba(df_proc)[0][1]
                else:
                    prob = 1.0 if pred_class == 1 else 0.0

                st.session_state['last_single_result'] = {
                    'class': "Churn" if pred_class == 1 else "No Churn",
                    'prob': round(float(prob) * 100, 2),
                    'model_used': selected_model_name,
                    'input_summary': input_dict
                }
                st.success("✅ Prediction generated! Switch to the 'Prediction Result' tab to view details.")
            else:
                st.error(f"Model {selected_model_name} could not be loaded.")

    # --- TAB 2: Prediction Result ---
    with tab_result:
        st.subheader("🎯 Prediction Output & Analysis")
        if 'last_single_result' in st.session_state:
            res = st.session_state['last_single_result']
            
            res_col1, res_col2 = st.columns([5, 7])

            with res_col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"#### Selected Model: **{res['model_used']}**")
                
                if res['class'] == "Churn":
                    st.markdown("""
                        <div style="text-align: center; padding: 20px;">
                            <span class="metric-badge badge-churn" style="font-size: 1.2rem; padding: 8px 20px;">⚠️ CHURN RISK HIGH</span>
                            <h2 style="color: #f43f5e; font-size: 3rem; margin: 15px 0;">{prob}%</h2>
                            <p style="color: #94a3b8;">Predicted Class: <strong style="color: #f43f5e;">CHURN</strong></p>
                        </div>
                    """.format(prob=res['prob']), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="text-align: center; padding: 20px;">
                            <span class="metric-badge badge-notchurn" style="font-size: 1.2rem; padding: 8px 20px;">✅ LOW CHURN RISK</span>
                            <h2 style="color: #10b981; font-size: 3rem; margin: 15px 0;">{prob}%</h2>
                            <p style="color: #94a3b8;">Predicted Class: <strong style="color: #10b981;">NO CHURN</strong></p>
                        </div>
                    """.format(prob=res['prob']), unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Confidence Score Gauge")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = res['prob'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability (%)", 'font': {'color': "#ffffff"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#6366f1"},
                        'bgcolor': "#0f172a",
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.3)'},
                            {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.3)'},
                            {'range': [65, 100], 'color': 'rgba(244, 63, 94, 0.3)'}
                        ],
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=230, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Submitted Profile Summary")
            st.json(res['input_summary'])
        else:
            st.info("ℹ️ No prediction generated yet. Please complete the form in the **Input Form** tab and click 'Predict Churn Risk'.")

    # --- TAB 3: Model Info / Performance ---
    with tab_model_info:
        st.subheader("📈 Pre-Computed Model Evaluation Metrics")
        st.write("Stored benchmark evaluation metrics across all four pretrained models:")

        # Summary Table
        metrics_data = []
        for m_name, m_info in MODEL_METRICS.items():
            metrics_data.append({
                "Model": m_name,
                "Accuracy": m_info["accuracy"],
                "Precision": m_info["precision"],
                "Recall": m_info["recall"],
                "F1 Score": m_info["f1"],
                "Recommendation": m_info["recommendation"] or "-"
            })

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Detailed Model Reports & Confusion Matrices")

        for m_name, m_info in MODEL_METRICS.items():
            with st.expander(f"📌 {m_name} Detailed Report & Confusion Matrix"):
                col_a, col_b = st.columns([5, 7])

                with col_a:
                    st.markdown("##### Confusion Matrix")
                    cm = m_info["cm"]
                    st.markdown(f"""
                        <div class="cm-box">
                            <div class="cm-cell cm-tn">
                                <strong>True Negative (TN)</strong><br>{cm['TN']}
                            </div>
                            <div class="cm-cell cm-fp">
                                <strong>False Positive (FP)</strong><br>{cm['FP']}
                            </div>
                            <div class="cm-cell cm-fn">
                                <strong>False Negative (FN)</strong><br>{cm['FN']}
                            </div>
                            <div class="cm-cell cm-tp">
                                <strong>True Positive (TP)</strong><br>{cm['TP']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown("##### Classification Report")
                    st.code(m_info["report"])

# =========================================================
# BATCH PREDICTION MODE
# =========================================================
elif st.session_state['mode'] == "Batch Prediction":
    # Top Navigation Bar
    n_col1, n_col2 = st.columns([8, 2])
    with n_col1:
        st.title("📊 Batch Customer Churn Prediction")
        st.caption("Bulk Dataset Inference, K-Means Customer Clustering & Export")
    with n_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Switch Mode", type="secondary", use_container_width=True):
            reset_mode()
            st.rerun()

    st.markdown("---")

    # Sidebar Model Selection
    st.sidebar.header("⚙️ Model Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Machine Learning Model",
        options=["Logistic Regression", "XGBoost", "Random Forest", "Decision Tree"],
        index=0
    )
    st.session_state['selected_model_name'] = selected_model_name

    rec = MODEL_METRICS[selected_model_name]["recommendation"]
    if rec:
        st.sidebar.info(rec)

    # ALL 5 Tabs for Batch Mode
    tab_upload, tab_batch_results, tab_model_comp, tab_kmeans, tab_download = st.tabs([
        "📁 CSV Upload",
        "📋 Batch Prediction Results",
        "📈 Model Comparison / Performance",
        "🧩 K-Means Clustering",
        "📥 Download Results"
    ])

    # --- TAB 1: CSV Upload ---
    with tab_upload:
        st.subheader("📤 Upload Customer Batch CSV File")
        st.write("Upload a `.csv` file containing customer records matching the 19 required feature columns:")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_batch_df = pd.read_csv(uploaded_file)
                st.session_state['raw_batch_df'] = raw_batch_df

                st.success(f"✅ CSV Uploaded Successfully! ({len(raw_batch_df)} rows loaded)")
                
                # Check for missing required columns
                missing_cols = [c for c in FEATURE_NAMES if c not in raw_batch_df.columns]
                if missing_cols:
                    st.warning(f"⚠️ Warning: The CSV is missing {len(missing_cols)} columns: {missing_cols}. Missing values will be auto-filled during preprocessing.")
                else:
                    st.info("✅ All 19 required feature columns present!")

                st.markdown("##### Uploaded Data Preview (First 5 Rows)")
                st.dataframe(raw_batch_df.head(), use_container_width=True)

                if st.button("⚡ Run Batch Predictions Now", type="primary", use_container_width=True):
                    df_proc = preprocess_features(raw_batch_df, encoders, FEATURE_NAMES)
                    selected_model = models.get(selected_model_name)

                    if selected_model:
                        preds = selected_model.predict(df_proc)
                        if hasattr(selected_model, "predict_proba"):
                            probs = selected_model.predict_proba(df_proc)[:, 1]
                        else:
                            probs = np.where(preds == 1, 1.0, 0.0)

                        result_batch_df = raw_batch_df.copy()
                        result_batch_df['Churn_Prediction'] = ["Churn" if p == 1 else "No Churn" for p in preds]
                        result_batch_df['Churn_Probability_Percent'] = np.round(probs * 100, 2)

                        st.session_state['batch_result_df'] = result_batch_df
                        st.session_state['batch_processed'] = True
                        st.success("🎉 Batch predictions completed! View results in the 'Batch Prediction Results' tab.")
                    else:
                        st.error(f"Model {selected_model_name} unavailable.")

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
        else:
            st.info("ℹ️ Upload a CSV file above to proceed with batch processing.")

    # --- TAB 2: Batch Prediction Results ---
    with tab_batch_results:
        st.subheader("📋 Batch Prediction Results Table")
        if st.session_state.get('batch_processed', False) and 'batch_result_df' in st.session_state:
            res_df = st.session_state['batch_result_df']

            total_cust = len(res_df)
            churn_cust = (res_df['Churn_Prediction'] == 'Churn').sum()
            churn_rate = round((churn_cust / total_cust) * 100, 2) if total_cust > 0 else 0

            # High-level Metrics Summary
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Customers Evaluated", f"{total_cust:,}")
            m2.metric("Predicted Churners", f"{churn_cust:,}", delta=f"{churn_rate}% Rate", delta_color="inverse")
            m3.metric("Model Selected", selected_model_name)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(res_df, use_container_width=True)
        else:
            st.info("ℹ️ No batch results available yet. Please upload a CSV and run predictions in the **CSV Upload** tab.")

    # --- TAB 3: Model Comparison / Performance ---
    with tab_model_comp:
        st.subheader("📈 Pre-Computed Model Comparison & Info")
        st.write("Stored evaluation metrics for all four pretrained models:")

        metrics_data = []
        for m_name, m_info in MODEL_METRICS.items():
            metrics_data.append({
                "Model": m_name,
                "Accuracy": m_info["accuracy"],
                "Precision": m_info["precision"],
                "Recall": m_info["recall"],
                "F1 Score": m_info["f1"],
                "Recommendation": m_info["recommendation"] or "-"
            })

        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Breakdown & Confusion Matrices")

        for m_name, m_info in MODEL_METRICS.items():
            with st.expander(f"📌 {m_name} Confusion Matrix & Classification Report"):
                col_a, col_b = st.columns([5, 7])

                with col_a:
                    st.markdown("##### Confusion Matrix")
                    cm = m_info["cm"]
                    st.markdown(f"""
                        <div class="cm-box">
                            <div class="cm-cell cm-tn"><strong>True Negative (TN)</strong><br>{cm['TN']}</div>
                            <div class="cm-cell cm-fp"><strong>False Positive (FP)</strong><br>{cm['FP']}</div>
                            <div class="cm-cell cm-fn"><strong>False Negative (FN)</strong><br>{cm['FN']}</div>
                            <div class="cm-cell cm-tp"><strong>True Positive (TP)</strong><br>{cm['TP']}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown("##### Classification Report")
                    st.code(m_info["report"])

    # --- TAB 4: K-Means Clustering (Batch Only Feature) ---
    with tab_kmeans:
        st.subheader("🧩 Customer Segmentation (K-Means Clustering)")
        st.write("Segment uploaded batch customers by tenure, MonthlyCharges, and TotalCharges:")

        if st.session_state.get('batch_processed', False) and 'batch_result_df' in st.session_state:
            df_cluster = st.session_state['batch_result_df'].copy()

            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            # Clean numeric cols
            for c in num_cols:
                df_cluster[c] = pd.to_numeric(df_cluster[c].astype(str).str.strip(), errors='coerce').fillna(0.0)

            num_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=4)

            scaler = StandardScaler()
            X_cluster = scaler.fit_transform(df_cluster[num_cols])

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df_cluster['Cluster'] = [f"Cluster {c+1}" for c in kmeans.fit_predict(X_cluster)]

            # Interactive Plotly Scatter Plot
            st.markdown("##### 📍 Customer Segmentation Scatter Plot (Monthly Charges vs Tenure)")
            fig = px.scatter(
                df_cluster,
                x='tenure',
                y='MonthlyCharges',
                color='Cluster',
                symbol='Churn_Prediction',
                size='TotalCharges',
                hover_data=['Churn_Probability_Percent', 'Contract', 'PaymentMethod'],
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"K-Means Customer Segments (K={num_clusters})"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.6)',
                font=dict(color='#f8fafc'),
                xaxis=dict(gridcolor='#334155', title="Tenure (Months)"),
                yaxis=dict(gridcolor='#334155', title="Monthly Charges ($)"),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster Profile Summary Table
            st.markdown("##### 📊 Cluster Profiles Summary")
            cluster_summary = df_cluster.groupby('Cluster').agg(
                Count=('tenure', 'count'),
                Avg_Tenure=('tenure', 'mean'),
                Avg_MonthlyCharges=('MonthlyCharges', 'mean'),
                Avg_TotalCharges=('TotalCharges', 'mean'),
                Avg_Churn_Risk_Pct=('Churn_Probability_Percent', 'mean')
            ).reset_index()

            cluster_summary['Avg_Tenure'] = cluster_summary['Avg_Tenure'].round(1)
            cluster_summary['Avg_MonthlyCharges'] = cluster_summary['Avg_MonthlyCharges'].round(2)
            cluster_summary['Avg_TotalCharges'] = cluster_summary['Avg_TotalCharges'].round(2)
            cluster_summary['Avg_Churn_Risk_Pct'] = cluster_summary['Avg_Churn_Risk_Pct'].round(2)

            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ K-Means Clustering requires batch data. Please upload a CSV file and run predictions in the **CSV Upload** tab.")

    # --- TAB 5: Download Results ---
    with tab_download:
        st.subheader("📥 Export Prediction Reports")
        if st.session_state.get('batch_processed', False) and 'batch_result_df' in st.session_state:
            df_export = st.session_state['batch_result_df']

            st.write("Download the generated predictions alongside all feature columns:")

            col_down1, col_down2 = st.columns(2)

            with col_down1:
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Results as CSV",
                    data=csv_data,
                    file_name="churn_predictions_export.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

            with col_down2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_export.to_excel(writer, index=False, sheet_name="Churn Predictions")
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    label="📊 Download Results as Excel (.xlsx)",
                    data=excel_data,
                    file_name="churn_predictions_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Export Preview")
            st.dataframe(df_export.head(10), use_container_width=True)
        else:
            st.info("ℹ️ No predictions to download. Upload a batch CSV and run predictions first.")
