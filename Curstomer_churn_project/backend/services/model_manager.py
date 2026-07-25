import os
import json
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")

class ModelManager:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = []
        self.cluster_gmm = None
        self.cluster_kmeans = None
        self.pca_2d = None
        self.cluster_labels = {}
        self.survival_cox = None
        self.metrics = {}
        self.elbow_metrics = []
        self.shap_global = []
        self.baseline_psi = {}
        self.survival_baseline = {}
        self.loaded = False

    def load_artifacts(self):
        if self.loaded:
            return

        print("--- Loading Trained ML Artifacts into FastAPI Memory ---")
        # Load encoders & scaler
        with open(os.path.join(MODELS_DIR, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
            self.feature_names = json.load(f)

        # Load models
        for m_name in ["logistic_regression", "decision_tree", "random_forest", "xgboost"]:
            m_path = os.path.join(MODELS_DIR, f"model_{m_name}.pkl")
            if os.path.exists(m_path):
                with open(m_path, "rb") as f:
                    self.models[m_name] = pickle.load(f)

        # Load cluster models
        with open(os.path.join(MODELS_DIR, "cluster_gmm.pkl"), "rb") as f:
            self.cluster_gmm = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "cluster_kmeans.pkl"), "rb") as f:
            self.cluster_kmeans = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "pca_2d.pkl"), "rb") as f:
            self.pca_2d = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "cluster_labels.json"), "r") as f:
            self.cluster_labels = json.load(f)
        with open(os.path.join(MODELS_DIR, "elbow_metrics.json"), "r") as f:
            self.elbow_metrics = json.load(f)

        # Load survival & metrics
        with open(os.path.join(MODELS_DIR, "survival_cox.pkl"), "rb") as f:
            self.survival_cox = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
            self.metrics = json.load(f)
        with open(os.path.join(MODELS_DIR, "shap_global.json"), "r") as f:
            self.shap_global = json.load(f)
        with open(os.path.join(MODELS_DIR, "baseline_psi.json"), "r") as f:
            self.baseline_psi = json.load(f)
        with open(os.path.join(MODELS_DIR, "survival_baseline.json"), "r") as f:
            self.survival_baseline = json.load(f)

        self.loaded = True
        print(f"--- ML Manager Artifacts Loaded Successfully! ({len(self.models)} models available) ---")

    def preprocess_dict(self, input_dict: dict):
        """Converts raw customer dict into encoded feature vector matching feature_names."""
        # Clean numeric
        monthly_charges = float(input_dict.get('MonthlyCharges', 50.0))
        tenure = int(input_dict.get('tenure', 10))
        total_charges = float(input_dict.get('TotalCharges', monthly_charges * tenure))

        # Engineer derived features
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        service_count = sum(1 for col in service_cols if input_dict.get(col) == 'Yes')
        charges_per_service = round(monthly_charges / (service_count + 1), 2)
        
        tenure_bucket = '0-12m'
        if tenure > 60: tenure_bucket = '60m+'
        elif tenure > 48: tenure_bucket = '48-60m'
        elif tenure > 24: tenure_bucket = '24-48m'
        elif tenure > 12: tenure_bucket = '12-24m'

        has_sec_sup = 1 if (input_dict.get('OnlineSecurity') == 'Yes' and input_dict.get('TechSupport') == 'Yes') else 0
        contract_risk = {'Month-to-month': 1.0, 'One year': 0.5, 'Two year': 0.1}.get(input_dict.get('Contract'), 0.8)

        processed = {
            'gender': input_dict.get('gender', 'Male'),
            'SeniorCitizen': int(input_dict.get('SeniorCitizen', 0)),
            'Partner': input_dict.get('Partner', 'No'),
            'Dependents': input_dict.get('Dependents', 'No'),
            'tenure': tenure,
            'PhoneService': input_dict.get('PhoneService', 'Yes'),
            'MultipleLines': input_dict.get('MultipleLines', 'No'),
            'InternetService': input_dict.get('InternetService', 'Fiber optic'),
            'OnlineSecurity': input_dict.get('OnlineSecurity', 'No'),
            'OnlineBackup': input_dict.get('OnlineBackup', 'No'),
            'DeviceProtection': input_dict.get('DeviceProtection', 'No'),
            'TechSupport': input_dict.get('TechSupport', 'No'),
            'StreamingTV': input_dict.get('StreamingTV', 'No'),
            'StreamingMovies': input_dict.get('StreamingMovies', 'No'),
            'Contract': input_dict.get('Contract', 'Month-to-month'),
            'PaperlessBilling': input_dict.get('PaperlessBilling', 'Yes'),
            'PaymentMethod': input_dict.get('PaymentMethod', 'Electronic check'),
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'service_count': service_count,
            'charges_per_service': charges_per_service,
            'tenure_bucket': tenure_bucket,
            'has_security_and_support': has_sec_sup,
            'contract_risk': contract_risk
        }

        df_row = pd.DataFrame([processed])
        
        # Apply encoders
        for col in df_row.columns:
            if col in self.encoders and col != 'Churn':
                val = str(df_row[col].iloc[0])
                le = self.encoders[col]
                if val in le.classes_:
                    df_row[col] = le.transform([val])[0]
                else:
                    df_row[col] = 0

        # Ensure order matches feature_names
        for feat in self.feature_names:
            if feat not in df_row.columns:
                df_row[feat] = 0
        
        df_row = df_row[self.feature_names]
        return df_row

model_manager = ModelManager()
