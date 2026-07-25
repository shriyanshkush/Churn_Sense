import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, silhouette_score
)
import shap
from lifelines import CoxPHFitter, KaplanMeierFitter

# Ensure directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_or_load_dataset():
    """Load existing dataset or synthesize a realistic Telco dataset if missing."""
    telco_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    csv_master = os.path.join(BASE_DIR, "customer_churn_dataset-training-master.csv")

    if os.path.exists(telco_path):
        print(f"Loading Telco dataset from {telco_path}")
        df = pd.read_csv(telco_path)
    elif os.path.exists(csv_master):
        print(f"Adapting master CSV dataset from {csv_master}")
        df_raw = pd.read_csv(csv_master).dropna().head(10000)
        # Adapt schema to standard telco fields
        df = pd.DataFrame()
        df['customerID'] = [f"CUST-{i:05d}" for i in range(len(df_raw))]
        df['gender'] = df_raw['Gender'] if 'Gender' in df_raw.columns else 'Male'
        df['SeniorCitizen'] = (df_raw['Age'] >= 60).astype(int) if 'Age' in df_raw.columns else 0
        df['Partner'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['Dependents'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['tenure'] = df_raw['Tenure'].astype(int)
        df['PhoneService'] = 'Yes'
        df['MultipleLines'] = 'No'
        df['InternetService'] = np.random.choice(['Fiber optic', 'DSL', 'No'], len(df_raw), p=[0.4, 0.4, 0.2])
        df['OnlineSecurity'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['OnlineBackup'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['DeviceProtection'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['TechSupport'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['StreamingTV'] = np.random.choice(['Yes', 'No'], len(df_raw))
        df['StreamingMovies'] = np.random.choice(['Yes', 'No'], len(df_raw))
        
        # Contract mapping
        if 'Contract Length' in df_raw.columns:
            contract_map = {'Monthly': 'Month-to-month', 'Quarterly': 'One year', 'Annual': 'Two year'}
            df['Contract'] = df_raw['Contract Length'].map(contract_map).fillna('Month-to-month')
        else:
            df['Contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], len(df_raw))
            
        df['PaperlessBilling'] = 'Yes'
        df['PaymentMethod'] = 'Electronic check'
        df['MonthlyCharges'] = np.round(df_raw['Total Spend'] / np.maximum(df_raw['Tenure'], 1), 2) if 'Total Spend' in df_raw.columns else np.random.uniform(20, 110, len(df_raw))
        df['MonthlyCharges'] = np.clip(df['MonthlyCharges'], 18.0, 120.0)
        df['TotalCharges'] = np.round(df['MonthlyCharges'] * df['tenure'], 2)
        df['Churn'] = df_raw['Churn'].map({1: 'Yes', 0: 'No', '1': 'Yes', '0': 'No', 'Yes': 'Yes', 'No': 'No'})
    else:
        print("Generating synthetic Telco dataset for training...")
        np.random.seed(42)
        n = 5000
        churn_prob = np.random.uniform(0.1, 0.8, n)
        df = pd.DataFrame({
            'customerID': [f"CUST-{i:05d}" for i in range(n)],
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'Dependents': np.random.choice(['Yes', 'No'], n),
            'tenure': np.random.randint(1, 73, n),
            'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.4, 0.45, 0.15]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n),
            'MonthlyCharges': np.random.uniform(18.25, 118.75, n),
        })
        df['TotalCharges'] = np.round(df['MonthlyCharges'] * df['tenure'], 2)
        
        # Realistic Churn Logic
        risk = (
            (df['Contract'] == 'Month-to-month') * 0.35 +
            (df['InternetService'] == 'Fiber optic') * 0.2 +
            (df['MonthlyCharges'] > 70) * 0.25 +
            (df['tenure'] < 12) * 0.25 -
            (df['TechSupport'] == 'Yes') * 0.2 -
            (df['OnlineSecurity'] == 'Yes') * 0.2
        )
        prob = 1 / (1 + np.exp(-risk))
        df['Churn'] = np.where(np.random.uniform(0, 1, n) < prob, 'Yes', 'No')

    return df

def engineer_features(df_input):
    """Engineers RFM and domain-specific features."""
    df = df_input.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Clean TotalCharges
    if df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)

    # Derived Features
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['service_count'] = 0
    for col in service_cols:
        if col in df.columns:
            df['service_count'] += (df[col] == 'Yes').astype(int)

    df['charges_per_service'] = np.round(df['MonthlyCharges'] / (df['service_count'] + 1), 2)
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 60, 100], 
                                 labels=['0-12m', '12-24m', '24-48m', '48-60m', '60m+']).astype(str)
    df['has_security_and_support'] = ((df.get('OnlineSecurity', '') == 'Yes') & (df.get('TechSupport', '') == 'Yes')).astype(int)
    
    contract_risk = {'Month-to-month': 1.0, 'One year': 0.5, 'Two year': 0.1}
    df['contract_risk'] = df['Contract'].map(contract_risk).fillna(0.8) if 'Contract' in df.columns else 0.5

    return df

def main():
    print("--- Starting ChurnSense ML Training Pipeline ---")
    df_raw = generate_or_load_dataset()
    
    # Save raw clean sample dataset for batch upload testing
    sample_csv_path = os.path.join(BASE_DIR, "backend", "sample_churn_data.csv")
    df_raw.head(500).to_csv(sample_csv_path, index=False)
    print(f"Saved demo batch CSV to {sample_csv_path}")

    # Feature Engineering
    df_feat = engineer_features(df_raw)
    
    # Encode Categoricals
    encoders = {}
    categorical_cols = df_feat.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')

    df_encoded = df_feat.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Encode Target
    target_le = LabelEncoder()
    y = target_le.fit_transform(df_raw['Churn'].astype(str))
    encoders['Churn'] = target_le

    X = df_encoded.drop(columns=['Churn'])
    feature_names = X.columns.tolist()

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled_all = scaler.transform(X)

    # Save encoders & scaler
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # --- 1. Model Training & Evaluation (4 Models) ---
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=42)
    }

    metrics_summary = {}

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        metrics_summary[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "roc_curve": {"fpr": np.round(fpr, 4).tolist()[::max(1, len(fpr)//50)], 
                          "tpr": np.round(tpr, 4).tolist()[::max(1, len(tpr)//50)]}
        }

        # Save model file
        with open(os.path.join(MODELS_DIR, f"model_{name}.pkl"), "wb") as f:
            pickle.dump(model, f)

    # Save metrics JSON
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # --- 2. Clustering & Soft Segmentation (K-Means + GMM) ---
    print("Building K-Means & GMM Clustering pipeline...")
    # Calculate Elbow & Silhouette scores for k=2..7
    elbow_data = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled_all)
        sil = silhouette_score(X_scaled_all, km.labels_) if len(X_scaled_all) <= 5000 else 0.5
        elbow_data.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})

    # Fit GMM with k=4
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_clusters = gmm.fit_predict(X_scaled_all)

    # Fit KMeans with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_clusters = kmeans.fit_predict(X_scaled_all)

    # PCA 2D Transformation
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled_all)

    # Auto-label Clusters based on cluster profiles
    df_cluster_analysis = df_raw.copy()
    df_cluster_analysis['cluster'] = gmm_clusters
    xgb_best = models['xgboost']
    df_cluster_analysis['churn_prob'] = xgb_best.predict_proba(X)[:, 1]

    cluster_labels = {}
    for c_id in range(4):
        c_sub = df_cluster_analysis[df_cluster_analysis['cluster'] == c_id]
        avg_churn = c_sub['churn_prob'].mean()
        avg_tenure = c_sub['tenure'].mean()
        avg_charges = c_sub['MonthlyCharges'].mean()

        if avg_churn >= 0.5 and avg_charges >= 65:
            label = "High-Risk Price-Sensitive"
        elif avg_churn < 0.25 and avg_tenure >= 35:
            label = "Stable High-Value"
        elif avg_churn >= 0.45 and avg_tenure < 18:
            label = "New & Vulnerable"
        else:
            label = "Loyal Low-Engagement"
        cluster_labels[c_id] = label

    # Save cluster artifacts
    with open(os.path.join(MODELS_DIR, "cluster_gmm.pkl"), "wb") as f:
        pickle.dump(gmm, f)
    with open(os.path.join(MODELS_DIR, "cluster_kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODELS_DIR, "pca_2d.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODELS_DIR, "cluster_labels.json"), "w") as f:
        json.dump(cluster_labels, f, indent=2)
    with open(os.path.join(MODELS_DIR, "elbow_metrics.json"), "w") as f:
        json.dump(elbow_data, f, indent=2)

    # --- 3. Survival Analysis (Cox Proportional Hazards + Kaplan-Meier) ---
    print("Fitting Survival Analysis models (lifelines)...")
    df_survival = pd.DataFrame({
        'tenure': df_raw['tenure'].clip(lower=1),
        'churn_event': (df_raw['Churn'] == 'Yes').astype(int),
        'MonthlyCharges': df_raw['MonthlyCharges'],
        'is_month_to_month': (df_feat['Contract'] == 'Month-to-month').astype(int) if 'Contract' in df_feat else 0,
        'has_tech_support': (df_feat.get('TechSupport', '') == 'Yes').astype(int)
    })

    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col='tenure', event_col='churn_event')

    kmf = KaplanMeierFitter()
    kmf.fit(df_survival['tenure'], event_observed=df_survival['churn_event'])

    with open(os.path.join(MODELS_DIR, "survival_cox.pkl"), "wb") as f:
        pickle.dump(cph, f)

    # Pre-calculate baseline survival curve
    timeline = list(range(1, 73))
    sf_baseline = kmf.survival_function_at_times(timeline).values.tolist()
    with open(os.path.join(MODELS_DIR, "survival_baseline.json"), "w") as f:
        json.dump({"timeline": timeline, "survival_probability": [round(x, 4) for x in sf_baseline]}, f)

    # --- 4. Global SHAP Summary ---
    print("Computing SHAP Summary values...")
    try:
        explainer = shap.TreeExplainer(models['xgboost'])
        shap_vals = explainer.shap_values(X_test.head(200))
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_summary = [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
        ]
        with open(os.path.join(MODELS_DIR, "shap_global.json"), "w") as f:
            json.dump(shap_summary, f, indent=2)
    except Exception as e:
        print(f"SHAP global generation notice: {e}")

    # --- 5. Data Drift Baseline (PSI) ---
    print("Calculating Population Stability Index (PSI) baseline...")
    psi_baseline = {}
    for col in X.columns:
        psi_baseline[col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "quantiles": [float(q) for q in np.quantile(X[col], [0.1, 0.25, 0.5, 0.75, 0.9])]
        }
    with open(os.path.join(MODELS_DIR, "baseline_psi.json"), "w") as f:
        json.dump(psi_baseline, f, indent=2)

    print("--- ML Training Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    main()
