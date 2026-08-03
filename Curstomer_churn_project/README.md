# ⚡ ChurnSense AI — Customer Churn & Retention Intelligence Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React_18-61DAFB.svg?style=flat&logo=react)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Build-Vite-646CFF.svg?style=flat&logo=vite)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind_CSS-38BDF8.svg?style=flat&logo=tailwindcss)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/AI-Google_Gemini-4285F4.svg?style=flat&logo=google)](https://ai.google.dev/)

**ChurnSense AI** is a full-stack Machine Learning & Generative AI web platform designed for customer churn prediction, interactive What-If scenario simulation, customer segmentation, and Gemini AI retention strategy generation.

---

## 🖼 Application Screenshots

### 1. Landing Page (Mode Selection)
Mode selection landing interface with **zero pre-loaded default data**, allowing users to choose between **Single Customer Prediction** and **Batch Prediction**.

![Landing Page](screenshots/landing_page.png)

---

### 2. Single Customer Prediction & Risk Gauge Dashboard
19-feature input form with real-time churn risk prediction, risk tier level assignment, and confidence gauge.

![Single Customer Input](screenshots/single_customer_input.png)
![Prediction Result](screenshots/prediction_result.png)

---

### 3. Interactive What-If Scenario Simulator
Real-time sliders for Monthly Charges, Tenure, Contract Type, TechSupport, and OnlineSecurity with live SHAP feature driver waterfall plots.

![What-If Simulator](screenshots/what_if_simulator.png)

---

### 4. Gemini Segment-Aware AI Retention Strategies & Promo Coupons
Generates personalized customer retention playbooks, intervention cost breakdowns, campaign ROI (+%), and custom promo coupons (e.g. `SAVE-HIGH-RISK-984`) incorporating customer call notes and prompt details.

![Gemini AI Retention Strategy](screenshots/gemini_ai_strategy.png)

---

### 5. Batch Scoring & 1-Click Sample CSV Dataset Loader
Upload batch customer `.csv` files for bulk churn scoring, risk tier assignment, and exported predictions. Includes a **Download Sample CSV** button and a **Use Sample Dataset (25 Rows)** 1-click test loader.

![Batch Scoring](screenshots/batch_scoring.png)

---

### 6. Multi-Model ML Benchmark & Model Comparison
Interactive model comparison evaluating **Logistic Regression** (*Best Recall: 78.55%*), **XGBoost** (*Best Accuracy: 79.00%*), **Random Forest**, and **Decision Tree** with pre-computed metrics tables, confusion matrices, and classification reports.

![Model Comparison](screenshots/model_comparison.png)

---

## 🚀 Quick Start Instructions

### 1. ML Model Pipeline Setup
```bash
python ml_pipeline/train_pipeline.py
```

### 2. Start FastAPI Backend (Port 8000)
```bash
python backend/main.py
```
- Interactive API Docs: `http://127.0.0.1:8000/docs`

### 3. Start React SPA Frontend (Port 3000)
```bash
cd frontend
npm install
npm run dev
```
- Application Interface: `http://localhost:3000`

### 4. Optional Streamlit Demo
```bash
streamlit run app.py
```
- Streamlit Interface: `http://localhost:8501`

---

## 🔌 Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check & API version status |
| `POST` | `/api/v1/predict` | Single customer churn prediction & SHAP drivers |
| `POST` | `/api/v1/batch-predict` | Upload CSV for bulk churn risk classification & predictions |
| `POST` | `/api/v1/ai-strategy` | Generate segment-aware Gemini retention strategy & promo coupon |
| `GET` | `/api/v1/clusters` | 2D PCA projection coordinates & cluster elbow metrics |
| `GET` | `/api/v1/model-insights` | Model metrics comparison & SHAP feature importances |

---

## 🔑 Gemini API Key Configuration

To enable live Google Gemini AI strategy generation:
```powershell
$env:GEMINI_API_KEY="your_google_gemini_api_key_here"
```
*(Automatic template fallback is used if no API key is provided).*
