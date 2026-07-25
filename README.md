# ⚡ ChurnSense AI — Enterprise Customer Churn & Retention Intelligence Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React_18-61DAFB.svg?style=flat&logo=react)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Build-Vite-646CFF.svg?style=flat&logo=vite)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind_CSS-38BDF8.svg?style=flat&logo=tailwindcss)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/AI-Google_Gemini-4285F4.svg?style=flat&logo=google)](https://ai.google.dev/)

**ChurnSense AI** is a production-ready, full-stack Machine Learning & Generative AI platform engineered to predict customer churn, segment user profiles, run survival/CLV analysis, monitor data drift, and generate personalized, AI-driven retention strategies.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [How to Run the Application](#-how-to-run-the-application)
  - [1. Run ML Training Pipeline](#1-run-ml-training-pipeline)
  - [2. Start FastAPI Backend](#2-start-fastapi-backend)
  - [3. Start React Frontend](#3-start-react-frontend)
  - [4. Start Streamlit Quick Demo (Optional)](#4-start-streamlit-quick-demo-optional)
- [API Endpoints Reference](#-api-endpoints-reference)
- [Environment Variables](#-environment-variables)
- [License & Support](#-license--support)

---

## ✨ Key Features

1. **Multi-Model Machine Learning Engine**:
   - Supports **XGBoost Classifier**, **Random Forest Ensemble**, **Decision Tree Baseline**, and **Logistic Regression**.
   - Outputs calibrated churn probabilities and dynamic risk tiers (*Low*, *Medium*, *High*, *Critical*).

2. **Explainable AI (XAI)**:
   - **SHAP Waterfall Plot**: Pinpoints top individual local feature drivers contributing to a customer's churn risk.
   - **SHAP Global Feature Importance**: Displays dataset-wide predictive feature impact.

3. **Customer Segmentation & Clustering**:
   - **GMM (Gaussian Mixture Model)** soft-clustering and **K-Means Clustering**.
   - 2D PCA visual projections of customer cohorts (e.g., *High-Risk Price-Sensitive*, *Loyal Power-Users*, *Low-Engagement Standard*).

4. **Cox Survival Analysis & CLV Estimation**:
   - Computes expected remaining customer tenure and estimates **Customer Lifetime Value (CLV)**.

5. **Interactive What-If Scenario Simulator**:
   - Allows retention managers to simulate contract changes, tech support additions, or pricing adjustments in real time to observe churn risk changes.

6. **Gemini Segment-Aware AI Retention Strategies**:
   - Integrates Google's Gemini API to generate personalized customer outreach strategies, tailored scripts, and discount promotional coupons based on customer segment & SHAP drivers.

7. **Data Drift & Model Monitoring (PSI)**:
   - Evaluates **Population Stability Index (PSI)** to detect dataset drift between baseline data and incoming production batches.

8. **Batch Scoring**:
   - Upload CSV datasets for bulk churn scoring, risk tier assignment, and exported predictions.

---

## 🏗 System Architecture

```
                                  +---------------------------------------+
                                  |    React SPA Frontend (Vite/Tailwind) |
                                  |    http://localhost:3000              |
                                  +-------------------+-------------------+
                                                      |
                                                      | HTTP / Axios Requests
                                                      v
                                  +---------------------------------------+
                                  |    FastAPI REST Backend               |
                                  |    http://127.0.0.1:8000              |
                                  +---------+-------------------+---------+
                                            |                   |
                     +----------------------+                   +---------------------+
                     |                                                                |
                     v                                                                v
   +------------------------------------+                           +-----------------------------------+
   |   Trained ML Artifacts             |                           |   Google Gemini AI Service        |
   |   (XGBoost, RF, GMM, Cox, SHAP)    |                           |   Automated Retention Strategies  |
   +------------------------------------+                           +-----------------------------------+
```

---

## 📁 Project Directory Structure

```
Customer_Churn/
├── README.md                           # Main Project Documentation
└── Curstomer_churn_project/
    ├── requirements.txt                # Python backend & ML dependencies
    ├── app.py                          # Streamlit quick demo application
    ├── customer_churn.py               # Legacy exploratory data analysis script
    ├── customer_churn_dataset-training-master.csv # Master dataset
    │
    ├── backend/                        # FastAPI Enterprise Backend
    │   ├── main.py                     # FastAPI entry point & CORS configuration
    │   ├── schemas.py                  # Pydantic data schemas
    │   ├── sample_churn_data.csv       # Sample dataset for batch testing
    │   ├── api/
    │   │   └── router.py               # API route definitions
    │   ├── services/
    │   │   ├── model_manager.py        # ML artifact loader & preprocessing
    │   │   ├── prediction_service.py   # Prediction & SHAP calculation logic
    │   │   ├── drift_service.py        # Population Stability Index (PSI) drift engine
    │   │   └── gemini_service.py       # Google Gemini AI strategy generator
    │   └── models/                     # Saved binary models (.pkl & .json)
    │       ├── model_xgboost.pkl
    │       ├── model_random_forest.pkl
    │       ├── model_decision_tree.pkl
    │       ├── model_logistic_regression.pkl
    │       ├── cluster_gmm.pkl
    │       ├── cluster_kmeans.pkl
    │       ├── survival_cox.pkl
    │       ├── encoders.pkl
    │       ├── scaler.pkl
    │       └── metrics.json
    │
    ├── frontend/                       # React 18 + Vite + Tailwind CSS Single Page App
    │   ├── package.json                # Frontend Node.js dependencies & scripts
    │   ├── vite.config.js              # Vite server configuration (Port 3000, API Proxy)
    │   ├── tailwind.config.js          # Tailwind CSS styling configuration
    │   ├── index.html                  # HTML root container
    │   └── src/
    │       ├── main.jsx                # React app initializer
    │       ├── App.jsx                 # Main layout & navigation container
    │       └── components/
    │           ├── Navbar.jsx          # Top navigation bar
    │           ├── Overview.jsx        # Model metrics & dashboard overview
    │           ├── CustomerAnalyzer.jsx# Single customer profile prediction & SHAP
    │           ├── WhatIfSimulator.jsx # Real-time interactive scenario simulator
    │           ├── BatchUpload.jsx     # CSV upload & bulk churn scoring
    │           ├── CustomerClustering.jsx # GMM/K-Means cohort visualization
    │           ├── DriftMonitoring.jsx # PSI data drift analysis
    │           └── ModelInsights.jsx   # Global feature impact & metrics
    │
    └── ml_pipeline/                    # Machine Learning Training Pipeline
        └── train_pipeline.py           # Training, clustering, survival & artifact generation script
```

---

## 🛠 Tech Stack

| Layer | Technology | Usage |
| :--- | :--- | :--- |
| **Frontend Framework** | **React 18**, **Vite** | Modern component framework & fast development bundler |
| **Styling & UI** | **Tailwind CSS**, **Lucide Icons** | Responsive dark-themed UI components |
| **Data Visualization** | **Recharts** | Interactive charts, bar plots, and scatter projections |
| **Backend API** | **FastAPI**, **Uvicorn** | High-performance asynchronous Python REST API |
| **Machine Learning** | **XGBoost**, **Scikit-Learn** | Churn prediction models & cluster algorithms |
| **Explainable AI** | **SHAP** | Local and global feature importance attribution |
| **Survival Analysis** | **Lifelines** | Cox Proportional Hazards for tenure & CLV estimation |
| **Generative AI** | **Google Gemini API** | Segment-aware retention strategies & promo generation |
| **Data Handling** | **Pandas**, **NumPy** | Data preprocessing, feature engineering & transformation |

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your machine:

- **Python**: Version `3.9` or higher
- **Node.js**: Version `18.0.0` or higher
- **npm**: Version `9.0.0` or higher
- **Google Gemini API Key** *(Optional, required for AI strategy generation feature)*

---

## ⚙️ Installation & Setup

### 1. Clone & Set Up Directory

Open your terminal and navigate to the project directory:

```bash
cd d:\Customer_Churn\Curstomer_churn_project
```

### 2. Set Up Python Virtual Environment

Create and activate a virtual environment:

```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Backend Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 🚀 How to Run the Application

To run the complete platform, follow these steps in order:

### Step 1: Run ML Training Pipeline (First-Time Setup)

Train the ML models and generate the binary model artifacts (.pkl/.json) required by the backend:

```bash
# Navigate to the project root directory
cd d:\Customer_Churn\Curstomer_churn_project

# Run training script
python ml_pipeline/train_pipeline.py
```

*Expected Output*: Saved artifacts will be placed inside `Curstomer_churn_project/backend/models/`.

---

### Step 2: Start FastAPI Backend Server

Launch the backend API server on `http://127.0.0.1:8000`:

```bash
# Ensure you are inside Curstomer_churn_project directory
cd d:\Customer_Churn\Curstomer_churn_project

# Option A: Run directly via main.py
python backend/main.py

# Option B: Run via Uvicorn CLI with live reload
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- **Backend API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

### Step 3: Start React Frontend Application

Open a **new terminal window** and launch the React development server:

```bash
# Navigate to the frontend directory
cd d:\Customer_Churn\Curstomer_churn_project\frontend

# Start Vite dev server
npm run dev
```

- **Frontend Application URL**: `http://localhost:3000` (or `http://localhost:5173`)
- Open your browser and navigate to `http://localhost:3000` to interact with ChurnSense AI.

---

### Step 4: Start Streamlit Quick Demo (Optional)

If you prefer a lightweight Streamlit interface instead of the React SPA:

```bash
# Navigate to project root directory
cd d:\Customer_Churn\Curstomer_churn_project

# Start Streamlit
streamlit run app.py
```

- **Streamlit Demo URL**: `http://localhost:8501`

---

## 🔌 API Endpoints Reference

The FastAPI backend exposes the following core endpoints under `/api/v1`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and version info |
| `POST` | `/api/v1/predict` | Predict churn probability, risk tier, GMM cluster, and SHAP drivers for a customer |
| `POST` | `/api/v1/batch-predict` | Upload CSV for bulk churn risk classification & predictions |
| `POST` | `/api/v1/simulate-what-if` | Simulate scenario adjustments and view modified churn risk |
| `POST` | `/api/v1/ai-strategy` | Generate segment-aware retention strategy & discount coupon via Gemini |
| `GET` | `/api/v1/model-metrics` | Retrieve model performance metrics (Accuracy, ROC-AUC, F1-Score) |
| `GET` | `/api/v1/cluster-insights` | Get cluster profiles and 2D PCA projection coordinates |
| `POST` | `/api/v1/check-drift` | Compute PSI data drift metrics against sample/production CSV |

---

## 🔑 Environment Variables

To enable **Google Gemini AI Strategy Generation**, set the `GEMINI_API_KEY` environment variable in your terminal session before launching the backend:

### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="your_google_gemini_api_key_here"
```

### Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=your_google_gemini_api_key_here
```

### macOS / Linux:
```bash
export GEMINI_API_KEY="your_google_gemini_api_key_here"
```

*Note: If `GEMINI_API_KEY` is not provided, the system will fall back to intelligent template-based AI retention strategies.*

---

## ❓ Troubleshooting & FAQs

- **Q: Backend throws `FileNotFoundError` for `.pkl` files.**
  - *Fix*: Run `python ml_pipeline/train_pipeline.py` first to generate models in `backend/models/`.

- **Q: Frontend CORS issues when calling API.**
  - *Fix*: Ensure FastAPI backend is running on `http://127.0.0.1:8000`. The backend CORS policy is set to allow `*`.

- **Q: Port 3000 or 8000 is already in use.**
  - *Fix*: Kill existing node/python processes or specify custom ports (`uvicorn backend.main:app --port 8001` or set `server.port` in `vite.config.js`).

---

## 📄 License & Credits

Built as an enterprise-grade AI solution for customer churn analytics, segmentation, and retention strategy automation.