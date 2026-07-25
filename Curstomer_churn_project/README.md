# ⚡ ChurnSense AI — Enterprise Customer Churn & Retention Intelligence Platform

Refer to the main [README.md](../README.md) at the repository root for complete instructions, architecture, tech stack details, and API documentation.

## Quick Start Summary

### 1. ML Model Training Pipeline
```bash
cd Curstomer_churn_project
python ml_pipeline/train_pipeline.py
```

### 2. Start FastAPI Backend (Port 8000)
```bash
cd Curstomer_churn_project
python backend/main.py
```
- API Docs: `http://127.0.0.1:8000/docs`

### 3. Start React SPA Frontend (Port 3000)
```bash
cd Curstomer_churn_project/frontend
npm install
npm run dev
```
- Frontend UI: `http://localhost:3000`

### 4. Start Streamlit Quick Demo (Optional)
```bash
cd Curstomer_churn_project
streamlit run app.py
```
