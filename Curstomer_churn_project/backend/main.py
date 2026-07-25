import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.router import router as api_router
from backend.services.model_manager import model_manager

app = FastAPI(
    title="ChurnSense AI API",
    description="Enterprise Machine Learning & Actionable AI Customer Churn Intelligence Platform",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev & demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Initializing ChurnSense FastAPI Backend...")
    model_manager.load_artifacts()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "ChurnSense AI API",
        "version": "2.0.0",
        "documentation": "/docs"
    }

# Register API Router
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
