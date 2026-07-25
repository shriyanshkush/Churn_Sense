import numpy as np
import pandas as pd
from typing import Dict, Any
from backend.services.model_manager import model_manager
from backend.schemas import DriftCheckResponse

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index between expected (training) and actual (batch) distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    expected_counts, _ = np.histogram(expected, buckets)
    actual_counts, _ = np.histogram(actual, buckets)

    expected_pct = np.maximum(expected_counts / len(expected), 0.0001)
    actual_pct = np.maximum(actual_counts / len(actual), 0.0001)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_val, 4))

def check_data_drift(df_batch: pd.DataFrame) -> DriftCheckResponse:
    model_manager.load_artifacts()
    baseline = model_manager.baseline_psi

    feature_scores = {}
    total_psi = 0.0
    valid_count = 0

    for col in model_manager.feature_names:
        if col in df_batch.columns:
            try:
                batch_vals = pd.to_numeric(df_batch[col], errors='coerce').dropna().values
                if len(batch_vals) > 0 and col in baseline:
                    b_mean = baseline[col]['mean']
                    b_std = baseline[col]['std']
                    # Approximate expected normal/quantile distribution
                    expected_vals = np.random.normal(b_mean, max(0.01, b_std), 1000)
                    psi_score = calculate_psi(expected_vals, batch_vals)
                    feature_scores[col] = psi_score
                    total_psi += psi_score
                    valid_count += 1
            except Exception:
                pass

    overall_psi = round(total_psi / max(1, valid_count), 4)
    if overall_psi >= 0.25:
        status = "High Data Drift (Model Retraining Recommended)"
    elif overall_psi >= 0.10:
        status = "Moderate Drift Detected (Monitor Closely)"
    else:
        status = "Low Drift (Population Distribution Stable)"

    return DriftCheckResponse(
        total_records=len(df_batch),
        overall_psi=overall_psi,
        drift_status=status,
        feature_psi_scores=feature_scores
    )
