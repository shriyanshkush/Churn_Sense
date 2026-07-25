import React, { useState } from 'react';
import axios from 'axios';
import { Activity, ShieldCheck, AlertTriangle, UploadCloud, CheckCircle } from 'lucide-react';

export default function DriftMonitoring() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [driftResult, setDriftResult] = useState(null);

  const handleDriftCheck = async () => {
    if (!file) return;
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post('/api/v1/drift-check', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDriftResult(res.data);
    } catch (err) {
      console.error("Drift Check Error:", err);
      alert("Error calculating PSI drift.");
    } finally {
      setLoading(false);
    }
  };

  const getPsiColor = (score) => {
    if (score >= 0.25) return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
    if (score >= 0.10) return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
  };

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="glass-panel p-6 border-indigo-500/20">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Activity className="w-5 h-5 text-indigo-400" />
          <span>MLOps Population Stability Index (PSI) Data Drift Monitoring</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Detect statistical population drift between production batch uploads and training baseline distributions to safeguard model integrity.
        </p>
      </div>

      {/* Upload & Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <div className="glass-panel p-6 space-y-4">
          <h3 className="text-sm font-semibold text-white">Upload Production Batch Dataset</h3>
          <p className="text-xs text-slate-400">Select a batch CSV file to compare feature quantiles against training baseline.</p>

          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
            className="w-full text-xs text-slate-400 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-indigo-300 hover:file:bg-slate-700"
          />

          <button
            onClick={handleDriftCheck}
            disabled={!file || loading}
            className="w-full py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-bold rounded-xl transition shadow-lg shadow-indigo-600/30 disabled:opacity-50"
          >
            {loading ? 'Calculating PSI Metrics...' : 'Calculate Population Stability Index'}
          </button>
        </div>

        {/* PSI Status Card */}
        <div className="lg:col-span-2 glass-panel p-6 flex flex-col justify-between">
          <div>
            <span className="text-xs font-medium text-slate-400">Overall Population Stability Index (PSI)</span>
            <div className="mt-2 flex items-baseline space-x-3">
              <span className="text-4xl font-extrabold text-white font-display">
                {driftResult ? driftResult.overall_psi : '0.042'}
              </span>
              <span className={`px-3 py-1 rounded-full text-xs font-bold border ${getPsiColor(driftResult?.overall_psi || 0.042)}`}>
                {driftResult ? driftResult.drift_status : 'Low Drift (Population Distribution Stable)'}
              </span>
            </div>
          </div>

          <div className="mt-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300 space-y-1">
            <p><strong>PSI &lt; 0.10:</strong> No significant distribution change; model predictions are fully reliable.</p>
            <p><strong>0.10 ≤ PSI &lt; 0.25:</strong> Moderate drift detected; monitor feature distributions closely.</p>
            <p><strong>PSI ≥ 0.25:</strong> Significant population drift; model retraining strongly recommended.</p>
          </div>
        </div>

      </div>

      {/* Feature-by-Feature PSI Table */}
      {driftResult?.feature_psi_scores && (
        <div className="glass-panel p-6 space-y-3">
          <h3 className="text-sm font-semibold text-white">Feature Drift Breakdown</h3>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {Object.entries(driftResult.feature_psi_scores).map(([feat, score]) => (
              <div key={feat} className="glass-card p-3 border border-slate-800 flex items-center justify-between">
                <span className="text-xs text-slate-300 font-mono">{feat}</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold font-mono border ${getPsiColor(score)}`}>
                  PSI: {score}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
