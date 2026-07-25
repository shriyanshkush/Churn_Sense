import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Cpu, ShieldCheck, BarChart2, Layers } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid } from 'recharts';

export default function ModelInsights() {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchInsights();
  }, []);

  const fetchInsights = async () => {
    try {
      setLoading(true);
      const res = await axios.get('/api/v1/model-insights');
      setInsights(res.data);
    } catch (err) {
      console.error("Model Insights API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const modelNames = {
    xgboost: "XGBoost Classifier",
    random_forest: "Random Forest Ensemble",
    decision_tree: "Decision Tree Baseline",
    logistic_regression: "Logistic Regression Baseline"
  };

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="glass-panel p-6 border-indigo-500/20">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-indigo-400" />
          <span>Multi-Model Insights & SHAP Explainability</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Comparative performance metrics (AUC, F1, Precision, Recall) across Logistic Regression, Decision Tree, Random Forest, and XGBoost, with Global SHAP Feature Importance.
        </p>
      </div>

      {/* Model Performance Comparison Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {insights?.models_metrics && Object.entries(insights.models_metrics).map(([mKey, mData]) => (
          <div key={mKey} className={`glass-card p-5 border ${mKey === 'xgboost' ? 'border-indigo-500/50 bg-indigo-950/20' : 'border-slate-800'}`}>
            <span className="text-xs font-bold text-white block">{modelNames[mKey] || mKey}</span>
            <div className="mt-3 space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">ROC-AUC:</span>
                <strong className="text-indigo-400 font-mono">{(mData.roc_auc * 100).toFixed(1)}%</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">F1 Score:</span>
                <strong className="text-purple-300 font-mono">{(mData.f1_score * 100).toFixed(1)}%</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Precision:</span>
                <strong className="text-slate-200 font-mono">{(mData.precision * 100).toFixed(1)}%</strong>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Recall:</span>
                <strong className="text-slate-200 font-mono">{(mData.recall * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Global SHAP Feature Importance Chart */}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold text-white mb-1">Global SHAP Feature Importance (Model Insights)</h3>
        <p className="text-xs text-slate-400 mb-4">Mean absolute SHAP impact across training dataset</p>

        <div className="h-64 w-full">
          {insights?.global_shap_importance ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={insights.global_shap_importance.slice(0, 10)} margin={{ left: 20, right: 20, top: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="feature" stroke="#94a3b8" fontSize={11} angle={-25} textAnchor="end" />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                <Bar dataKey="importance" fill="#6366f1" radius={[4, 4, 0, 0]}>
                  {insights.global_shap_importance.slice(0, 10).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={`hsl(${240 + index * 12}, 80%, 65%)`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-400 text-xs">Loading Global SHAP data...</div>
          )}
        </div>
      </div>

    </div>
  );
}
