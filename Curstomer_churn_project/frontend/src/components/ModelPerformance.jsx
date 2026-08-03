import React, { useState } from 'react';
import { Award, CheckCircle2, ChevronDown, ChevronUp, Cpu, Info, ShieldAlert, Sparkles } from 'lucide-react';

export const MODEL_METRICS_DATA = {
  logistic_regression: {
    id: "logistic_regression",
    name: "Logistic Regression",
    accuracy: "77%",
    precision: "54.16%",
    recall: "78.55%",
    f1: "64.11%",
    badge: "🏆 Best Performing (Highest Recall & F1 Score)",
    recommendation: "Recommended for max recall (catches 78.55% of all churning customers).",
    cm: { TN: 782, FP: 251, FN: 81, TP: 295 },
    report: `              precision    recall  f1-score   support

        No       0.91      0.76      0.83      1033
       Yes       0.54      0.79      0.64       376

  accuracy                           0.77      1409
 macro avg       0.72      0.77      0.73      1409
weighted avg       0.81      0.77      0.78      1409`
  },
  xgboost: {
    id: "xgboost",
    name: "XGBoost",
    accuracy: "79%",
    precision: "59.52%",
    recall: "59.52%",
    f1: "59.52%",
    badge: "⭐ Recommended for Precision & Accuracy",
    recommendation: "Recommended for highest overall accuracy (79.00%) & precision (59.52%).",
    cm: { TN: 881, FP: 152, FN: 152, TP: 224 },
    report: `              precision    recall  f1-score   support

        No       0.85      0.85      0.85      1033
       Yes       0.60      0.60      0.60       376

  accuracy                           0.79      1409
 macro avg       0.72      0.72      0.72      1409
weighted avg       0.79      0.79      0.79      1409`
  },
  random_forest: {
    id: "random_forest",
    name: "Random Forest",
    accuracy: "78%",
    precision: "56.80%",
    recall: "62.73%",
    f1: "59.62%",
    badge: null,
    recommendation: "Strong ensemble performance across balanced customer distributions.",
    cm: { TN: 854, FP: 179, FN: 140, TP: 236 },
    report: `              precision    recall  f1-score   support

        No       0.86      0.83      0.84      1033
       Yes       0.57      0.63      0.60       376

  accuracy                           0.78      1409
 macro avg       0.71      0.73      0.72      1409
weighted avg       0.78      0.78      0.78      1409`
  },
  decision_tree: {
    id: "decision_tree",
    name: "Decision Tree",
    accuracy: "74%",
    precision: "51.02%",
    recall: "60.59%",
    f1: "55.39%",
    badge: null,
    recommendation: "Fast baseline classifier for high-interpretability decision rules.",
    cm: { TN: 810, FP: 223, FN: 147, TP: 229 },
    report: `              precision    recall  f1-score   support

        No       0.85      0.78      0.81      1033
       Yes       0.51      0.61      0.55       376

  accuracy                           0.74      1409
 macro avg       0.68      0.70      0.68      1409
weighted avg       0.76      0.74      0.74      1409`
  }
};

export default function ModelPerformance({ selectedModel, setSelectedModel }) {
  const [openExpander, setOpenExpander] = useState(null);

  const toggleExpander = (id) => {
    setOpenExpander(openExpander === id ? null : id);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-xl">
        <div className="flex items-center space-x-3 mb-2">
          <Cpu className="w-6 h-6 text-indigo-400" />
          <h2 className="text-xl font-bold text-white tracking-tight">
            Pre-Computed Model Evaluation & Metrics
          </h2>
        </div>
        <p className="text-sm text-slate-400">
          Stored benchmark evaluation results across all four pre-trained machine learning algorithms:
        </p>
      </div>

      {/* Benchmark Metrics Comparison Table */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl overflow-x-auto">
        <h3 className="text-base font-bold text-slate-200 mb-4 flex items-center space-x-2">
          <Award className="w-5 h-5 text-amber-400" />
          <span>Model Benchmark Metrics Comparison Table</span>
        </h3>
        
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-xs font-semibold text-slate-400 uppercase tracking-wider bg-slate-950/60">
              <th className="py-3.5 px-4">Model</th>
              <th className="py-3.5 px-4">Accuracy</th>
              <th className="py-3.5 px-4">Precision</th>
              <th className="py-3.5 px-4">Recall</th>
              <th className="py-3.5 px-4">F1 Score</th>
              <th className="py-3.5 px-4">Recommendation</th>
              <th className="py-3.5 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-sm text-slate-200">
            {Object.values(MODEL_METRICS_DATA).map((m) => {
              const isSelected = selectedModel === m.id;
              return (
                <tr 
                  key={m.id} 
                  className={`transition-colors ${isSelected ? 'bg-indigo-950/30 border-l-4 border-l-indigo-500' : 'hover:bg-slate-800/40'}`}
                >
                  <td className="py-4 px-4 font-bold text-white flex items-center space-x-2">
                    <span>{m.name}</span>
                    {isSelected && (
                      <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded-full border border-indigo-500/30">
                        Selected
                      </span>
                    )}
                  </td>
                  <td className="py-4 px-4 font-semibold text-emerald-400">{m.accuracy}</td>
                  <td className="py-4 px-4 font-semibold text-slate-300">{m.precision}</td>
                  <td className="py-4 px-4 font-semibold text-cyan-400">{m.recall}</td>
                  <td className="py-4 px-4 font-bold text-indigo-400">{m.f1}</td>
                  <td className="py-4 px-4 text-xs text-slate-300">
                    {m.badge ? (
                      <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium">
                        <Sparkles className="w-3 h-3 text-amber-400" />
                        <span>{m.badge}</span>
                      </span>
                    ) : (
                      <span className="text-slate-500">-</span>
                    )}
                  </td>
                  <td className="py-4 px-4 text-right">
                    <button
                      onClick={() => setSelectedModel(m.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                        isSelected 
                          ? 'bg-indigo-600 text-white shadow-md' 
                          : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                      }`}
                    >
                      {isSelected ? 'Active Model' : 'Select Model'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Expandable Sections for Confusion Matrix & Classification Reports */}
      <div className="space-y-4">
        <h3 className="text-base font-bold text-slate-200 flex items-center space-x-2">
          <Info className="w-5 h-5 text-indigo-400" />
          <span>Detailed Model Reports & Confusion Matrices</span>
        </h3>

        {Object.values(MODEL_METRICS_DATA).map((m) => {
          const isOpen = openExpander === m.id;
          return (
            <div 
              key={m.id} 
              className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-lg transition"
            >
              <button
                onClick={() => toggleExpander(m.id)}
                className="w-full p-4 flex items-center justify-between bg-slate-950/40 hover:bg-slate-800/40 text-left transition"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 font-bold">
                    {m.name[0]}
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white">{m.name} Detailed Report</h4>
                    <p className="text-xs text-slate-400">
                      Accuracy: {m.accuracy} &bull; Recall: {m.recall} &bull; F1: {m.f1}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {m.badge && (
                    <span className="hidden sm:inline-block text-[11px] px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 font-medium">
                      {m.badge}
                    </span>
                  )}
                  {isOpen ? (
                    <ChevronUp className="w-5 h-5 text-slate-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-slate-400" />
                  )}
                </div>
              </button>

              {isOpen && (
                <div className="p-6 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-950/60">
                  {/* Confusion Matrix Grid */}
                  <div>
                    <h5 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                      Confusion Matrix (Test Set N=1,409)
                    </h5>
                    <div className="grid grid-cols-2 gap-3 text-center">
                      <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-500/30 text-emerald-300">
                        <span className="text-xs font-semibold block text-emerald-400/80">True Negative (TN)</span>
                        <span className="text-xl font-extrabold">{m.cm.TN}</span>
                      </div>
                      <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/30 text-amber-300">
                        <span className="text-xs font-semibold block text-amber-400/80">False Positive (FP)</span>
                        <span className="text-xl font-extrabold">{m.cm.FP}</span>
                      </div>
                      <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-500/30 text-rose-300">
                        <span className="text-xs font-semibold block text-rose-400/80">False Negative (FN)</span>
                        <span className="text-xl font-extrabold">{m.cm.FN}</span>
                      </div>
                      <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-500/30 text-indigo-300">
                        <span className="text-xs font-semibold block text-indigo-400/80">True Positive (TP)</span>
                        <span className="text-xl font-extrabold">{m.cm.TP}</span>
                      </div>
                    </div>
                  </div>

                  {/* Classification Report */}
                  <div>
                    <h5 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                      Full Classification Report
                    </h5>
                    <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto">
                      {m.report}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
