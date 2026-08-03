import React from 'react';
import { UserCheck, Layers, ArrowRight, ShieldCheck, Zap, Database } from 'lucide-react';

export default function LandingPage({ onSelectMode }) {
  return (
    <div className="max-w-6xl mx-auto py-12 px-4 space-y-12">
      {/* Hero Header */}
      <div className="text-center space-y-4 max-w-3xl mx-auto">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs font-semibold">
          <Zap className="w-3.5 h-3.5 text-indigo-400" />
          <span>Churn Analytics Platform &bull; v2.0 Enterprise</span>
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight leading-tight">
          Customer Churn Prediction Platform
        </h1>

        <p className="text-slate-400 text-base sm:text-lg leading-relaxed">
          Enterprise machine learning inference & customer analytics. Select your prediction mode below to begin scoring customer churn risk.
        </p>
      </div>

      {/* 2 Mode Selection Clickable Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-4">
        {/* Option 1: Single Customer Prediction */}
        <div 
          onClick={() => onSelectMode('single')}
          className="group relative bg-slate-900/90 border border-slate-800 hover:border-indigo-500/60 rounded-3xl p-8 shadow-2xl transition-all duration-300 hover:-translate-y-2 cursor-pointer flex flex-col justify-between"
        >
          <div className="space-y-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30 group-hover:scale-110 transition-transform">
              <UserCheck className="w-8 h-8" />
            </div>

            <div>
              <h2 className="text-2xl font-bold text-white group-hover:text-indigo-400 transition-colors">
                Single Customer Prediction
              </h2>
              <p className="text-slate-400 text-sm mt-3 leading-relaxed">
                Interactively enter individual customer demographic, subscription, and contract attributes to compute real-time churn probabilities, risk levels, and confidence scores.
              </p>
            </div>

            <ul className="space-y-2 text-xs text-slate-300 pt-2">
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                <span>Manual 19-Feature Interactive Input Form</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                <span>Instant Probability Gauge & Churn Class Result</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400"></span>
                <span>Stored Model Metrics & Evaluation Reports</span>
              </li>
            </ul>
          </div>

          <div className="pt-8">
            <button className="w-full py-3.5 px-6 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-bold text-sm rounded-xl shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2 group-hover:bg-indigo-500 transition">
              <span>Start Single Customer Prediction</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>

        {/* Option 2: Batch Prediction */}
        <div 
          onClick={() => onSelectMode('batch')}
          className="group relative bg-slate-900/90 border border-slate-800 hover:border-teal-500/60 rounded-3xl p-8 shadow-2xl transition-all duration-300 hover:-translate-y-2 cursor-pointer flex flex-col justify-between"
        >
          <div className="space-y-6">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-teal-500 to-emerald-400 flex items-center justify-center text-white shadow-lg shadow-teal-500/30 group-hover:scale-110 transition-transform">
              <Layers className="w-8 h-8" />
            </div>

            <div>
              <h2 className="text-2xl font-bold text-white group-hover:text-teal-400 transition-colors">
                Batch Prediction
              </h2>
              <p className="text-slate-400 text-sm mt-3 leading-relaxed">
                Upload batch CSV datasets for multi-customer inference. Generate bulk predictions, execute K-Means customer segmentation clustering, and export results to CSV or Excel.
              </p>
            </div>

            <ul className="space-y-2 text-xs text-slate-300 pt-2">
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>
                <span>Bulk CSV File Upload & Column Validation</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>
                <span>Batch Inference & Predictions Results Table</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>
                <span>K-Means Customer Segmentation Scatter Plot</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>
                <span>One-Click Export to CSV & Excel (.xlsx)</span>
              </li>
            </ul>
          </div>

          <div className="pt-8">
            <button className="w-full py-3.5 px-6 bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-500 hover:to-emerald-500 text-white font-bold text-sm rounded-xl shadow-lg shadow-teal-600/30 flex items-center justify-center space-x-2 group-hover:bg-teal-500 transition">
              <span>Start Batch Prediction</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="text-center pt-8 border-t border-slate-800/80 text-xs text-slate-500 flex items-center justify-center space-x-2">
        <ShieldCheck className="w-4 h-4 text-emerald-500" />
        <span>No preloaded sample data is shown on initial load. All prediction data comes directly from user inputs or uploaded CSV files.</span>
      </div>
    </div>
  );
}
