import React from 'react';
import { ArrowLeft, ShieldCheck, Sparkles, Zap } from 'lucide-react';
import { MODEL_METRICS_DATA } from './ModelPerformance';

export default function HeaderNav({ mode, onResetMode, selectedModel, setSelectedModel }) {
  const currentModelInfo = MODEL_METRICS_DATA[selectedModel] || MODEL_METRICS_DATA.logistic_regression;

  return (
    <header className="sticky top-0 z-40 bg-slate-950/90 border-b border-slate-800 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-18 py-3">
          
          {/* Brand Logo & Mode Badge */}
          <div className="flex items-center space-x-4">
            <div 
              onClick={onResetMode}
              className="flex items-center space-x-3 cursor-pointer group"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25 group-hover:scale-105 transition-transform">
                <Zap className="w-6 h-6" />
              </div>
              <div>
                <span className="text-xl font-bold tracking-tight text-white group-hover:text-indigo-400 transition-colors">
                  ChurnSense
                </span>
                <span className="ml-2 text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
                  v2.0 ML
                </span>
              </div>
            </div>

            {mode && (
              <span className="hidden sm:inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-semibold text-slate-300">
                <span className={`w-2 h-2 rounded-full ${mode === 'single' ? 'bg-indigo-400' : 'bg-teal-400'}`}></span>
                <span>{mode === 'single' ? 'Single Customer Mode' : 'Batch Prediction Mode'}</span>
              </span>
            )}
          </div>

          {/* Model Selector & Switch Mode Button */}
          <div className="flex items-center space-x-4">
            {mode && (
              <div className="flex items-center space-x-2">
                <span className="hidden lg:inline text-xs text-slate-400 font-semibold">Model:</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="bg-slate-900 border border-slate-700/80 text-slate-200 text-xs rounded-xl px-3 py-2 font-medium focus:outline-none focus:border-indigo-500 transition"
                >
                  <option value="logistic_regression">Logistic Regression (Best F1 & Recall)</option>
                  <option value="xgboost">XGBoost (Best Accuracy & Precision)</option>
                  <option value="random_forest">Random Forest Ensemble</option>
                  <option value="decision_tree">Decision Tree Baseline</option>
                </select>
              </div>
            )}

            {mode && (
              <button
                onClick={onResetMode}
                className="flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-bold text-slate-300 hover:text-white transition active:scale-95"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>Switch Mode</span>
              </button>
            )}
          </div>

        </div>

        {/* Selected Model Recommendation Bar */}
        {mode && currentModelInfo.badge && (
          <div className="pb-2.5 pt-1 flex items-center justify-between text-xs border-t border-slate-900 text-amber-300">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span className="font-semibold">{currentModelInfo.badge}</span>
              <span className="hidden md:inline text-slate-400">&bull; {currentModelInfo.recommendation}</span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
