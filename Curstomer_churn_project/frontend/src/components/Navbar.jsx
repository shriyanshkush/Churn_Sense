import React from 'react';
import { 
  Sliders, BarChart3, Users, UploadCloud, Cpu, Activity, ShieldCheck, Zap
} from 'lucide-react';

const navItems = [
  { id: 'analyzer', label: 'Customer Predictor & Analyzer', icon: Sliders },
  { id: 'overview', label: 'Executive Analytics', icon: BarChart3 },
  { id: 'clustering', label: 'Customer Segmentation', icon: Users },
  { id: 'batch', label: 'CSV Batch Predictor', icon: UploadCloud },
  { id: 'insights', label: 'Model Insights (SHAP)', icon: Cpu },
  { id: 'drift', label: 'MLOps Drift Monitor', icon: Activity },
];

export default function Navbar({ activeTab, setActiveTab, selectedModel, setSelectedModel }) {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 rounded-none bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Brand Logo */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('analyzer')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/25">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight gradient-text">ChurnSense</span>
              <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">v2.0 Enterprise</span>
            </div>
          </div>

          {/* Model Selector */}
          <div className="hidden md:flex items-center space-x-3">
            <span className="text-xs text-slate-400 font-medium">Model Engine:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-slate-900 border border-slate-700/70 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-500"
            >
              <option value="xgboost">XGBoost Classifier (Best AUC)</option>
              <option value="random_forest">Random Forest Ensemble</option>
              <option value="decision_tree">Decision Tree Baseline</option>
              <option value="logistic_regression">Logistic Regression Baseline</option>
            </select>
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs border border-emerald-500/20">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>FastAPI Connected</span>
            </div>
          </div>

        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto py-2 scrollbar-none border-t border-slate-900">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-md shadow-indigo-600/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
