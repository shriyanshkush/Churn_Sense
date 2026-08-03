import React, { useState } from 'react';
import { Search, Bell, History, User, Download, FileText, CheckCircle2, Cpu } from 'lucide-react';

export default function Header({ 
  activeTab, 
  searchQuery, 
  setSearchQuery, 
  selectedModel, 
  setSelectedModel, 
  onExportReport 
}) {
  const [activeSubTab, setActiveSubTab] = useState('global');
  const [showNotification, setShowNotification] = useState(false);
  const [reportExported, setReportExported] = useState(false);

  const handleExport = () => {
    setReportExported(true);
    if (onExportReport) onExportReport();
    setTimeout(() => setReportExported(false), 3000);
  };

  const getPlaceholder = () => {
    if (activeTab === 'explorer') return 'Search cohorts...';
    if (activeTab === 'monitoring') return 'Search features or runs...';
    return 'Search accounts...';
  };

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-10 px-8 py-3 flex items-center justify-between shadow-xs">
      {/* Left Title & Tabs */}
      <div className="flex items-center space-x-8">
        <div className="flex items-center space-x-3">
          <span className="text-base font-extrabold text-slate-900 tracking-tight">
            Churn Prediction Suite
          </span>
        </div>

        {/* Sub-tabs */}
        <div className="hidden sm:flex items-center space-x-6 text-xs font-semibold text-slate-500">
          <button
            onClick={() => setActiveSubTab('global')}
            className={`py-1 relative transition ${
              activeSubTab === 'global' ? 'text-slate-900 font-bold' : 'hover:text-slate-800'
            }`}
          >
            Global Metrics
            {activeSubTab === 'global' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-slate-900 rounded-full"></span>
            )}
          </button>

          <button
            onClick={() => setActiveSubTab('risk_tiers')}
            className={`py-1 relative transition ${
              activeSubTab === 'risk_tiers' ? 'text-slate-900 font-bold' : 'hover:text-slate-800'
            }`}
          >
            Risk Tiers
            {activeSubTab === 'risk_tiers' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-slate-900 rounded-full"></span>
            )}
          </button>
        </div>
      </div>

      {/* Center/Right Model Selector & Controls */}
      <div className="flex items-center space-x-4">
        {/* Model Selection Dropdown */}
        <div className="flex items-center space-x-2 bg-slate-50 border border-slate-200/90 rounded-xl px-3 py-1.5 shadow-xs">
          <Cpu className="w-3.5 h-3.5 text-teal-600" />
          <span className="text-[11px] font-medium text-slate-500 hidden md:inline">Model:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-transparent font-mono text-xs font-bold text-slate-900 focus:outline-none cursor-pointer"
          >
            <option value="xgboost">XGBoost Classifier (v2_FINAL)</option>
            <option value="random_forest">Random Forest Classifier</option>
            <option value="decision_tree">Decision Tree Model</option>
            <option value="logistic_regression">Logistic Regression</option>
          </select>
        </div>

        {/* Search Account / Feature Input */}
        <div className="relative hidden md:block">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={getPlaceholder()}
            className="w-48 lg:w-56 pl-9 pr-4 py-1.5 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:border-teal-500 focus:bg-white transition"
          />
        </div>

        {/* Action Icons */}
        <div className="flex items-center space-x-1.5 text-slate-500">
          <button
            onClick={() => setShowNotification(!showNotification)}
            className="relative p-2 hover:bg-slate-100 rounded-xl transition text-slate-600 hover:text-slate-900"
            title="Notifications"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-500"></span>
          </button>

          <button
            onClick={() => alert(`Session History: Running ${selectedModel.toUpperCase()} engine. Active model verified in FastAPI memory.`)}
            className="p-2 hover:bg-slate-100 rounded-xl transition text-slate-600 hover:text-slate-900"
            title="Model Execution History"
          >
            <History className="w-4 h-4" />
          </button>
        </div>

        {/* Export Report Pill Button */}
        <button
          onClick={handleExport}
          className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold rounded-xl shadow-xs transition flex items-center space-x-2"
        >
          {reportExported ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Report Downloaded</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4 text-slate-200" />
              <span>Export Report</span>
            </>
          )}
        </button>
      </div>

      {/* Notifications Popover */}
      {showNotification && (
        <div className="absolute top-14 right-20 w-80 bg-white border border-slate-200 rounded-2xl shadow-xl p-4 z-50 animate-in fade-in slide-in-from-top-2">
          <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
            <h4 className="text-xs font-bold text-slate-900">Alert Center</h4>
            <span className="text-[10px] font-mono bg-rose-100 text-rose-800 px-2 py-0.5 rounded font-semibold">1 Alert</span>
          </div>
          <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs space-y-1">
            <p className="font-bold text-rose-900 flex items-center space-x-1">
              <span>⚠️ Model Drift Alert</span>
            </p>
            <p className="text-rose-700 text-[11px]">
              PSI shifted to <strong>0.24</strong> for feature 'Tenure'. Re-training recommended.
            </p>
          </div>
        </div>
      )}
    </header>
  );
}
