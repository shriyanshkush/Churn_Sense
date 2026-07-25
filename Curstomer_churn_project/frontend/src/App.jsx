import React, { useState } from 'react';
import Navbar from './components/Navbar';
import CustomerAnalyzer from './components/CustomerAnalyzer';
import Overview from './components/Overview';
import CustomerClustering from './components/CustomerClustering';
import BatchUpload from './components/BatchUpload';
import ModelInsights from './components/ModelInsights';
import DriftMonitoring from './components/DriftMonitoring';

export default function App() {
  const [activeTab, setActiveTab] = useState('analyzer');
  const [selectedModel, setSelectedModel] = useState('xgboost');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
      />

      {/* Main Content Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {activeTab === 'analyzer' && <CustomerAnalyzer selectedModel={selectedModel} />}
        {activeTab === 'overview' && <Overview onNavigate={setActiveTab} />}
        {activeTab === 'clustering' && <CustomerClustering />}
        {activeTab === 'batch' && <BatchUpload selectedModel={selectedModel} />}
        {activeTab === 'insights' && <ModelInsights />}
        {activeTab === 'drift' && <DriftMonitoring />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <p>ChurnSense AI v2.0 Enterprise &bull; FastAPI Backend &bull; React Dashboard &bull; Streamlit Demo Mode</p>
      </footer>
    </div>
  );
}
