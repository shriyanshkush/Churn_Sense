import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import HeaderNav from './components/HeaderNav';
import SingleCustomerPrediction from './components/SingleCustomerPrediction';
import BatchPrediction from './components/BatchPrediction';

export default function App() {
  // mode: null (Landing Page), 'single' (Single Customer Prediction), 'batch' (Batch Prediction)
  const [mode, setMode] = useState(null);
  const [selectedModel, setSelectedModel] = useState('logistic_regression');

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 font-sans selection:bg-indigo-500 selection:text-white flex flex-col antialiased">
      {/* Top Header Navigation (visible when mode is selected or on landing page) */}
      <HeaderNav
        mode={mode}
        onResetMode={() => setMode(null)}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
      />

      {/* Main Dynamic View */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {mode === null && (
          <LandingPage onSelectMode={(selectedModeName) => setMode(selectedModeName)} />
        )}

        {mode === 'single' && (
          <SingleCustomerPrediction
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />
        )}

        {mode === 'batch' && (
          <BatchPrediction
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-600">
        ChurnSense AI Enterprise Platform &bull; Pretrained Model Inference Engine
      </footer>
    </div>
  );
}
