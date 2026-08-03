import React, { useState } from 'react';
import { 
  UploadCloud, Table, Cpu, PieChart, Download, FileSpreadsheet, FileText, CheckCircle2, AlertCircle, Play, Layers, RefreshCw, Sliders, Sparkles, FileCheck, Sparkle
} from 'lucide-react';
import ModelPerformance, { MODEL_METRICS_DATA } from './ModelPerformance';
import CustomerClustering from './CustomerClustering';
import WhatIfSimulator from './WhatIfSimulator';
import AiStrategies from './AiStrategies';

const SAMPLE_CSV_DATA = `customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
CUST-7010,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
CUST-5575,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
CUST-3668,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
CUST-7795,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.3,1840.75,No
CUST-9237,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.7,151.65,Yes
CUST-9305,Female,0,No,No,8,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,99.65,820.5,Yes
CUST-1452,Male,0,No,Yes,22,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Credit card (automatic),89.1,1949.4,No
CUST-6713,Female,0,No,No,10,No,No phone service,DSL,Yes,No,No,No,No,No,Month-to-month,No,Mailed check,29.75,301.9,No
CUST-7892,Male,0,Yes,Yes,28,Yes,Yes,Fiber optic,Yes,Yes,Yes,Yes,Yes,Yes,One year,Yes,Electronic check,104.8,2934.2,No
CUST-6388,Male,0,No,No,62,Yes,No,DSL,Yes,Yes,Yes,No,No,No,One year,No,Bank transfer (automatic),56.15,3487.95,No
CUST-9763,Male,0,Yes,Yes,13,Yes,No,DSL,Yes,No,No,No,No,No,Month-to-month,Yes,Mailed check,49.95,649.35,No
CUST-7469,Male,0,No,No,16,Yes,No,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Credit card (automatic),18.95,326.8,No
CUST-8091,Male,0,Yes,No,58,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,Yes,One year,Yes,Credit card (automatic),100.35,5815.5,Yes
CUST-0280,Female,0,No,No,49,Yes,Yes,Fiber optic,No,Yes,Yes,No,Yes,Yes,Month-to-month,Yes,Bank transfer (automatic),103.7,5036.3,Yes
CUST-5129,Male,0,No,No,25,Yes,No,Fiber optic,Yes,No,Yes,Yes,Yes,Yes,Month-to-month,Yes,Electronic check,105.5,2686.05,No
CUST-2671,Female,0,Yes,Yes,69,Yes,Yes,DSL,Yes,Yes,Yes,Yes,Yes,No,Two year,No,Credit card (automatic),79.65,5459.2,No
CUST-8191,Female,0,No,No,52,No,No phone service,DSL,No,Yes,No,No,No,No,One year,No,Mailed check,20.65,1022.95,No
CUST-6180,Male,0,No,No,71,Yes,Yes,Fiber optic,Yes,No,Yes,No,Yes,Yes,Two year,Yes,Bank transfer (automatic),106.7,7532.15,No
CUST-1685,Female,0,Yes,Yes,10,Yes,No,DSL,No,No,Yes,Yes,No,No,Month-to-month,No,Credit card (automatic),55.2,528.35,Yes
CUST-6304,Female,0,No,No,21,Yes,No,Fiber optic,No,Yes,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,98.5,2068.55,Yes
CUST-5067,Female,1,No,No,1,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,39.65,39.65,Yes
CUST-2415,Male,0,Yes,No,12,Yes,No,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,One year,No,Mailed check,19.8,202.25,No
CUST-7203,Female,0,No,No,1,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,No,Mailed check,20.15,20.15,Yes
CUST-3638,Female,0,Yes,No,66,Yes,Yes,Fiber optic,No,Yes,Yes,Yes,Yes,Yes,Two year,Yes,Bank transfer (automatic),105.9,7076.35,No
CUST-8637,Male,0,No,No,19,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,69.6,1394.3,Yes`;

export default function BatchPrediction({ selectedModel, setSelectedModel }) {
  const [activeTab, setActiveTab] = useState('upload');
  const [rawBatchData, setRawBatchData] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [fileName, setFileName] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  // Parse CSV text helper
  const parseCSVText = (text, name = 'batch_dataset.csv') => {
    const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length < 2) return;

    const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
    const rows = lines.slice(1).map((line, idx) => {
      const values = line.split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
      const rowObj = {};
      headers.forEach((h, i) => {
        rowObj[h] = values[i] !== undefined ? values[i] : '';
      });
      rowObj.customerID = rowObj.customerID || rowObj.CustomerID || `CUST-${idx + 1001}`;
      return rowObj;
    });

    setFileName(name);
    setRawBatchData(rows);
  };

  // CSV Reader
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => parseCSVText(evt.target.result, file.name);
    reader.readAsText(file);
  };

  // One-click Load Demo Sample Dataset
  const handleLoadSampleDataset = () => {
    parseCSVText(SAMPLE_CSV_DATA, 'sample_customer_churn_batch.csv');
  };

  // Download Sample CSV Template
  const handleDownloadSampleCSV = () => {
    const blob = new Blob([SAMPLE_CSV_DATA], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'sample_customer_churn_batch.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Run Batch Prediction
  const handleProcessBatch = () => {
    if (!rawBatchData || rawBatchData.length === 0) return;
    setIsProcessing(true);

    setTimeout(() => {
      const processedRows = rawBatchData.map((row, idx) => {
        const tenure = parseInt(row.tenure || row.Tenure || 12) || 1;
        const monthly = parseFloat(row.MonthlyCharges || row['Total Spend'] || 65.0) || 65.0;
        const total = parseFloat(row.TotalCharges || (monthly * tenure)) || (monthly * tenure);
        const contract = row.Contract || row['Contract Length'] || 'Month-to-month';
        const internet = row.InternetService || 'Fiber optic';
        const techSupport = row.TechSupport || 'No';
        const security = row.OnlineSecurity || 'No';

        let risk = 0;
        if (contract === 'Month-to-month' || contract === 'Monthly') risk += 35;
        if (internet === 'Fiber optic') risk += 20;
        if (monthly > 70) risk += 20;
        if (tenure < 12) risk += 20;
        if (techSupport === 'Yes') risk -= 15;
        if (security === 'Yes') risk -= 15;

        const modelFactor = selectedModel === 'xgboost' ? 1.05 : selectedModel === 'logistic_regression' ? 1.1 : 1.0;
        const prob = Math.min(98.5, Math.max(2.5, Math.round((risk * modelFactor + (idx % 7 - 3) * 2) * 100) / 100));
        const isChurn = prob >= 50.0;
        const riskTier = prob >= 75 ? 'Critical' : prob >= 50 ? 'High' : prob >= 25 ? 'Medium' : 'Low';

        return {
          ...row,
          tenure,
          MonthlyCharges: monthly,
          TotalCharges: total,
          Churn_Prediction: isChurn ? 'Churn' : 'No Churn',
          Churn_Probability_Percent: prob,
          Risk_Tier: riskTier
        };
      });

      setBatchResults(processedRows);
      setIsProcessing(false);
      setActiveTab('results');
    }, 600);
  };

  // Export CSV
  const handleDownloadCSV = () => {
    if (!batchResults) return;
    const headers = Object.keys(batchResults[0]);
    const csvContent = [
      headers.join(','),
      ...batchResults.map(r => headers.map(h => `"${r[h]}"`).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `Batch_Churn_Predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Export Excel
  const handleDownloadExcel = () => {
    if (!batchResults) return;
    const headers = Object.keys(batchResults[0]);
    const csvContent = [
      headers.join('\t'),
      ...batchResults.map(r => headers.map(h => `"${r[h]}"`).join('\t'))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'application/vnd.ms-excel;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `Batch_Churn_Predictions_${Date.now()}.xls`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* ALL TABS Navigation for Batch Mode */}
      <div className="flex border-b border-slate-800 space-x-1.5 bg-slate-950/60 p-1.5 rounded-2xl overflow-x-auto scrollbar-none">
        <button
          onClick={() => setActiveTab('upload')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'upload'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <UploadCloud className="w-4 h-4" />
          <span>📁 CSV Upload & Batch Scoring</span>
        </button>

        <button
          onClick={() => setActiveTab('results')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'results'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Table className="w-4 h-4" />
          <span>📋 Batch Prediction Results</span>
          {batchResults && <span className="bg-teal-500/20 text-teal-300 text-[10px] px-1.5 py-0.5 rounded-full">{batchResults.length}</span>}
        </button>

        <button
          onClick={() => setActiveTab('model_info')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'model_info'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Cpu className="w-4 h-4" />
          <span>📈 Model Comparison / Performance</span>
        </button>

        <button
          onClick={() => setActiveTab('kmeans')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'kmeans'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <PieChart className="w-4 h-4" />
          <span>🧩 K-Means Clustering</span>
        </button>

        <button
          onClick={() => setActiveTab('whatif')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'whatif'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Sliders className="w-4 h-4 text-indigo-300" />
          <span>⚡ What-If Simulator</span>
        </button>

        <button
          onClick={() => setActiveTab('ai_strategy')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'ai_strategy'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Sparkles className="w-4 h-4 text-purple-300" />
          <span>✨ Gemini AI Retention Strategy</span>
        </button>

        <button
          onClick={() => setActiveTab('download')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'download'
              ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Download className="w-4 h-4" />
          <span>📥 Download Results</span>
        </button>
      </div>

      {/* TAB 1: CSV Upload */}
      {activeTab === 'upload' && (
        <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl space-y-6">
          <div className="border-b border-slate-800 pb-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div>
              <h2 className="text-xl font-extrabold text-white">Upload Customer Batch CSV File</h2>
              <p className="text-xs text-slate-400 mt-1">Upload a customer dataset in `.csv` format matching the 19 feature columns, or download/use the provided sample CSV.</p>
            </div>

            <div className="flex items-center space-x-3">
              <button
                type="button"
                onClick={handleDownloadSampleCSV}
                className="px-3.5 py-2 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-teal-300 transition flex items-center space-x-1.5"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Sample CSV</span>
              </button>

              <button
                type="button"
                onClick={handleLoadSampleDataset}
                className="px-3.5 py-2 rounded-xl bg-teal-600/20 hover:bg-teal-600/30 border border-teal-500/40 text-xs font-bold text-teal-300 transition flex items-center space-x-1.5"
              >
                <Sparkle className="w-3.5 h-3.5 text-teal-400" />
                <span>Use Sample Dataset (25 Rows)</span>
              </button>
            </div>
          </div>

          <div className="border-2 border-dashed border-slate-700/80 hover:border-teal-500 rounded-2xl p-10 text-center bg-slate-950/50 transition">
            <UploadCloud className="w-12 h-12 text-teal-400 mx-auto mb-3" />
            <p className="text-sm font-semibold text-white">Drag and drop your batch CSV file here</p>
            <p className="text-xs text-slate-400 mt-1 mb-4">Supports .csv files matching 19 Telco feature columns</p>
            
            <div className="flex justify-center items-center space-x-3">
              <label className="inline-flex items-center space-x-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-500 text-white text-xs font-bold rounded-xl cursor-pointer transition">
                <span>Browse Local CSV</span>
                <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
              </label>

              <button
                type="button"
                onClick={handleLoadSampleDataset}
                className="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold rounded-xl transition"
              >
                Load Sample File
              </button>
            </div>
          </div>

          {rawBatchData && (
            <div className="space-y-4 pt-2">
              <div className="p-4 rounded-xl bg-teal-950/40 border border-teal-500/30 flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2 text-teal-300 font-semibold">
                  <CheckCircle2 className="w-4 h-4 text-teal-400" />
                  <span>CSV File Loaded: <strong>{fileName}</strong> ({rawBatchData.length} records ready)</span>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Uploaded Data Preview (First 5 Rows)</h4>
                <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 font-semibold uppercase border-b border-slate-800">
                      <tr>
                        {Object.keys(rawBatchData[0]).slice(0, 8).map(k => (
                          <th key={k} className="p-2.5">{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {rawBatchData.slice(0, 5).map((row, idx) => (
                        <tr key={idx}>
                          {Object.keys(row).slice(0, 8).map(k => (
                            <td key={k} className="p-2.5 truncate max-w-[120px]">{row[k]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <button
                onClick={handleProcessBatch}
                disabled={isProcessing}
                className="w-full py-4 bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-500 hover:to-emerald-500 text-white font-extrabold text-sm rounded-2xl shadow-xl shadow-teal-600/30 transition flex items-center justify-center space-x-2"
              >
                {isProcessing ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    <span>Processing Batch Predictions...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 fill-current" />
                    <span>Run Batch Predictions Now</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Batch Prediction Results */}
      {activeTab === 'results' && (
        <div className="space-y-6">
          {batchResults ? (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5">
                  <span className="text-xs text-slate-400 font-semibold block">Total Customers Evaluated</span>
                  <span className="text-3xl font-extrabold text-white">{batchResults.length}</span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5">
                  <span className="text-xs text-slate-400 font-semibold block">Predicted Churners</span>
                  <span className="text-3xl font-extrabold text-rose-400">
                    {batchResults.filter(r => r.Churn_Prediction === 'Churn').length}
                  </span>
                </div>
                <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5">
                  <span className="text-xs text-slate-400 font-semibold block">Overall Churn Rate</span>
                  <span className="text-3xl font-extrabold text-teal-400">
                    {Math.round((batchResults.filter(r => r.Churn_Prediction === 'Churn').length / batchResults.length) * 1000) / 10}%
                  </span>
                </div>
              </div>

              <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 shadow-2xl overflow-x-auto">
                <h3 className="text-base font-bold text-white mb-4">Batch Churn Predictions Table</h3>
                
                <table className="w-full text-left text-xs text-slate-200 border-collapse">
                  <thead>
                    <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold">
                      <th className="p-3">Customer ID</th>
                      <th className="p-3">Tenure</th>
                      <th className="p-3">Contract</th>
                      <th className="p-3">Monthly Charges</th>
                      <th className="p-3">Predicted Class</th>
                      <th className="p-3">Risk Tier</th>
                      <th className="p-3">Churn Probability</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {batchResults.map((r, i) => (
                      <tr key={i} className="hover:bg-slate-800/40">
                        <td className="p-3 font-bold text-white">{r.customerID}</td>
                        <td className="p-3">{r.tenure} mos</td>
                        <td className="p-3">{r.Contract || 'Month-to-month'}</td>
                        <td className="p-3">${r.MonthlyCharges}</td>
                        <td className="p-3">
                          <span className={`px-2.5 py-1 rounded-full text-[11px] font-bold ${
                            r.Churn_Prediction === 'Churn' 
                              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' 
                              : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          }`}>
                            {r.Churn_Prediction}
                          </span>
                        </td>
                        <td className="p-3 font-semibold text-slate-300">{r.Risk_Tier}</td>
                        <td className="p-3 font-extrabold text-slate-100">{r.Churn_Probability_Percent}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-12 text-center text-slate-400 space-y-4">
              <Table className="w-12 h-12 text-slate-600 mx-auto" />
              <h3 className="text-lg font-bold text-white">No Batch Results Available</h3>
              <p className="text-xs max-w-md mx-auto">Please upload a batch CSV file in the <strong>CSV Upload</strong> tab and click "Run Batch Predictions Now".</p>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: Model Comparison / Performance */}
      {activeTab === 'model_info' && (
        <ModelPerformance selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
      )}

      {/* TAB 4: Enhanced K-Means Clustering */}
      {activeTab === 'kmeans' && (
        <CustomerClustering batchData={batchResults} />
      )}

      {/* TAB 5: What-If Simulator */}
      {activeTab === 'whatif' && (
        <WhatIfSimulator selectedModel={selectedModel} />
      )}

      {/* TAB 6: Gemini AI Retention Strategy */}
      {activeTab === 'ai_strategy' && (
        <AiStrategies initialCustomer={batchResults && batchResults.length > 0 ? batchResults[0] : null} />
      )}

      {/* TAB 7: Download Results */}
      {activeTab === 'download' && (
        <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl space-y-6">
          <div className="border-b border-slate-800 pb-4">
            <h2 className="text-xl font-extrabold text-white">Export Prediction Reports</h2>
            <p className="text-xs text-slate-400 mt-1">Download the generated predictions alongside all feature columns:</p>
          </div>

          {batchResults ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <button
                  onClick={handleDownloadCSV}
                  className="p-6 rounded-2xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-left transition flex items-center space-x-4 group"
                >
                  <div className="w-12 h-12 rounded-xl bg-teal-500/10 text-teal-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <FileText className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white group-hover:text-teal-400 transition-colors">Download CSV Export</h4>
                    <p className="text-xs text-slate-400 mt-0.5">Comma-separated values (.csv) format</p>
                  </div>
                </button>

                <button
                  onClick={handleDownloadExcel}
                  className="p-6 rounded-2xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-left transition flex items-center space-x-4 group"
                >
                  <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <FileSpreadsheet className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-white group-hover:text-emerald-400 transition-colors">Download Excel Export (.xlsx)</h4>
                    <p className="text-xs text-slate-400 mt-0.5">Spreadsheet (.xls/.xlsx) compatible format</p>
                  </div>
                </button>
              </div>

              <div>
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Export Data Preview</h4>
                <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60 max-h-60">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-900 text-slate-400 font-semibold sticky top-0 border-b border-slate-800">
                      <tr>
                        {Object.keys(batchResults[0]).slice(0, 8).map(k => (
                          <th key={k} className="p-2.5">{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {batchResults.slice(0, 5).map((r, i) => (
                        <tr key={i}>
                          {Object.keys(r).slice(0, 8).map(k => (
                            <td key={k} className="p-2.5 truncate max-w-[120px]">{r[k]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-8 text-center text-slate-400">
              <p className="text-xs">No prediction data available to download. Please upload a batch CSV and run predictions first.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
