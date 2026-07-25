import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, FileSpreadsheet, Download, Search, CheckCircle, AlertCircle, Sparkles, Copy, Layers } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

export default function BatchUpload({ selectedModel }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchResults, setBatchResults] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [batchStrategy, setBatchStrategy] = useState(null);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDownloadSampleCsv = async () => {
    try {
      const response = await axios.get('/api/v1/sample-csv', { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'ChurnSense_Sample_Dataset.csv');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error("Error downloading sample CSV:", err);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(`/api/v1/batch-predict?selected_model=${selectedModel}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setBatchResults(res.data);
    } catch (err) {
      console.error("Batch Upload Error:", err);
      alert("Error processing batch CSV. Please ensure valid customer CSV structure.");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateBatchStrategy = async () => {
    if (!batchResults || !batchResults.predictions) return;
    try {
      setStrategyLoading(true);
      // Find dominant high-risk cluster in batch
      const clusterCounts = {};
      batchResults.predictions.forEach(p => {
        clusterCounts[p.cluster_label] = (clusterCounts[p.cluster_label] || 0) + 1;
      });

      const dominantCluster = Object.keys(clusterCounts).reduce((a, b) => clusterCounts[a] > clusterCounts[b] ? a : b, "High-Risk Price-Sensitive");

      const res = await axios.post('/api/v1/ai-strategy', {
        gender: "Mixed Batch",
        tenure: 12,
        Contract: "Month-to-month",
        MonthlyCharges: 75.0,
        churn_probability: batchResults.overall_churn_rate,
        risk_tier: batchResults.overall_churn_rate >= 50 ? "High" : "Medium",
        cluster_label: dominantCluster,
        top_shap_drivers: ["Contract", "MonthlyCharges", "TechSupport"],
        clv_estimate: batchResults.average_clv,
        custom_notes: `Batch dataset upload containing ${batchResults.total_rows} customer records with ${batchResults.high_risk_customers} at-risk customers.`
      });
      setBatchStrategy(res.data);
    } catch (err) {
      console.error("Batch Strategy Error:", err);
    } finally {
      setStrategyLoading(false);
    }
  };

  const handleDownloadCsv = () => {
    if (!batchResults || !batchResults.predictions) return;
    const headers = ['ID', 'Gender', 'Tenure', 'Contract', 'Monthly Charges', 'Churn Prob (%)', 'Risk Tier', 'CLV ($)', 'Cluster Label', 'Top Driver'];
    const rows = batchResults.predictions.map(p => [
      p.id, p.gender, p.tenure, p.contract, p.monthly_charges, p.churn_probability, p.risk_tier, p.clv_estimate, p.cluster_label, p.top_risk_driver
    ]);

    const csvContent = "data:text/csv;charset=utf-8," 
      + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `ChurnSense_Batch_Predictions_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Compute cluster breakdown pie chart data for batch
  const getBatchClusterData = () => {
    if (!batchResults || !batchResults.predictions) return [];
    const counts = {};
    batchResults.predictions.forEach(p => {
      counts[p.cluster_label] = (counts[p.cluster_label] || 0) + 1;
    });
    const colors = {
      "High-Risk Price-Sensitive": "#ef4444",
      "Stable High-Value": "#10b981",
      "New & Vulnerable": "#f59e0b",
      "Loyal Low-Engagement": "#8b5cf6"
    };
    return Object.keys(counts).map(lbl => ({
      name: lbl,
      value: counts[lbl],
      color: colors[lbl] || "#6366f1"
    }));
  };

  const filteredPredictions = batchResults?.predictions?.filter(p => {
    const matchesSearch = p.id.toLowerCase().includes(searchTerm.toLowerCase()) || p.cluster_label.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = riskFilter === 'all' || p.risk_tier === riskFilter;
    return matchesSearch && matchesRisk;
  }) || [];

  const getBadgeStyle = (tier) => {
    if (tier === 'Critical') return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
    if (tier === 'High') return 'bg-red-500/20 text-red-300 border-red-500/40';
    if (tier === 'Medium') return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
    return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
  };

  return (
    <div className="space-y-6">
      
      {/* Header Banner */}
      <div className="glass-panel p-6 border-indigo-500/20">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
              <UploadCloud className="w-5 h-5 text-indigo-400" />
              <span>Multi-Customer CSV Batch Upload, Clustering & Campaign Engine</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Upload multi-person CSV datasets to run automated batch predictions, GMM cluster segmentation, CLV estimations, and batch AI retention campaigns.
            </p>
          </div>

          {/* Sample CSV Download Button */}
          <button
            onClick={handleDownloadSampleCsv}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-indigo-300 text-xs font-semibold rounded-xl transition border border-indigo-500/30 flex items-center space-x-2 whitespace-nowrap"
          >
            <Download className="w-4 h-4 text-indigo-400" />
            <span>Download Sample CSV Dataset</span>
          </button>
        </div>
      </div>

      {/* Upload Dropzone & Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Dropzone */}
        <div className="glass-panel p-6 border-dashed border-2 border-slate-700 hover:border-indigo-500/50 transition text-center flex flex-col items-center justify-center space-y-3">
          <FileSpreadsheet className="w-10 h-10 text-indigo-400" />
          <div>
            <p className="text-xs font-semibold text-white">Upload Multi-Person Customer CSV</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Supports Telco Churn CSV or standard customer schema</p>
          </div>

          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
            id="csv-upload-input"
          />
          <label
            htmlFor="csv-upload-input"
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs rounded-xl font-medium cursor-pointer border border-slate-700"
          >
            {file ? file.name : 'Select CSV File'}
          </label>

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-bold rounded-xl transition shadow-lg shadow-indigo-600/30 disabled:opacity-50"
          >
            {loading ? 'Processing Batch CSV...' : 'Run Batch Predictions & Clustering'}
          </button>
        </div>

        {/* Batch Summary Stats */}
        <div className="lg:col-span-2 glass-panel p-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="glass-card p-4 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Total Customers</span>
            <span className="text-2xl font-bold text-white font-display mt-1">
              {batchResults ? batchResults.total_rows : '--'}
            </span>
          </div>

          <div className="glass-card p-4 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">High Risk Count</span>
            <span className="text-2xl font-bold text-rose-400 font-display mt-1">
              {batchResults ? batchResults.high_risk_customers : '--'}
            </span>
          </div>

          <div className="glass-card p-4 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Batch Churn Rate</span>
            <span className="text-2xl font-bold text-amber-400 font-display mt-1">
              {batchResults ? `${batchResults.overall_churn_rate}%` : '--'}
            </span>
          </div>

          <div className="glass-card p-4 border border-slate-800">
            <span className="text-[11px] text-slate-400 block">Average Preserved CLV</span>
            <span className="text-2xl font-bold text-emerald-400 font-display mt-1">
              {batchResults ? `$${batchResults.average_clv.toLocaleString()}` : '--'}
            </span>
          </div>
        </div>

      </div>

      {/* Multi-Customer Clustering Breakdown & Batch Campaign Strategy */}
      {batchResults && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Batch Clustering Distribution */}
          <div className="glass-panel p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Batch Cluster Segmentation Distribution</h3>
              <p className="text-xs text-slate-400 mt-0.5">GMM Soft Cluster assignment across uploaded batch</p>
            </div>

            <div className="h-48 my-2">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={getBatchClusterData()}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={75}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {getBatchClusterData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="space-y-1.5 text-xs">
              {getBatchClusterData().map((item, idx) => (
                <div key={idx} className="flex items-center justify-between text-slate-300">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                    <span>{item.name}</span>
                  </div>
                  <strong className="text-white">{item.value} ({Math.round((item.value / batchResults.total_rows)*100)}%)</strong>
                </div>
              ))}
            </div>
          </div>

          {/* Batch AI Retention Campaign Strategy */}
          <div className="lg:col-span-2 glass-panel p-6 space-y-4 border-indigo-500/30 bg-gradient-to-r from-indigo-950/30 to-purple-950/20 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-purple-400" />
                    <span>Batch Campaign Retention Strategy Generator</span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">Generates targeted campaign offers for the highest risk customer segment in this batch.</p>
                </div>
                
                <button
                  onClick={handleGenerateBatchStrategy}
                  disabled={strategyLoading}
                  className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-semibold rounded-xl transition shadow-md shadow-indigo-600/30 disabled:opacity-50"
                >
                  {strategyLoading ? 'Generating Campaign...' : 'Generate Batch Campaign Strategy'}
                </button>
              </div>

              {batchStrategy && (
                <div className="mt-4 p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3 text-xs">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="font-bold text-indigo-300 text-sm">{batchStrategy.strategy_title}</span>
                    <div className="flex items-center space-x-2 bg-slate-800 px-3 py-1 rounded-lg">
                      <span className="text-slate-400">Campaign Code:</span>
                      <span className="font-mono text-purple-300 font-bold">{batchStrategy.discount_coupon}</span>
                    </div>
                  </div>

                  <p className="text-slate-300 leading-relaxed">{batchStrategy.executive_summary}</p>
                  <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-200">
                    <strong>Batch Retention Offer:</strong> {batchStrategy.retention_offer}
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center pt-1">
                    <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Est Campaign Cost</span>
                      <span className="font-bold text-slate-200">${batchStrategy.estimated_cost}</span>
                    </div>
                    <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Risk Reduction</span>
                      <span className="font-bold text-emerald-400">-{batchStrategy.expected_risk_reduction}%</span>
                    </div>
                    <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block text-[10px]">Estimated Campaign ROI</span>
                      <span className="font-bold text-indigo-400">+{batchStrategy.estimated_roi}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      )}

      {/* Predictions Table */}
      {batchResults && (
        <div className="glass-panel p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            
            {/* Search & Filter */}
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search ID or Cluster..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-xl pl-9 pr-3 py-2 focus:outline-none focus:border-indigo-500 w-48"
                />
              </div>

              <select
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
              >
                <option value="all">All Risk Tiers</option>
                <option value="Critical">Critical Risk (>85%)</option>
                <option value="High">High Risk (60-85%)</option>
                <option value="Medium">Medium Risk (30-60%)</option>
                <option value="Low">Low Risk (&lt;30%)</option>
              </select>
            </div>

            {/* Export CSV Button */}
            <button
              onClick={handleDownloadCsv}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-xl transition shadow-md flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Export Predictions CSV</span>
            </button>
          </div>

          {/* Table */}
          <div className="overflow-x-auto border border-slate-800 rounded-xl">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900/90 text-slate-400 uppercase font-semibold text-[10px] border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Customer ID</th>
                  <th className="px-4 py-3">Contract</th>
                  <th className="px-4 py-3">Monthly ($)</th>
                  <th className="px-4 py-3">Churn Prob</th>
                  <th className="px-4 py-3">Risk Tier</th>
                  <th className="px-4 py-3">Estimated CLV</th>
                  <th className="px-4 py-3">Cluster Label</th>
                  <th className="px-4 py-3">Top Risk Driver</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                {filteredPredictions.slice(0, 50).map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/60 transition">
                    <td className="px-4 py-2.5 font-mono text-slate-200">{row.id}</td>
                    <td className="px-4 py-2.5">{row.contract}</td>
                    <td className="px-4 py-2.5">${row.monthly_charges}</td>
                    <td className="px-4 py-2.5 font-bold font-mono">
                      <span className={row.churn_probability >= 50 ? 'text-rose-400' : 'text-emerald-400'}>
                        {row.churn_probability}%
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${getBadgeStyle(row.risk_tier)}`}>
                        {row.risk_tier}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-bold text-indigo-300">${row.clv_estimate.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-purple-300 font-medium">{row.cluster_label}</td>
                    <td className="px-4 py-2.5 text-slate-400 font-mono">{row.top_risk_driver}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}
