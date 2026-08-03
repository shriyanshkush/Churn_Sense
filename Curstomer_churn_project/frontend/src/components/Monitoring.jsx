import React, { useState } from 'react';
import axios from 'axios';
import { 
  Filter, 
  Calendar, 
  UploadCloud, 
  Download, 
  ExternalLink, 
  AlertCircle, 
  RefreshCw, 
  CheckCircle2, 
  FileText,
  ChevronDown
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function Monitoring({ onRunComplete }) {
  const [selectedFeature, setSelectedFeature] = useState('usage_freq');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  // Global SHAP Feature Importance data matching Reference 4
  const shapData = [
    { feature: 'usage_velocity', score: 0.421, pct: '84%' },
    { feature: 'contract_tenure', score: 0.312, pct: '65%' },
    { feature: 'support_tickets', score: 0.285, pct: '58%' },
    { feature: 'avg_session_time', score: 0.194, pct: '40%' },
    { feature: 'mrr_delta_pct', score: 0.120, pct: '25%' },
  ];

  // Feature Distribution Shift (Baseline vs Current Q1..Q5)
  const shiftDistributionData = [
    { quantile: 'Q1', baseline: 40, current: 28 },
    { quantile: 'Q2', baseline: 85, current: 62 },
    { quantile: 'Q3', baseline: 65, current: 75 },
    { quantile: 'Q4', baseline: 50, current: 42 },
    { quantile: 'Q5', baseline: 25, current: 35 },
  ];

  // Recent Batch Runs table data
  const [batchRuns, setBatchRuns] = useState([
    { id: '#RUN-88492', timestamp: 'Oct 24, 2023 14:20', size: '12,402 rows', status: 'COMPLETED' },
    { id: '#RUN-88490', timestamp: 'Oct 24, 2023 10:05', size: '4,500 rows', status: 'COMPLETED' },
    { id: '#RUN-88481', timestamp: 'Oct 23, 2023 18:44', size: '1,200 rows', status: 'FAILED' },
  ]);

  const handleFileUpload = async (file) => {
    if (!file) return;
    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post('/api/v1/batch-predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadResult(res.data);

      // Add new run to table
      const newRun = {
        id: `#RUN-${Math.floor(10000 + Math.random() * 90000)}`,
        timestamp: 'Just now',
        size: `${res.data.total_rows} rows`,
        status: 'COMPLETED'
      };
      setBatchRuns([newRun, ...batchRuns]);

      if (onRunComplete) onRunComplete(res.data);
    } catch (err) {
      console.error("Batch Scoring Upload Error:", err);
      alert("Error processing batch scoring CSV.");
    } fontFinally: {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300 pb-12">
      
      {/* 1. Global SHAP Feature Importance Card (Top Full Width) */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900">Global SHAP Feature Importance</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Contribution of top 10 features to the churn model output.
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs font-semibold">
            <button 
              onClick={() => alert("Filtering feature importance by customer segment")}
              className="bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-xl border border-slate-200 transition flex items-center space-x-1.5"
            >
              <Filter className="w-3.5 h-3.5 text-slate-500" />
              <span>Filter Segments</span>
            </button>

            <button 
              onClick={() => alert("Time horizon set to Last 30 Days (L30D)")}
              className="bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded-xl border border-slate-200 transition flex items-center space-x-1.5"
            >
              <Calendar className="w-3.5 h-3.5 text-slate-500" />
              <span>L30D</span>
            </button>
          </div>
        </div>

        {/* Custom Horizontal Bar List matching Reference Screen 4 */}
        <div className="space-y-3.5 pt-1">
          {shapData.map((item, idx) => (
            <div key={idx} className="flex items-center space-x-4 text-xs">
              <span className="w-36 font-mono text-slate-700 font-semibold truncate">{item.feature}</span>
              <div className="flex-1 bg-slate-100 rounded-full h-5 overflow-hidden relative">
                <div 
                  className="bg-[#047857] h-full rounded-full transition-all duration-500" 
                  style={{ width: item.pct }}
                ></div>
              </div>
              <span className="w-12 font-mono font-bold text-slate-900 text-right">{item.score.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 2. Middle Row: Stability Index PSI & Feature Distribution Shift */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Card: Stability Index (PSI) */}
        <div className="lg:col-span-5 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between space-y-6">
          <div>
            <h3 className="text-base font-bold text-slate-900">Stability Index (PSI)</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Population Stability Index over training baseline.
            </p>

            <div className="space-y-3 mt-6">
              {/* GLOBAL DRIFT */}
              <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3.5 flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider block">GLOBAL DRIFT</span>
                  <span className="text-xl font-extrabold font-mono text-slate-900">0.082</span>
                </div>
                <span className="badge-healthy">STABLE</span>
              </div>

              {/* USAGE_FREQ */}
              <div className="bg-rose-50/60 border border-rose-200/80 rounded-xl p-3.5 flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono font-bold text-rose-500 uppercase tracking-wider block">USAGE_FREQ</span>
                  <span className="text-xl font-extrabold font-mono text-rose-900">0.241</span>
                </div>
                <span className="badge-drifted">DRIFTED</span>
              </div>

              {/* GEO_REGION */}
              <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3.5 flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono font-bold text-slate-400 uppercase tracking-wider block">GEO_REGION</span>
                  <span className="text-xl font-extrabold font-mono text-slate-900">0.045</span>
                </div>
                <span className="badge-healthy">STABLE</span>
              </div>
            </div>
          </div>

          <button
            onClick={() => alert("Full PSI Feature Matrix: baseline vs target distributions fully calibrated.")}
            className="w-full py-2.5 bg-white border border-slate-300 hover:bg-slate-50 text-slate-800 text-xs font-bold rounded-xl transition shadow-xs"
          >
            View Detailed Metrics
          </button>
        </div>

        {/* Right Card: Feature Distribution Shift */}
        <div className="lg:col-span-7 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-900">Feature Distribution Shift</h3>

              <div className="flex items-center space-x-4 text-xs font-semibold text-slate-600">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-300"></span>
                    <span>Baseline</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#047857]"></span>
                    <span>Current</span>
                  </div>
                </div>

                <select
                  value={selectedFeature}
                  onChange={(e) => setSelectedFeature(e.target.value)}
                  className="bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-800 font-mono text-xs focus:outline-none"
                >
                  <option value="usage_freq">usage_freq</option>
                  <option value="tenure">tenure</option>
                  <option value="MonthlyCharges">MonthlyCharges</option>
                </select>
              </div>
            </div>

            {/* Distribution Bar Chart */}
            <div className="h-48 w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={shiftDistributionData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="quantile" stroke="#94a3b8" fontSize={11} />
                  <YAxis stroke="#94a3b8" fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="baseline" fill="#e2e8f0" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="current" fill="#047857" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <p className="text-xs text-slate-500 border-t border-slate-100 pt-3">
            Comparing <code className="font-mono text-slate-800 font-bold">{selectedFeature}</code> distribution across the last 7 days vs model training dataset.
          </p>
        </div>

      </div>

      {/* 3. Bottom Row: Drop Batch File & Recent Batch Runs */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Drop Batch File (4 cols) */}
        <div className="lg:col-span-4 bg-slate-50/70 border-2 border-dashed border-slate-300 rounded-2xl p-6 text-center flex flex-col items-center justify-center space-y-4 hover:border-teal-500 transition">
          <div className="w-12 h-12 rounded-2xl bg-white border border-slate-200 flex items-center justify-center shadow-xs">
            <UploadCloud className="w-6 h-6 text-teal-600" />
          </div>

          <div>
            <h4 className="text-sm font-bold text-slate-900">Drop Batch File</h4>
            <p className="text-xs text-slate-500 mt-1">
              Support .csv, .parquet files up to 2GB
            </p>
          </div>

          <label className="cursor-pointer px-5 py-2.5 bg-[#047857] hover:bg-[#065F46] text-white text-xs font-bold rounded-xl shadow-xs transition">
            <span>{uploading ? 'Processing File...' : 'Select Files'}</span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => handleFileUpload(e.target.files[0])}
              className="hidden"
            />
          </label>
        </div>

        {/* Recent Batch Runs Table (8 cols) */}
        <div className="lg:col-span-8 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="text-base font-bold text-slate-900">Recent Batch Runs</h3>
            <button 
              onClick={() => alert("Viewing full logs for all 48 historical batch scoring pipelines.")}
              className="text-xs font-bold text-teal-700 hover:text-teal-800 flex items-center space-x-1"
            >
              <RefreshCw className="w-3.5 h-3.5 mr-1" />
              <span>View All Runs</span>
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 font-mono uppercase text-[10px] tracking-wider">
                  <th className="pb-3 font-semibold">RUN_ID</th>
                  <th className="pb-3 font-semibold">TIMESTAMP</th>
                  <th className="pb-3 font-semibold">INPUT_SIZE</th>
                  <th className="pb-3 font-semibold">STATUS</th>
                  <th className="pb-3 font-semibold text-right">ACTIONS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-800">
                {batchRuns.map((run, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/80 transition">
                    <td className="py-3.5 font-mono font-semibold text-slate-900">{run.id}</td>
                    <td className="py-3.5 text-slate-600 font-mono text-[11px]">{run.timestamp}</td>
                    <td className="py-3.5 font-mono text-slate-700">{run.size}</td>
                    <td className="py-3.5">
                      {run.status === 'COMPLETED' ? (
                        <span className="badge-healthy">COMPLETED</span>
                      ) : (
                        <span className="badge-drifted">FAILED</span>
                      )}
                    </td>
                    <td className="py-3.5 text-right space-x-2">
                      {run.status === 'COMPLETED' ? (
                        <>
                          <a
                            href="/api/v1/sample-csv"
                            download
                            className="p-1.5 text-slate-600 hover:text-slate-900 inline-block"
                            title="Download Batch Output CSV"
                          >
                            <Download className="w-4 h-4" />
                          </a>
                          <button 
                            onClick={() => alert(`Opening execution report for ${run.id}`)}
                            className="p-1.5 text-slate-600 hover:text-slate-900 inline-block" 
                            title="Open Run Details"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </button>
                        </>
                      ) : (
                        <>
                          <button 
                            onClick={() => alert("Failure reason: Missing required column 'Contract' on row 1,192.")}
                            className="p-1.5 text-slate-600 hover:text-slate-900 inline-block" 
                            title="View Error Traceback"
                          >
                            <AlertCircle className="w-4 h-4 text-rose-500" />
                          </button>
                          <button 
                            onClick={() => alert(`Retrying batch scoring job ${run.id}...`)}
                            className="p-1.5 text-slate-600 hover:text-slate-900 inline-block" 
                            title="Retry Run"
                          >
                            <RefreshCw className="w-4 h-4" />
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>

    </div>
  );
}
