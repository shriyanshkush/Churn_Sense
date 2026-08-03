import React, { useState } from 'react';
import { 
  TrendingDown, 
  Zap, 
  CreditCard, 
  TrendingUp, 
  AlertTriangle, 
  ArrowRight, 
  Download, 
  ChevronDown, 
  BarChart2,
  CheckCircle2,
  Sparkles
} from 'lucide-react';

export default function Overview({ selectedModel, onNavigate, searchQuery, onSelectCustomer }) {
  const [timeframe, setTimeframe] = useState('weekly');
  const [filterType, setFilterType] = useState('high_impact');
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainedSuccess, setRetrainedSuccess] = useState(false);

  const modelDisplayName = {
    'xgboost': 'XGBoost_v2_FINAL',
    'random_forest': 'Random_Forest_v1',
    'decision_tree': 'Decision_Tree_v1',
    'logistic_regression': 'Logistic_Regression_v1'
  }[selectedModel] || 'XGBoost_v2_FINAL';

  // Table Data
  const recentBatchData = [
    { id: '#USR-8910', segment: 'Enterprise Gold', prob: 0.94, risk: 'CRITICAL', driver: 'Declining API Usage' },
    { id: '#USR-2245', segment: 'SME Core', prob: 0.78, risk: 'HIGH', driver: 'Support Ticket Overload' },
    { id: '#USR-1102', segment: 'Enterprise Platinum', prob: 0.12, risk: 'LOW', driver: 'Stable - High Engagement' },
    { id: '#USR-4491', segment: 'Mid-Market Growth', prob: 0.65, risk: 'HIGH', driver: 'Payment Failure Warning' },
    { id: '#USR-9012', segment: 'Enterprise Silver', prob: 0.32, risk: 'MEDIUM', driver: 'Recent Contract Renewal' },
  ];

  const filteredRows = recentBatchData.filter(row => {
    if (searchQuery) {
      return row.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
             row.segment.toLowerCase().includes(searchQuery.toLowerCase()) ||
             row.driver.toLowerCase().includes(searchQuery.toLowerCase());
    }
    if (filterType === 'high_impact') {
      return row.risk === 'CRITICAL' || row.risk === 'HIGH';
    }
    return true;
  });

  const handleRetrain = () => {
    setIsRetraining(true);
    setTimeout(() => {
      setIsRetraining(false);
      setRetrainedSuccess(true);
      setTimeout(() => setRetrainedSuccess(false), 5000);
    }, 1800);
  };

  const getRiskBadge = (risk) => {
    switch (risk) {
      case 'CRITICAL': return 'badge-critical';
      case 'HIGH': return 'badge-high';
      case 'MEDIUM': return 'badge-medium';
      default: return 'badge-low';
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Subheader Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-slate-200/60">
        <div>
          <div className="flex items-center space-x-2 text-emerald-600 font-mono text-[11px] font-bold tracking-wider uppercase">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>LIVE PREDICTION ENGINE</span>
          </div>
          <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight mt-0.5">
            Executive Overview
          </h2>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <div className="bg-white border border-slate-200 rounded-xl px-3 py-1.5 flex items-center space-x-2 shadow-xs">
            <span className="text-slate-500 font-medium">Active Model:</span>
            <span className="font-mono font-bold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
              {modelDisplayName}
            </span>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl px-3 py-1.5 flex items-center space-x-2 shadow-xs">
            <span className="text-slate-500 font-medium">Last Run:</span>
            <span className="font-mono font-bold text-slate-800">14 Oct, 09:12</span>
          </div>
        </div>
      </div>

      {/* KPI Cards (4 Column Grid) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        
        {/* KPI 1: Overall Churn Rate */}
        <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs relative overflow-hidden flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <span className="text-xs font-semibold text-slate-500">Overall Churn Rate</span>
            <TrendingDown className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold font-mono text-slate-900">12.4%</div>
            <div className="text-[11px] font-semibold text-emerald-600 font-mono mt-1">
              -2.1% from last month
            </div>
          </div>
          {/* Sparkline simulation */}
          <div className="absolute right-4 bottom-4 w-16 h-10 opacity-20 flex items-end justify-between space-x-1">
            <div className="w-3 bg-slate-800 h-8 rounded-t"></div>
            <div className="w-3 bg-slate-800 h-6 rounded-t"></div>
            <div className="w-3 bg-slate-800 h-4 rounded-t"></div>
            <div className="w-3 bg-slate-800 h-3 rounded-t"></div>
          </div>
        </div>

        {/* KPI 2: Avg Risk Score */}
        <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <span className="text-xs font-semibold text-slate-500">Avg Risk Score</span>
            <Zap className="w-4 h-4 text-rose-500" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold font-mono text-slate-900">68.2</div>
            <div className="text-[11px] font-semibold text-rose-600 font-mono mt-1">
              Elevated threshold detected
            </div>
          </div>
          <div className="mt-3 w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
            <div className="bg-gradient-to-r from-teal-500 via-amber-500 to-rose-500 h-full w-[68%]"></div>
          </div>
        </div>

        {/* KPI 3: At-Risk Revenue */}
        <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <span className="text-xs font-semibold text-slate-500">At-Risk Revenue</span>
            <CreditCard className="w-4 h-4 text-slate-600" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold font-mono text-slate-900">$2.4M</div>
            <div className="text-[11px] font-medium text-slate-500 mt-1">
              Across 412 high-risk accounts
            </div>
          </div>
        </div>

        {/* KPI 4: CLV Forecast */}
        <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-start justify-between">
            <span className="text-xs font-semibold text-slate-500">CLV Forecast (12mo)</span>
            <TrendingUp className="w-4 h-4 text-teal-600" />
          </div>
          <div className="mt-3">
            <div className="text-3xl font-extrabold font-mono text-slate-900">$18.9M</div>
            <div className="text-[11px] font-semibold text-teal-600 font-mono mt-1">
              +5.4% predicted growth
            </div>
          </div>
        </div>

      </div>

      {/* Middle Row (8/4 Layout) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Risk Tier Distribution Chart (8 cols) */}
        <div className="lg:col-span-7 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-slate-900">Risk Tier Distribution</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Population breakdown by predictive churn probability
                </p>
              </div>

              {/* Timeframe Toggle Buttons */}
              <div className="bg-slate-100 p-1 rounded-xl flex items-center space-x-1 text-xs">
                <button
                  onClick={() => setTimeframe('daily')}
                  className={`px-3 py-1 rounded-lg font-medium transition ${
                    timeframe === 'daily' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Daily
                </button>
                <button
                  onClick={() => setTimeframe('weekly')}
                  className={`px-3 py-1 rounded-lg font-medium transition ${
                    timeframe === 'weekly' ? 'bg-slate-900 text-white shadow-xs font-bold' : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Weekly
                </button>
              </div>
            </div>

            {/* Custom Bar Visualization matching Reference Screenshot 1 */}
            <div className="my-8 pt-6 pb-2 px-6 flex items-end justify-between h-48 border-b border-slate-100">
              
              {/* LOW Tier */}
              <div className="flex flex-col items-center flex-1 space-y-3">
                <div className="w-16 bg-[#047857] rounded-t-sm h-36 transition-all duration-500 hover:opacity-90"></div>
                <span className="badge-low">LOW</span>
              </div>

              {/* MEDIUM Tier */}
              <div className="flex flex-col items-center flex-1 space-y-3">
                <div className="w-16 bg-[#BFDBFE] rounded-t-sm h-24 transition-all duration-500 hover:opacity-90"></div>
                <span className="badge-medium">MEDIUM</span>
              </div>

              {/* HIGH Tier */}
              <div className="flex flex-col items-center flex-1 space-y-3">
                <div className="w-16 bg-[#FCA5A5] rounded-t-sm h-16 transition-all duration-500 hover:opacity-90"></div>
                <span className="badge-high">HIGH</span>
              </div>

              {/* CRITICAL Tier */}
              <div className="flex flex-col items-center flex-1 space-y-3">
                <div className="w-16 bg-[#B91C1C] rounded-t-sm h-10 transition-all duration-500 hover:opacity-90"></div>
                <span className="badge-critical">CRITICAL</span>
              </div>

            </div>
          </div>

          {/* Legend Bottom */}
          <div className="pt-2 flex items-center space-x-6 text-xs text-slate-600 font-medium">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#047857]"></span>
              <span>Retained</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#B91C1C]"></span>
              <span>Predicted Churn</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-[#BFDBFE]"></span>
              <span>Pending Renewal</span>
            </div>
          </div>
        </div>

        {/* Right Column: Model Performance & Drift Alert (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          
          {/* Card 1: Model Performance */}
          <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-900">Model Performance</h3>
              <span className="badge-healthy">HEALTHY</span>
            </div>
            <p className="text-xs text-slate-500">Current XGBoost validation</p>

            {/* Metrics Bars */}
            <div className="space-y-3.5 pt-1">
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span className="text-slate-600">Precision / Recall</span>
                  <span className="font-mono text-slate-900">0.89 / 0.84</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-[#047857] h-full rounded-full" style={{ width: '87%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span className="text-slate-600">AUC-ROC Score</span>
                  <span className="font-mono text-slate-900">0.92</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-[#047857] h-full rounded-full" style={{ width: '92%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span className="text-slate-600">F1 Score</span>
                  <span className="font-mono text-slate-900">0.864</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-[#047857] h-full rounded-full" style={{ width: '86%' }}></div>
                </div>
              </div>
            </div>

            <button
              onClick={() => onNavigate('monitoring')}
              className="w-full py-2 bg-white border border-slate-300 hover:bg-slate-50 text-slate-800 text-xs font-bold rounded-xl transition mt-2 shadow-xs"
            >
              View Detailed Metrics
            </button>
          </div>

          {/* Card 2: Model Drift Alert */}
          <div className="bg-rose-50/80 border border-rose-200 rounded-2xl p-5 shadow-xs space-y-3">
            <div className="flex items-center space-x-2 text-rose-900 font-bold text-xs">
              <AlertTriangle className="w-4 h-4 text-rose-600" />
              <span>Model Drift Alert</span>
            </div>

            <p className="text-xs text-rose-800 leading-relaxed">
              PSI (Population Stability Index) has shifted to <strong className="font-mono text-rose-900">0.24</strong> for the 'Tenure' feature. Re-training recommended.
            </p>

            <button
              onClick={handleRetrain}
              disabled={isRetraining}
              className="text-xs font-bold text-rose-700 hover:text-rose-900 flex items-center space-x-1 transition disabled:opacity-50"
            >
              <span>{isRetraining ? 'Retraining XGBoost Pipeline...' : 'Initiate Re-training'}</span>
              <ArrowRight className="w-3.5 h-3.5 ml-1" />
            </button>

            {retrainedSuccess && (
              <div className="p-2 bg-emerald-100 border border-emerald-300 text-emerald-900 text-[11px] rounded-lg font-semibold flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Model XGBoost_v2_FINAL successfully retrained!</span>
              </div>
            )}
          </div>

        </div>

      </div>

      {/* Bottom Row: Recent Batch Scores Table */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <h3 className="text-base font-bold text-slate-900">Recent Batch Scores</h3>

          <div className="flex items-center space-x-4 text-xs font-medium text-slate-600">
            <div className="flex items-center space-x-2">
              <span className="text-slate-500">Filter:</span>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-800 font-medium focus:outline-none"
              >
                <option value="high_impact">High Impact Only</option>
                <option value="all">All Accounts</option>
              </select>
            </div>

            <span className="text-slate-300">|</span>

            <a
              href="/api/v1/sample-csv"
              download
              className="text-teal-700 hover:text-teal-800 font-bold flex items-center space-x-1 transition"
            >
              <span>Download CSV</span>
              <Download className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>

        {/* Data Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-mono uppercase text-[10px] tracking-wider">
                <th className="pb-3 font-semibold">CUSTOMER ID</th>
                <th className="pb-3 font-semibold">SEGMENT</th>
                <th className="pb-3 font-semibold">CHURN PROBABILITY</th>
                <th className="pb-3 font-semibold">RISK LEVEL</th>
                <th className="pb-3 font-semibold">KEY DRIVER</th>
                <th className="pb-3 font-semibold text-right">ACTIONS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-medium text-slate-800">
              {filteredRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50/80 transition">
                  <td className="py-3.5 font-mono font-semibold text-slate-900">{row.id}</td>
                  <td className="py-3.5 text-slate-700">{row.segment}</td>
                  <td className="py-3.5 font-mono font-bold text-slate-900">{row.prob.toFixed(2)}</td>
                  <td className="py-3.5">
                    <span className={getRiskBadge(row.risk)}>{row.risk}</span>
                  </td>
                  <td className="py-3.5 text-slate-600">{row.driver}</td>
                  <td className="py-3.5 text-right">
                    <button
                      onClick={() => {
                        if (onSelectCustomer) onSelectCustomer(row);
                        onNavigate('customer_detail');
                      }}
                      className="px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold rounded-lg transition"
                    >
                      Analyze
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
