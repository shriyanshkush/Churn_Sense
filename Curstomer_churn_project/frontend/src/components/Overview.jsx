import React from 'react';
import { Users, AlertTriangle, DollarSign, Activity, ArrowUpRight, CheckCircle2, TrendingDown } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

export default function Overview({ onNavigate }) {
  const kpiData = [
    { title: 'Global Churn Rate', value: '24.8%', delta: '-2.4% vs last mo', icon: TrendingDown, color: 'from-rose-500/20 to-red-500/10', text: 'text-rose-400' },
    { title: 'High-Risk Customers', value: '1,240', delta: 'Requires action', icon: AlertTriangle, color: 'from-amber-500/20 to-orange-500/10', text: 'text-amber-400' },
    { title: 'Average CLV Preserved', value: '$1,850', delta: '+12% with AI strategy', icon: DollarSign, color: 'from-emerald-500/20 to-teal-500/10', text: 'text-emerald-400' },
    { title: 'Population Stability Index', value: '0.042', delta: 'Low Data Drift', icon: Activity, color: 'from-indigo-500/20 to-purple-500/10', text: 'text-indigo-400' },
  ];

  const riskDistribution = [
    { name: 'Low Risk (<30%)', value: 45, color: '#10b981' },
    { name: 'Medium Risk (30-60%)', value: 28, color: '#f59e0b' },
    { name: 'High Risk (60-85%)', value: 18, color: '#ef4444' },
    { name: 'Critical Risk (>85%)', value: 9, color: '#881337' },
  ];

  const clusterOverview = [
    { segment: 'High-Risk Price-Sensitive', count: 1420, avgChurn: '78.4%', topContract: 'Month-to-month' },
    { segment: 'Stable High-Value', count: 2150, avgChurn: '8.2%', topContract: 'Two year' },
    { segment: 'New & Vulnerable', count: 980, avgChurn: '58.9%', topContract: 'Month-to-month' },
    { segment: 'Loyal Low-Engagement', count: 1820, avgChurn: '18.5%', topContract: 'One year' },
  ];

  return (
    <div className="space-y-6">
      
      {/* Header Banner */}
      <div className="glass-panel p-6 bg-gradient-to-r from-indigo-950/60 via-slate-900/80 to-purple-950/40 border border-indigo-500/20">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Enterprise Customer Churn Intelligence</h1>
            <p className="text-sm text-slate-400 mt-1">Multi-model ML predictions, GMM soft clustering, SHAP explainability, and ROI-driven retention strategies.</p>
          </div>
          <div className="flex space-x-3">
            <button 
              onClick={() => onNavigate('analyzer')}
              className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-indigo-600/30 flex items-center space-x-2"
            >
              <span>Test Customer Predictor</span>
              <ArrowUpRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpiData.map((kpi, idx) => {
          const Icon = kpi.icon;
          return (
            <div key={idx} className="glass-card p-5 border border-slate-800/80 hover:border-indigo-500/30 transition">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400">{kpi.title}</span>
                <div className={`p-2 rounded-xl bg-gradient-to-br ${kpi.color}`}>
                  <Icon className={`w-5 h-5 ${kpi.text}`} />
                </div>
              </div>
              <div className="mt-3">
                <span className="text-2xl font-bold text-white tracking-tight">{kpi.value}</span>
                <span className="block text-xs font-medium text-slate-400 mt-1">{kpi.delta}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Risk Distribution Donut */}
        <div className="glass-panel p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">Customer Risk Distribution</h3>
            <p className="text-xs text-slate-400 mt-0.5">Categorized by predicted churn probability tiers</p>
          </div>

          <div className="h-56 my-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                  itemStyle={{ color: '#e2e8f0', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            {riskDistribution.map((item, idx) => (
              <div key={idx} className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-slate-300 truncate">{item.name}: {item.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Cluster Segment Summaries */}
        <div className="lg:col-span-2 glass-panel p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">GMM Actionable Customer Clusters</h3>
              <p className="text-xs text-slate-400">Behavioral & Churn Risk Segmentation</p>
            </div>
            <button 
              onClick={() => onNavigate('clustering')} 
              className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center space-x-1"
            >
              <span>Explore 2D PCA Scatter</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {clusterOverview.map((item, idx) => (
              <div key={idx} className="glass-card p-4 border border-slate-800/80 hover:border-slate-700">
                <span className="text-xs font-semibold text-indigo-400 block">{item.segment}</span>
                <div className="mt-2 flex items-center justify-between text-xs text-slate-300">
                  <span>Customer Volume: <strong className="text-white">{item.count}</strong></span>
                  <span>Avg Risk: <strong className={parseFloat(item.avgChurn) > 50 ? 'text-rose-400' : 'text-emerald-400'}>{item.avgChurn}</strong></span>
                </div>
                <div className="mt-1 text-xs text-slate-400">
                  Dominant Contract: <span className="text-slate-200">{item.topContract}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

    </div>
  );
}
