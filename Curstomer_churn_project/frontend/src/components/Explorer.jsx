import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Sparkles, 
  CheckCircle2, 
  TrendingUp, 
  DollarSign, 
  Activity, 
  Layers, 
  X,
  AlertCircle
} from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Explorer({ onGenerateStrategy }) {
  const [densityMap, setDensityMap] = useState(true);
  const [showOutliers, setShowOutliers] = useState(false);
  const [selectedSegment, setSelectedSegment] = useState('High-Risk Price-Sensitive');
  const [showToast, setShowToast] = useState(true);
  const [pcaData, setPcaData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Segment Profiles mapping
  const segmentsInfo = {
    'High-Risk Price-Sensitive': {
      userCount: 422,
      tenure: '14.2 Mo',
      engagement: '18.4%',
      ltv: '$1,240',
      color: '#0D9488',
      drivers: [
        { name: 'Competitor Price Match', impact: '42% Impact', icon: '📈', color: 'text-rose-600' },
        { name: 'Feature Latency Issues', impact: '18% Impact', icon: '🔄', color: 'text-slate-600' },
        { name: 'Involuntary Credit Failure', impact: '12% Impact', icon: '💳', color: 'text-slate-600' },
      ]
    },
    'Low-Engagement': {
      userCount: 680,
      tenure: '28.5 Mo',
      engagement: '24.1%',
      ltv: '$2,450',
      color: '#1E3A8A',
      drivers: [
        { name: 'Infrequent Logins', impact: '35% Impact', icon: '📉', color: 'text-rose-600' },
        { name: 'Unresolved Support Tickets', impact: '22% Impact', icon: '🛠️', color: 'text-slate-600' },
        { name: 'Lack of Onboarding', impact: '15% Impact', icon: '🎓', color: 'text-slate-600' },
      ]
    },
    'Critical Risk': {
      userCount: 295,
      tenure: '4.1 Mo',
      engagement: '8.2%',
      ltv: '$680',
      color: '#DC2626',
      drivers: [
        { name: 'Contract Expiration Near', impact: '58% Impact', icon: '⏳', color: 'text-rose-600' },
        { name: 'High Monthly Bill', impact: '24% Impact', icon: '💸', color: 'text-rose-600' },
        { name: 'Zero Security Addons', impact: '10% Impact', icon: '🔒', color: 'text-slate-600' },
      ]
    }
  };

  useEffect(() => {
    fetchClusterData();
  }, []);

  const fetchClusterData = async () => {
    try {
      setLoading(true);
      const res = await axios.get('/api/v1/clusters');
      if (res.data?.pca_scatter && res.data.pca_scatter.length > 0) {
        setPcaData(res.data.pca_scatter);
      } else {
        // Fallback demo scatter points matching Reference 3
        setPcaData(getFallbackScatterPoints());
      }
    } catch (err) {
      console.warn("PCA API error, loading mock scatter:", err);
      setPcaData(getFallbackScatterPoints());
    } finally {
      setLoading(false);
    }
  };

  const getFallbackScatterPoints = () => [
    // Segment A (Price-Sensitive) around (-2, 0.5)
    { x: -2.1, y: 0.6, cluster_label: 'High-Risk Price-Sensitive', churn_probability: 78.4 },
    { x: -1.9, y: 0.4, cluster_label: 'High-Risk Price-Sensitive', churn_probability: 82.1 },
    { x: -2.3, y: 0.8, cluster_label: 'High-Risk Price-Sensitive', churn_probability: 74.0 },
    { x: -2.0, y: 0.2, cluster_label: 'High-Risk Price-Sensitive', churn_probability: 79.5 },
    { x: -1.7, y: 0.7, cluster_label: 'High-Risk Price-Sensitive', churn_probability: 85.0 },

    // Segment B (Low-Engagement) around (0.5, -1.2)
    { x: 0.3, y: -1.1, cluster_label: 'Low-Engagement', churn_probability: 45.2 },
    { x: 0.6, y: -1.4, cluster_label: 'Low-Engagement', churn_probability: 52.0 },
    { x: 0.4, y: -0.9, cluster_label: 'Low-Engagement', churn_probability: 41.8 },
    { x: 0.7, y: -1.2, cluster_label: 'Low-Engagement', churn_probability: 48.6 },

    // Segment C (Critical Risk) around (1.8, 1.2)
    { x: 1.7, y: 1.1, cluster_label: 'Critical Risk', churn_probability: 91.2 },
    { x: 1.9, y: 1.3, cluster_label: 'Critical Risk', churn_probability: 94.5 },
    { x: 1.6, y: 1.4, cluster_label: 'Critical Risk', churn_probability: 88.0 },
  ];

  const currentInfo = segmentsInfo[selectedSegment] || segmentsInfo['High-Risk Price-Sensitive'];

  return (
    <div className="space-y-6 animate-in fade-in duration-300 relative pb-12">
      
      {/* Subheader */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-slate-200/60">
        <div>
          <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Dimensional Explorer
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            PCA Cluster Projection: High-dimensional customer features mapped to 2D space
          </p>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <div className="bg-white border border-slate-200 rounded-xl px-3 py-1.5 flex items-center space-x-2 shadow-xs">
            <span className="w-2 h-2 rounded-full bg-teal-500"></span>
            <span className="text-slate-500 font-medium">Variance Explained:</span>
            <span className="font-mono font-bold text-slate-900">84.2%</span>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl px-3 py-1.5 flex items-center space-x-2 shadow-xs">
            <span className="text-slate-500 font-medium">Algorithm:</span>
            <span className="font-mono font-bold text-slate-900">GMM (k=4)</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Left Scatter Canvas (8 cols) + Right Selected Segment (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Scatter Plot Canvas */}
        <div className="lg:col-span-8 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between relative min-h-[520px]">
          <div>
            {/* Top Canvas Controls */}
            <div className="flex items-center justify-between border-b border-slate-100 pb-4">
              <span className="text-xs font-mono font-bold text-slate-500 uppercase tracking-wider">
                2D FEATURE PROJECTION (PC1 VS PC2)
              </span>

              <div className="flex items-center space-x-5 text-xs text-slate-600 font-medium">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={densityMap}
                    onChange={(e) => setDensityMap(e.target.checked)}
                    className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                  />
                  <span>Density Map</span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showOutliers}
                    onChange={(e) => setShowOutliers(e.target.checked)}
                    className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                  />
                  <span>Show Outliers</span>
                </label>
              </div>
            </div>

            {/* Scatter Graph Body matching Reference Screen 3 */}
            <div className="h-[380px] w-full mt-4 relative bg-[#fafbfc] rounded-xl border border-slate-100 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="PC1" 
                    stroke="#94a3b8" 
                    fontSize={10} 
                    label={{ value: 'PC1 (Dominant Variance)', position: 'insideBottom', offset: -15, fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }} 
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="PC2" 
                    stroke="#94a3b8" 
                    fontSize={10} 
                    label={{ value: 'PC2 (Sub-Variance)', angle: -90, position: 'insideLeft', offset: 10, fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }} 
                  />
                  <ZAxis type="number" dataKey="churn_probability" range={[80, 260]} />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const pt = payload[0].payload;
                      return (
                        <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs font-sans space-y-1">
                          <p className="font-bold text-teal-300">{pt.cluster_label || selectedSegment}</p>
                          <p className="text-slate-300 font-mono">Churn Risk: <span className="text-rose-400 font-bold">{pt.churn_probability}%</span></p>
                          <p className="text-slate-400">PC1: {pt.x} | PC2: {pt.y}</p>
                        </div>
                      );
                    }}
                  />
                  <Scatter 
                    data={pcaData} 
                    onClick={(entry) => {
                      if (entry && entry.cluster_label) {
                        setSelectedSegment(entry.cluster_label);
                      }
                    }}
                  >
                    {pcaData.map((entry, index) => {
                      const isSelected = entry.cluster_label === selectedSegment;
                      let fill = '#0D9488';
                      if (entry.cluster_label?.includes('Low-Engagement')) fill = '#1E3A8A';
                      if (entry.cluster_label?.includes('Critical')) fill = '#DC2626';

                      return (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={fill} 
                          stroke={isSelected ? '#000000' : 'none'} 
                          strokeWidth={isSelected ? 2 : 0}
                          opacity={densityMap ? 0.85 : 0.6} 
                        />
                      );
                    })}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>

              {/* Dotted Cluster Legend inside scatter frame */}
              <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur border border-slate-200 rounded-xl p-3 shadow-xs space-y-1.5 text-[11px] font-medium text-slate-700">
                <button
                  onClick={() => setSelectedSegment('High-Risk Price-Sensitive')}
                  className={`flex items-center space-x-2 w-full text-left rounded px-1.5 py-0.5 transition ${
                    selectedSegment === 'High-Risk Price-Sensitive' ? 'bg-teal-50 font-bold' : ''
                  }`}
                >
                  <span className="w-2.5 h-2.5 rounded-full bg-[#0D9488]"></span>
                  <span>Segment A: Price-Sensitive</span>
                </button>

                <button
                  onClick={() => setSelectedSegment('Low-Engagement')}
                  className={`flex items-center space-x-2 w-full text-left rounded px-1.5 py-0.5 transition ${
                    selectedSegment === 'Low-Engagement' ? 'bg-blue-50 font-bold' : ''
                  }`}
                >
                  <span className="w-2.5 h-2.5 rounded-full bg-[#1E3A8A]"></span>
                  <span>Segment B: Low-Engagement</span>
                </button>

                <button
                  onClick={() => setSelectedSegment('Critical Risk')}
                  className={`flex items-center space-x-2 w-full text-left rounded px-1.5 py-0.5 transition ${
                    selectedSegment === 'Critical Risk' ? 'bg-red-50 font-bold' : ''
                  }`}
                >
                  <span className="w-2.5 h-2.5 rounded-full bg-[#DC2626]"></span>
                  <span>Segment C: Critical Risk</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel: SELECTED SEGMENT Card */}
        <div className="lg:col-span-4 bg-blue-50/60 border border-blue-200/80 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
          <div className="space-y-5">
            
            {/* Header Badge */}
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-extrabold text-blue-900 uppercase tracking-wider">
                  SELECTED SEGMENT
                </span>
                <span className="bg-teal-200/80 text-teal-900 font-mono text-[11px] font-bold px-2 py-0.5 rounded">
                  {currentInfo.userCount} Users
                </span>
              </div>
              <h3 className="text-lg font-extrabold text-slate-900 mt-1">
                {selectedSegment}
              </h3>
            </div>

            {/* Metrics Breakdown Cards */}
            <div className="space-y-2.5">
              <div className="bg-white border border-slate-200 rounded-xl p-3 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-600">Avg. Tenure</span>
                <span className="text-base font-extrabold font-mono text-slate-900">{currentInfo.tenure}</span>
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-3 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-600">Engagement Rate</span>
                <span className="text-base font-extrabold font-mono text-slate-900">{currentInfo.engagement}</span>
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-3 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-600">LTV Forecast</span>
                <span className="text-base font-extrabold font-mono text-slate-900">{currentInfo.ltv}</span>
              </div>
            </div>

            {/* TOP CHURN DRIVERS */}
            <div className="space-y-3 pt-2">
              <span className="text-[11px] font-mono font-bold text-slate-400 tracking-wider uppercase block">
                TOP CHURN DRIVERS
              </span>

              <div className="space-y-2.5">
                {currentInfo.drivers.map((drv, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs font-semibold text-slate-800">
                    <div className="flex items-center space-x-2">
                      <span>{drv.icon}</span>
                      <span>{drv.name}</span>
                    </div>
                    <span className={`font-mono ${drv.color}`}>{drv.impact}</span>
                  </div>
                ))}
              </div>
            </div>

          </div>

          {/* Action Button */}
          <button
            onClick={() => onGenerateStrategy && onGenerateStrategy(selectedSegment)}
            className="w-full mt-6 py-3 bg-[#0f172a] hover:bg-slate-800 text-white text-xs font-bold rounded-xl shadow-md transition flex items-center justify-center space-x-2"
          >
            <Sparkles className="w-4 h-4 text-teal-400" />
            <span>Generate Retention Strategy</span>
          </button>
        </div>

      </div>

      {/* Floating Toast Notification matching Reference Screen 3 */}
      {showToast && (
        <div className="fixed bottom-6 right-8 bg-[#0f172a] text-white px-4 py-3 rounded-xl shadow-2xl flex items-center space-x-3 border border-slate-700 animate-in slide-in-from-bottom-3 z-50">
          <CheckCircle2 className="w-4 h-4 text-teal-400 flex-shrink-0" />
          <span className="text-xs font-medium">Cohort data updated successfully.</span>
          <button 
            onClick={() => setShowToast(false)}
            className="text-slate-400 hover:text-white ml-2"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

    </div>
  );
}
