import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Users, Layers, Activity, Info, Filter } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid } from 'recharts';

export default function CustomerClustering() {
  const [clustersData, setClustersData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedCluster, setSelectedCluster] = useState('all');

  useEffect(() => {
    fetchClusters();
  }, []);

  const fetchClusters = async () => {
    try {
      setLoading(true);
      const res = await axios.get('/api/v1/clusters');
      setClustersData(res.data);
    } catch (err) {
      console.error("Cluster API Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const getClusterColor = (label) => {
    if (label.includes("High-Risk")) return "#ef4444";
    if (label.includes("Stable")) return "#10b981";
    if (label.includes("New")) return "#f59e0b";
    return "#8b5cf6";
  };

  // Get unique cluster labels to eliminate duplicates
  const uniqueClusterLabels = clustersData?.cluster_labels
    ? Array.from(new Set(Object.values(clustersData.cluster_labels)))
    : [];

  const filteredScatter = clustersData?.pca_scatter?.filter(p => {
    if (selectedCluster === 'all') return true;
    return p.cluster_label === selectedCluster;
  }) || [];

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="glass-panel p-6 border-indigo-500/20">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
              <Users className="w-5 h-5 text-indigo-400" />
              <span>Customer Segmentation (GMM Soft Clustering & 2D PCA)</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Soft clustering via Gaussian Mixture Models (GMM) combined with 2D PCA projection coordinates.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <Filter className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-xs text-slate-400 font-medium">Filter Segment:</span>
            <select
              value={selectedCluster}
              onChange={(e) => setSelectedCluster(e.target.value)}
              className="bg-slate-900 border border-slate-700/80 text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500"
            >
              <option value="all">All Segments ({clustersData?.pca_scatter?.length || 0})</option>
              {uniqueClusterLabels.map((lbl, idx) => (
                <option key={idx} value={lbl}>{lbl}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Main Scatter Plot & Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* 2D PCA Scatter Plot */}
        <div className="lg:col-span-2 glass-panel p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">2D PCA Visual Scatter Projection</h3>
              <p className="text-xs text-slate-400">Position = PCA Component 1 vs 2, Size = Churn Risk %</p>
            </div>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 font-mono">
              {filteredScatter.length} Points
            </span>
          </div>

          <div className="h-96 w-full">
            {loading ? (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">
                Loading PCA Scatter Projection...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis type="number" dataKey="x" name="PCA-1" stroke="#64748b" fontSize={11} />
                  <YAxis type="number" dataKey="y" name="PCA-2" stroke="#64748b" fontSize={11} />
                  <ZAxis type="number" dataKey="churn_probability" range={[30, 200]} name="Churn Risk" />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-700 p-3 rounded-xl shadow-xl text-xs space-y-1">
                          <p className="font-bold text-indigo-300">{data.cluster_label}</p>
                          <p className="text-slate-300">Churn Risk: <strong className="text-rose-400">{data.churn_probability}%</strong></p>
                          <p className="text-slate-400">Monthly Charges: ${data.monthly_charges}</p>
                          <p className="text-slate-400">Tenure: {data.tenure} mos</p>
                        </div>
                      );
                    }}
                  />
                  <Scatter data={filteredScatter}>
                    {filteredScatter.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getClusterColor(entry.cluster_label)} opacity={0.8} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Unique Cluster Legend & GMM Explanation */}
        <div className="glass-panel p-6 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">Actionable Segment Profiles</h3>
            
            <div className="space-y-3">
              {uniqueClusterLabels.map((lbl, idx) => (
                <div 
                  key={idx}
                  onClick={() => setSelectedCluster(selectedCluster === lbl ? 'all' : lbl)}
                  className={`p-3 rounded-xl border transition cursor-pointer ${
                    selectedCluster === lbl ? 'bg-slate-800/90 border-indigo-500' : 'bg-slate-900/40 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: getClusterColor(lbl) }} />
                    <span className="text-xs font-bold text-white">{lbl}</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    {lbl.includes('High-Risk') && 'High monthly bill, short tenure, month-to-month contract.'}
                    {lbl.includes('Stable') && 'Long tenure, active security add-ons, annual contract.'}
                    {lbl.includes('New') && 'Recent signup, vulnerable to early churn without support.'}
                    {lbl.includes('Loyal') && 'Long tenure with basic services, low churn risk.'}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 flex items-start space-x-2">
            <Info className="w-4 h-4 text-indigo-400 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Soft Clustering Note:</strong> GMM assigns probability weights (e.g. 70% High-Risk, 30% Loyal) enabling segment-aware AI strategy prompts.
            </span>
          </div>
        </div>

      </div>

      {/* Optimal K Selection (Elbow Method & Silhouette Scores) */}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold text-white mb-1">Optimal Cluster k Diagnostics</h3>
        <p className="text-xs text-slate-400 mb-4">Elbow Method (Inertia SSE) & Silhouette Scores for k=2..7</p>

        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={clustersData?.elbow_metrics || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="k" stroke="#64748b" fontSize={11} label={{ value: 'Number of Clusters (k)', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 11 }} />
              <YAxis yAxisId="left" stroke="#6366f1" fontSize={11} label={{ value: 'Inertia (SSE)', angle: -90, position: 'insideLeft', fill: '#6366f1', fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" stroke="#10b981" fontSize={11} label={{ value: 'Silhouette Score', angle: 90, position: 'insideRight', fill: '#10b981', fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              <Line yAxisId="left" type="monotone" dataKey="inertia" stroke="#6366f1" strokeWidth={2} name="Inertia (SSE)" />
              <Line yAxisId="right" type="monotone" dataKey="silhouette" stroke="#10b981" strokeWidth={2} name="Silhouette Score" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
