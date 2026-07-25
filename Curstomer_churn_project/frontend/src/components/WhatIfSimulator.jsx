import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Sliders, Sparkles, AlertCircle, ArrowDown, ArrowUp, DollarSign, ShieldAlert, CheckCircle2, Copy } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LineChart, Line, CartesianGrid } from 'recharts';

export default function WhatIfSimulator({ selectedModel }) {
  const [customer, setCustomer] = useState({
    gender: 'Female',
    SeniorCitizen: 0,
    Partner: 'Yes',
    Dependents: 'No',
    tenure: 10,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'Yes',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'Yes',
    StreamingMovies: 'Yes',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 85.0,
    TotalCharges: 850.0,
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [aiStrategy, setAiStrategy] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    runPrediction();
  }, [customer, selectedModel]);

  const runPrediction = async () => {
    try {
      setLoading(true);
      const res = await axios.post('/api/v1/predict', {
        ...customer,
        selected_model: selectedModel
      });
      setPrediction(res.data);
    } catch (err) {
      console.error("Prediction Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSliderChange = (field, value) => {
    setCustomer(prev => {
      const updated = { ...prev, [field]: value };
      if (field === 'tenure' || field === 'MonthlyCharges') {
        updated.TotalCharges = Math.round(updated.tenure * updated.MonthlyCharges * 100) / 100;
      }
      return updated;
    });
  };

  const handleGenerateAiStrategy = async () => {
    if (!prediction) return;
    try {
      setAiLoading(true);
      const topDrivers = prediction.shap_waterfall.slice(0, 3).map(item => item.feature);
      const res = await axios.post('/api/v1/ai-strategy', {
        gender: customer.gender,
        tenure: customer.tenure,
        Contract: customer.Contract,
        MonthlyCharges: customer.MonthlyCharges,
        churn_probability: prediction.churn_probability,
        risk_tier: prediction.risk_tier,
        cluster_label: prediction.cluster_label,
        top_shap_drivers: topDrivers,
        clv_estimate: prediction.clv_estimate,
        gemini_api_key: apiKey.trim() || undefined
      });
      setAiStrategy(res.data);
    } catch (err) {
      console.error("AI Strategy Error:", err);
    } finally {
      setAiLoading(false);
    }
  };

  const copyCoupon = () => {
    if (aiStrategy?.discount_coupon) {
      navigator.clipboard.writeText(aiStrategy.discount_coupon);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getRiskColor = (tier) => {
    if (tier === 'Critical' || tier === 'High') return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
    if (tier === 'Medium') return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
  };

  return (
    <div className="space-y-6">
      
      {/* Header */}
      <div className="glass-panel p-6 border-indigo-500/20">
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Sliders className="w-5 h-5 text-indigo-400" />
          <span>Interactive What-If Simulator & AI Retention Engine</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Adjust customer attributes in real-time to watch churn risk, SHAP drivers, CLV, and survival curves recalculate dynamically.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Interactive Controls */}
        <div className="lg:col-span-5 glass-panel p-6 space-y-4">
          <h3 className="text-sm font-semibold text-white border-b border-slate-800 pb-2">Customer Attributes</h3>

          {/* Monthly Charges Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Monthly Charges:</span>
              <span className="text-indigo-400 font-mono font-bold">${customer.MonthlyCharges} / mo</span>
            </div>
            <input
              type="range"
              min="18.0"
              max="120.0"
              step="0.5"
              value={customer.MonthlyCharges}
              onChange={(e) => handleSliderChange('MonthlyCharges', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Tenure Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Customer Tenure:</span>
              <span className="text-indigo-400 font-mono font-bold">{customer.tenure} months</span>
            </div>
            <input
              type="range"
              min="1"
              max="72"
              step="1"
              value={customer.tenure}
              onChange={(e) => handleSliderChange('tenure', parseInt(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Contract Type Select */}
          <div className="space-y-1.5">
            <label className="text-xs text-slate-300 font-medium block">Contract Type:</label>
            <select
              value={customer.Contract}
              onChange={(e) => handleSliderChange('Contract', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700/80 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
            >
              <option value="Month-to-month">Month-to-month (High Risk)</option>
              <option value="One year">One year (Moderate Risk)</option>
              <option value="Two year">Two year (Low Risk)</option>
            </select>
          </div>

          {/* Internet Service */}
          <div className="space-y-1.5">
            <label className="text-xs text-slate-300 font-medium block">Internet Service:</label>
            <select
              value={customer.InternetService}
              onChange={(e) => handleSliderChange('InternetService', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700/80 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
            >
              <option value="Fiber optic">Fiber optic</option>
              <option value="DSL">DSL</option>
              <option value="No">No Internet</option>
            </select>
          </div>

          {/* Security & Support Toggles */}
          <div className="grid grid-cols-2 gap-3 pt-2">
            <div className="space-y-1">
              <label className="text-xs text-slate-400 block">Tech Support:</label>
              <button
                onClick={() => handleSliderChange('TechSupport', customer.TechSupport === 'Yes' ? 'No' : 'Yes')}
                className={`w-full py-1.5 px-3 rounded-xl text-xs font-semibold border transition ${
                  customer.TechSupport === 'Yes'
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : 'bg-slate-900 text-slate-400 border-slate-800'
                }`}
              >
                {customer.TechSupport === 'Yes' ? 'Active ✓' : 'Inactive ✗'}
              </button>
            </div>

            <div className="space-y-1">
              <label className="text-xs text-slate-400 block">Online Security:</label>
              <button
                onClick={() => handleSliderChange('OnlineSecurity', customer.OnlineSecurity === 'Yes' ? 'No' : 'Yes')}
                className={`w-full py-1.5 px-3 rounded-xl text-xs font-semibold border transition ${
                  customer.OnlineSecurity === 'Yes'
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : 'bg-slate-900 text-slate-400 border-slate-800'
                }`}
              >
                {customer.OnlineSecurity === 'Yes' ? 'Active ✓' : 'Inactive ✗'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Live Results, SHAP Waterfall & AI Strategy */}
        <div className="lg:col-span-7 space-y-6">
          
          {/* Risk Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            
            {/* Churn Probability */}
            <div className="glass-card p-5 border border-slate-800 flex flex-col justify-between">
              <span className="text-xs text-slate-400 font-medium">Predicted Churn Risk</span>
              <div className="mt-2 flex items-baseline space-x-2">
                <span className={`text-3xl font-extrabold font-display ${prediction?.churn_probability >= 50 ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {prediction ? `${prediction.churn_probability}%` : '--'}
                </span>
              </div>
              <span className={`mt-2 inline-block text-xs px-2.5 py-0.5 rounded-full border font-semibold w-fit ${getRiskColor(prediction?.risk_tier)}`}>
                {prediction?.risk_tier || 'Medium'} Risk Tier
              </span>
            </div>

            {/* Estimated CLV */}
            <div className="glass-card p-5 border border-slate-800 flex flex-col justify-between">
              <span className="text-xs text-slate-400 font-medium">Estimated CLV Preserved</span>
              <span className="text-3xl font-extrabold text-indigo-300 font-display mt-2">
                {prediction ? `$${prediction.clv_estimate.toLocaleString()}` : '--'}
              </span>
              <span className="text-xs text-slate-400 mt-2">
                Expected Tenure: <strong>{prediction?.expected_remaining_tenure || '--'} mos</strong>
              </span>
            </div>

            {/* Cluster Segment */}
            <div className="glass-card p-5 border border-slate-800 flex flex-col justify-between">
              <span className="text-xs text-slate-400 font-medium">GMM Soft Cluster Label</span>
              <span className="text-sm font-bold text-purple-300 mt-2 leading-tight">
                {prediction?.cluster_label || 'Segmenting...'}
              </span>
              <span className="text-xs text-slate-400 mt-2">
                Segment-aware AI prompt enabled
              </span>
            </div>

          </div>

          {/* Local SHAP Waterfall Chart */}
          <div className="glass-panel p-6">
            <h3 className="text-sm font-semibold text-white mb-1">Local SHAP Feature Impact Breakdown</h3>
            <p className="text-xs text-slate-400 mb-4">Positive bars increase churn risk; negative bars lower churn risk.</p>

            <div className="h-56 w-full">
              {prediction?.shap_waterfall ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart layout="vertical" data={prediction.shap_waterfall} margin={{ left: 40, right: 20, top: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" stroke="#64748b" fontSize={11} />
                    <YAxis type="category" dataKey="feature" stroke="#94a3b8" fontSize={11} width={100} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                    <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                      {prediction.shap_waterfall.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? "#ef4444" : "#10b981"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-slate-400 text-xs">Computing SHAP Waterfall...</div>
              )}
            </div>
          </div>

          {/* Gemini Segment-Aware AI Retention Strategy */}
          <div className="glass-panel p-6 border-indigo-500/30 bg-gradient-to-r from-indigo-950/30 to-purple-950/20">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div>
                <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
                  <Sparkles className="w-4 h-4 text-purple-400" />
                  <span>Segment-Aware Gemini AI Retention Strategy</span>
                </h3>
                <p className="text-xs text-slate-400">Generates custom discount coupons & ROI estimate for this segment.</p>
              </div>

              <button
                onClick={handleGenerateAiStrategy}
                disabled={aiLoading}
                className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-semibold rounded-xl transition shadow-md shadow-indigo-600/30 flex items-center justify-center space-x-2 disabled:opacity-50"
              >
                {aiLoading ? 'Generating Strategy...' : 'Generate AI Offer & Coupon'}
              </button>
            </div>

            {/* Generated Strategy View */}
            {aiStrategy && (
              <div className="mt-4 p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3 text-xs">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <span className="font-bold text-indigo-300 text-sm">{aiStrategy.strategy_title}</span>
                  <div className="flex items-center space-x-2 bg-slate-800 px-3 py-1 rounded-lg">
                    <span className="text-slate-400">Coupon:</span>
                    <span className="font-mono text-purple-300 font-bold">{aiStrategy.discount_coupon}</span>
                    <button onClick={copyCoupon} className="text-indigo-400 hover:text-white ml-1">
                      {copied ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </div>

                <p className="text-slate-300 leading-relaxed">{aiStrategy.executive_summary}</p>
                <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-200">
                  <strong>Recommended Intervention Offer:</strong> {aiStrategy.retention_offer}
                </div>

                <div className="grid grid-cols-3 gap-2 text-center pt-1">
                  <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Estimated Cost</span>
                    <span className="font-bold text-slate-200">${aiStrategy.estimated_cost}</span>
                  </div>
                  <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Risk Reduction</span>
                    <span className="font-bold text-emerald-400">-{aiStrategy.expected_risk_reduction}%</span>
                  </div>
                  <div className="bg-slate-950 p-2 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block text-[10px]">Campaign ROI</span>
                    <span className="font-bold text-indigo-400">+{aiStrategy.estimated_roi}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

        </div>

      </div>

    </div>
  );
}
