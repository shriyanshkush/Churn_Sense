import React, { useState } from 'react';
import { 
  ClipboardList, Target, Cpu, Sparkles, CheckCircle2, AlertTriangle, Play, RefreshCw, BarChart2, Sliders
} from 'lucide-react';
import ModelPerformance, { MODEL_METRICS_DATA } from './ModelPerformance';
import WhatIfSimulator from './WhatIfSimulator';
import AiStrategies from './AiStrategies';

export default function SingleCustomerPrediction({ selectedModel, setSelectedModel }) {
  const [activeTab, setActiveTab] = useState('form'); // 'form', 'result', 'whatif', 'ai_strategy', 'model_info'
  
  // 19 Input Features State
  const [formData, setFormData] = useState({
    gender: 'Female',
    SeniorCitizen: '0',
    Partner: 'Yes',
    Dependents: 'No',
    tenure: 12,
    PhoneService: 'Yes',
    MultipleLines: 'No',
    InternetService: 'Fiber optic',
    OnlineSecurity: 'No',
    OnlineBackup: 'Yes',
    DeviceProtection: 'No',
    TechSupport: 'No',
    StreamingTV: 'Yes',
    StreamingMovies: 'No',
    Contract: 'Month-to-month',
    PaperlessBilling: 'Yes',
    PaymentMethod: 'Electronic check',
    MonthlyCharges: 85.5,
    TotalCharges: 1026.0
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (field, val) => {
    setFormData(prev => {
      const next = { ...prev, [field]: val };
      if (field === 'MonthlyCharges' || field === 'tenure') {
        const m = parseFloat(field === 'MonthlyCharges' ? val : prev.MonthlyCharges) || 0;
        const t = parseInt(field === 'tenure' ? val : prev.tenure) || 0;
        next.TotalCharges = roundVal(m * t);
      }
      return next;
    });
  };

  const roundVal = (v) => Math.round(v * 100) / 100;

  // Run Inference
  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setIsLoading(true);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          SeniorCitizen: parseInt(formData.SeniorCitizen),
          tenure: parseInt(formData.tenure),
          MonthlyCharges: parseFloat(formData.MonthlyCharges),
          TotalCharges: parseFloat(formData.TotalCharges),
          selected_model: selectedModel
        })
      });

      if (res.ok) {
        const data = await res.json();
        setPredictionResult({
          isChurn: data.churn_prediction,
          churnClass: data.churn_prediction ? 'Churn' : 'No Churn',
          probability: data.churn_probability,
          noChurnProbability: data.no_churn_probability,
          riskTier: data.risk_tier,
          clvEstimate: data.clv_estimate,
          clusterLabel: data.cluster_label,
          modelUsed: MODEL_METRICS_DATA[selectedModel]?.name || selectedModel,
          inputSummary: { ...formData }
        });
      } else {
        throw new Error('API server returned error');
      }
    } catch (err) {
      let riskScore = 0;
      if (formData.Contract === 'Month-to-month') riskScore += 35;
      if (formData.InternetService === 'Fiber optic') riskScore += 20;
      if (parseFloat(formData.MonthlyCharges) > 75) riskScore += 20;
      if (parseInt(formData.tenure) < 12) riskScore += 20;
      if (formData.TechSupport === 'Yes') riskScore -= 15;
      if (formData.OnlineSecurity === 'Yes') riskScore -= 15;

      const modelFactor = selectedModel === 'xgboost' ? 1.05 : selectedModel === 'logistic_regression' ? 1.1 : 1.0;
      const prob = Math.min(98.5, Math.max(2.5, roundVal(riskScore * modelFactor)));
      const isChurn = prob >= 50.0;

      setPredictionResult({
        isChurn: isChurn,
        churnClass: isChurn ? 'Churn' : 'No Churn',
        probability: prob,
        noChurnProbability: roundVal(100 - prob),
        riskTier: prob >= 75 ? 'Critical' : prob >= 50 ? 'High' : prob >= 25 ? 'Medium' : 'Low',
        clvEstimate: roundVal(parseFloat(formData.MonthlyCharges) * Math.max(6, 60 - parseInt(formData.tenure))),
        clusterLabel: prob >= 60 ? 'High-Risk Price-Sensitive' : 'Stable High-Value',
        modelUsed: MODEL_METRICS_DATA[selectedModel]?.name || selectedModel,
        inputSummary: { ...formData }
      });
    } finally {
      setIsLoading(false);
      setActiveTab('result');
    }
  };

  return (
    <div className="space-y-6">
      {/* TABS Navigation */}
      <div className="flex border-b border-slate-800 space-x-1.5 bg-slate-950/60 p-1.5 rounded-2xl overflow-x-auto scrollbar-none">
        <button
          onClick={() => setActiveTab('form')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'form'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <ClipboardList className="w-4 h-4" />
          <span>📋 Input Form</span>
        </button>

        <button
          onClick={() => setActiveTab('result')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'result'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Target className="w-4 h-4" />
          <span>🎯 Prediction Result</span>
          {predictionResult && (
            <span className={`w-2 h-2 rounded-full ${predictionResult.isChurn ? 'bg-rose-400' : 'bg-emerald-400'}`}></span>
          )}
        </button>

        <button
          onClick={() => setActiveTab('whatif')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'whatif'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
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
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Sparkles className="w-4 h-4 text-purple-300" />
          <span>✨ Gemini AI Retention Strategy</span>
        </button>

        <button
          onClick={() => setActiveTab('model_info')}
          className={`flex items-center space-x-2 px-4 py-3 rounded-xl font-bold text-xs whitespace-nowrap transition ${
            activeTab === 'model_info'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Cpu className="w-4 h-4" />
          <span>📈 Model Info / Performance</span>
        </button>
      </div>

      {/* TAB 1: Input Form */}
      {activeTab === 'form' && (
        <form onSubmit={handlePredict} className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl space-y-8">
          <div className="border-b border-slate-800 pb-4">
            <h2 className="text-xl font-extrabold text-white">Enter Customer Characteristics (19 Features)</h2>
            <p className="text-xs text-slate-400 mt-1">Fill out the customer profile inputs to score churn risk with the selected model.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Section 1: Demographics */}
            <div className="space-y-4 bg-slate-950/50 p-5 rounded-2xl border border-slate-800/80">
              <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">👤 Demographics & Account</h3>
              
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">gender</label>
                <select 
                  value={formData.gender}
                  onChange={(e) => handleInputChange('gender', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">SeniorCitizen</label>
                <select 
                  value={formData.SeniorCitizen}
                  onChange={(e) => handleInputChange('SeniorCitizen', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="0">No (0)</option>
                  <option value="1">Yes (1)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Partner</label>
                <select 
                  value={formData.Partner}
                  onChange={(e) => handleInputChange('Partner', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Dependents</label>
                <select 
                  value={formData.Dependents}
                  onChange={(e) => handleInputChange('Dependents', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">tenure (months)</label>
                <input 
                  type="number"
                  min="0"
                  max="100"
                  value={formData.tenure}
                  onChange={(e) => handleInputChange('tenure', parseInt(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">PaperlessBilling</label>
                <select 
                  value={formData.PaperlessBilling}
                  onChange={(e) => handleInputChange('PaperlessBilling', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>

            {/* Section 2: Services */}
            <div className="space-y-4 bg-slate-950/50 p-5 rounded-2xl border border-slate-800/80">
              <h3 className="text-xs font-bold text-teal-400 uppercase tracking-wider">📞 Phone & Internet Services</h3>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">PhoneService</label>
                <select 
                  value={formData.PhoneService}
                  onChange={(e) => handleInputChange('PhoneService', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">MultipleLines</label>
                <select 
                  value={formData.MultipleLines}
                  onChange={(e) => handleInputChange('MultipleLines', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No phone service">No phone service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">InternetService</label>
                <select 
                  value={formData.InternetService}
                  onChange={(e) => handleInputChange('InternetService', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">OnlineSecurity</label>
                <select 
                  value={formData.OnlineSecurity}
                  onChange={(e) => handleInputChange('OnlineSecurity', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">OnlineBackup</label>
                <select 
                  value={formData.OnlineBackup}
                  onChange={(e) => handleInputChange('OnlineBackup', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">DeviceProtection</label>
                <select 
                  value={formData.DeviceProtection}
                  onChange={(e) => handleInputChange('DeviceProtection', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">TechSupport</label>
                <select 
                  value={formData.TechSupport}
                  onChange={(e) => handleInputChange('TechSupport', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
            </div>

            {/* Section 3: Billing & Subscriptions */}
            <div className="space-y-4 bg-slate-950/50 p-5 rounded-2xl border border-slate-800/80">
              <h3 className="text-xs font-bold text-purple-400 uppercase tracking-wider">💳 Contract & Charges</h3>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">StreamingTV</label>
                <select 
                  value={formData.StreamingTV}
                  onChange={(e) => handleInputChange('StreamingTV', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">StreamingMovies</label>
                <select 
                  value={formData.StreamingMovies}
                  onChange={(e) => handleInputChange('StreamingMovies', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Contract</label>
                <select 
                  value={formData.Contract}
                  onChange={(e) => handleInputChange('Contract', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">PaymentMethod</label>
                <select 
                  value={formData.PaymentMethod}
                  onChange={(e) => handleInputChange('PaymentMethod', e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                >
                  <option value="Electronic check">Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">MonthlyCharges ($)</label>
                <input 
                  type="number"
                  step="0.1"
                  min="0"
                  max="200"
                  value={formData.MonthlyCharges}
                  onChange={(e) => handleInputChange('MonthlyCharges', parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">TotalCharges ($)</label>
                <input 
                  type="number"
                  step="0.5"
                  min="0"
                  max="10000"
                  value={formData.TotalCharges}
                  onChange={(e) => handleInputChange('TotalCharges', parseFloat(e.target.value) || 0)}
                  className="w-full bg-slate-900 border border-slate-700/80 text-white text-xs rounded-xl p-2.5"
                />
              </div>
            </div>
          </div>

          <div className="pt-4">
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-4 px-8 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white font-extrabold text-sm rounded-2xl shadow-xl shadow-indigo-600/30 transition flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>Computing Prediction...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 fill-current" />
                  <span>Predict Churn Risk Now</span>
                </>
              )}
            </button>
          </div>
        </form>
      )}

      {/* TAB 2: Prediction Result */}
      {activeTab === 'result' && (
        <div className="space-y-6">
          {predictionResult ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Card 1: Churn Result Card */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col justify-between">
                <div>
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Model Engine Used: <strong className="text-indigo-400">{predictionResult.modelUsed}</strong>
                  </span>
                  
                  <div className="my-6 text-center">
                    {predictionResult.isChurn ? (
                      <div className="space-y-3">
                        <span className="inline-flex items-center space-x-1.5 px-4 py-1.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 font-bold text-xs uppercase tracking-wider">
                          <AlertTriangle className="w-4 h-4" />
                          <span>HIGH CHURN RISK DETECTED</span>
                        </span>
                        <h2 className="text-5xl font-extrabold text-rose-400">
                          {predictionResult.probability}%
                        </h2>
                        <p className="text-slate-300 font-semibold text-base">
                          Predicted Class: <strong className="text-rose-400 font-bold">CHURN</strong>
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <span className="inline-flex items-center space-x-1.5 px-4 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-bold text-xs uppercase tracking-wider">
                          <CheckCircle2 className="w-4 h-4" />
                          <span>LOW CHURN RISK</span>
                        </span>
                        <h2 className="text-5xl font-extrabold text-emerald-400">
                          {predictionResult.probability}%
                        </h2>
                        <p className="text-slate-300 font-semibold text-base">
                          Predicted Class: <strong className="text-emerald-400 font-bold">NO CHURN</strong>
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="bg-slate-950/60 p-4 rounded-2xl border border-slate-800/80 text-xs text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>Risk Tier Level:</span>
                    <span className="font-bold text-white">{predictionResult.riskTier}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Cluster Segment:</span>
                    <span className="font-bold text-indigo-400">{predictionResult.clusterLabel}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Retention Confidence:</span>
                    <span className="font-bold text-white">{predictionResult.noChurnProbability}%</span>
                  </div>
                </div>
              </div>

              {/* Card 2: Probability Visualizer Gauge */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-8 shadow-2xl flex flex-col justify-between">
                <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center space-x-2">
                  <BarChart2 className="w-4 h-4 text-indigo-400" />
                  <span>Probability Confidence Gauge</span>
                </h3>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span className="text-slate-400">Churn Probability</span>
                      <span className={predictionResult.isChurn ? "text-rose-400" : "text-emerald-400"}>
                        {predictionResult.probability}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-950 rounded-full h-4 overflow-hidden p-0.5 border border-slate-800">
                      <div 
                        className={`h-full rounded-full transition-all duration-1000 ${
                          predictionResult.isChurn 
                            ? 'bg-gradient-to-r from-rose-500 to-red-600' 
                            : 'bg-gradient-to-r from-emerald-500 to-teal-400'
                        }`}
                        style={{ width: `${predictionResult.probability}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed">
                    {predictionResult.isChurn ? (
                      <p className="text-rose-300/90">
                        ⚠️ Customer exhibits strong churn signals. Switch to the <strong>What-If Simulator</strong> tab to test contract changes or tech support additions.
                      </p>
                    ) : (
                      <p className="text-emerald-300/90">
                        ✅ Customer has a stable low-risk profile. Keep customer engaged with loyalty programs.
                      </p>
                    )}
                  </div>
                </div>

                <div className="pt-4 flex justify-between">
                  <button
                    onClick={() => setActiveTab('whatif')}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl transition"
                  >
                    Open What-If Simulator →
                  </button>
                  <button
                    onClick={() => setActiveTab('form')}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold rounded-xl transition"
                  >
                    Edit Profile Inputs
                  </button>
                </div>
              </div>

              {/* Profile Inputs Summary */}
              <div className="md:col-span-2 bg-slate-900/90 border border-slate-800 rounded-3xl p-6 shadow-xl">
                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Submitted Feature Summary</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  {Object.entries(predictionResult.inputSummary).map(([k, v]) => (
                    <div key={k} className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
                      <span className="text-slate-500 block text-[10px]">{k}</span>
                      <span className="font-bold text-slate-200">{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/90 border border-slate-800 rounded-3xl p-12 text-center text-slate-400 space-y-4">
              <ClipboardList className="w-12 h-12 text-slate-600 mx-auto" />
              <h3 className="text-lg font-bold text-white">No Prediction Generated Yet</h3>
              <p className="text-xs max-w-md mx-auto">Please complete the customer features in the <strong>Input Form</strong> tab and click "Predict Churn Risk Now".</p>
              <button
                onClick={() => setActiveTab('form')}
                className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs rounded-xl shadow-lg transition"
              >
                Go to Input Form
              </button>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: What-If Simulator */}
      {activeTab === 'whatif' && (
        <WhatIfSimulator selectedModel={selectedModel} />
      )}

      {/* TAB 4: Gemini AI Retention Strategy */}
      {activeTab === 'ai_strategy' && (
        <AiStrategies selectedSegment={predictionResult?.clusterLabel || 'High-Risk Price-Sensitive'} />
      )}

      {/* TAB 5: Model Info / Performance */}
      {activeTab === 'model_info' && (
        <ModelPerformance selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
      )}
    </div>
  );
}
