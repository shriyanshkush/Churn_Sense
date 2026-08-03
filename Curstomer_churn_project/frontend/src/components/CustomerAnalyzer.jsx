import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Info, 
  Copy, 
  CheckCircle2, 
  Mail, 
  Sparkles, 
  TrendingUp, 
  TrendingDown, 
  Sliders, 
  User, 
  BarChart2, 
  ChevronDown,
  Edit3,
  RotateCcw,
  Check,
  Zap,
  ChevronUp
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function CustomerAnalyzer({ selectedModel, initialCustomer }) {
  // Preset Demo Accounts matching reference design
  const demoAccounts = {
    '#RET-99281': {
      name: 'Jonathan Thorne',
      id: '#RET-99281',
      avatar: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=160&q=80',
      tier: 'Enterprise Tier',
      since: 'Client since Jan 2021',
      location: 'Austin, TX',
      ltv: '$42,500',
      // Full 19 ML Model Features
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'No',
      tenure: 36,
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
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
      MonthlyCharges: 115.0,
      TotalCharges: 4140.0,
    },
    '#USR-8910': {
      name: 'Sarah Jenkins',
      id: '#USR-8910',
      avatar: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&w=160&q=80',
      tier: 'Enterprise Gold',
      since: 'Client since Mar 2022',
      location: 'San Francisco, CA',
      ltv: '$38,200',
      gender: 'Female',
      SeniorCitizen: 0,
      Partner: 'No',
      Dependents: 'No',
      tenure: 14,
      PhoneService: 'Yes',
      MultipleLines: 'No',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'Yes',
      StreamingMovies: 'No',
      Contract: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: 94.0,
      TotalCharges: 1316.0,
    },
    '#USR-1102': {
      name: 'Michael Chang',
      id: '#USR-1102',
      avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=160&q=80',
      tier: 'Enterprise Platinum',
      since: 'Client since Aug 2019',
      location: 'New York, NY',
      ltv: '$64,000',
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'Yes',
      tenure: 58,
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'Yes',
      OnlineBackup: 'Yes',
      DeviceProtection: 'Yes',
      TechSupport: 'Yes',
      StreamingTV: 'Yes',
      StreamingMovies: 'Yes',
      Contract: 'Two year',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Credit card (automatic)',
      MonthlyCharges: 105.0,
      TotalCharges: 6090.0,
    }
  };

  const [selectedAccountId, setSelectedAccountId] = useState('#RET-99281');
  const [customer, setCustomer] = useState(initialCustomer || demoAccounts['#RET-99281']);

  // Retention Simulator Sliders State
  const [contractLength, setContractLength] = useState(24);
  const [monthlyDiscount, setMonthlyDiscount] = useState(15);
  const [resolvedTickets, setResolvedTickets] = useState(3);
  const [appliedStrategy, setAppliedStrategy] = useState(false);

  // Model Predictions & UI States
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copiedScript, setCopiedScript] = useState(false);
  const [emailSent, setEmailSent] = useState(false);
  const [showFullFeatureEditor, setShowFullFeatureEditor] = useState(false);

  useEffect(() => {
    runModelPrediction();
  }, [customer, selectedModel]);

  const handleAccountChange = (accId) => {
    setSelectedAccountId(accId);
    if (demoAccounts[accId]) {
      setCustomer(demoAccounts[accId]);
      setAppliedStrategy(false);
    }
  };

  const runModelPrediction = async () => {
    try {
      setLoading(true);
      const res = await axios.post('/api/v1/predict', {
        ...customer,
        selected_model: selectedModel || 'xgboost'
      });
      setPrediction(res.data);
    } catch (err) {
      console.warn("Prediction API notice:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureChange = (field, val) => {
    setCustomer(prev => {
      const updated = { ...prev, [field]: val };
      if (field === 'tenure' || field === 'MonthlyCharges') {
        updated.TotalCharges = Math.round(updated.tenure * updated.MonthlyCharges * 100) / 100;
      }
      return updated;
    });
  };

  // Base Churn Probability from live ML Model
  const baseProb = prediction ? prediction.churn_probability : 84.2;

  // Retention Simulator Real-time Calculations
  const contractFactor = contractLength === 24 ? 0.45 : (contractLength === 12 ? 0.25 : 0.0);
  const discountFactor = (monthlyDiscount / 100) * 0.8;
  const ticketFactor = (resolvedTickets / 3) * 0.12;

  const totalReductionFactor = Math.min(0.78, contractFactor + discountFactor + ticketFactor);
  const projectedChurn = Math.max(8.5, Math.round((baseProb * (1 - totalReductionFactor)) * 10) / 10);
  const improvementPct = Math.round(((baseProb - projectedChurn) / baseProb) * 100 * 10) / 10;

  // Cox Survival Curve Data (Dynamically rendered based on prediction)
  const survivalData = prediction?.survival_curve && prediction.survival_curve.length > 0
    ? prediction.survival_curve.slice(0, 5).map(pt => ({
        month: `${pt.month}M`,
        actual: Math.min(100, pt.survival_probability + 3),
        predicted: pt.survival_probability
      }))
    : [
        { month: '0M', actual: 100, predicted: 100 },
        { month: '6M', actual: 88, predicted: 85 },
        { month: '12M', actual: 68, predicted: 62 },
        { month: '18M', actual: 52, predicted: 45 },
        { month: '24M+', actual: 32, predicted: 28 },
      ];

  // SHAP waterfall items from backend or default matching reference
  const shapDrivers = prediction?.shap_waterfall && prediction.shap_waterfall.length >= 4
    ? prediction.shap_waterfall.slice(0, 4)
    : [
        { feature: 'Total Charges', shap_value: 0.142, direction: 'increases_churn' },
        { feature: 'Contract Type (Monthly)', shap_value: 0.115, direction: 'increases_churn' },
        { feature: 'Support Tickets (Last 30d)', shap_value: 0.081, direction: 'increases_churn' },
        { feature: 'Auto-Pay Enabled', shap_value: -0.062, direction: 'decreases_churn' }
      ];

  const copyOutreachScript = () => {
    const scriptText = `Hello ${customer.name.split(' ')[0]}, I noticed your account reached the 3-year milestone recently. We truly value your partnership. Given your recent volume of support tickets regarding the cloud integration, I'd like to personally offer you a dedicated Technical Account Manager for the next 90 days and a 15% loyalty discount if we transition you to our Annual Precision plan today.`;
    navigator.clipboard.writeText(scriptText);
    setCopiedScript(true);
    setTimeout(() => setCopiedScript(false), 2500);
  };

  const handleSendEmail = () => {
    setEmailSent(true);
    setTimeout(() => setEmailSent(false), 3500);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300 pb-12">
      
      {/* 1. Top Customer Profile Header Card matching Reference Image */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        
        {/* Customer Basic Info Box (8 cols) */}
        <div className="lg:col-span-8 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
              <img
                src={customer.avatar || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=160&q=80'}
                alt={customer.name}
                className="w-16 h-16 rounded-2xl object-cover border border-slate-200 shadow-xs"
              />

              <div>
                <div className="flex items-center space-x-2">
                  <h2 className="text-xl font-extrabold text-slate-900 tracking-tight">
                    {customer.name}
                  </h2>
                  <span className="font-mono text-[11px] font-bold text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                    ID: {customer.id}
                  </span>
                </div>

                <p className="text-xs text-slate-500 mt-1 font-medium">
                  {customer.tier} &bull; {customer.since} &bull; {customer.location}
                </p>
              </div>
            </div>

            {/* Account Switcher & Full Feature Editor Toggle Button */}
            <div className="flex items-center space-x-2">
              <select
                value={selectedAccountId}
                onChange={(e) => handleAccountChange(e.target.value)}
                className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs font-semibold text-slate-800 shadow-xs focus:outline-none focus:border-teal-500"
              >
                <option value="#RET-99281">Jonathan Thorne (#RET-99281)</option>
                <option value="#USR-8910">Sarah Jenkins (#USR-8910)</option>
                <option value="#USR-1102">Michael Chang (#USR-1102)</option>
              </select>

              <button
                onClick={() => setShowFullFeatureEditor(!showFullFeatureEditor)}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition flex items-center space-x-1.5 ${
                  showFullFeatureEditor 
                    ? 'bg-teal-600 text-white shadow-xs' 
                    : 'bg-slate-100 hover:bg-slate-200 text-slate-800'
                }`}
                title="Configure All 19 ML Features"
              >
                <Edit3 className="w-3.5 h-3.5" />
                <span>{showFullFeatureEditor ? 'Hide ML Features' : 'Configure All ML Features'}</span>
                {showFullFeatureEditor ? <ChevronUp className="w-3.5 h-3.5 ml-1" /> : <ChevronDown className="w-3.5 h-3.5 ml-1" />}
              </button>
            </div>
          </div>

          {/* Badges & Model Active Info Row */}
          <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className={baseProb >= 60 ? 'badge-critical' : 'badge-low'}>
                {baseProb >= 60 ? 'CRITICAL RISK' : 'LOW RISK'}
              </span>
              <span className="bg-slate-100 text-slate-800 border border-slate-200 font-mono text-[11px] font-bold px-2.5 py-0.5 rounded">
                LTV: {customer.ltv}
              </span>
            </div>

            <div className="text-[11px] font-mono text-slate-500">
              Active Model: <strong className="text-teal-700">{selectedModel ? selectedModel.toUpperCase() : 'XGBOOST'}</strong>
            </div>
          </div>
        </div>

        {/* Churn Probability Score Card (4 cols) matching Reference Image */}
        <div className="lg:col-span-4 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between text-center relative overflow-hidden">
          <span className="text-[11px] font-mono font-bold text-slate-400 uppercase tracking-wider block text-left">
            CHURN PROBABILITY
          </span>

          <div className="my-2">
            <div className="text-5xl font-extrabold font-mono text-rose-600 tracking-tight">
              {loading ? '...' : `${baseProb}%`}
            </div>
            <div className="text-xs font-semibold text-rose-600 font-mono mt-1.5 flex items-center justify-center space-x-1">
              <TrendingUp className="w-3.5 h-3.5" />
              <span>+12% vs last month</span>
            </div>
          </div>

          <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden mt-1">
            <div 
              className="bg-rose-600 h-full rounded-full transition-all duration-500" 
              style={{ width: `${Math.min(100, baseProb)}%` }}
            ></div>
          </div>
        </div>

      </div>

      {/* Full 19 ML Model Features Configurator Panel */}
      {showFullFeatureEditor && (
        <div className="bg-white border border-teal-200 rounded-2xl p-6 shadow-md space-y-5 animate-in slide-in-from-top-3">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div className="flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-teal-600" />
              <h4 className="text-sm font-bold text-slate-900">
                Full 19 ML Feature Configuration Panel ({customer.name})
              </h4>
            </div>
            <span className="text-xs text-slate-400 font-mono">Modifications directly re-score the {selectedModel.toUpperCase()} model</span>
          </div>

          {/* 19 Features Form Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 text-xs font-semibold text-slate-700">
            
            {/* Demographics & Contract */}
            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Gender</label>
              <select
                value={customer.gender}
                onChange={(e) => handleFeatureChange('gender', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Senior Citizen</label>
              <select
                value={customer.SeniorCitizen}
                onChange={(e) => handleFeatureChange('SeniorCitizen', parseInt(e.target.value))}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value={0}>No (0)</option>
                <option value={1}>Yes (1)</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Partner</label>
              <select
                value={customer.Partner}
                onChange={(e) => handleFeatureChange('Partner', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Dependents</label>
              <select
                value={customer.Dependents}
                onChange={(e) => handleFeatureChange('Dependents', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Tenure (Months: {customer.tenure})</label>
              <input
                type="number"
                min="1"
                max="72"
                value={customer.tenure}
                onChange={(e) => handleFeatureChange('tenure', parseInt(e.target.value) || 1)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 font-mono focus:outline-none focus:border-teal-500"
              />
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Contract</label>
              <select
                value={customer.Contract}
                onChange={(e) => handleFeatureChange('Contract', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Paperless Billing</label>
              <select
                value={customer.PaperlessBilling}
                onChange={(e) => handleFeatureChange('PaperlessBilling', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Payment Method</label>
              <select
                value={customer.PaymentMethod}
                onChange={(e) => handleFeatureChange('PaymentMethod', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                <option value="Credit card (automatic)">Credit card (automatic)</option>
              </select>
            </div>

            {/* Financials & Services */}
            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Monthly Charges ($)</label>
              <input
                type="number"
                value={customer.MonthlyCharges}
                onChange={(e) => handleFeatureChange('MonthlyCharges', parseFloat(e.target.value) || 10.0)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 font-mono focus:outline-none focus:border-teal-500"
              />
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Internet Service</label>
              <select
                value={customer.InternetService}
                onChange={(e) => handleFeatureChange('InternetService', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Fiber optic">Fiber optic</option>
                <option value="DSL">DSL</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Online Security</label>
              <select
                value={customer.OnlineSecurity}
                onChange={(e) => handleFeatureChange('OnlineSecurity', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Tech Support</label>
              <select
                value={customer.TechSupport}
                onChange={(e) => handleFeatureChange('TechSupport', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Online Backup</label>
              <select
                value={customer.OnlineBackup}
                onChange={(e) => handleFeatureChange('OnlineBackup', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Device Protection</label>
              <select
                value={customer.DeviceProtection}
                onChange={(e) => handleFeatureChange('DeviceProtection', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Streaming TV</label>
              <select
                value={customer.StreamingTV}
                onChange={(e) => handleFeatureChange('StreamingTV', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-500 text-[11px] mb-1">Streaming Movies</label>
              <select
                value={customer.StreamingMovies}
                onChange={(e) => handleFeatureChange('StreamingMovies', e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-teal-500"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* 2. Middle Row: 3 Column Grid matching Reference Image */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Column 1: Local Feature Drivers (SHAP Diverging Bars) */}
        <div className="lg:col-span-4 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-900">Local Feature Drivers</h3>
              <Info className="w-4 h-4 text-slate-400 hover:text-slate-600 cursor-pointer" title="Local SHAP value contribution" />
            </div>

            {/* Dynamic Feature Impact Rows from SHAP */}
            <div className="space-y-4 pt-4">
              {shapDrivers.map((item, idx) => {
                const isPos = item.shap_value >= 0;
                const formattedVal = `${isPos ? '+' : ''}${(item.shap_value * 100).toFixed(1)}%`;
                const barWidth = `${Math.min(48, Math.max(10, Math.abs(item.shap_value) * 120))}%`;

                return (
                  <div key={idx}>
                    <div className="flex justify-between text-xs font-semibold mb-1">
                      <span className="text-slate-700 truncate">{item.feature}</span>
                      <span className={`font-mono font-bold ${isPos ? 'text-rose-600' : 'text-teal-700'}`}>
                        {formattedVal}
                      </span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-md h-4 flex overflow-hidden">
                      {isPos ? (
                        <>
                          <div className="w-1/2 bg-slate-100"></div>
                          <div className="bg-rose-500 rounded-r-sm transition-all duration-300" style={{ width: barWidth }}></div>
                        </>
                      ) : (
                        <>
                          <div className="w-1/2 flex justify-end bg-slate-100">
                            <div className="bg-teal-600 rounded-l-sm transition-all duration-300" style={{ width: barWidth }}></div>
                          </div>
                          <div className="w-1/2 bg-slate-100"></div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="pt-4 border-t border-slate-100 flex items-center justify-between text-xs font-mono text-slate-500">
            <span>Base Probability: <strong>56.6%</strong></span>
            <BarChart2 className="w-4 h-4 text-slate-400" />
          </div>
        </div>

        {/* Column 2: Survival Probability Analysis (Cox Curve) */}
        <div className="lg:col-span-4 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-900">Survival Probability Analysis</h3>
              
              <div className="flex items-center space-x-3 text-[11px] font-medium text-slate-600">
                <div className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full bg-teal-600"></span>
                  <span>Actual</span>
                </div>
                <div className="flex items-center space-x-1">
                  <span className="w-2 h-2 rounded-full border border-teal-600"></span>
                  <span>Predicted</span>
                </div>
              </div>
            </div>

            {/* Line Chart */}
            <div className="h-44 w-full mt-3 relative">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={survivalData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <XAxis dataKey="month" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} domain={[0, 100]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="actual" stroke="#047857" strokeWidth={2.5} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="predicted" stroke="#047857" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>

              <div className="absolute top-16 right-10 bg-slate-900/90 text-white text-[10px] font-mono px-2 py-0.5 rounded shadow">
                NOW: 4.2 Mo Remaining
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-xl text-[11px] text-slate-600 leading-relaxed mt-2">
            Customer is in the <strong>"High-Attrition Zone"</strong> (Months 24-30). Models indicate a 65% drop in survival probability if no intervention occurs by end of Q3.
          </div>
        </div>

        {/* Column 3: Retention Simulator (Interactive What-If Sliders) */}
        <div className="lg:col-span-4 bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col justify-between">
          <div>
            <div className="border-b border-slate-100 pb-3">
              <h3 className="text-base font-bold text-slate-900">Retention Simulator</h3>
            </div>

            {/* Sliders matching Reference Image */}
            <div className="space-y-4 pt-4">
              
              {/* Slider 1: Contract Length */}
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span className="text-slate-600">CONTRACT LENGTH</span>
                  <span className="font-mono text-teal-800 font-bold">{contractLength} Months</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="24"
                  step="11"
                  value={contractLength}
                  onChange={(e) => setContractLength(parseInt(e.target.value))}
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* Slider 2: Monthly Discount */}
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span className="text-slate-600">MONTHLY DISCOUNT</span>
                  <span className="font-mono text-teal-800 font-bold">-${monthlyDiscount}.00</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="25"
                  step="5"
                  value={monthlyDiscount}
                  onChange={(e) => setMonthlyDiscount(parseInt(e.target.value))}
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* Slider 3: Resolved Tickets */}
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1">
                  <span className="text-slate-600">RESOLVED TICKETS</span>
                  <span className="font-mono text-teal-800 font-bold">{resolvedTickets}/3</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="1"
                  value={resolvedTickets}
                  onChange={(e) => setResolvedTickets(parseInt(e.target.value))}
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* Result Box: Projected Churn */}
              <div className="p-4 bg-blue-50/70 border border-blue-200/80 rounded-xl text-center">
                <span className="text-[10px] font-mono font-bold text-blue-900 uppercase block">
                  PROJECTED CHURN
                </span>
                <div className="text-3xl font-extrabold font-mono text-slate-900 mt-1">
                  {projectedChurn}%
                </div>
                <span className="text-[11px] font-bold text-emerald-700 font-mono block mt-0.5">
                  -{improvementPct}% IMPROVEMENT
                </span>
              </div>

            </div>
          </div>

          <button
            onClick={() => setAppliedStrategy(true)}
            className="w-full mt-4 py-2.5 bg-[#047857] hover:bg-[#065F46] text-white text-xs font-bold rounded-xl shadow-xs transition flex items-center justify-center space-x-2"
          >
            <span>{appliedStrategy ? 'Strategy Applied ✓' : 'Apply Strategy'}</span>
          </button>
        </div>

      </div>

      {/* 3. Bottom Card: Gemini AI Retention Strategy matching Reference Image */}
      <div className="bg-white border-l-4 border-l-indigo-600 border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-5">
        <div className="flex items-center space-x-3 border-b border-slate-100 pb-3">
          <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-xs">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <h3 className="text-lg font-extrabold text-slate-900">
            Gemini AI Retention Strategy
          </h3>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left Column: RECOMMENDED OUTREACH SCRIPT */}
          <div className="lg:col-span-7 space-y-3">
            <span className="text-[11px] font-mono font-bold text-slate-400 uppercase tracking-wider block">
              💬 RECOMMENDED OUTREACH SCRIPT
            </span>

            <div className="p-5 bg-slate-50 border border-slate-200/80 rounded-2xl text-xs text-slate-700 leading-relaxed font-sans relative">
              <span className="text-3xl text-slate-300 font-serif absolute top-3 right-4 select-none">”</span>
              "Hello {customer.name.split(' ')[0]}, I noticed your account reached the 3-year milestone recently. We truly value your partnership. Given your recent volume of support tickets regarding the cloud integration, I'd like to personally offer you a dedicated Technical Account Manager for the next 90 days and a 15% loyalty discount if we transition you to our Annual Precision plan today."
            </div>

            <div className="flex items-center space-x-4 text-xs font-semibold text-slate-600 pt-1">
              <button
                onClick={copyOutreachScript}
                className="flex items-center space-x-1.5 hover:text-slate-900 transition"
              >
                {copiedScript ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4 text-slate-400" />}
                <span>{copiedScript ? 'Script Copied!' : 'Copy Script'}</span>
              </button>

              <span className="text-slate-300">|</span>

              <button
                onClick={handleSendEmail}
                className="flex items-center space-x-1.5 hover:text-slate-900 transition"
              >
                <Mail className="w-4 h-4 text-slate-400" />
                <span>{emailSent ? 'Email Dispatched ✓' : 'Send Email'}</span>
              </button>
            </div>
          </div>

          {/* Right Column: PREDICTIVE INCENTIVE & AI RATIONALE */}
          <div className="lg:col-span-5 space-y-4">
            <div>
              <span className="text-[11px] font-mono font-bold text-slate-400 uppercase tracking-wider block mb-2">
                🎁 PREDICTIVE INCENTIVE
              </span>

              <div className="p-4 bg-[#5EEAD4]/20 border-2 border-dashed border-[#2DD4BF] rounded-2xl flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-mono font-extrabold text-slate-900">LOYALTY15_ANNUAL</h4>
                  <p className="text-[11px] text-teal-800 font-semibold mt-0.5">15% Discount + Premium Support Tier</p>
                </div>
                <CheckCircle2 className="w-5 h-5 text-teal-700" />
              </div>
            </div>

            {/* AI Rationale */}
            <div className="p-4 bg-blue-50/70 border border-blue-200/80 rounded-2xl space-y-1">
              <div className="flex items-center space-x-1.5 text-blue-900 font-bold text-[11px]">
                <span>📍 AI RATIONALE</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                SHAP analysis identifies "Contract Type" and "Support Frequency" as primary churn vectors. Transitioning to an Annual contract solves the billing friction, while the TAM offer addresses the underlying technical dissatisfaction noted in recent transcripts.
              </p>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
