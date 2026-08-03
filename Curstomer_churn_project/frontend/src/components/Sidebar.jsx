import React from 'react';
import { 
  LayoutDashboard, 
  Compass, 
  UserCheck, 
  Sparkles, 
  Activity, 
  Rocket, 
  Settings, 
  HelpCircle,
  ShieldCheck
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, onRunBatch }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'explorer', label: 'Explorer', icon: Compass },
    { id: 'customer_detail', label: 'Customer Detail', icon: UserCheck },
    { id: 'ai_strategies', label: 'AI Strategies', icon: Sparkles },
    { id: 'monitoring', label: 'Monitoring', icon: Activity },
  ];

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col justify-between min-h-screen sticky top-0 h-screen z-20 select-none shadow-sm">
      <div>
        {/* Brand Header */}
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-teal-600 to-emerald-400 flex items-center justify-center text-white font-bold text-lg shadow-md shadow-teal-500/20">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 tracking-tight leading-none">
                ChurnAnalytics Pro
              </h1>
              <span className="text-[11px] font-medium text-slate-400 tracking-wide block mt-1">
                Enterprise ML v2.4
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-[#5EEAD4] text-slate-900 shadow-sm font-bold'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-slate-900' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer & Actions */}
      <div className="p-4 space-y-4 border-t border-slate-100">
        {/* Run Batch Scoring Button */}
        <button
          onClick={onRunBatch || (() => setActiveTab('monitoring'))}
          className="w-full py-2.5 px-4 bg-[#047857] hover:bg-[#065F46] text-white text-xs font-bold rounded-xl shadow-md transition flex items-center justify-center space-x-2 active:scale-95"
        >
          <Rocket className="w-4 h-4 text-emerald-300" />
          <span>Run Batch Scoring</span>
        </button>

        {/* Sub-links */}
        <div className="space-y-1">
          <button 
            onClick={() => alert("Settings configuration panel")}
            className="w-full flex items-center space-x-3 px-3.5 py-2 rounded-lg text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition"
          >
            <Settings className="w-4 h-4 text-slate-400" />
            <span>Settings</span>
          </button>

          <button 
            onClick={() => alert("Enterprise ChurnAnalytics Support & Docs")}
            className="w-full flex items-center space-x-3 px-3.5 py-2 rounded-lg text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition"
          >
            <HelpCircle className="w-4 h-4 text-slate-400" />
            <span>Support</span>
          </button>
        </div>

        {/* User Profile */}
        <div className="pt-2 border-t border-slate-100 flex items-center space-x-3 px-1">
          <div className="relative">
            <img
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=120&q=80"
              alt="Alex Chen"
              className="w-9 h-9 rounded-full object-cover border border-slate-200"
            />
            <span className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-white"></span>
          </div>
          <div className="overflow-hidden">
            <h4 className="text-xs font-bold text-slate-900 truncate">Alex Chen</h4>
            <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider truncate">
              SR. ML LEAD
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
