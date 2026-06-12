import { Database, Layers3, ListChecks, ScanSearch } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import StatusIndicator from '../components/StatusIndicator';

const Dashboard = () => {
  const overview = [
    { label: 'Pairs Analyzed', value: '6,000', detail: 'Across the current dataset', icon: Database },
    { label: 'Difficulty Groups', value: '4', detail: 'Easy, medium, hard and mixed', icon: Layers3 },
    { label: 'Signals Evaluated', value: '6', detail: 'Similarity and identity features', icon: ScanSearch },
    { label: 'Pipeline Stages', value: '7', detail: 'Processing workflow completed', icon: ListChecks },
  ];

  const clusterData = [
    { name: 'Cluster 0 (Easy)', value: 2134, percentage: 35.6 },
    { name: 'Cluster 1 (Medium)', value: 1334, percentage: 22.2 },
    { name: 'Cluster 2 (Hard)', value: 790, percentage: 13.1 },
    { name: 'Cluster 3 (Mixed)', value: 1742, percentage: 29.0 },
  ];

  const featureImportance = [
    { name: 'Cosine Sim.', value: 28.7 },
    { name: 'Content Sim.', value: 22.4 },
    { name: 'Embedding Sim.', value: 18.9 },
    { name: 'TFIDF Sim.', value: 15.6 },
    { name: 'Metadata Sim.', value: 9.8 },
    { name: 'SHA256 Match', value: 4.6 },
  ];

  const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];

  return (
    <div className="p-4 md:p-8 space-y-8 animate-fade-in-up">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Dashboard</h1>
        <p className="text-slate-400">Dataset, pipeline and model evaluation overview</p>
      </div>

      {/* Dataset Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {overview.map(({ label, value, detail, icon: Icon }) => (
          <div
            key={label}
            className="group bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6 transition-all duration-300 hover:border-blue-500/50 hover:-translate-y-1 hover:shadow-lg hover:shadow-blue-500/10"
          >
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-slate-400 text-sm font-semibold">{label}</h3>
              <div className="rounded-lg bg-blue-500/10 p-2 text-blue-400 ring-1 ring-blue-500/20 transition-colors group-hover:bg-blue-500/20">
                <Icon size={18} />
              </div>
            </div>
            <p className="text-3xl font-bold text-white">{value}</p>
            <p className="mt-2 text-xs text-slate-500">{detail}</p>
          </div>
        ))}
      </div>

      {/* Status Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-6">Pipeline Status</h2>
          <div className="space-y-4">
            <StatusIndicator label="Data Validation" status="completed" percentage={100} />
            <StatusIndicator label="Geometric Clustering" status="completed" percentage={100} />
            <StatusIndicator label="Active Learning" status="completed" percentage={100} />
            <StatusIndicator label="Genetic Programming" status="completed" percentage={100} />
            <StatusIndicator label="Bloom Filter Setup" status="completed" percentage={100} />
            <StatusIndicator label="Role Hierarchy" status="completed" percentage={100} />
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-6">System Status</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Model Status</span>
              <span className="status-badge status-success">Trained</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">API Server</span>
              <span className="status-badge status-success">Active</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Database</span>
              <span className="status-badge status-success">Online</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Data Cache</span>
              <span className="status-badge status-success">Ready</span>
            </div>
            <div className="mt-6 pt-4 border-t border-slate-700 text-sm text-slate-400">
              <p>Last Updated: Just now</p>
              <p className="mt-1">Total Predictions: 6,000</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feature Importance */}
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-6">Top Features by Importance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="name" stroke="#94a3b8" style={{ fontSize: '12px' }} />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                formatter={(value) => `${(value as number).toFixed(1)}%`}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Evaluation Readiness */}
        <div className="relative overflow-hidden bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <div className="absolute -right-16 -top-16 h-40 w-40 rounded-full bg-blue-500/10 blur-3xl" />
          <div className="relative">
            <div className="mb-6 flex items-start justify-between gap-4">
              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-blue-400">Evaluation guardrail</p>
                <h2 className="text-xl font-semibold">Performance Metrics Pending</h2>
              </div>
              <span className="status-badge status-warning whitespace-nowrap">Needs validation</span>
            </div>
            <p className="max-w-xl text-sm leading-6 text-slate-400">
              Accuracy, precision, recall and F1 are intentionally hidden until the model is tested on an independent holdout set. This avoids presenting training performance as real-world quality.
            </p>
            <div className="mt-6 space-y-3">
              {[
                'Use a separate, unseen test split',
                'Report class-level and aggregate results',
                'Compare training and validation performance',
              ].map((item) => (
                <div key={item} className="flex items-center gap-3 rounded-lg border border-slate-700/80 bg-slate-900/40 px-4 py-3">
                  <span className="h-2 w-2 rounded-full bg-blue-400" />
                  <span className="text-sm text-slate-300">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Cluster Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-6">Cluster Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={clusterData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ percentage }) => `${percentage}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {clusterData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-6">Cluster Details</h2>
          <div className="space-y-4">
            {clusterData.map((cluster, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700">
                <div>
                  <p className="font-semibold">{cluster.name}</p>
                  <p className="text-sm text-slate-400">{cluster.value.toLocaleString()} samples</p>
                </div>
                <div className="text-right">
                  <p className="font-bold text-lg">{cluster.percentage}%</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-xl p-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-sm font-semibold text-slate-400 mb-2">Total Datasets</h3>
            <p className="text-3xl font-bold">1</p>
            <p className="text-xs text-slate-500 mt-1">AG News + BBC News</p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-400 mb-2">Total Pairs Analyzed</h3>
            <p className="text-3xl font-bold">6,000</p>
            <p className="text-xs text-slate-500 mt-1">Training & Validation</p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-400 mb-2">Pipeline Phases</h3>
            <p className="text-3xl font-bold">7</p>
            <p className="text-xs text-slate-500 mt-1">All successfully completed</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
