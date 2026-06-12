import { ArrowUpRight } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string;
  trend: string;
  status: 'success' | 'warning' | 'error' | 'info';
}

const MetricCard = ({ label, value, trend, status }: MetricCardProps) => {
  const statusColors = {
    success: 'text-green-400 bg-green-500/20',
    warning: 'text-yellow-400 bg-yellow-500/20',
    error: 'text-red-400 bg-red-500/20',
    info: 'text-blue-400 bg-blue-500/20',
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300">
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-slate-400 text-sm font-semibold">{label}</h3>
        <div className={`flex items-center gap-1 text-sm font-semibold px-2 py-1 rounded ${statusColors[status]}`}>
          <ArrowUpRight size={14} />
          {trend}
        </div>
      </div>
      <p className="text-3xl font-bold text-white mb-2">{value}</p>
      <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full w-full transition-all ${statusColors[status]}`} />
      </div>
    </div>
  );
};

export default MetricCard;
