interface StatusIndicatorProps {
  label: string;
  status: 'completed' | 'in-progress' | 'pending';
  percentage: number;
}

const StatusIndicator = ({ label, status, percentage }: StatusIndicatorProps) => {
  const statusConfig = {
    completed: { color: 'bg-green-500', textColor: 'text-green-400', bgColor: 'bg-green-500/20' },
    'in-progress': { color: 'bg-blue-500', textColor: 'text-blue-400', bgColor: 'bg-blue-500/20' },
    pending: { color: 'bg-yellow-500', textColor: 'text-yellow-400', bgColor: 'bg-yellow-500/20' },
  };

  const config = statusConfig[status];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-300">{label}</span>
        <span className={`text-xs font-semibold px-2 py-1 rounded ${config.bgColor} ${config.textColor}`}>
          {status.charAt(0).toUpperCase() + status.slice(1)}
        </span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${config.color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

export default StatusIndicator;
