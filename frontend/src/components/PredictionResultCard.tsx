import { Check, X } from 'lucide-react';

interface Prediction {
  id: number;
  pair: string;
  confidence: number;
  isDuplicate: boolean;
  cluster: string;
  uncertainty: number;
}

interface PredictionResultCardProps {
  prediction: Prediction;
}

const PredictionResultCard = ({ prediction }: PredictionResultCardProps) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 95) return 'text-green-400 bg-green-500/20';
    if (confidence >= 85) return 'text-blue-400 bg-blue-500/20';
    if (confidence >= 70) return 'text-yellow-400 bg-yellow-500/20';
    return 'text-red-400 bg-red-500/20';
  };

  const getUncertaintyColor = (uncertainty: number) => {
    if (uncertainty <= 0.05) return 'bg-green-500/20 text-green-400';
    if (uncertainty <= 0.15) return 'bg-blue-500/20 text-blue-400';
    if (uncertainty <= 0.25) return 'bg-yellow-500/20 text-yellow-400';
    return 'bg-red-500/20 text-red-400';
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-6 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        {/* Left Section - Pair Info */}
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-lg ${prediction.isDuplicate ? 'bg-red-500/20' : 'bg-green-500/20'}`}>
              {prediction.isDuplicate ? (
                <X size={20} className="text-red-400" />
              ) : (
                <Check size={20} className="text-green-400" />
              )}
            </div>
            <div>
              <p className="font-semibold text-white">{prediction.pair}</p>
              <p className="text-xs text-slate-400">ID: {prediction.id}</p>
            </div>
          </div>
        </div>

        {/* Right Section - Metrics */}
        <div className="flex flex-col sm:flex-row gap-4 lg:gap-6">
          {/* Confidence */}
          <div className="text-center">
            <p className="text-xs text-slate-400 mb-1">Confidence</p>
            <div className={`text-2xl font-bold px-3 py-2 rounded-lg ${getConfidenceColor(prediction.confidence)}`}>
              {prediction.confidence.toFixed(1)}%
            </div>
          </div>

          {/* Uncertainty */}
          <div className="text-center">
            <p className="text-xs text-slate-400 mb-1">Uncertainty</p>
            <div className={`text-lg font-bold px-3 py-2 rounded-lg ${getUncertaintyColor(prediction.uncertainty)}`}>
              {(prediction.uncertainty * 100).toFixed(1)}%
            </div>
          </div>

          {/* Cluster */}
          <div className="text-center">
            <p className="text-xs text-slate-400 mb-1">Cluster</p>
            <div className="px-3 py-2 rounded-lg bg-purple-500/20 text-purple-400 font-semibold text-sm">
              {prediction.cluster}
            </div>
          </div>

          {/* Status Badge */}
          <div className="text-center">
            <p className="text-xs text-slate-400 mb-1">Status</p>
            <div className={`px-4 py-2 rounded-lg font-semibold ${prediction.isDuplicate ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
              {prediction.isDuplicate ? 'Duplicate' : 'Unique'}
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar for Confidence */}
      <div className="mt-4 pt-4 border-t border-slate-700">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs text-slate-400">Confidence Level</span>
          <span className="text-xs font-semibold text-slate-300">{prediction.confidence.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
            style={{ width: `${prediction.confidence}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default PredictionResultCard;
