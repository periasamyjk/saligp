import { useState } from 'react';
import { AlertCircle, ChevronDown, FileSearch, Upload } from 'lucide-react';
import FileUploadZone from '../components/FileUploadZone';
import PredictionResultCard from '../components/PredictionResultCard';
import { usePredictions } from '../hooks/usePredictions';
import { exportToCSV, generateReport } from '../utils/exportUtils';

type Filter = 'all' | 'duplicate' | 'unique';
type Sort = 'confidence' | 'uncertainty';

const Predictions = () => {
  const { predictions, isLoading, error, processPredictions, clearPredictions } = usePredictions();
  const [showUpload, setShowUpload] = useState(true);
  const [sortBy, setSortBy] = useState<Sort>('confidence');
  const [filterDuplicate, setFilterDuplicate] = useState<Filter>('all');
  const [processingError, setProcessingError] = useState<string | null>(null);

  const handleFileProcess = async (files: File[]) => {
    setProcessingError(null);
    try {
      await processPredictions(files);
      setShowUpload(false);
    } catch (err) {
      setProcessingError(err instanceof Error ? err.message : 'Failed to process file');
      throw err;
    }
  };

  const filteredResults = [...predictions]
    .sort((a, b) => sortBy === 'confidence'
      ? b.confidence - a.confidence
      : a.uncertainty - b.uncertainty)
    .filter((prediction) => {
      if (filterDuplicate === 'duplicate') return prediction.isDuplicate;
      if (filterDuplicate === 'unique') return !prediction.isDuplicate;
      return true;
    });

  const duplicateCount = predictions.filter((prediction) => prediction.isDuplicate).length;
  const uniqueCount = predictions.length - duplicateCount;
  const avgConfidence = predictions.length > 0
    ? (predictions.reduce((sum, prediction) => sum + prediction.confidence, 0) / predictions.length).toFixed(2)
    : '0.00';

  return (
    <div className="p-4 md:p-8 space-y-8 animate-fade-in-up">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Predictions &amp; Results</h1>
        <p className="text-slate-400">Duplicate detection results with confidence scores</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard label="Total Predictions" value={predictions.length} />
        <SummaryCard label="Duplicates Found" value={duplicateCount} valueClassName="text-red-400" />
        <SummaryCard label="Unique Items" value={uniqueCount} valueClassName="text-green-400" />
        <SummaryCard label="Avg Confidence" value={`${avgConfidence}%`} valueClassName="text-blue-400" />
      </div>

      <section>
        <div className="flex flex-wrap gap-4">
          <button
            type="button"
            onClick={() => setShowUpload((open) => !open)}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors"
          >
            <Upload size={20} />
            Process New File
            <ChevronDown size={16} className={`transition-transform ${showUpload ? 'rotate-180' : ''}`} />
          </button>
          {predictions.length > 0 && (
            <button
              type="button"
              onClick={clearPredictions}
              className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-semibold transition-colors"
            >
              Clear Results
            </button>
          )}
        </div>

        {showUpload && (
          <div className="mt-4 animate-slide-up">
            <FileUploadZone onFileProcess={handleFileProcess} isProcessing={isLoading} />
          </div>
        )}

        {(error || processingError) && (
          <div className="mt-4 p-4 bg-red-500/20 border border-red-500/30 rounded-lg flex gap-3">
            <AlertCircle size={20} className="text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-red-400">Error Processing File</p>
              <p className="text-xs text-red-300">{processingError || error}</p>
            </div>
          </div>
        )}
      </section>

      {predictions.length === 0 && !showUpload && (
        <div className="min-h-64 flex flex-col items-center justify-center text-center bg-slate-800/30 border border-dashed border-slate-700 rounded-xl px-6 py-12">
          <div className="p-4 rounded-full bg-blue-500/10 text-blue-400 mb-4">
            <FileSearch size={36} />
          </div>
          <h2 className="text-xl font-semibold mb-2">No predictions yet</h2>
          <p className="max-w-md text-sm text-slate-400">
            Upload documents or a CSV file to analyze duplicate pairs. The summary cards and result list will update after processing.
          </p>
          <button
            type="button"
            onClick={() => setShowUpload(true)}
            className="mt-6 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors"
          >
            Choose files
          </button>
        </div>
      )}

      {predictions.length > 0 && (
        <>
          <div className="flex flex-col md:flex-row gap-4 bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4">
            <SelectControl
              label="Filter by Type"
              value={filterDuplicate}
              onChange={(value) => setFilterDuplicate(value as Filter)}
              options={[
                ['all', 'All Results'],
                ['duplicate', 'Duplicates Only'],
                ['unique', 'Unique Only'],
              ]}
            />
            <SelectControl
              label="Sort by"
              value={sortBy}
              onChange={(value) => setSortBy(value as Sort)}
              options={[
                ['confidence', 'Confidence (High to Low)'],
                ['uncertainty', 'Uncertainty (Low to High)'],
              ]}
            />
          </div>

          <div className="space-y-4">
            {filteredResults.length > 0 ? filteredResults.map((prediction) => (
              <PredictionResultCard key={prediction.id} prediction={prediction} />
            )) : (
              <div className="text-center py-12 bg-slate-800/50 border border-slate-700 rounded-xl">
                <p className="text-slate-400">No predictions match the selected filter</p>
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-4 justify-center">
            <button
              type="button"
              onClick={() => exportToCSV(predictions)}
              className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-semibold transition-colors"
            >
              Export to CSV
            </button>
            <button
              type="button"
              onClick={() => generateReport(predictions)}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors"
            >
              Generate Report
            </button>
          </div>
        </>
      )}
    </div>
  );
};

interface SummaryCardProps {
  label: string;
  value: number | string;
  valueClassName?: string;
}

const SummaryCard = ({ label, value, valueClassName = '' }: SummaryCardProps) => (
  <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4">
    <p className="text-xs text-slate-400 mb-2">{label}</p>
    <p className={`text-2xl font-bold ${valueClassName}`}>{value}</p>
  </div>
);

interface SelectControlProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<[string, string]>;
}

const SelectControl = ({ label, value, onChange, options }: SelectControlProps) => (
  <label className="flex-1 text-sm text-slate-400">
    <span className="block mb-2">{label}</span>
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
    >
      {options.map(([optionValue, optionLabel]) => (
        <option key={optionValue} value={optionValue}>{optionLabel}</option>
      ))}
    </select>
  </label>
);

export default Predictions;
