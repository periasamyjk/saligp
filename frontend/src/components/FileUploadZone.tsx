import { useState } from 'react';
import { Upload, File, AlertCircle, CheckCircle } from 'lucide-react';

interface FileUploadZoneProps {
  onFileProcess?: (files: File[]) => Promise<void>;
  isProcessing?: boolean;
}

const FileUploadZone = ({ onFileProcess, isProcessing: externalProcessing = false }: FileUploadZoneProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFiles(Array.from(e.target.files));
    }
  };

  const validateAndSetFiles = (selectedFiles: File[]) => {
    setError(null);
    setSuccess(false);

    const validExtensions = ['csv', 'tsv', 'txt', 'md', 'markdown', 'html', 'htm', 'json', 'jsonl', 'docx', 'pdf'];
    const invalidFile = selectedFiles.find((selectedFile) => {
      const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase();
      return !fileExtension || !validExtensions.includes(fileExtension);
    });

    if (invalidFile) {
      setError(`Invalid file format: ${invalidFile.name}`);
      return;
    }

    const oversizedFile = selectedFiles.find((selectedFile) => selectedFile.size > 100 * 1024 * 1024);
    if (oversizedFile) {
      setError(`File is too large: ${oversizedFile.name}. Max size is 100MB.`);
      return;
    }

    setFiles(selectedFiles);
  };

  const handleProcess = async () => {
    if (files.length === 0) return;
    
    setIsProcessing(true);
    setError(null);

    try {
      if (onFileProcess) {
        await onFileProcess(files);
      }
      setSuccess(true);
      setTimeout(() => {
        setFiles([]);
        setSuccess(false);
      }, 2000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process file';
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-8 space-y-6">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
          dragActive
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-slate-600 bg-slate-900/20 hover:border-slate-500'
        }`}
      >
        <input
          type="file"
          id="file-input"
          accept=".csv,.tsv,.txt,.md,.markdown,.html,.htm,.json,.jsonl,.docx,.pdf,text/csv,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          multiple
          onChange={handleChange}
          className="hidden"
        />

        {files.length === 0 || success ? (
          <label htmlFor="file-input" className="cursor-pointer">
            {success ? (
              <>
                <CheckCircle size={40} className="mx-auto mb-4 text-green-400 animate-pulse" />
                <h3 className="text-lg font-semibold mb-2 text-green-400">File Processed Successfully!</h3>
                <p className="text-slate-400 mb-4">Your predictions are ready to view</p>
              </>
            ) : (
              <>
                <Upload size={40} className="mx-auto mb-4 text-blue-400" />
                <h3 className="text-lg font-semibold mb-2 text-white">Upload Documents</h3>
                <p className="text-slate-400 mb-4">Drag and drop documents or a CSV file here</p>
              </>
            )}
            {!success && (
              <>
                <span className="inline-block px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors">
                  Browse Files
                </span>
                <p className="text-xs text-slate-500 mt-4">Supported: TXT, PDF, DOCX, CSV, TSV, MD, HTML, JSON | Max 100MB each</p>
              </>
            )}
          </label>
        ) : (
          <div>
            <File size={40} className="mx-auto mb-4 text-green-400" />
            <p className="font-semibold text-white mb-2">
              {files.length === 1 ? files[0].name : `${files.length} files selected`}
            </p>
            <p className="text-sm text-slate-400 mb-6">
              {(files.reduce((total, selectedFile) => total + selectedFile.size, 0) / 1024).toFixed(2)} KB
            </p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={() => {
                  setFiles([]);
                  setError(null);
                }}
                disabled={isProcessing || externalProcessing}
                className="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg font-semibold transition-colors disabled:opacity-50"
              >
                Change File
              </button>
              <button
                onClick={handleProcess}
                disabled={isProcessing || externalProcessing}
                className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProcessing || externalProcessing ? 'Processing...' : 'Process File'}
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-lg flex gap-3">
          <AlertCircle size={20} className="text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-red-400 mb-1">Error</p>
            <p className="text-xs text-red-300">{error}</p>
          </div>
        </div>
      )}

      <div className="p-4 bg-blue-500/20 border border-blue-500/30 rounded-lg flex gap-3">
        <AlertCircle size={20} className="text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-blue-400 mb-1">Data Format Requirements</p>
          <p className="text-xs text-blue-300">
            Upload two or more documents to compare them pairwise, or upload a CSV with feature columns or text pair columns.
          </p>
          <p className="text-xs text-blue-300 mt-2">
            <a href="/sample_data.csv" download className="underline hover:text-blue-200">
              Download sample data →
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default FileUploadZone;
