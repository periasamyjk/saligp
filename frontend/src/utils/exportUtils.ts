import { MockPrediction } from './mockData';

export const exportToCSV = (predictions: MockPrediction[], filename = 'saligp_results.csv') => {
  const headers = [
    'ID',
    'Pair',
    'Is Duplicate',
    'Confidence %',
    'Uncertainty',
    'Cluster',
    ...Object.keys(predictions[0]?.features || {}),
  ];

  const rows = predictions.map(p => [
    p.id,
    p.pair,
    p.isDuplicate ? 'Yes' : 'No',
    p.confidence.toFixed(2),
    p.uncertainty.toFixed(4),
    p.cluster,
    ...Object.values(p.features).map(v => v.toFixed(4)),
  ]);

  const csv = [
    headers.join(','),
    ...rows.map(row => row.join(',')),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const generateReport = (predictions: MockPrediction[]) => {
  const headers = [
    'input_size_kb',
    'input_file',
    'duplicate_file',
    'actual_size_bytes',
    'content_profile',
    'sha256_prefix',
    'saligp_prediction',
    'gp_score',
    'prediction_confidence',
    'processing_time_seconds',
    'filename_similarity',
    'content_similarity',
    'metadata_similarity',
    'size_similarity',
    'tfidf_similarity',
    'embedding_similarity',
    'sha256_match',
    'overall_similarity',
  ];

  const rows = predictions.map((prediction) => {
    const { inputFile, duplicateFile } = resolvePairFiles(prediction);
    const features = prediction.features || {};

    return [
      prediction.input_size_kb ?? inferSizeKb(inputFile),
      prediction.input_file ?? inputFile,
      prediction.duplicate_file ?? duplicateFile,
      prediction.actual_size_bytes ?? inferActualSizeBytes(prediction.input_size_kb ?? inferSizeKb(inputFile)),
      prediction.content_profile ?? 'uploaded document',
      prediction.sha256_prefix ?? '',
      prediction.isDuplicate ? 'Duplicate' : 'Unique',
      formatNumber(prediction.gp_score ?? prediction.prediction_confidence ?? prediction.confidence / 100),
      formatNumber(prediction.prediction_confidence ?? prediction.confidence / 100),
      formatNumber(prediction.processing_time_seconds),
      formatNumber(features.filename_similarity),
      formatNumber(features.content_similarity),
      formatNumber(features.metadata_similarity),
      formatNumber(features.size_similarity),
      formatNumber(features.tfidf_similarity),
      formatNumber(features.embedding_similarity),
      formatNumber(features.sha256_match),
      formatNumber(features.overall_similarity),
    ];
  });

  const csv = [
    headers.join(','),
    ...rows.map((row) => row.map(csvEscape).join(',')),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `saligp_report_${new Date().getTime()}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

const csvEscape = (value: string | number | null | undefined) => {
  const text = value === null || value === undefined ? '' : String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
};

const formatNumber = (value: number | null | undefined) => (
  typeof value === 'number' && Number.isFinite(value) ? Number(value.toFixed(6)) : ''
);

const resolvePairFiles = (prediction: MockPrediction) => {
  if (prediction.input_file || prediction.duplicate_file) {
    return {
      inputFile: prediction.input_file || '',
      duplicateFile: prediction.duplicate_file || '',
    };
  }

  const [inputFile = prediction.pair, duplicateFile = ''] = prediction.pair.split(' <-> ');
  return { inputFile, duplicateFile };
};

const inferSizeKb = (filename: string) => {
  const match = filename.match(/(\d+)\s*kb/i);
  return match ? Number(match[1]) : '';
};

const inferActualSizeBytes = (sizeKb: number | string) => (
  typeof sizeKb === 'number' && Number.isFinite(sizeKb) ? sizeKb * 1024 : ''
);
