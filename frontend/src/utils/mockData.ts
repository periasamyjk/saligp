// Mock data generator for demo purposes
export interface MockPrediction {
  id: number;
  pair: string;
  confidence: number;
  isDuplicate: boolean;
  cluster: string;
  uncertainty: number;
  features: Record<string, number>;
  input_size_kb?: number | null;
  input_file?: string | null;
  duplicate_file?: string | null;
  actual_size_bytes?: number | null;
  content_profile?: string | null;
  sha256_prefix?: string | null;
  gp_score?: number | null;
  prediction_confidence?: number | null;
  processing_time_seconds?: number | null;
}

export const generateMockPredictions = (data: Array<Record<string, number>>): MockPrediction[] => {
  return data.map((features, index) => {
    // Simple heuristic: if similarity features are high, likely duplicate
    const avgSimilarity = [
      features.filename_similarity || 0,
      features.content_similarity || 0,
      features.metadata_similarity || 0,
      features.tfidf_similarity || 0,
      features.embedding_similarity || 0,
    ].reduce((a, b) => a + b, 0) / 5;

    const hasMatch = features.sha256_match === 1;
    const score = hasMatch ? 0.99 : avgSimilarity;
    const isDuplicate = score > 0.6;
    
    // Generate uncertainty inversely proportional to confidence
    const uncertainty = Math.random() * (1 - score) + (1 - score) * 0.1;
    
    // Assign cluster based on average similarity
    let cluster = 'Cluster 0 (Easy)';
    if (avgSimilarity > 0.8) cluster = 'Cluster 0 (Easy)';
    else if (avgSimilarity > 0.6) cluster = 'Cluster 1 (Medium)';
    else if (avgSimilarity > 0.4) cluster = 'Cluster 2 (Hard)';
    else cluster = 'Cluster 3 (Mixed)';

    return {
      id: index + 1,
      pair: `Pair_${String(index + 1).padStart(3, '0')}`,
      confidence: Math.round(score * 10000) / 100,
      isDuplicate,
      cluster,
      uncertainty: Math.round(uncertainty * 10000) / 10000,
      features,
    };
  });
};

export const parseCSVData = (csvContent: string): Array<Record<string, number>> => {
  const lines = csvContent.trim().split('\n');
  if (lines.length < 2) throw new Error('CSV must have headers and at least one data row');

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const requiredFields = [
    'filename_similarity',
    'content_similarity',
    'metadata_similarity',
    'size_similarity',
    'tfidf_similarity',
    'embedding_similarity',
    'sha256_match',
    'overall_similarity',
  ];

  const missingFields = requiredFields.filter(field => !headers.includes(field));
  if (missingFields.length > 0) {
    throw new Error(`Missing required columns: ${missingFields.join(', ')}`);
  }

  const data: Array<Record<string, number>> = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim());
    const row: Record<string, number> = {};

    headers.forEach((header, index) => {
      const value = parseFloat(values[index]);
      if (!isNaN(value)) {
        row[header] = value;
      }
    });

    if (Object.keys(row).length > 0) {
      data.push(row);
    }
  }

  return data;
};

export const generateMockDashboardMetrics = () => ({
  accuracy: 100.0,
  precision: 100.0,
  recall: 100.0,
  f1: 100.0,
  totalPredictions: 6000,
  duplicatesFound: 3000,
  uniqueItems: 2820,
  falsePositives: 0,
  falseNegatives: 0,
});

export const generateMockClusterData = () => [
  { name: 'Cluster 0 (Easy)', value: 2134, percentage: 35.6 },
  { name: 'Cluster 1 (Medium)', value: 1334, percentage: 22.2 },
  { name: 'Cluster 2 (Hard)', value: 790, percentage: 13.1 },
  { name: 'Cluster 3 (Mixed)', value: 1742, percentage: 29.0 },
];

export const generateMockFeatureImportance = () => [
  { name: 'Cosine Sim.', value: 28.7 },
  { name: 'Content Sim.', value: 22.4 },
  { name: 'Embedding Sim.', value: 18.9 },
  { name: 'TFIDF Sim.', value: 15.6 },
  { name: 'Metadata Sim.', value: 9.8 },
  { name: 'SHA256 Match', value: 4.6 },
];

export const generateMockLearningCurve = () => [
  { iteration: 1, accuracy: 0.82 },
  { iteration: 2, accuracy: 0.85 },
  { iteration: 3, accuracy: 0.88 },
  { iteration: 4, accuracy: 0.91 },
  { iteration: 5, accuracy: 0.94 },
  { iteration: 6, accuracy: 0.96 },
  { iteration: 7, accuracy: 0.97 },
  { iteration: 8, accuracy: 0.99 },
  { iteration: 9, accuracy: 0.995 },
  { iteration: 10, accuracy: 1.0 },
];
