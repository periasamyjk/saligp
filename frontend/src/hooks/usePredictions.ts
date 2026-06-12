import { useState } from 'react';
import APIClient from '../utils/apiClient';
import { MockPrediction } from '../utils/mockData';

interface PredictionState {
  predictions: MockPrediction[];
  isLoading: boolean;
  error: string | null;
  totalProcessed: number;
}

export const usePredictions = () => {
  const [state, setState] = useState<PredictionState>({
    predictions: [],
    isLoading: false,
    error: null,
    totalProcessed: 0,
  });

  const processPredictions = async (files: File[]) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await APIClient.uploadFiles(files);
      const predictions = response.results || [];

      setState(prev => ({
        ...prev,
        predictions,
        totalProcessed: predictions.length,
        isLoading: false,
      }));

      return predictions;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to process file';
      setState(prev => ({
        ...prev,
        error: errorMessage,
        isLoading: false,
      }));
      throw error;
    }
  };

  const clearPredictions = () => {
    setState({
      predictions: [],
      isLoading: false,
      error: null,
      totalProcessed: 0,
    });
  };

  return {
    ...state,
    processPredictions,
    clearPredictions,
  };
};

export const useDashboardMetrics = () => {
  const [metrics, setMetrics] = useState({
    accuracy: 100.0,
    precision: 100.0,
    recall: 100.0,
    f1: 100.0,
  });

  return { metrics, setMetrics };
};
