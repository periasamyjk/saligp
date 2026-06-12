// Environment variables for development
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  predict: `${API_BASE_URL}/predict`,
  upload: `${API_BASE_URL}/upload`,
  ownership: `${API_BASE_URL}/ownership`,
  metrics: `${API_BASE_URL}/metrics`,
};

export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';
