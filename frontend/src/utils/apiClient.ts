import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Prediction endpoints
  async predict(features: Record<string, number>) {
    try {
      const response = await this.client.post('/predict', {
        features,
      });
      return response.data;
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  }

  async predictBatch(data: Array<Record<string, number>>) {
    try {
      const response = await this.client.post('/predict-batch', { data });
      return response.data;
    } catch (error) {
      console.error('Batch prediction error:', error);
      throw error;
    }
  }

  // Metrics endpoints
  async getMetrics() {
    try {
      const response = await this.client.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Metrics error:', error);
      throw error;
    }
  }

  async getDashboardData() {
    try {
      const response = await this.client.get('/dashboard');
      return response.data;
    } catch (error) {
      console.error('Dashboard data error:', error);
      throw error;
    }
  }

  // File upload
  async uploadFile(file: File) {
    return this.uploadFiles([file]);
  }

  async uploadFiles(files: File[]) {
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file));
      const response = await this.client.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  // Analytics
  async getAnalytics() {
    try {
      const response = await this.client.get('/analytics');
      return response.data;
    } catch (error) {
      console.error('Analytics error:', error);
      throw error;
    }
  }
}

export default new APIClient();
