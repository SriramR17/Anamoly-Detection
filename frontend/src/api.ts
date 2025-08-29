import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export interface DashboardData {
  total_samples: number;
  total_anomalies: number;
  anomaly_rate: number;
  last_updated: string;
  recent_anomalies?: any[];
}

export interface Prediction {
  Predicted_Anomaly: number;
  Anomaly_Probability?: number;
  CellName?: string;
  Time?: string;
  [key: string]: any;
}

export interface Statistics {
  total_records: number;
  anomalies: number;
  normal: number;
  avg_anomaly_prob?: number;
  max_anomaly_prob?: number;
  min_anomaly_prob?: number;
}

export interface ApiResponse<T> {
  status: string;
  data: T;
  message?: string;
}

export const apiService = {
  // Get dashboard overview data
  getDashboard: async (): Promise<ApiResponse<DashboardData>> => {
    try {
      const response = await api.get('/api/dashboard');
      return response.data;
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      throw error;
    }
  },

  // Get all predictions
  getPredictions: async (): Promise<ApiResponse<Prediction[]>> => {
    try {
      const response = await api.get('/api/predictions');
      return response.data;
    } catch (error) {
      console.error('Error fetching predictions:', error);
      throw error;
    }
  },

  // Get statistics
  getStats: async (): Promise<ApiResponse<Statistics>> => {
    try {
      const response = await api.get('/api/stats');
      return response.data;
    } catch (error) {
      console.error('Error fetching statistics:', error);
      throw error;
    }
  },

  // Run anomaly detection
  runDetection: async (): Promise<ApiResponse<any>> => {
    try {
      const response = await api.post('/api/run-detection');
      return response.data;
    } catch (error) {
      console.error('Error running detection:', error);
      throw error;
    }
  }
};

export default apiService;
