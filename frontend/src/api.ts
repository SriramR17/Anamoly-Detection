// api.ts
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
  peak_hours:number;
  peak_rate:number;
  last_updated: string;
  recent_anomalies?: any[];
}

export interface EvaluationMetricsData {
  Accuracy: number;
  F1_Score: number;
  Precision: number;
  Recall: number;
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

export interface BestModelData {
  [modelName: string]: {
    Accuracy_Mean: number;
    F1_Mean: number;
    Precision_Mean: number;
    Recall_Mean: number;
  };
}



export const apiService = {
  getDashboard: async (): Promise<ApiResponse<DashboardData>> => {
    const response = await api.get<ApiResponse<DashboardData>>('/api/dashboard');
    return response.data;
  },

  getPredictions: async (): Promise<ApiResponse<Prediction[]>> => {
    const response = await api.get<ApiResponse<Prediction[]>>('/api/predictions');
    return response.data;
  },

  getStats: async (): Promise<ApiResponse<Statistics>> => {
    const response = await api.get<ApiResponse<Statistics>>('/api/stats');
    return response.data;
  },


  getBestModels: async (): Promise<ApiResponse<BestModelData[]>> => {
    const response = await api.get<ApiResponse<BestModelData[]>>('/api/best_models');
    return response.data;
  },
};

// Normalize: your endpoint returns raw metrics; wrap it into ApiResponse
export const evaluationService = {
  getEvaluationMetrics: async (): Promise<EvaluationMetricsData> => {
    try {
      const response = await api.get<EvaluationMetricsData>('/api/get_evaluation_metrics');
      return response.data; // ✅ directly the metrics
    } catch (error) {
      console.error('Error fetching Evaluation Metrics Data:', error);
      throw error;
    }
  }
};




