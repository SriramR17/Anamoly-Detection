import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export const apiService = {
  // Get dashboard overview data
  getDashboard: async () => {
    try {
      const response = await api.get('/api/dashboard');
      return response.data;
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      throw error;
    }
  },

  // Get all predictions
  getPredictions: async () => {
    try {
      const response = await api.get('/api/predictions');
      return response.data;
    } catch (error) {
      console.error('Error fetching predictions:', error);
      throw error;
    }
  },

  // Get statistics
  getStats: async () => {
    try {
      const response = await api.get('/api/stats');
      return response.data;
    } catch (error) {
      console.error('Error fetching statistics:', error);
      throw error;
    }
  },

  // Run anomaly detection
  runDetection: async () => {
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
