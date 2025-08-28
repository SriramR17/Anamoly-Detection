import React, { useState, useEffect } from 'react';
import { apiService } from '../api';

const Statistics = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStatistics();
  }, []);

  const fetchStatistics = async () => {
    try {
      setLoading(true);
      const result = await apiService.getStats();
      if (result.status === 'success') {
        setStats(result.data);
      }
      setError(null);
    } catch (error) {
      setError('Failed to load statistics');
      console.error('Statistics error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading">🔄 Loading statistics...</div>;
  if (error) return <div className="error">❌ {error}</div>;
  if (!stats) return <div className="no-data">📭 No statistics available</div>;

  const anomalyRate = stats.total_records > 0 ? (stats.anomalies / stats.total_records * 100) : 0;
  const normalRate = stats.total_records > 0 ? (stats.normal / stats.total_records * 100) : 0;

  return (
    <div className="statistics">
      <h2>📈 Detailed Statistics</h2>
      
      <div className="stats-grid">
        <div className="stat-card">
          <h3>📊 Total Records</h3>
          <div className="stat-value large">{stats.total_records?.toLocaleString() || 0}</div>
        </div>
        
        <div className="stat-card anomaly">
          <h3>⚠️ Anomalies</h3>
          <div className="stat-value large">{stats.anomalies?.toLocaleString() || 0}</div>
          <div className="stat-subtitle">{anomalyRate.toFixed(2)}% of total</div>
        </div>
        
        <div className="stat-card success">
          <h3>✅ Normal Records</h3>
          <div className="stat-value large">{stats.normal?.toLocaleString() || 0}</div>
          <div className="stat-subtitle">{normalRate.toFixed(2)}% of total</div>
        </div>
        
        <div className="stat-card">
          <h3>📈 Detection Rate</h3>
          <div className="stat-value">{anomalyRate.toFixed(3)}%</div>
        </div>
      </div>

      {(stats.avg_anomaly_prob !== undefined) && (
        <div className="probability-stats">
          <h3>🎯 Probability Statistics</h3>
          <div className="prob-grid">
            <div className="prob-item">
              <span className="prob-label">Average Probability:</span>
              <span className="prob-value">{(stats.avg_anomaly_prob * 100).toFixed(2)}%</span>
            </div>
            <div className="prob-item">
              <span className="prob-label">Maximum Probability:</span>
              <span className="prob-value">{(stats.max_anomaly_prob * 100).toFixed(2)}%</span>
            </div>
            <div className="prob-item">
              <span className="prob-label">Minimum Probability:</span>
              <span className="prob-value">{(stats.min_anomaly_prob * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      )}

      <div className="additional-metrics">
        <h3>📊 Additional Metrics</h3>
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Anomaly-to-Normal Ratio:</span>
            <span className="metric-value">
              {stats.normal > 0 ? `1:${(stats.normal / Math.max(stats.anomalies, 1)).toFixed(1)}` : 'N/A'}
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Data Quality Score:</span>
            <span className="metric-value">
              {stats.total_records > 0 ? 'Good' : 'No Data'}
            </span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Sample Size Category:</span>
            <span className="metric-value">
              {stats.total_records > 10000 ? 'Large' : 
               stats.total_records > 1000 ? 'Medium' : 
               stats.total_records > 0 ? 'Small' : 'Empty'}
            </span>
          </div>
        </div>
      </div>

      <div className="insights-section">
        <h3>💡 Key Insights</h3>
        <div className="insights">
          {anomalyRate > 10 && (
            <div className="insight warning">
              ⚠️ High anomaly rate detected ({anomalyRate.toFixed(1)}%). Review system health.
            </div>
          )}
          {anomalyRate < 1 && stats.total_records > 0 && (
            <div className="insight success">
              ✅ Low anomaly rate ({anomalyRate.toFixed(1)}%). System appears stable.
            </div>
          )}
          {stats.total_records === 0 && (
            <div className="insight info">
              ℹ️ No data processed yet. Run anomaly detection to see results.
            </div>
          )}
          {stats.total_records > 0 && (
            <div className="insight info">
              📊 {stats.total_records.toLocaleString()} samples processed successfully.
            </div>
          )}
        </div>
      </div>

      <div className="refresh-section">
        <button onClick={fetchStatistics} className="refresh-btn">
          🔄 Refresh Statistics
        </button>
      </div>
    </div>
  );
};

export default Statistics;
