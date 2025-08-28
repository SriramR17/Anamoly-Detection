import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { apiService } from '../api';

const Dashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const result = await apiService.getDashboard();
      setData(result.data);
      setError(null);
    } catch (error) {
      setError('Failed to load dashboard data');
      console.error('Dashboard error:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading">🔄 Loading dashboard...</div>;
  if (error) return <div className="error">❌ {error}</div>;
  if (!data) return <div className="no-data">📭 No data available</div>;

  const pieData = [
    { name: 'Normal', value: data.total_samples - data.total_anomalies, color: '#4CAF50' },
    { name: 'Anomalies', value: data.total_anomalies, color: '#f44336' }
  ];

  return (
    <div className="dashboard">
      <div className="stats-grid">
        <div className="stat-card">
          <h3>📊 Total Samples</h3>
          <div className="stat-value">{data.total_samples.toLocaleString()}</div>
        </div>
        
        <div className="stat-card anomaly">
          <h3>⚠️ Anomalies Detected</h3>
          <div className="stat-value">{data.total_anomalies.toLocaleString()}</div>
        </div>
        
        <div className="stat-card">
          <h3>📈 Anomaly Rate</h3>
          <div className="stat-value">{data.anomaly_rate}%</div>
        </div>
        
        <div className="stat-card">
          <h3>🕐 Last Updated</h3>
          <div className="stat-value small">{data.last_updated}</div>
        </div>
      </div>

      <div className="charts-section">
        <div className="chart-container">
          <h3>📊 Distribution Overview</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                dataKey="value"
                label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(1)}%)`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>📊 Summary Bar Chart</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={pieData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#2196F3" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {data.recent_anomalies && data.recent_anomalies.length > 0 && (
        <div className="recent-anomalies">
          <h3>🚨 Recent Anomalies</h3>
          <div className="anomalies-list">
            {data.recent_anomalies.slice(0, 5).map((anomaly, index) => (
              <div key={index} className="anomaly-item">
                <span className="anomaly-id">#{index + 1}</span>
                <span className="anomaly-prob">
                  {anomaly.Anomaly_Probability ? `${(anomaly.Anomaly_Probability * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="refresh-section">
        <button onClick={fetchDashboardData} className="refresh-btn">
          🔄 Refresh Data
        </button>
      </div>
    </div>
  );
};

export default Dashboard;
