import React, { useState, useEffect } from 'react';
import { apiService } from '../api';

const PredictionsTable = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all'); // 'all', 'anomalies', 'normal'

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const result = await apiService.getPredictions();
      if (result.status === 'success') {
        setPredictions(result.data);
      } else {
        setPredictions([]);
      }
      setError(null);
    } catch (error) {
      setError('Failed to load predictions');
      console.error('Predictions error:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredPredictions = predictions.filter(prediction => {
    if (filter === 'anomalies') return prediction.Predicted_Anomaly === 1;
    if (filter === 'normal') return prediction.Predicted_Anomaly === 0;
    return true;
  });

  if (loading) return <div className="loading">🔄 Loading predictions...</div>;
  if (error) return <div className="error">❌ {error}</div>;

  return (
    <div className="predictions-table">
      <div className="table-header">
        <h2>📋 Predictions</h2>
        <div className="filter-controls">
          <label>Filter: </label>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All ({predictions.length})</option>
            <option value="anomalies">
              Anomalies ({predictions.filter(p => p.Predicted_Anomaly === 1).length})
            </option>
            <option value="normal">
              Normal ({predictions.filter(p => p.Predicted_Anomaly === 0).length})
            </option>
          </select>
        </div>
      </div>

      {filteredPredictions.length === 0 ? (
        <div className="no-data">📭 No predictions found</div>
      ) : (
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Status</th>
                <th>Anomaly Probability</th>
                <th>Cell Name</th>
                <th>Time</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {filteredPredictions.slice(0, 50).map((prediction, index) => (
                <tr key={index} className={prediction.Predicted_Anomaly ? 'anomaly-row' : 'normal-row'}>
                  <td>{index + 1}</td>
                  <td>
                    <span className={`status-badge ${prediction.Predicted_Anomaly ? 'anomaly' : 'normal'}`}>
                      {prediction.Predicted_Anomaly ? '⚠️ Anomaly' : '✅ Normal'}
                    </span>
                  </td>
                  <td>
                    <span className="probability">
                      {prediction.Anomaly_Probability ? 
                        `${(prediction.Anomaly_Probability * 100).toFixed(2)}%` : 
                        'N/A'
                      }
                    </span>
                  </td>
                  <td>{prediction.CellName || 'N/A'}</td>
                  <td>{prediction.Time || 'N/A'}</td>
                  <td>
                    <button className="details-btn" onClick={() => alert(JSON.stringify(prediction, null, 2))}>
                      👁️ View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredPredictions.length > 50 && (
            <div className="table-footer">
              <p>Showing first 50 of {filteredPredictions.length} results</p>
            </div>
          )}
        </div>
      )}

      <div className="refresh-section">
        <button onClick={fetchPredictions} className="refresh-btn">
          🔄 Refresh Predictions
        </button>
      </div>
    </div>
  );
};

export default PredictionsTable;
