import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import PredictionsTable from './components/PredictionsTable';
import Statistics from './components/Statistics';
import { apiService } from './api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  const handleRunDetection = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.runDetection();
      if (result.status === 'success') {
        setLastUpdate(new Date().toLocaleString());
        // Refresh current tab data
        window.location.reload();
      } else {
        setError(result.message || 'Detection failed');
      }
    } catch (error) {
      setError('Failed to run detection: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'predictions':
        return <PredictionsTable />;
      case 'statistics':
        return <Statistics />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔍 Anomaly Detection Dashboard</h1>
        <div className="header-actions">
          {lastUpdate && <span className="last-update">Last updated: {lastUpdate}</span>}
          <button 
            onClick={handleRunDetection} 
            disabled={loading}
            className="run-detection-btn"
          >
            {loading ? '🔄 Running...' : '🚀 Run Detection'}
          </button>
        </div>
      </header>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      <nav className="nav-tabs">
        <button 
          className={activeTab === 'dashboard' ? 'active' : ''} 
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Dashboard
        </button>
        <button 
          className={activeTab === 'predictions' ? 'active' : ''} 
          onClick={() => setActiveTab('predictions')}
        >
          📋 Predictions
        </button>
        <button 
          className={activeTab === 'statistics' ? 'active' : ''} 
          onClick={() => setActiveTab('statistics')}
        >
          📈 Statistics
        </button>
      </nav>

      <main className="main-content">
        {renderContent()}
      </main>

      <footer className="app-footer">
        <p>Network Anomaly Detection System - Real-time Dashboard</p>
      </footer>
    </div>
  );
}

export default App;
