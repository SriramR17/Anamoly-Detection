import React, { useState, useEffect } from 'react';
import { 
  Filter, 
  Eye, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  Search,
  Download,
  Clock
} from 'lucide-react';
import { apiService, Prediction } from '../api';

const PredictionsTable: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'anomalies' | 'normal'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

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
    const matchesFilter = 
      filter === 'all' || 
      (filter === 'anomalies' && prediction.Predicted_Anomaly === 1) ||
      (filter === 'normal' && prediction.Predicted_Anomaly === 0);
    
    const matchesSearch = 
      !searchTerm ||
      (prediction.CellName && prediction.CellName.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (prediction.Time && prediction.Time.toLowerCase().includes(searchTerm.toLowerCase()));
    
    return matchesFilter && matchesSearch;
  });

  const totalPages = Math.ceil(filteredPredictions.length / itemsPerPage);
  const paginatedPredictions = filteredPredictions.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const getStatusBadge = (prediction: Prediction) => {
    const isAnomaly = prediction.Predicted_Anomaly === 1;
    return (
      <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
        isAnomaly 
          ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
          : 'bg-green-500/20 text-green-400 border border-green-500/30'
      }`}>
        {isAnomaly ? (
          <AlertTriangle className="w-4 h-4" />
        ) : (
          <CheckCircle className="w-4 h-4" />
        )}
        <span>{isAnomaly ? 'Anomaly' : 'Normal'}</span>
      </div>
    );
  };

  const getProbabilityColor = (probability?: number) => {
    if (!probability) return 'text-gray-400';
    if (probability >= 0.8) return 'text-red-400';
    if (probability >= 0.6) return 'text-orange-400';
    if (probability >= 0.4) return 'text-yellow-400';
    return 'text-green-400';
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-xl p-8 border border-gray-700">
        <div className="flex items-center justify-center space-x-4">
          <RefreshCw className="w-6 h-6 text-blue-400 animate-spin" />
          <span className="text-white text-lg">Loading predictions...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-xl p-8 border border-red-500/30">
        <div className="flex items-center justify-center space-x-4 text-red-400">
          <AlertTriangle className="w-6 h-6" />
          <span className="text-lg">{error}</span>
        </div>
        <div className="text-center mt-4">
          <button 
            onClick={fetchPredictions}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center">
              <Eye className="mr-3 h-6 w-6 text-blue-400" />
              Prediction Results
            </h2>
            <p className="text-gray-400 mt-1">
              Showing {filteredPredictions.length} of {predictions.length} predictions
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search by cell name or time..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-gray-700 text-white pl-10 pr-4 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none w-64"
              />
            </div>
            
            {/* Filter */}
            <div className="flex items-center space-x-2">
              <Filter className="text-gray-400 w-4 h-4" />
              <select 
                value={filter} 
                onChange={(e) => setFilter(e.target.value as 'all' | 'anomalies' | 'normal')}
                className="bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                <option value="all">All ({predictions.length})</option>
                <option value="anomalies">
                  Anomalies ({predictions.filter(p => p.Predicted_Anomaly === 1).length})
                </option>
                <option value="normal">
                  Normal ({predictions.filter(p => p.Predicted_Anomaly === 0).length})
                </option>
              </select>
            </div>
            
            {/* Refresh Button */}
            <button 
              onClick={fetchPredictions}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
        </div>
      </div>

      {/* Table */}
      {filteredPredictions.length === 0 ? (
        <div className="bg-gray-800 rounded-xl p-8 border border-gray-700 text-center">
          <div className="text-gray-400 text-lg">No predictions found</div>
          <p className="text-gray-500 mt-2">Try adjusting your filters or running detection first</p>
        </div>
      ) : (
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="text-left py-4 px-6 text-gray-300 font-semibold">#</th>
                  <th className="text-left py-4 px-6 text-gray-300 font-semibold">Status</th>
                  <th className="text-left py-4 px-6 text-gray-300 font-semibold">Probability</th>
                  <th className="text-left py-4 px-6 text-gray-300 font-semibold">Cell Name</th>
                  <th className="text-left py-4 px-6 text-gray-300 font-semibold">Time</th>
                  <th className="text-left py-4 px-6 text-gray-300 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {paginatedPredictions.map((prediction, index) => (
                  <tr 
                    key={index} 
                    className={`border-b border-gray-700 hover:bg-gray-750 transition-colors ${
                      prediction.Predicted_Anomaly ? 'bg-red-900/10' : 'bg-green-900/10'
                    }`}
                  >
                    <td className="py-4 px-6 text-gray-300 font-mono text-sm">
                      {(currentPage - 1) * itemsPerPage + index + 1}
                    </td>
                    <td className="py-4 px-6">
                      {getStatusBadge(prediction)}
                    </td>
                    <td className="py-4 px-6">
                      <span className={`font-bold text-lg ${getProbabilityColor(prediction.Anomaly_Probability)}`}>
                        {prediction.Anomaly_Probability ? 
                          `${(prediction.Anomaly_Probability * 100).toFixed(2)}%` : 
                          'N/A'
                        }
                      </span>
                    </td>
                    <td className="py-4 px-6 text-gray-300 font-mono">
                      {prediction.CellName || 'N/A'}
                    </td>
                    <td className="py-4 px-6 text-gray-300 flex items-center">
                      <Clock className="w-4 h-4 mr-2 text-gray-500" />
                      {prediction.Time || 'N/A'}
                    </td>
                    <td className="py-4 px-6">
                      <button 
                        onClick={() => setSelectedPrediction(prediction)}
                        className="bg-gray-600 hover:bg-gray-500 text-white px-3 py-1 rounded-lg transition-colors flex items-center space-x-1 text-sm"
                      >
                        <Eye className="w-4 h-4" />
                        <span>View</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Pagination */}
          {totalPages > 1 && (
            <div className="bg-gray-700 px-6 py-4 flex items-center justify-between">
              <div className="text-gray-400 text-sm">
                Showing {(currentPage - 1) * itemsPerPage + 1} to{' '}
                {Math.min(currentPage * itemsPerPage, filteredPredictions.length)} of{' '}
                {filteredPredictions.length} results
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1 bg-gray-600 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-500"
                >
                  Previous
                </button>
                <span className="px-3 py-1 text-gray-300">
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 bg-gray-600 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-500"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Detail Modal */}
      {selectedPrediction && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-xl border border-gray-700 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">Prediction Details</h3>
                <button 
                  onClick={() => setSelectedPrediction(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Status:</span>
                  {getStatusBadge(selectedPrediction)}
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Anomaly Probability:</span>
                  <span className={`font-bold text-lg ${getProbabilityColor(selectedPrediction.Anomaly_Probability)}`}>
                    {selectedPrediction.Anomaly_Probability ? 
                      `${(selectedPrediction.Anomaly_Probability * 100).toFixed(4)}%` : 
                      'N/A'
                    }
                  </span>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="text-white font-semibold mb-2">Raw Data:</h4>
                  <pre className="text-gray-300 text-sm overflow-x-auto">
                    {JSON.stringify(selectedPrediction, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionsTable;
