import React, { useState, useEffect } from 'react';
import { 
  Filter, 
  Eye, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  Search,
  Clock,
  Wifi,
  TrendingUp,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Calendar,
  MapPin
} from 'lucide-react';
import { apiService, Prediction } from '../api';

interface GroupedPrediction {
  key: string;
  anomalies: Prediction[];
  totalPredictions: number;
  anomalyRate: number;
  latestTime?: string;
  avgProbability: number;
}

const PredictionsTable: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'cell' | 'time'>('cell');
  const [filter, setFilter] = useState<'all' | 'anomalies' | 'normal'>('anomalies');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 12;

  useEffect(() => {
    fetchPredictions();
  }, [viewMode]);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const result = await apiService.getPredictions(viewMode);
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

  const groupPredictions = (): GroupedPrediction[] => {
    const groups: { [key: string]: Prediction[] } = {};
    
    predictions.forEach(prediction => {
      let groupKey: string;
      
      if (viewMode === 'cell') {
        groupKey = prediction.CellName || 'Unknown Cell';
      } else {
        // Group by hour for time-wise view
        const time = prediction.Time || '';
        const hour = time.split(':')[0] + ':00';
        groupKey = hour || 'Unknown Time';
      }
      
      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(prediction);
    });

    return Object.entries(groups).map(([key, groupPredictions]) => {
      const anomalies = groupPredictions.filter(p => p.Predicted_Anomaly === 1);
      const avgProbability = groupPredictions.reduce((sum, p) => 
        sum + (p.Anomaly_Probability || 0), 0) / groupPredictions.length;
      
      return {
        key,
        anomalies,
        totalPredictions: groupPredictions.length,
        anomalyRate: (anomalies.length / groupPredictions.length) * 100,
        latestTime: groupPredictions[groupPredictions.length - 1]?.Time,
        avgProbability
      };
    }).sort((a, b) => b.anomalyRate - a.anomalyRate);
  };

  const filteredGroups = groupPredictions().filter(group => {
    const matchesFilter = 
      filter === 'all' || 
      (filter === 'anomalies' && group.anomalies.length > 0) ||
      (filter === 'normal' && group.anomalies.length === 0);
    
    const matchesSearch = 
      !searchTerm || 
      group.key.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  const totalPages = Math.ceil(filteredGroups.length / itemsPerPage);
  const paginatedGroups = filteredGroups.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const truncateNetworkType = (text: string, maxLength: number = 20) => {
    if (!text) return 'N/A';
    return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
  };

  const getSeverityColor = (rate: number) => {
    if (rate >= 80) return 'border-red-500/50 bg-red-500/10';
    if (rate >= 60) return 'border-orange-500/50 bg-orange-500/10';
    if (rate >= 40) return 'border-yellow-500/50 bg-yellow-500/10';
    return 'border-green-500/50 bg-green-500/10';
  };

  const getSeverityIcon = (rate: number) => {
    if (rate >= 80) return <AlertTriangle className="w-5 h-5 text-red-400" />;
    if (rate >= 60) return <TrendingUp className="w-5 h-5 text-orange-400" />;
    if (rate >= 40) return <BarChart3 className="w-5 h-5 text-yellow-400" />;
    return <CheckCircle className="w-5 h-5 text-green-400" />;
  };

  const toggleExpanded = (key: string) => {
    const newExpanded = new Set(expandedCards);
    if (newExpanded.has(key)) {
      newExpanded.delete(key);
    } else {
      newExpanded.add(key);
    }
    setExpandedCards(newExpanded);
  };

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-8 border border-gray-700">
        <div className="flex items-center justify-center space-x-4">
          <RefreshCw className="w-6 h-6 text-blue-400 animate-spin" />
          <span className="text-white text-lg">Loading predictions...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gradient-to-br from-red-900/20 to-gray-800 rounded-xl p-8 border border-red-500/30">
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
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700">
        <div className="flex flex-col space-y-4">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
            <div>
              <h2 className="text-3xl font-bold text-white flex items-center">
                <Eye className="mr-3 h-8 w-8 text-blue-400" />
                Anomaly Detection Results
              </h2>
              <p className="text-gray-400 mt-2">
                Showing {filteredGroups.length} groups with {predictions.filter(p => p.Predicted_Anomaly === 1).length} total anomalies
              </p>
            </div>
            
            <button 
              onClick={fetchPredictions}
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-6 py-3 rounded-lg transition-all duration-200 flex items-center space-x-2 shadow-lg"
            >
              <RefreshCw className="w-5 h-5" />
              <span>Refresh Data</span>
            </button>
          </div>

          {/* View Mode Toggle */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-6">
            <div className="flex items-center space-x-3">
              <span className="text-gray-300 font-medium">View Mode:</span>
              <div className="flex bg-gray-700 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('cell')}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    viewMode === 'cell' 
                      ? 'bg-blue-600 text-white shadow-md' 
                      : 'text-gray-300 hover:text-white hover:bg-gray-600'
                  }`}
                >
                  <MapPin className="w-4 h-4" />
                  <span>Cell-wise</span>
                </button>
                <button
                  onClick={() => setViewMode('time')}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    viewMode === 'time' 
                      ? 'bg-blue-600 text-white shadow-md' 
                      : 'text-gray-300 hover:text-white hover:bg-gray-600'
                  }`}
                >
                  <Calendar className="w-4 h-4" />
                  <span>Time-wise</span>
                </button>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  placeholder={`Search by ${viewMode === 'cell' ? 'cell name' : 'time'}...`}
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-gray-700 text-white pl-10 pr-4 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none w-64 transition-colors"
                />
              </div>
              
              {/* Filter */}
              <div className="flex items-center space-x-2">
                <Filter className="text-gray-400 w-4 h-4" />
                <select 
                  value={filter} 
                  onChange={(e) => setFilter(e.target.value as 'all' | 'anomalies' | 'normal')}
                  className="bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none transition-colors"
                >
                  <option value="all">All Groups</option>
                  <option value="anomalies">With Anomalies</option>
                  <option value="normal">Normal Only</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 rounded-xl p-4 border border-blue-500/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-300 text-sm font-medium">Total Groups</p>
              <p className="text-2xl font-bold text-white">{filteredGroups.length}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-red-600/20 to-red-800/20 rounded-xl p-4 border border-red-500/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-300 text-sm font-medium">High Risk Groups</p>
              <p className="text-2xl font-bold text-white">
                {filteredGroups.filter(g => g.anomalyRate >= 80).length}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-red-400" />
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-orange-600/20 to-orange-800/20 rounded-xl p-4 border border-orange-500/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-300 text-sm font-medium">Medium Risk Groups</p>
              <p className="text-2xl font-bold text-white">
                {filteredGroups.filter(g => g.anomalyRate >= 40 && g.anomalyRate < 80).length}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-orange-400" />
          </div>
        </div>
        
        <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 rounded-xl p-4 border border-green-500/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-300 text-sm font-medium">Normal Groups</p>
              <p className="text-2xl font-bold text-white">
                {filteredGroups.filter(g => g.anomalyRate < 40).length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>
        </div>
      </div>

      {/* Groups Grid */}
      {filteredGroups.length === 0 ? (
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-12 border border-gray-700 text-center">
          <div className="text-gray-400 text-xl mb-2">No data found</div>
          <p className="text-gray-500">Try adjusting your filters or running detection first</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {paginatedGroups.map((group) => {
              const isExpanded = expandedCards.has(group.key);
              const severityColor = getSeverityColor(group.anomalyRate);
              
              return (
                <div 
                  key={group.key}
                  className={`bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border ${severityColor} transition-all duration-300 hover:scale-[1.02] hover:shadow-xl`}
                >
                  <div className="p-6">
                    {/* Header */}
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        {getSeverityIcon(group.anomalyRate)}
                        <div>
                          <h3 className="text-lg font-bold text-white truncate max-w-[200px]" title={group.key}>
                            {group.key}
                          </h3>
                          <p className="text-gray-400 text-sm flex items-center mt-1">
                            {viewMode === 'cell' ? (
                              <><Wifi className="w-4 h-4 mr-1" /> Cell Tower</>
                            ) : (
                              <><Clock className="w-4 h-4 mr-1" /> Time Period</>
                            )}
                          </p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-2xl font-bold text-white">
                          {group.anomalyRate.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-400">
                          Anomaly Rate
                        </div>
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-gray-700/50 rounded-lg p-3">
                        <div className="text-red-400 font-semibold text-lg">
                          {group.anomalies.length}
                        </div>
                        <div className="text-gray-400 text-xs">Anomalies</div>
                      </div>
                      <div className="bg-gray-700/50 rounded-lg p-3">
                        <div className="text-blue-400 font-semibold text-lg">
                          {group.totalPredictions}
                        </div>
                        <div className="text-gray-400 text-xs">Total</div>
                      </div>
                    </div>

                    {/* Latest Time */}
                    {group.latestTime && (
                      <div className="flex items-center text-gray-400 text-sm mb-4">
                        <Clock className="w-4 h-4 mr-2" />
                        Latest: {group.latestTime}
                      </div>
                    )}

                    {/* Expand Button */}
                    <button
                      onClick={() => toggleExpanded(group.key)}
                      className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center space-x-2"
                    >
                      {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      <span>{isExpanded ? 'Hide' : 'Show'} Details</span>
                    </button>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <div className="border-t border-gray-700 p-6 bg-gray-800/50">
                      <div className="space-y-3 max-h-64 overflow-y-auto">
                        {group.anomalies.length > 0 ? (
                          group.anomalies.slice(0, 5).map((prediction, idx) => (
                            <div 
                              key={idx}
                              className="bg-gray-700 rounded-lg p-3 hover:bg-gray-600 transition-colors cursor-pointer"
                              onClick={() => setSelectedPrediction(prediction)}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                  <AlertTriangle className="w-4 h-4 text-red-400" />
                                  <span className="text-white text-sm font-medium">
                                    {viewMode === 'cell' ? prediction.Time : prediction.CellName}
                                  </span>
                                </div>
                                <div className="text-red-400 font-bold text-sm">
                                  {prediction.Anomaly_Probability ? 
                                    `${(prediction.Anomaly_Probability * 100).toFixed(1)}%` : 
                                    'N/A'
                                  }
                                </div>
                              </div>
                              {prediction.NetworkType && (
                                <div className="text-gray-400 text-xs mt-1 flex items-center">
                                  <Wifi className="w-3 h-3 mr-1" />
                                  {truncateNetworkType(prediction.NetworkType, 30)}
                                </div>
                              )}
                            </div>
                          ))
                        ) : (
                          <div className="text-gray-400 text-center py-4">
                            No anomalies in this group
                          </div>
                        )}
                        
                        {group.anomalies.length > 5 && (
                          <div className="text-center text-gray-400 text-sm">
                            ... and {group.anomalies.length - 5} more anomalies
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <div className="text-gray-400 text-sm">
                  Showing {(currentPage - 1) * itemsPerPage + 1} to{' '}
                  {Math.min(currentPage * itemsPerPage, filteredGroups.length)} of{' '}
                  {filteredGroups.length} groups
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="px-4 py-2 bg-gray-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                  >
                    Previous
                  </button>
                  <span className="px-4 py-2 text-gray-300 flex items-center">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                    className="px-4 py-2 bg-gray-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Detail Modal */}
      {selectedPrediction && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border border-gray-700 max-w-3xl w-full max-h-[80vh] overflow-y-auto shadow-2xl">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white">Prediction Details</h3>
                <button 
                  onClick={() => setSelectedPrediction(null)}
                  className="text-gray-400 hover:text-white transition-colors text-2xl"
                >
                  ×
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <span className="text-gray-400 text-sm">Status</span>
                    <div className="mt-2">
                      <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium ${
                        selectedPrediction.Predicted_Anomaly === 1
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                          : 'bg-green-500/20 text-green-400 border border-green-500/30'
                      }`}>
                        {selectedPrediction.Predicted_Anomaly === 1 ? (
                          <AlertTriangle className="w-4 h-4" />
                        ) : (
                          <CheckCircle className="w-4 h-4" />
                        )}
                        <span>{selectedPrediction.Predicted_Anomaly === 1 ? 'Anomaly Detected' : 'Normal'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <span className="text-gray-400 text-sm">Anomaly Probability</span>
                    <div className="mt-2">
                      <div className="text-2xl font-bold text-red-400">
                        {selectedPrediction.Anomaly_Probability ? 
                          `${(selectedPrediction.Anomaly_Probability * 100).toFixed(2)}%` : 
                          'N/A'
                        }
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <span className="text-gray-400 text-sm">Cell Information</span>
                    <div className="mt-2 text-white">
                      <p className="font-medium">{selectedPrediction.CellName || 'N/A'}</p>
                      <p className="text-gray-400 text-sm mt-1 flex items-center">
                        <Clock className="w-4 h-4 mr-1" />
                        {selectedPrediction.Time || 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <span className="text-gray-400 text-sm">Network Type</span>
                    <div className="mt-2">
                      <div className="text-white font-medium">
                        {selectedPrediction.NetworkType ? (
                          <div className="bg-gray-600 rounded-lg p-2 text-sm max-h-20 overflow-y-auto">
                            {selectedPrediction.NetworkType}
                          </div>
                        ) : (
                          'N/A'
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <span className="text-gray-400 text-sm">Technical Details</span>
                    <div className="mt-2">
                      <div className="bg-gray-600 rounded-lg p-3 max-h-40 overflow-y-auto">
                        <pre className="text-gray-300 text-xs leading-relaxed">
                          {JSON.stringify(selectedPrediction, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </div>
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