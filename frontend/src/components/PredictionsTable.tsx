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
  MapPin,
  ArrowUpDown,
  SortAsc,
  SortDesc,
  Activity,
  Shield,
  Target
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

type FilterType = 'all' | 'anomalies' | 'normal';
type SortType = 'default' | 'low-to-high' | 'high-to-low';

const PredictionsTable: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'cell' | 'time'>('cell');
  const [filter, setFilter] = useState<FilterType>('anomalies');
  const [sortOrder, setSortOrder] = useState<SortType>('default');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
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
        const time = prediction.Time || '';
        const hour = time.split(':')[0] + ':00';
        groupKey = hour || 'Unknown Time';
      }
      
      if (!groups[groupKey]) {
        groups[groupKey] = [];
      }
      groups[groupKey].push(prediction);
    });

    let groupedData = Object.entries(groups).map(([key, groupPredictions]) => {
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
    });

    // Apply sorting based on sortOrder
    switch (sortOrder) {
      case 'low-to-high':
        groupedData.sort((a, b) => a.anomalyRate - b.anomalyRate);
        break;
      case 'high-to-low':
        groupedData.sort((a, b) => b.anomalyRate - a.anomalyRate);
        break;
      case 'default':
      default:
        groupedData.sort((a, b) => b.anomalyRate - a.anomalyRate);
        break;
    }

    return groupedData;
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
    if (rate >= 80) return 'border-red-500/50 bg-red-500/10 shadow-red-500/20';
    if (rate >= 60) return 'border-orange-500/50 bg-orange-500/10 shadow-orange-500/20';
    if (rate >= 40) return 'border-yellow-500/50 bg-yellow-500/10 shadow-yellow-500/20';
    return 'border-green-500/50 bg-green-500/10 shadow-green-500/20';
  };

  const getSeverityIcon = (rate: number) => {
    if (rate >= 80) return <AlertTriangle className="w-5 h-5 text-red-400" />;
    if (rate >= 60) return <TrendingUp className="w-5 h-5 text-orange-400" />;
    if (rate >= 40) return <BarChart3 className="w-5 h-5 text-yellow-400" />;
    return <CheckCircle className="w-5 h-5 text-green-400" />;
  };

  const getSeverityLabel = (rate: number) => {
    if (rate >= 80) return { label: 'Critical', color: 'text-red-400', bg: 'bg-red-500/20' };
    if (rate >= 60) return { label: 'High', color: 'text-orange-400', bg: 'bg-orange-500/20' };
    if (rate >= 40) return { label: 'Medium', color: 'text-yellow-400', bg: 'bg-yellow-500/20' };
    return { label: 'Low', color: 'text-green-400', bg: 'bg-green-500/20' };
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

  const getSortIcon = () => {
    switch (sortOrder) {
      case 'low-to-high': return <SortAsc className="w-4 h-4" />;
      case 'high-to-low': return <SortDesc className="w-4 h-4" />;
      default: return <ArrowUpDown className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6">
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-12 border border-gray-700 shadow-2xl">
          <div className="flex items-center justify-center space-x-4">
            <RefreshCw className="w-8 h-8 text-blue-400 animate-spin" />
            <div className="text-center">
              <div className="text-white text-2xl font-bold mb-2">Loading Predictions</div>
              <div className="text-gray-400">Analyzing network anomalies...</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6">
        <div className="bg-gradient-to-br from-red-900/20 to-gray-800 rounded-xl p-12 border border-red-500/30 shadow-2xl">
          <div className="flex items-center justify-center space-x-4 text-red-400 mb-6">
            <AlertTriangle className="w-8 h-8" />
            <div className="text-center">
              <div className="text-2xl font-bold mb-2">Error Loading Data</div>
              <div className="text-lg">{error}</div>
            </div>
          </div>
          <div className="text-center">
            <button 
              onClick={fetchPredictions}
              className="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white px-6 py-3 rounded-lg transition-all duration-200 shadow-lg font-semibold"
            >
              <RefreshCw className="w-5 h-5 inline mr-2" />
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6">
      <div className="space-y-6">
        {/* Enhanced Header */}
        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-8 border border-gray-700 shadow-2xl">
          <div className="flex flex-col space-y-6">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
              <div className="flex items-center space-x-4">
                <div className="p-4 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
                  <Eye className="h-10 w-10 text-white" />
                </div>
                <div>
                  <h1 className="text-4xl font-bold text-white">Anomaly Detection Results</h1>
                  <p className="text-gray-400 mt-2 text-lg">
                    Real-time analysis of {filteredGroups.length} groups with{' '}
                    <span className="text-red-400 font-semibold">
                      {predictions.filter(p => p.Predicted_Anomaly === 1).length}
                    </span>{' '}
                    total anomalies detected
                  </p>
                </div>
              </div>
              
              <button 
                onClick={fetchPredictions}
                className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-8 py-4 rounded-xl transition-all duration-200 flex items-center space-x-3 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                <RefreshCw className="w-6 h-6" />
                <span className="font-semibold">Refresh Data</span>
              </button>
            </div>

            {/* Enhanced View Mode Toggle */}
            <div className="bg-gray-700/50 rounded-xl p-6">
              <div className="flex flex-col lg:flex-row items-start lg:items-center space-y-6 lg:space-y-0 lg:space-x-8">
                <div className="flex items-center space-x-4">
                  <span className="text-gray-300 font-semibold text-lg">View Mode:</span>
                  <div className="flex bg-gray-800 rounded-xl p-2 shadow-inner">
                    <button
                      onClick={() => setViewMode('cell')}
                      className={`flex items-center space-x-3 px-6 py-3 rounded-lg transition-all duration-200 font-medium ${
                        viewMode === 'cell' 
                          ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg' 
                          : 'text-gray-300 hover:text-white hover:bg-gray-700'
                      }`}
                    >
                      <MapPin className="w-5 h-5" />
                      <span>Cell-wise Analysis</span>
                    </button>
                    <button
                      onClick={() => setViewMode('time')}
                      className={`flex items-center space-x-3 px-6 py-3 rounded-lg transition-all duration-200 font-medium ${
                        viewMode === 'time' 
                          ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg' 
                          : 'text-gray-300 hover:text-white hover:bg-gray-700'
                      }`}
                    >
                      <Calendar className="w-5 h-5" />
                      <span>Time-based Analysis</span>
                    </button>
                  </div>
                </div>

                {/* Enhanced Filters */}
                <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-6">
                  {/* Search */}
                  <div className="relative">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                    <input
                      type="text"
                      placeholder={`Search by ${viewMode === 'cell' ? 'cell name' : 'time'}...`}
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="bg-gray-800 text-white pl-12 pr-6 py-3 rounded-xl border border-gray-600 focus:border-blue-500 focus:outline-none w-72 transition-all duration-200 shadow-inner"
                    />
                  </div>
                  
                  {/* Enhanced Filter & Sort Dropdown */}
                  <div className="flex items-center space-x-4">
                    

                    {/* Sort Dropdown */}
                    <div className="flex items-center space-x-3">
                      {getSortIcon()}
                      <select 
                        value={sortOrder} 
                        onChange={(e) => setSortOrder(e.target.value as SortType)}
                        className="bg-gray-800 text-white px-4 py-3 rounded-xl border border-gray-600 focus:border-blue-500 focus:outline-none transition-all duration-200 font-medium min-w-[180px]"
                      >
                        <option value="default">Default (High to Low)</option>
                        <option value="high-to-low">Anomaly Rate: High to Low</option>
                        <option value="low-to-high">Anomaly Rate: Low to High</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Summary Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 rounded-xl p-6 border border-blue-500/30 shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-300 text-sm font-semibold uppercase tracking-wide">Total Groups</p>
                <p className="text-3xl font-bold text-white mt-2">{filteredGroups.length}</p>
                <div className="flex items-center mt-2 text-blue-400 text-sm">
                  <Activity className="w-4 h-4 mr-1" />
                  Active monitoring
                </div>
              </div>
              <div className="p-4 bg-blue-500/20 rounded-xl">
                <BarChart3 className="w-8 h-8 text-blue-400" />
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-red-600/20 to-red-800/20 rounded-xl p-6 border border-red-500/30 shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-red-300 text-sm font-semibold uppercase tracking-wide">Critical Risk</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {filteredGroups.filter(g => g.anomalyRate >= 80).length}
                </p>
                <div className="flex items-center mt-2 text-red-400 text-sm">
                  <Shield className="w-4 h-4 mr-1" />
                  Immediate attention
                </div>
              </div>
              <div className="p-4 bg-red-500/20 rounded-xl">
                <AlertTriangle className="w-8 h-8 text-red-400" />
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-orange-600/20 to-orange-800/20 rounded-xl p-6 border border-orange-500/30 shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-orange-300 text-sm font-semibold uppercase tracking-wide">Medium Risk</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {filteredGroups.filter(g => g.anomalyRate >= 40 && g.anomalyRate < 80).length}
                </p>
                <div className="flex items-center mt-2 text-orange-400 text-sm">
                  <Target className="w-4 h-4 mr-1" />
                  Monitor closely
                </div>
              </div>
              <div className="p-4 bg-orange-500/20 rounded-xl">
                <TrendingUp className="w-8 h-8 text-orange-400" />
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 rounded-xl p-6 border border-green-500/30 shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-300 text-sm font-semibold uppercase tracking-wide">Normal Status</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {filteredGroups.filter(g => g.anomalyRate < 40).length}
                </p>
                <div className="flex items-center mt-2 text-green-400 text-sm">
                  <CheckCircle className="w-4 h-4 mr-1" />
                  Operating normally
                </div>
              </div>
              <div className="p-4 bg-green-500/20 rounded-xl">
                <CheckCircle className="w-8 h-8 text-green-400" />
              </div>
            </div>
          </div>
        </div>

        {/* Groups Grid */}
        {filteredGroups.length === 0 ? (
          <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-16 border border-gray-700 text-center shadow-2xl">
            <div className="mb-6">
              <Search className="w-16 h-16 text-gray-500 mx-auto mb-4" />
              <div className="text-gray-400 text-2xl font-bold mb-2">No Data Found</div>
              <p className="text-gray-500 text-lg">Try adjusting your filters or running detection first</p>
            </div>
            <button 
              onClick={fetchPredictions}
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-6 py-3 rounded-lg transition-all duration-200 font-semibold"
            >
              Refresh Data
            </button>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
              {paginatedGroups.map((group, index) => {
                const isExpanded = expandedCards.has(group.key);
                const severityColor = getSeverityColor(group.anomalyRate);
                const severity = getSeverityLabel(group.anomalyRate);
                
                return (
                  <div 
                    key={group.key}
                    className={`bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl border-2 ${severityColor} transition-all duration-300 hover:scale-105 hover:shadow-2xl overflow-hidden`}
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className="p-6">
                      {/* Enhanced Header */}
                      <div className="flex items-start justify-between mb-6">
                        <div className="flex items-center space-x-4">
                          <div className="p-3 bg-gray-700/50 rounded-xl">
                            {getSeverityIcon(group.anomalyRate)}
                          </div>
                          <div>
                            <h3 className="text-xl font-bold text-white truncate max-w-[200px]" title={group.key}>
                              {group.key}
                            </h3>
                            <p className="text-gray-400 text-sm flex items-center mt-2">
                              {viewMode === 'cell' ? (
                                <><Wifi className="w-4 h-4 mr-2" /> Network Cell</>
                              ) : (
                                <><Clock className="w-4 h-4 mr-2" /> Time Period</>
                              )}
                            </p>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className="text-3xl font-bold text-white mb-1">
                            {group.anomalyRate.toFixed(1)}%
                          </div>
                          <div className={`px-3 py-1 rounded-full text-xs font-bold ${severity.color} ${severity.bg}`}>
                            {severity.label} RISK
                          </div>
                        </div>
                      </div>

                      {/* Enhanced Stats */}
                      <div className="grid grid-cols-2 gap-4 mb-6">
                        <div className="bg-gradient-to-br from-red-500/10 to-red-600/10 border border-red-500/20 rounded-xl p-4">
                          <div className="flex items-center justify-between">
                            <AlertTriangle className="w-6 h-6 text-red-400" />
                            <div className="text-right">
                              <div className="text-red-400 font-bold text-xl">
                                {group.anomalies.length}
                              </div>
                              <div className="text-gray-400 text-xs font-medium">Anomalies</div>
                            </div>
                          </div>
                        </div>
                        <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 border border-blue-500/20 rounded-xl p-4">
                          <div className="flex items-center justify-between">
                            <BarChart3 className="w-6 h-6 text-blue-400" />
                            <div className="text-right">
                              <div className="text-blue-400 font-bold text-xl">
                                {group.totalPredictions}
                              </div>
                              <div className="text-gray-400 text-xs font-medium">Total</div>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Enhanced Latest Time */}
                      {group.latestTime && (
                        <div className="flex items-center justify-between text-gray-400 text-sm mb-6 bg-gray-700/30 rounded-lg p-3">
                          <div className="flex items-center">
                            <Clock className="w-4 h-4 mr-2" />
                            <span className="font-medium">Latest Detection:</span>
                          </div>
                          <span className="text-white font-semibold">{group.latestTime}</span>
                        </div>
                      )}

                      {/* Enhanced Expand Button */}
                      <button
                        onClick={() => toggleExpanded(group.key)}
                        className="w-full bg-gradient-to-r from-gray-700 to-gray-600 hover:from-gray-600 hover:to-gray-500 text-white px-6 py-4 rounded-xl transition-all duration-200 flex items-center justify-center space-x-3 font-semibold shadow-lg hover:shadow-xl"
                      >
                        {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                        <span>{isExpanded ? 'Hide' : 'Show'} Detailed Analysis</span>
                      </button>
                    </div>

                    {/* Enhanced Expanded Content */}
                    {isExpanded && (
                      <div className="border-t border-gray-700 bg-gray-800/50 backdrop-blur-sm">
                        <div className="p-6">
                          <h4 className="text-white font-bold text-lg mb-4 flex items-center">
                            <Eye className="w-5 h-5 mr-2 text-blue-400" />
                            Anomaly Details
                          </h4>
                          <div className="space-y-4 max-h-80 overflow-y-auto custom-scrollbar">
                            {group.anomalies.length > 0 ? (
                              group.anomalies.slice(0, 8).map((prediction, idx) => (
                                <div 
                                  key={idx}
                                  className="bg-gradient-to-r from-gray-700 to-gray-600 rounded-lg p-4 hover:from-gray-600 hover:to-gray-500 transition-all duration-200 cursor-pointer border border-gray-600 hover:border-gray-500 shadow-lg"
                                  onClick={() => setSelectedPrediction(prediction)}
                                >
                                  <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center space-x-3">
                                      <div className="p-2 bg-red-500/20 rounded-lg">
                                        <AlertTriangle className="w-4 h-4 text-red-400" />
                                      </div>
                                      <div>
                                        <span className="text-white text-sm font-semibold">
                                          {viewMode === 'cell' ? prediction.Time : prediction.CellName}
                                        </span>
                                        <div className="text-gray-400 text-xs">
                                          {viewMode === 'cell' ? 'Detection Time' : 'Cell Name'}
                                        </div>
                                      </div>
                                    </div>
                                    <div className="text-right">
                                      <div className="text-red-400 font-bold text-lg">
                                        {prediction.Anomaly_Probability ? 
                                          `${(prediction.Anomaly_Probability * 100).toFixed(1)}%` : 
                                          'N/A'
                                        }
                                      </div>
                                      <div className="text-gray-400 text-xs">Confidence</div>
                                    </div>
                                  </div>
                                  {prediction.NetworkType && (
                                    <div className="text-gray-400 text-xs flex items-center bg-gray-800/50 rounded-lg p-2">
                                      <Wifi className="w-3 h-3 mr-2" />
                                      <span className="font-medium">Network:</span>
                                      <span className="ml-2 text-gray-300">
                                        {truncateNetworkType(prediction.NetworkType, 40)}
                                      </span>
                                    </div>
                                  )}
                                </div>
                              ))
                            ) : (
                              <div className="text-center py-8">
                                <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-3" />
                                <div className="text-gray-400 text-lg">No anomalies detected</div>
                                <div className="text-gray-500 text-sm">This group is operating normally</div>
                              </div>
                            )}
                            
                            {group.anomalies.length > 8 && (
                              <div className="text-center bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                                <div className="text-gray-400 text-sm font-medium">
                                  ... and <span className="text-white font-bold">{group.anomalies.length - 8}</span> more anomalies
                                </div>
                                <div className="text-gray-500 text-xs mt-1">Click individual items for detailed analysis</div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Enhanced Pagination */}
            {totalPages > 1 && (
              <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-xl p-6 border border-gray-700 shadow-2xl">
                <div className="flex flex-col sm:flex-row items-center justify-between space-y-4 sm:space-y-0">
                  <div className="text-gray-400 text-sm">
                    Showing <span className="text-white font-semibold">{(currentPage - 1) * itemsPerPage + 1}</span> to{' '}
                    <span className="text-white font-semibold">{Math.min(currentPage * itemsPerPage, filteredGroups.length)}</span> of{' '}
                    <span className="text-white font-semibold">{filteredGroups.length}</span> groups
                  </div>
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                      disabled={currentPage === 1}
                      className="px-6 py-3 bg-gray-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors font-semibold shadow-lg"
                    >
                      Previous
                    </button>
                    <div className="flex items-center space-x-2">
                      {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                        const pageNum = i + 1;
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setCurrentPage(pageNum)}
                            className={`px-4 py-2 rounded-lg transition-colors font-semibold ${
                              currentPage === pageNum
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                      disabled={currentPage === totalPages}
                      className="px-6 py-3 bg-gray-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-600 transition-colors font-semibold shadow-lg"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* Enhanced Detail Modal */}
        {selectedPrediction && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center p-4 z-50">
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl border border-gray-700 max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
              <div className="p-8">
                <div className="flex items-center justify-between mb-8">
                  <h3 className="text-3xl font-bold text-white flex items-center">
                    <AlertTriangle className="w-8 h-8 mr-3 text-red-400" />
                    Detailed Anomaly Analysis
                  </h3>
                  <button 
                    onClick={() => setSelectedPrediction(null)}
                    className="text-gray-400 hover:text-white transition-colors text-3xl hover:bg-gray-700 rounded-lg p-2"
                  >
                    ×
                  </button>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="space-y-6">
                    <div className="bg-gradient-to-br from-gray-700 to-gray-600 rounded-xl p-6 border border-gray-600">
                      <span className="text-gray-400 text-sm font-semibold uppercase tracking-wide">Detection Status</span>
                      <div className="mt-4">
                        <div className={`flex items-center space-x-3 px-4 py-3 rounded-xl text-lg font-bold ${
                          selectedPrediction.Predicted_Anomaly === 1
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                            : 'bg-green-500/20 text-green-400 border border-green-500/30'
                        }`}>
                          {selectedPrediction.Predicted_Anomaly === 1 ? (
                            <AlertTriangle className="w-6 h-6" />
                          ) : (
                            <CheckCircle className="w-6 h-6" />
                          )}
                          <span>{selectedPrediction.Predicted_Anomaly === 1 ? 'ANOMALY DETECTED' : 'NORMAL BEHAVIOR'}</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-gray-700 to-gray-600 rounded-xl p-6 border border-gray-600">
                      <span className="text-gray-400 text-sm font-semibold uppercase tracking-wide">Confidence Level</span>
                      <div className="mt-4">
                        <div className="text-4xl font-bold text-red-400 mb-2">
                          {selectedPrediction.Anomaly_Probability ? 
                            `${(selectedPrediction.Anomaly_Probability * 100).toFixed(2)}%` : 
                            'N/A'
                          }
                        </div>
                        <div className="w-full bg-gray-800 rounded-full h-3">
                          <div 
                            className="h-3 bg-gradient-to-r from-red-500 to-red-600 rounded-full transition-all duration-1000"
                            style={{ 
                              width: `${selectedPrediction.Anomaly_Probability ? (selectedPrediction.Anomaly_Probability * 100) : 0}%` 
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-gray-700 to-gray-600 rounded-xl p-6 border border-gray-600">
                      <span className="text-gray-400 text-sm font-semibold uppercase tracking-wide">Location Details</span>
                      <div className="mt-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300 font-medium">Cell Name:</span>
                          <span className="text-white font-bold">{selectedPrediction.CellName || 'N/A'}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-gray-300 font-medium">Detection Time:</span>
                          <span className="text-blue-400 font-semibold flex items-center">
                            <Clock className="w-4 h-4 mr-2" />
                            {selectedPrediction.Time || 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-6">
                    <div className="bg-gradient-to-br from-gray-700 to-gray-600 rounded-xl p-6 border border-gray-600">
                      <span className="text-gray-400 text-sm font-semibold uppercase tracking-wide">Network Information</span>
                      <div className="mt-4">
                        {selectedPrediction.NetworkType ? (
                          <div className="bg-gray-800 rounded-xl p-4 max-h-32 overflow-y-auto border border-gray-600">
                            <div className="text-white font-mono text-sm leading-relaxed">
                              {selectedPrediction.NetworkType}
                            </div>
                          </div>
                        ) : (
                          <div className="text-gray-400 text-center py-4">No network type information available</div>
                        )}
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-gray-700 to-gray-600 rounded-xl p-6 border border-gray-600">
                      <span className="text-gray-400 text-sm font-semibold uppercase tracking-wide">Raw Data Analysis</span>
                      <div className="mt-4">
                        <div className="bg-gray-900 rounded-xl p-4 max-h-60 overflow-y-auto border border-gray-600">
                          <pre className="text-gray-300 text-xs leading-relaxed font-mono">
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

      {/* Custom Styles */}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #1f2937;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #4b5563;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #6b7280;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn { animation: fadeIn 0.4s ease-out forwards; }
      `}</style>
    </div>
  );
};

export default PredictionsTable;