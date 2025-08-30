import React, { useState, useEffect } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';
import { 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  BarChart3,
  PieChart as PieChartIcon,
  Activity,
  Info,
  Lightbulb
} from 'lucide-react';
import { apiService, Statistics as StatisticsData } from '../api';

const Statistics: React.FC = () => {
  const [stats, setStats] = useState<StatisticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-xl p-8 border border-gray-700">
        <div className="flex items-center justify-center space-x-4">
          <RefreshCw className="w-6 h-6 text-blue-400 animate-spin" />
          <span className="text-white text-lg">Loading statistics...</span>
        </div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="bg-gray-800 rounded-xl p-8 border border-red-500/30">
        <div className="flex items-center justify-center space-x-4 text-red-400">
          <AlertTriangle className="w-6 h-6" />
          <span className="text-lg">{error || 'No statistics available'}</span>
        </div>
        <div className="text-center mt-4">
          <button 
            onClick={fetchStatistics}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const anomalyRate = stats.total_records > 0 ? (stats.anomalies / stats.total_records * 100) : 0;
  const normalRate = stats.total_records > 0 ? (stats.normal / stats.total_records * 100) : 0;

  // Data for charts
  const distributionData = [
    { name: 'Normal', value: stats.normal, color: '#10B981', percentage: normalRate.toFixed(2) },
    { name: 'Anomalies', value: stats.anomalies, color: '#EF4444', percentage: anomalyRate.toFixed(2) }
  ];

  const detailsData = [
    { name: 'Total Records', value: stats.total_records },
    { name: 'Normal', value: stats.normal },
    { name: 'Anomalies', value: stats.anomalies }
  ];

  const StatCard: React.FC<{ 
    title: string; 
    value: string; 
    subtitle?: string; 
    icon: React.ElementType; 
    color: string;
    trend?: number;
  }> = ({ title, value, subtitle, icon: Icon, color, trend }) => (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all duration-300 transform hover:scale-105">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Icon className={`w-5 h-5 ${color}`} />
            <h3 className="text-gray-300 font-medium">{title}</h3>
          </div>
          <div className="text-2xl font-bold text-white mb-1">{value}</div>
          {subtitle && (
            <div className="text-sm text-gray-400">{subtitle}</div>
          )}
          {trend !== undefined && (
            <div className={`text-xs mt-1 ${trend > 0 ? 'text-green-400' : trend < 0 ? 'text-red-400' : 'text-gray-400'}`}>
              {trend > 0 ? '↗' : trend < 0 ? '↙' : '→'} {Math.abs(trend).toFixed(1)}% from baseline
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const InsightCard: React.FC<{
    type: 'success' | 'warning' | 'info';
    title: string;
    description: string;
  }> = ({ type, title, description }) => {
    const colorMap = {
      success: 'border-green-500/30 bg-green-500/10 text-green-400',
      warning: 'border-orange-500/30 bg-orange-500/10 text-orange-400',
      info: 'border-blue-500/30 bg-blue-500/10 text-blue-400'
    };

    const iconMap = {
      success: CheckCircle,
      warning: AlertTriangle,
      info: Info
    };

    const Icon = iconMap[type];

    return (
      <div className={`border rounded-lg p-4 ${colorMap[type]}`}>
        <div className="flex items-start space-x-3">
          <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-medium mb-1">{title}</h4>
            <p className="text-sm opacity-90">{description}</p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center">
              <BarChart3 className="mr-3 h-6 w-6 text-purple-400" />
              Detailed Statistics
            </h2>
            <p className="text-gray-400 mt-1">
              Comprehensive analysis of anomaly detection results
            </p>
          </div>
          <button 
            onClick={fetchStatistics}
            className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Key Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Records"
          value={stats.total_records.toLocaleString()}
          icon={BarChart3}
          color="text-blue-400"
          trend={0}
        />
        <StatCard
          title="Anomalies Detected"
          value={stats.anomalies.toLocaleString()}
          subtitle={`${anomalyRate.toFixed(2)}% of total`}
          icon={AlertTriangle}
          color="text-red-400"
          trend={anomalyRate > 5 ? 15 : anomalyRate < 1 ? -5 : 0}
        />
        <StatCard
          title="Normal Records"
          value={stats.normal.toLocaleString()}
          subtitle={`${normalRate.toFixed(2)}% of total`}
          icon={CheckCircle}
          color="text-green-400"
          trend={normalRate > 95 ? 2 : normalRate < 90 ? -8 : 0}
        />
        <StatCard
          title="Detection Rate"
          value={`${anomalyRate.toFixed(3)}%`}
          subtitle="Overall anomaly rate"
          icon={Activity}
          color="text-orange-400"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Distribution Pie Chart */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <PieChartIcon className="mr-2 h-5 w-5 text-green-400" />
            Data Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={distributionData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                dataKey="value"
                label={({ name, percentage }) => `${name}: ${percentage}%`}
                labelLine={false}
              >
                {distributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value, name) => [value.toLocaleString(), name]}
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F9FAFB'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center mt-4 space-x-6">
            {distributionData.map((item, index) => (
              <div key={index} className="text-center">
                <div className="flex items-center justify-center mb-1">
                  <div 
                    className="w-3 h-3 rounded-full mr-2" 
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <span className="text-gray-300 text-sm">{item.name}</span>
                </div>
                <div className="text-lg font-bold text-white">{item.value.toLocaleString()}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Bar Chart */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <BarChart3 className="mr-2 h-5 w-5 text-blue-400" />
            Records Overview
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={detailsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                formatter={(value) => [value.toLocaleString(), 'Count']}
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#F9FAFB'
                }}
              />
              <Bar dataKey="value" fill="#3B82F6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Probability Statistics */}
      {stats.avg_anomaly_prob !== undefined && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
            <TrendingUp className="mr-2 h-5 w-5 text-yellow-400" />
            Probability Statistics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">Average Probability</div>
              <div className="text-2xl font-bold text-yellow-400">
                {(stats.avg_anomaly_prob * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">Maximum Probability</div>
              <div className="text-2xl font-bold text-red-400">
                {(stats.max_anomaly_prob! * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="text-gray-400 text-sm mb-1">Minimum Probability</div>
              <div className="text-2xl font-bold text-green-400">
                {(stats.min_anomaly_prob! * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      

      
      
    </div>
  );
};

export default Statistics;
