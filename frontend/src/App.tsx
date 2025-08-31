// App.tsx
import React, { useEffect, useState } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, LineChart, Line
} from 'recharts';
import GaugeChart from "react-gauge-chart";
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  Clock,
  Eye,
  Zap,
  Database,
  Target,
  Search,
  BarChart3,
} from 'lucide-react';

import PredictionsTable from './components/PredictionsTable';
import Statistics from './components/Statistics';
import { apiService, evaluationService, DashboardData } from './api';

// ---------- Local types ----------
type IconType = React.ElementType;

interface MetricItem {
  name: string;
  value: number;      // percentage value (0-100)
  deviation: number;  // ± percentage
  color: string;
  icon: IconType;
}

type AnomalyTimePoint = {
  hour: number;
  anomalies: number;
};

type StatsCardProps = {
  title: string;
  value: string | number;
  icon: IconType;
  trend?: number;
  color?: 'blue' | 'red' | 'green' | 'orange' | 'purple';
};

type EvaluationMetricsProps = {
  metrics: MetricItem[];
};

type AlgoPerf = { name: string; accuracy: number; f1: number; precision: number; recall: number };

type AlgorithmComparisonProps = {
  data: AlgoPerf[];
  selectedAlgorithm: number | null;
  setSelectedAlgorithm: (v: number | null) => void;
};

type PieDatum = { name: string; value: number; color: string };
type AnomalyPieChartProps = { data: PieDatum[] };

// ---------- UI Components ----------
const StatsCard: React.FC<StatsCardProps> = ({ title, value, icon: Icon, color = 'blue' }) => {
  const colorMap: Record<NonNullable<StatsCardProps['color']>, string> = {
    blue: 'from-blue-500 to-blue-600',
    red: 'from-red-500 to-red-600',
    green: 'from-green-500 to-green-600',
    orange: 'from-orange-500 to-orange-600',
    purple: 'from-purple-500 to-purple-600',
  };

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all duration-300 transform hover:scale-105">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm font-medium">{title}</p>
          <p className="text-3xl font-bold text-white mt-2">{value}</p>
        </div>
        <div className={`p-3 rounded-lg bg-gradient-to-r ${colorMap[color]}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
      </div>
    </div>
  );
};

// ---------- Metric Gauge ----------
type MetricGaugeProps = {
  name: string;
  value: number;
  color: string;
};

const MetricGauge: React.FC<MetricGaugeProps> = ({ name, value }) => {
  const getNeedleColor = (val: number) => {
    if (val >= 90) return "#10B981";
    if (val >= 70) return "#F59E0B";
    return "#EF4444";
  };

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 flex flex-col items-center w-full h-full">
      <div className="text-white font-medium mb-3">{name}</div>
      <GaugeChart
        id={`gauge-${name}`}
        nrOfLevels={5}
        colors={["#10B981", "#F59E0B", "#EF4444"]}
        arcWidth={0.3}
        percent={value / 100}
        textColor="#F9FAFB"
        needleColor={getNeedleColor(value)}
        needleBaseColor="#374151"
        formatTextValue={() => `${value.toFixed(2)}%`}
        style={{ width: "100%", height: "160px" }}
      />
    </div>
  );
};

// ---------- Evaluation Metrics ----------
const EvaluationMetrics: React.FC<EvaluationMetricsProps> = ({ metrics }) => {
  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 h-full">
      <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
        <TrendingUp className="mr-2 h-5 w-5 text-blue-400" />
        Evaluation Metrics
      </h3>
      <div className="grid grid-cols-2 gap-6">
        {metrics.map((metric, idx) => (
          <MetricGauge
            key={idx}
            name={metric.name}
            value={metric.value}
            color={metric.color}
          />
        ))}
      </div>
    </div>
  );
};

// ---------- Algorithm Comparison ----------
const AlgorithmComparison: React.FC<AlgorithmComparisonProps> = ({ data, selectedAlgorithm, setSelectedAlgorithm }) => {
  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
        <Activity className="mr-2 h-5 w-5 text-purple-400" />
        Algorithm Performance Comparison
      </h3>

      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" stroke="#9CA3AF" angle={-45} textAnchor="end" height={80} fontSize={12} />
          <YAxis stroke="#9CA3AF" domain={[80, 100]} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#F9FAFB'
            }}
          />
          <Legend />
          <Bar dataKey="accuracy" fill="#3B82F6" name="Accuracy" radius={[2, 2, 0, 0]} />
          <Bar dataKey="f1" fill="#10B981" name="F1-Score" radius={[2, 2, 0, 0]} />
          <Bar dataKey="precision" fill="#8B5CF6" name="Precision" radius={[2, 2, 0, 0]} />
          <Bar dataKey="recall" fill="#F59E0B" name="Recall" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

// ---------- Anomaly Pie Chart ----------
const AnomalyPieChart: React.FC<AnomalyPieChartProps> = ({ data }) => (
  <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
      <Eye className="mr-2 h-5 w-5 text-green-400" />
      Anomaly vs Normal Distribution
    </h3>
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          outerRadius={100}
          dataKey="value"
          startAngle={90}
          endAngle={450}
        >
          {data.map((entry, index) => (
            <Cell key={index} fill={entry.color} stroke={entry.color} strokeWidth={2} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: '#1F2937',
            border: '1px solid #374151',
            borderRadius: '8px',
            color: '#F9FAFB'
          }}
          formatter={(value: any) => [`${value}%`]}
        />
      </PieChart>
    </ResponsiveContainer>
  </div>
);

// ---------- Time vs Anomalies Graph ----------
const TimeByAnomalyGraph: React.FC<{ data: AnomalyTimePoint[] }> = ({ data }) => (
  <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
      <Clock className="mr-2 h-5 w-5 text-blue-400" />
      Time vs Anomalies
    </h3>
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="hour" stroke="#9CA3AF" />
        <YAxis stroke="#9CA3AF" />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1F2937',
            border: '1px solid #374151',
            borderRadius: '8px',
            color: '#F9FAFB'
          }}
        />
        <Legend />
        <Line 
          type="monotone" 
          dataKey="anomalies" 
          name="Anomalies"
          stroke="#3B82F6" 
          strokeWidth={3} 
          dot={{ r: 4 }} 
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);


// ---------- Main App ----------
function App() {
  const [evaluationMetrics, setEvaluationMetrics] = useState<MetricItem[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'predictions' | 'statistics'>('dashboard');
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const [distributionData, setDistributionData] = useState<PieDatum[]>([
    { name: 'Normal Traffic', value: 0, color: '#10B981' },
    { name: 'Anomalous Traffic', value: 0, color: '#EF4444' }
  ]);
  const [stats, setStats] = useState({
    totalSamples: '0',
    anomaliesDetected: '0',
    detectionRate: '0%',
    responseTime: '0ms',
  });
  const [algorithmPerformanceData, setAlgorithmPerformanceData] = useState<AlgoPerf[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<AnomalyTimePoint[]>([]);

  useEffect(() => {
    fetchDashboardData();
    fetchEvaluationMetrics();
    fetchAlgorithmData();
    fetchTimeSeriesData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const result = await apiService.getDashboard();
      if (result.status === 'success') {
        setStats({
          totalSamples: result.data.total_samples.toLocaleString(),
          anomaliesDetected: result.data.total_anomalies.toLocaleString(),
          detectionRate: `${result.data.anomaly_rate}%`,
          responseTime: '0.23ms',
        });

        const anomalyRate = result.data.anomaly_rate;
        const normalRate = 100 - anomalyRate;

        setDistributionData([
          { name: 'Normal Traffic', value: parseFloat(normalRate.toFixed(2)), color: '#10B981' },
          { name: 'Anomalous Traffic', value: parseFloat(anomalyRate.toFixed(2)), color: '#EF4444' }
        ]);
        setLastUpdate(new Date().toLocaleString());
      }
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
    }
  };

  const fetchEvaluationMetrics = async () => {
    try {
      const result = await evaluationService.getEvaluationMetrics();
      const raw = (result as any)?.data ?? result;

      const formatted = [
        { name: 'Accuracy', value: Number((raw?.Accuracy ?? 0) * 100), deviation: 0.15, color: '#10B981', icon: Target },
        { name: 'F1-Score', value: Number((raw?.F1_Score ?? 0) * 100), deviation: 0.29, color: '#8B5CF6', icon: BarChart3 },
        { name: 'Precision', value: Number((raw?.Precision ?? 0) * 100), deviation: 0.13, color: '#3B82F6', icon: Search },
        { name: 'Recall', value: Number((raw?.Recall ?? 0) * 100), deviation: 0.53, color: '#F59E0B', icon: Eye },
      ];
      setEvaluationMetrics(formatted);
    } catch (err) {
      console.error('Error fetching evaluation metrics:', err);
    }
  };

  const fetchAlgorithmData = async () => {
    try {
      const result = await apiService.getBestModels();
      if (result.status === 'success') {
        const transformedData: AlgoPerf[] = result.data.map((modelObj) => {
          const modelName = Object.keys(modelObj)[0];
          const metrics = modelObj[modelName];
          return {
            name: modelName,
            accuracy: parseFloat((metrics.Accuracy_Mean * 100).toFixed(2)),
            f1: parseFloat((metrics.F1_Mean * 100).toFixed(2)),
            precision: parseFloat((metrics.Precision_Mean * 100).toFixed(2)),
            recall: parseFloat((metrics.Recall_Mean * 100).toFixed(2))
          };
        });
        setAlgorithmPerformanceData(transformedData);
      }
    } catch (err) {
      console.error('Error fetching algorithm data:', err);
    }
  };

  const fetchTimeSeriesData = async () => {
    try {
      const result = await apiService.getAnomalyTimeSeries();
      if (result.status === "success") {
        setTimeSeriesData(result.data);
      }
    } catch (err) {
      console.error("Error fetching time series:", err);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'predictions':
        return <PredictionsTable />;
      case 'statistics':
        return <Statistics />;
      case 'dashboard':
      default:
        return (
          <div className="space-y-8">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatsCard title="Total Samples" value={stats.totalSamples} icon={Database} color="blue" />
              <StatsCard title="Anomalies Detected" value={stats.anomaliesDetected} icon={AlertTriangle} color="red" />
              <StatsCard title="Detection Rate" value={stats.detectionRate} icon={TrendingUp} color="green" />
              <StatsCard title="Response Time" value={stats.responseTime} icon={Zap} color="purple" />
            </div>

            {/* Evaluation Metrics + Right Column */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <EvaluationMetrics metrics={evaluationMetrics} />
              <AnomalyPieChart data={distributionData} />
            </div>

            {/* Algorithm Performance */}
            <AlgorithmComparison
              data={algorithmPerformanceData}
              selectedAlgorithm={selectedAlgorithm}
              setSelectedAlgorithm={setSelectedAlgorithm}
            />

            {/* Time vs Anomalies */}
            <TimeByAnomalyGraph data={timeSeriesData} />
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Header */}
      <div className="bg-gray-800/80 backdrop-blur-sm border-b border-gray-700 px-6 py-6 sticky top-0 z-10">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between space-y-4 lg:space-y-0">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
              <Activity className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Anomaly Detection Dashboard</h1>
              <p className="text-gray-400 mt-1">Real-time network security monitoring & threat analysis</p>
            </div>
          </div>
          {lastUpdate && <div className="text-sm text-gray-400">Last updated: {lastUpdate}</div>}
        </div>

        {error && (
          <div className="mt-4 bg-red-900/20 border border-red-500/30 rounded-lg p-3">
            <div className="flex items-center space-x-2 text-red-400">
              <AlertTriangle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="bg-gray-800/50 backdrop-blur-sm border-b border-gray-700 px-6 py-4">
        <nav className="flex space-x-8">
          <button
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'dashboard' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white hover:bg-gray-700'
            }`}
            onClick={() => setActiveTab('dashboard')}
          >
            <Activity className="w-4 h-4" />
            <span>Dashboard</span>
          </button>
          <button
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'predictions' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white hover:bg-gray-700'
            }`}
            onClick={() => setActiveTab('predictions')}
          >
            <Eye className="w-4 h-4" />
            <span>Predictions</span>
          </button>
          <button
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'statistics' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white hover:bg-gray-700'
            }`}
            onClick={() => setActiveTab('statistics')}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Statistics</span>
          </button>
        </nav>
      </div>

      <div className="p-6">{renderContent()}</div>
    </div>
  );
}

export default App;
