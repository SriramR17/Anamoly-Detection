// App.tsx
import React, { useEffect, useState } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar
} from 'recharts';
import GaugeChart from "react-gauge-chart";
import {
  Activity,
  AlertTriangle,
  Shield,
  TrendingUp,
  Clock,
  Network,
  Eye,
  Zap,
  Database,
  Target,
  Search,
  BarChart3,
  Play,
  RefreshCw
} from 'lucide-react';

import PredictionsTable from './components/PredictionsTable';
import Statistics from './components/Statistics';
import { apiService, evaluationService, DashboardData, BestModelData } from './api';

// ---------- Local types ----------
type IconType = React.ElementType;

interface MetricItem {
  name: string;
  value: number;      // percentage value (0-100)
  deviation: number;  // ± percentage
  color: string;
  icon: IconType;
}

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

type Severity = 'high' | 'medium' | 'low';
type NetworkInfo = {
  name: string;
  totalSamples: number;
  anomaliesDetected: number;
  anomalyRate: number;
  lastUpdate: string;
  recentAnomaly: string;
  severity: Severity;
};
type NetworkTileProps = {
  network: NetworkInfo;
  index: number;
  isSelected: boolean;
  onClick: (index: number) => void;
};
type TopNetworksProps = {
  networks: NetworkInfo[];
  selectedNetwork: number | null;
  setSelectedNetwork: (v: number | null) => void;
};

// ---------- Sample data ----------
const topAnomalyNetworks: NetworkInfo[] = [
  { name: 'Network-Alpha-001', totalSamples: 125847, anomaliesDetected: 2341, anomalyRate: 1.86, lastUpdate: '2 min ago', recentAnomaly: 'Port Scan Attack', severity: 'high' },
  { name: 'Network-Beta-002', totalSamples: 98563, anomaliesDetected: 1897, anomalyRate: 1.92, lastUpdate: '5 min ago', recentAnomaly: 'DDoS Attempt', severity: 'high' },
  { name: 'Network-Gamma-003', totalSamples: 87234, anomaliesDetected: 1654, anomalyRate: 1.90, lastUpdate: '8 min ago', recentAnomaly: 'Malware Traffic', severity: 'medium' },
  { name: 'Network-Delta-004', totalSamples: 76895, anomaliesDetected: 1423, anomalyRate: 1.85, lastUpdate: '12 min ago', recentAnomaly: 'Brute Force', severity: 'medium' },
  { name: 'Network-Epsilon-005', totalSamples: 65432, anomaliesDetected: 1198, anomalyRate: 1.83, lastUpdate: '15 min ago', recentAnomaly: 'SQL Injection', severity: 'low' },
  { name: 'Network-Zeta-006', totalSamples: 54321, anomaliesDetected: 987, anomalyRate: 1.82, lastUpdate: '18 min ago', recentAnomaly: 'XSS Attack', severity: 'low' }
];

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

// ---------- Metric Gauge (new) ----------
type MetricGaugeProps = {
  name: string;
  value: number;
  color: string;
};

const MetricGauge: React.FC<MetricGaugeProps> = ({ name, value }) => {
  // Needle color logic
  const getNeedleColor = (val: number) => {
    if (val >= 90) return "#10B981";   // green
    if (val >= 70) return "#F59E0B";   // yellow
    return "#EF4444";                  // red
  };

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 flex flex-col items-center w-full h-full">
      <div className="text-white font-medium mb-3">{name}</div>
      <GaugeChart
        id={`gauge-${name}`}
        nrOfLevels={5}                              // 5 thick segments
        colors={["#10B981", "#F59E0B", "#EF4444"]}  // arc: green → yellow → red
        arcWidth={0.3}
        percent={value / 100}
        textColor="#F9FAFB"
        needleColor={getNeedleColor(value)}         // dynamic needle color
        needleBaseColor="#374151"
        formatTextValue={() => `${value.toFixed(2)}%`}
        style={{ width: "100%", height: "160px" }}
      />
    </div>
  );
};




// ---------- Evaluation Metrics (now gauges) ----------
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

      <div className="mb-4 flex flex-wrap gap-2">
        {data.map((algo, idx) => (
          <button
            key={idx}
            onClick={() => setSelectedAlgorithm(selectedAlgorithm === idx ? null : idx)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${
              selectedAlgorithm === idx
                ? 'bg-blue-600 text-white ring-2 ring-blue-400'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {algo.name}
          </button>
        ))}
      </div>

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

      {selectedAlgorithm !== null && (
        <div className="mt-4 p-4 bg-gray-700 rounded-lg animate-fadeIn">
          <h4 className="text-white font-semibold mb-2">{data[selectedAlgorithm].name} Details</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div><span className="text-gray-400">Accuracy:</span><span className="text-blue-400 font-medium ml-2">{data[selectedAlgorithm].accuracy}%</span></div>
            <div><span className="text-gray-400">F1-Score:</span><span className="text-green-400 font-medium ml-2">{data[selectedAlgorithm].f1}%</span></div>
            <div><span className="text-gray-400">Precision:</span><span className="text-purple-400 font-medium ml-2">{data[selectedAlgorithm].precision}%</span></div>
            <div><span className="text-gray-400">Recall:</span><span className="text-orange-400 font-medium ml-2">{data[selectedAlgorithm].recall}%</span></div>
          </div>
        </div>
      )}
    </div>
  );
};

// ---------- Anomaly Pie Chart ----------
const AnomalyPieChart: React.FC<AnomalyPieChartProps> = ({ data }) => {
  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={14} fontWeight="bold">
        {`${(percent * 100).toFixed(1)}%`}
      </text>
    );
  };

  return (
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
            label={renderCustomizedLabel}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            startAngle={90}
            endAngle={450}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} stroke={entry.color} strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#F9FAFB'
            }}
            formatter={(value: any, name: any) => [`${value}%`, name]}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="flex justify-center mt-4 space-x-8">
        {data.map((item, index) => (
          <div key={index} className="text-center">
            <div className="flex items-center justify-center mb-1">
              <div className="w-4 h-4 rounded-full mr-2" style={{ backgroundColor: item.color }} />
              <span className="text-gray-300 text-sm">{item.name}</span>
            </div>
            <div className="text-2xl font-bold text-white">{item.value}%</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ---------- Peak Hours Clock ----------
const PeakHoursClock: React.FC = () => {
  const [peakHour, setPeakHour] = useState<number>(0);
  const [rotation, setRotation] = useState(0);
  const [loading, setLoading] = useState(true);
  const [anomalyPercentage, setAnomalyPercentage] = useState(85);

  useEffect(() => {
    const fetchPeakHour = async () => {
      try {
        const response = await fetch('/api/dashboard');
        const result = await response.json();
        
        if (result.status === 'success') {
          setPeakHour(result.data.peak_hours);
          setAnomalyPercentage(result.data.peak_rate);
        }
      } catch (error) {
        console.error('Failed to fetch peak hour:', error);
        setPeakHour(14); // Fallback to default
      } finally {
        setLoading(false);
      }
    };

    fetchPeakHour();
    
    // Optional: Poll for updates
    const interval = setInterval(fetchPeakHour, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const deg = (peakHour % 12) * 30;
    const timer = setTimeout(() => setRotation(deg), 1000);
    return () => clearTimeout(timer);
  }, [peakHour]);

  const formatHour = (hour: number): string => {
    if (hour === 0) return '12:00 AM';
    if (hour < 12) return `${hour}:00 AM`;
    if (hour === 12) return '12:00 PM';
    return `${hour - 12}:00 PM`;
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4"></div>
          <div className="w-60 h-60 bg-gray-700 rounded-full mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
        <Clock className="mr-2 h-5 w-5 text-orange-400" />
        Peak Anomaly Hours
      </h3>
      <div className="flex justify-center">
        <div className="relative w-60 h-60 rounded-full border-4 border-gray-600 flex items-center justify-center">
          {[...Array(12)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-4 bg-gray-400"
              style={{
                top: "6px",
                left: "50%",
                transform: `rotate(${i * 30}deg) translateX(-50%)`,
                transformOrigin: "center 120px",
              }}
            />
          ))}
          <div
            className="absolute w-1 bg-orange-400 origin-bottom rounded"
            style={{
              height: "40%",
              bottom: "50%",
              left: "50%",
              transform: `translateX(-50%) rotate(${rotation}deg)`,
              transition: "transform 1s ease-in-out",
            }}
          />
        </div>
      </div>
      <div className="text-center mt-4">
        <div className="text-3xl font-bold text-white">{formatHour(peakHour)}</div>
        <div className="text-gray-400">Peak Anomaly Detection Time</div>
        <div className="mt-2 text-sm text-orange-400">
          {anomalyPercentage}% of daily anomalies occur during this hour
        </div>
      </div>
    </div>
  );
};


// ---------- Network Tile ----------
const NetworkTile: React.FC<NetworkTileProps> = ({ network, index, isSelected, onClick }) => {
  const severityColors: Record<Severity, string> = {
    high: 'border-red-500 bg-red-500/10 shadow-red-500/20',
    medium: 'border-orange-500 bg-orange-500/10 shadow-orange-500/20',
    low: 'border-green-500 bg-green-500/10 shadow-green-500/20',
  };

  const severityTextColors: Record<Severity, string> = {
    high: 'text-red-400',
    medium: 'text-orange-400',
    low: 'text-green-400',
  };

  return (
    <div
      className={`bg-gray-800 rounded-xl p-5 border-2 cursor-pointer transition-all duration-300 transform hover:scale-105 ${
        severityColors[network.severity]
      } ${isSelected ? 'ring-2 ring-blue-500 shadow-lg' : 'hover:shadow-lg'}`}
      onClick={() => onClick(index)}
    >
      <div className="flex items-center justify-between mb-4">
        <Network className={`h-6 w-6 ${severityTextColors[network.severity]}`} />
        <span className={`px-3 py-1 text-xs rounded-full font-medium ${severityTextColors[network.severity]} bg-gray-700`}>
          {network.severity.toUpperCase()}
        </span>
      </div>

      <h4 className="text-white font-bold text-lg mb-3 truncate">{network.name}</h4>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Total Samples:</span>
          <span className="text-white font-semibold">{network.totalSamples.toLocaleString()}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Anomalies Detected:</span>
          <span className="text-red-400 font-semibold">{network.anomaliesDetected.toLocaleString()}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-400 text-sm">Anomaly Rate:</span>
          <span className="text-orange-400 font-bold">{network.anomalyRate}%</span>
        </div>

        <div className="pt-3 border-t border-gray-700">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400 text-xs">Last Update:</span>
            <span className="text-gray-300 text-xs">{network.lastUpdate}</span>
          </div>
          <div>
            <span className="text-gray-400 text-xs">Recent Anomaly: </span>
            <span className={`text-xs font-medium ${severityTextColors[network.severity]}`}>{network.recentAnomaly}</span>
          </div>
        </div>

        <div className="mt-3">
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="h-2 rounded-full transition-all duration-1000"
              style={{
                width: `${(network.anomalyRate / 3) * 100}%`,
                backgroundColor:
                  network.severity === 'high' ? '#EF4444' : network.severity === 'medium' ? '#F59E0B' : '#10B981'
              }}
            />
          </div>
        </div>
      </div>

      {isSelected && (
        <div className="mt-4 p-3 bg-gray-700 rounded-lg animate-fadeIn">
          <div className="text-sm text-gray-300">
            <div className="grid grid-cols-2 gap-2">
              <div>Detection Accuracy: <span className="text-green-400">98.5%</span></div>
              <div>False Positives: <span className="text-yellow-400">1.2%</span></div>
              <div>Response Time: <span className="text-blue-400">0.3ms</span></div>
              <div>Threat Level: <span className={severityTextColors[network.severity]}>{network.severity}</span></div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ---------- Top Networks ----------
const TopNetworks: React.FC<TopNetworksProps> = ({ networks, selectedNetwork, setSelectedNetwork }) => (
  <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
      <Shield className="mr-2 h-5 w-5 text-red-400" />
      Top Anomaly Networks
    </h3>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {networks.map((network, index) => (
        <NetworkTile
          key={index}
          network={network}
          index={index}
          isSelected={selectedNetwork === index}
          onClick={(i) => setSelectedNetwork(selectedNetwork === i ? null : i)}
        />
      ))}
    </div>
  </div>
);

// ---------- Main App ----------
function App() {
  const [evaluationMetrics, setEvaluationMetrics] = useState<MetricItem[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<number | null>(null);
  const [selectedNetwork, setSelectedNetwork] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'predictions' | 'statistics'>('dashboard');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
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
  const [isLoadingAlgorithms, setIsLoadingAlgorithms] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    fetchEvaluationMetrics();
    fetchAlgorithmData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const result = await apiService.getDashboard();
      if (result.status === 'success') {
        setDashboardData(result.data);
        setStats({
          totalSamples: result.data.total_samples.toLocaleString(),
          anomaliesDetected: result.data.total_anomalies.toLocaleString(),
          detectionRate: `${result.data.anomaly_rate}%`,
          responseTime: '0.23ms',
        });

        const anomalyRate = result.data.anomaly_rate;
        const normalRate = 100 - anomalyRate;

        const peak_hour=result.data.peak_hours;

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
      setIsLoadingAlgorithms(true);
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
    } finally {
      setIsLoadingAlgorithms(false);
    }
  };

  const handleRunDetection = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.runDetection();
      if (result.status === 'success') {
        setLastUpdate(new Date().toLocaleString());
        await fetchDashboardData();
      } else {
        setError(result.message || 'Detection failed');
      }
    } catch (err: any) {
      setError('Failed to run detection: ' + err.message);
    } finally {
      setLoading(false);
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
              <div className="grid grid-cols-1 gap-6">
                <AnomalyPieChart data={distributionData} />
                <PeakHoursClock />
              </div>
            </div>

            {/* Algorithm Performance */}
            <AlgorithmComparison
              data={algorithmPerformanceData}
              selectedAlgorithm={selectedAlgorithm}
              setSelectedAlgorithm={setSelectedAlgorithm}
            />

            {/* Top Anomaly Networks */}
            <TopNetworks
              networks={topAnomalyNetworks}
              selectedNetwork={selectedNetwork}
              setSelectedNetwork={setSelectedNetwork}
            />
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

          <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-6">
            {lastUpdate && <div className="text-sm text-gray-400">Last updated: {lastUpdate}</div>}
          </div>
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

      {/* Styles */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out forwards; }
        .hover\\:scale-105:hover { transform: scale(1.05); }
        .transition-all {
          transition-property: all;
          transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
        }
      `}</style>
    </div>
  );
}

export default App;
