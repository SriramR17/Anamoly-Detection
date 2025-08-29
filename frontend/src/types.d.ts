declare module 'react-gauge-chart';

export interface Network {
  name: string;
  totalSamples: number;
  anomaliesDetected: number;
  anomalyRate: number;
  lastUpdate: string;
  recentAnomaly: string;
  severity: 'high' | 'medium' | 'low';
}

export interface Metric {
  name: string;
  value: number;
  deviation: number;
  color: string;
  icon: React.ComponentType<any>;
}

export interface AlgorithmPerformance {
  name: string;
  accuracy: number;
  f1: number;
  precision: number;
  recall: number;
}
