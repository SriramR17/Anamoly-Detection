// src/react-gauge-chart.d.ts
declare module 'react-gauge-chart' {
    import * as React from 'react';
  
    interface GaugeChartProps {
      id?: string;
      nrOfLevels?: number;
      colors?: string[];
      arcWidth?: number;
      percent?: number;
      textColor?: string;
      needleColor?: string;
      needleBaseColor?: string;
      hideText?: boolean;
      formatTextValue?: (value: string) => string;
      animDelay?: number;
      animateDuration?: number;
      [key: string]: any;
    }
  
    export default class GaugeChart extends React.Component<GaugeChartProps> {}
  }
  