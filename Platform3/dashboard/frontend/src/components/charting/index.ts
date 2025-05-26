/**
 * Customizable Charting Tools Module
 * Professional-grade charting components for forex trading analysis
 * 
 * This module provides comprehensive charting capabilities:
 * - Advanced chart types (Candlestick, Line, Area, OHLC, Heikin-Ashi)
 * - 50+ technical indicators with real-time calculations
 * - Professional drawing tools and annotations
 * - Multiple timeframes and data sources
 * - Chart templates and customization
 * - Real-time data streaming
 */

// Main Chart Components
export { default as AdvancedChart } from './AdvancedChart';
export { default as IndicatorLibrary } from './IndicatorLibrary';
export { default as DrawingTools } from './DrawingTools';

// Type Definitions
export interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Indicator {
  id: string;
  name: string;
  type: 'overlay' | 'oscillator';
  visible: boolean;
  settings: Record<string, any>;
  color: string;
  lineWidth: number;
}

export interface DrawingTool {
  id: string;
  type: 'trendline' | 'horizontal' | 'vertical' | 'rectangle' | 'circle' | 'fibonacci' | 'text' | 'arrow' | 'channel';
  name: string;
  points: Array<{ x: number; y: number; time?: number; price?: number }>;
  style: {
    color: string;
    lineWidth: number;
    lineStyle: 'solid' | 'dashed' | 'dotted';
    fillColor?: string;
    fillOpacity?: number;
  };
  text?: string;
  visible: boolean;
  locked: boolean;
  created: Date;
}

export interface ChartTemplate {
  id: string;
  name: string;
  chartType: string;
  indicators: Indicator[];
  drawings: DrawingTool[];
  settings: ChartSettings;
}

export interface ChartSettings {
  backgroundColor: string;
  gridColor: string;
  textColor: string;
  candleUpColor: string;
  candleDownColor: string;
  volumeUpColor: string;
  volumeDownColor: string;
  crosshairMode: 'normal' | 'magnet';
  timeScale: {
    visible: boolean;
    timeVisible: boolean;
    secondsVisible: boolean;
  };
  priceScale: {
    visible: boolean;
    autoScale: boolean;
    invertScale: boolean;
    alignLabels: boolean;
  };
}

export interface IndicatorConfig {
  id: string;
  name: string;
  category: 'trend' | 'momentum' | 'volatility' | 'volume' | 'support_resistance';
  description: string;
  parameters: IndicatorParameter[];
  defaultSettings: Record<string, any>;
  calculation: (data: number[], params: Record<string, any>) => number[];
}

export interface IndicatorParameter {
  name: string;
  type: 'number' | 'select' | 'boolean' | 'color';
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  default: any;
  description: string;
}

export interface ActiveIndicator {
  id: string;
  configId: string;
  name: string;
  visible: boolean;
  settings: Record<string, any>;
  color: string;
  lineWidth: number;
  style: 'solid' | 'dashed' | 'dotted';
  overlay: boolean;
}

// Configuration Types
export interface ChartingConfig {
  defaultTimeframe: string;
  defaultChartType: 'candlestick' | 'line' | 'area' | 'ohlc' | 'heikin-ashi';
  enableRealTime: boolean;
  updateInterval: number;
  maxDataPoints: number;
  enableDrawingTools: boolean;
  enableIndicators: boolean;
  saveTemplates: boolean;
}

// Utility Functions
export const getDefaultChartSettings = (): ChartSettings => ({
  backgroundColor: '#1e1e1e',
  gridColor: '#2a2a2a',
  textColor: '#ffffff',
  candleUpColor: '#26a69a',
  candleDownColor: '#ef5350',
  volumeUpColor: '#26a69a80',
  volumeDownColor: '#ef535080',
  crosshairMode: 'normal',
  timeScale: {
    visible: true,
    timeVisible: true,
    secondsVisible: false
  },
  priceScale: {
    visible: true,
    autoScale: true,
    invertScale: false,
    alignLabels: true
  }
});

export const getRandomColor = (): string => {
  const colors = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
  ];
  return colors[Math.floor(Math.random() * colors.length)];
};

export const formatPrice = (price: number, decimals: number = 5): string => {
  return price.toFixed(decimals);
};

export const formatVolume = (volume: number): string => {
  if (volume >= 1000000) {
    return `${(volume / 1000000).toFixed(1)}M`;
  } else if (volume >= 1000) {
    return `${(volume / 1000).toFixed(1)}K`;
  }
  return volume.toString();
};

export const formatTime = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleTimeString();
};

export const calculatePriceChange = (current: number, previous: number): { change: number; percentage: number } => {
  const change = current - previous;
  const percentage = (change / previous) * 100;
  return { change, percentage };
};

// Constants
export const DEFAULT_CHARTING_CONFIG: ChartingConfig = {
  defaultTimeframe: 'H1',
  defaultChartType: 'candlestick',
  enableRealTime: true,
  updateInterval: 1000,
  maxDataPoints: 1000,
  enableDrawingTools: true,
  enableIndicators: true,
  saveTemplates: true
};

export const TIMEFRAMES = [
  { value: 'M1', label: '1 Minute', seconds: 60 },
  { value: 'M5', label: '5 Minutes', seconds: 300 },
  { value: 'M15', label: '15 Minutes', seconds: 900 },
  { value: 'M30', label: '30 Minutes', seconds: 1800 },
  { value: 'H1', label: '1 Hour', seconds: 3600 },
  { value: 'H4', label: '4 Hours', seconds: 14400 },
  { value: 'D1', label: '1 Day', seconds: 86400 },
  { value: 'W1', label: '1 Week', seconds: 604800 },
  { value: 'MN1', label: '1 Month', seconds: 2592000 }
];

export const CHART_TYPES = [
  { value: 'candlestick', label: 'Candlestick', description: 'OHLC candlestick chart' },
  { value: 'line', label: 'Line', description: 'Simple line chart' },
  { value: 'area', label: 'Area', description: 'Filled area chart' },
  { value: 'ohlc', label: 'OHLC', description: 'Open-High-Low-Close bars' },
  { value: 'heikin-ashi', label: 'Heikin-Ashi', description: 'Modified candlestick chart' }
];

export const INDICATOR_CATEGORIES = [
  { value: 'trend', label: 'Trend', description: 'Trend following indicators' },
  { value: 'momentum', label: 'Momentum', description: 'Momentum oscillators' },
  { value: 'volatility', label: 'Volatility', description: 'Volatility measures' },
  { value: 'volume', label: 'Volume', description: 'Volume-based indicators' },
  { value: 'support_resistance', label: 'Support/Resistance', description: 'Key levels' }
];

export const DRAWING_TOOL_TYPES = [
  { value: 'trendline', label: 'Trend Line', description: 'Draw trend lines' },
  { value: 'horizontal', label: 'Horizontal Line', description: 'Support/resistance levels' },
  { value: 'vertical', label: 'Vertical Line', description: 'Time-based lines' },
  { value: 'rectangle', label: 'Rectangle', description: 'Rectangular areas' },
  { value: 'circle', label: 'Circle', description: 'Circular areas' },
  { value: 'fibonacci', label: 'Fibonacci', description: 'Fibonacci retracements' },
  { value: 'text', label: 'Text', description: 'Text annotations' },
  { value: 'arrow', label: 'Arrow', description: 'Directional arrows' },
  { value: 'channel', label: 'Channel', description: 'Parallel channels' }
];

// Color Schemes
export const COLOR_SCHEMES = {
  dark: {
    backgroundColor: '#1e1e1e',
    gridColor: '#2a2a2a',
    textColor: '#ffffff',
    candleUpColor: '#26a69a',
    candleDownColor: '#ef5350'
  },
  light: {
    backgroundColor: '#ffffff',
    gridColor: '#e0e0e0',
    textColor: '#000000',
    candleUpColor: '#4caf50',
    candleDownColor: '#f44336'
  },
  blue: {
    backgroundColor: '#0d1421',
    gridColor: '#1e2a3a',
    textColor: '#ffffff',
    candleUpColor: '#00bcd4',
    candleDownColor: '#ff5722'
  }
};

// Performance Optimization
export const CHART_PERFORMANCE_CONFIG = {
  MAX_VISIBLE_CANDLES: 500,
  RENDER_THROTTLE_MS: 16, // 60 FPS
  DATA_COMPRESSION_THRESHOLD: 10000,
  INDICATOR_CALCULATION_BATCH_SIZE: 100
};

// Version Information
export const CHARTING_VERSION = '1.0.0';
export const CHARTING_DESCRIPTION = 'Professional Customizable Charting Tools for Platform3 Forex Trading';

// Export all components as a bundle
export const ChartingComponents = {
  AdvancedChart,
  IndicatorLibrary,
  DrawingTools
};

// Export utility functions as a bundle
export const ChartingUtils = {
  getDefaultChartSettings,
  getRandomColor,
  formatPrice,
  formatVolume,
  formatTime,
  calculatePriceChange
};

// Export constants as a bundle
export const ChartingConstants = {
  DEFAULT_CHARTING_CONFIG,
  TIMEFRAMES,
  CHART_TYPES,
  INDICATOR_CATEGORIES,
  DRAWING_TOOL_TYPES,
  COLOR_SCHEMES,
  CHART_PERFORMANCE_CONFIG,
  CHARTING_VERSION,
  CHARTING_DESCRIPTION
};
