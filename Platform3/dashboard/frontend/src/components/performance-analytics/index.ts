/**
 * Performance Analytics & Reporting Module
 * Comprehensive trading performance analysis and reporting system
 * 
 * This module provides detailed performance analytics:
 * - Real-time performance metrics and KPIs
 * - Risk-adjusted performance measures
 * - Detailed trade analysis and statistics
 * - Risk analytics and monitoring
 * - Custom report generation
 * - Export capabilities
 */

// Main Components
export { default as PerformanceAnalyticsDashboard } from './PerformanceAnalyticsDashboard';
export { default as RiskAnalytics } from './RiskAnalytics';

// Type Definitions
export interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPercentage: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  maxDrawdownPercentage: number;
  winRate: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  largestWin: number;
  largestLoss: number;
  averageTradeReturn: number;
  volatility: number;
  calmarRatio: number;
  recoveryFactor: number;
  payoffRatio: number;
}

export interface TradeData {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  entryTime: Date;
  exitTime: Date;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercentage: number;
  commission: number;
  duration: number; // in minutes
  strategy: string;
  tags: string[];
}

export interface RiskMetrics {
  valueAtRisk95: number;
  valueAtRisk99: number;
  conditionalVaR: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  volatility: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  trackingError: number;
  downside_deviation: number;
  upside_capture: number;
  downside_capture: number;
  calmarRatio: number;
  sterlingRatio: number;
  burkeRatio: number;
}

export interface DrawdownPeriod {
  start: Date;
  end: Date;
  duration: number; // days
  peak: number;
  trough: number;
  drawdown: number;
  recovery: Date | null;
  recoveryDuration: number | null;
}

export interface RiskExposure {
  symbol: string;
  exposure: number;
  exposurePercentage: number;
  var95: number;
  correlation: number;
  beta: number;
  riskContribution: number;
}

export interface PerformanceReport {
  id: string;
  name: string;
  period: {
    start: Date;
    end: Date;
  };
  metrics: PerformanceMetrics;
  riskMetrics: RiskMetrics;
  trades: TradeData[];
  charts: {
    equityCurve: any;
    drawdownChart: any;
    monthlyReturns: any;
    riskMetrics: any;
  };
  generated: Date;
  format: 'pdf' | 'excel' | 'html';
}

// Configuration Types
export interface PerformanceAnalyticsConfig {
  defaultPeriod: string;
  autoRefresh: boolean;
  refreshInterval: number;
  riskFreeRate: number;
  benchmarkSymbol: string;
  confidenceLevel: number;
  enableRiskAnalytics: boolean;
  enableReporting: boolean;
}

// Utility Functions
export const calculateSharpeRatio = (returns: number[], riskFreeRate: number = 0): number => {
  const excessReturns = returns.map(r => r - riskFreeRate / 252);
  const meanExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
  const stdDev = Math.sqrt(excessReturns.reduce((sum, r) => sum + Math.pow(r - meanExcessReturn, 2), 0) / excessReturns.length);
  return stdDev > 0 ? (meanExcessReturn * Math.sqrt(252)) / (stdDev * Math.sqrt(252)) : 0;
};

export const calculateSortinoRatio = (returns: number[], riskFreeRate: number = 0): number => {
  const excessReturns = returns.map(r => r - riskFreeRate / 252);
  const meanExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
  const downsideReturns = excessReturns.filter(r => r < 0);
  const downsideDeviation = downsideReturns.length > 0 
    ? Math.sqrt(downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length)
    : 0;
  return downsideDeviation > 0 ? (meanExcessReturn * Math.sqrt(252)) / (downsideDeviation * Math.sqrt(252)) : 0;
};

export const calculateMaxDrawdown = (returns: number[]): { maxDrawdown: number; duration: number } => {
  const cumulativeReturns = returns.reduce((acc, r, i) => {
    acc.push((acc[i - 1] || 1) * (1 + r));
    return acc;
  }, [] as number[]);
  
  let maxDrawdown = 0;
  let maxDuration = 0;
  let peak = cumulativeReturns[0];
  let peakIndex = 0;
  
  for (let i = 1; i < cumulativeReturns.length; i++) {
    if (cumulativeReturns[i] > peak) {
      peak = cumulativeReturns[i];
      peakIndex = i;
    } else {
      const drawdown = (peak - cumulativeReturns[i]) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        maxDuration = i - peakIndex;
      }
    }
  }
  
  return { maxDrawdown, duration: maxDuration };
};

export const calculateVaR = (returns: number[], confidenceLevel: number = 0.95): number => {
  const sortedReturns = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
  return -sortedReturns[index];
};

export const calculateBeta = (returns: number[], benchmarkReturns: number[]): number => {
  if (returns.length !== benchmarkReturns.length) return 1;
  
  const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const meanBenchmark = benchmarkReturns.reduce((sum, r) => sum + r, 0) / benchmarkReturns.length;
  
  const covariance = returns.reduce((sum, r, i) => 
    sum + (r - meanReturn) * (benchmarkReturns[i] - meanBenchmark), 0) / returns.length;
  const benchmarkVariance = benchmarkReturns.reduce((sum, r) => 
    sum + Math.pow(r - meanBenchmark, 2), 0) / benchmarkReturns.length;
  
  return benchmarkVariance > 0 ? covariance / benchmarkVariance : 1;
};

export const calculateAlpha = (returns: number[], benchmarkReturns: number[], riskFreeRate: number = 0): number => {
  const beta = calculateBeta(returns, benchmarkReturns);
  const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const meanBenchmark = benchmarkReturns.reduce((sum, r) => sum + r, 0) / benchmarkReturns.length;
  
  return (meanReturn - riskFreeRate / 252) - beta * (meanBenchmark - riskFreeRate / 252);
};

export const calculateWinRate = (trades: TradeData[]): number => {
  const winningTrades = trades.filter(t => t.pnl > 0).length;
  return trades.length > 0 ? winningTrades / trades.length : 0;
};

export const calculateProfitFactor = (trades: TradeData[]): number => {
  const grossProfit = trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
  const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
  return grossLoss > 0 ? grossProfit / grossLoss : 0;
};

export const formatPercentage = (value: number, decimals: number = 2): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

export const formatCurrency = (value: number, currency: string = 'USD'): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency
  }).format(value);
};

export const formatNumber = (value: number, decimals: number = 2): string => {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
};

export const getRiskLevel = (value: number, thresholds: number[]): 'low' | 'medium' | 'high' => {
  if (value <= thresholds[0]) return 'low';
  if (value <= thresholds[1]) return 'medium';
  return 'high';
};

export const getPerformanceGrade = (sharpeRatio: number): 'A' | 'B' | 'C' | 'D' | 'F' => {
  if (sharpeRatio >= 2.0) return 'A';
  if (sharpeRatio >= 1.5) return 'B';
  if (sharpeRatio >= 1.0) return 'C';
  if (sharpeRatio >= 0.5) return 'D';
  return 'F';
};

// Constants
export const DEFAULT_PERFORMANCE_CONFIG: PerformanceAnalyticsConfig = {
  defaultPeriod: '1M',
  autoRefresh: true,
  refreshInterval: 60000,
  riskFreeRate: 0.02, // 2% annual
  benchmarkSymbol: 'SPY',
  confidenceLevel: 0.95,
  enableRiskAnalytics: true,
  enableReporting: true
};

export const PERFORMANCE_PERIODS = [
  { value: '1D', label: '1 Day' },
  { value: '1W', label: '1 Week' },
  { value: '1M', label: '1 Month' },
  { value: '3M', label: '3 Months' },
  { value: '6M', label: '6 Months' },
  { value: '1Y', label: '1 Year' },
  { value: 'YTD', label: 'Year to Date' },
  { value: 'ALL', label: 'All Time' }
];

export const RISK_THRESHOLDS = {
  VAR_95: [1000, 5000], // Low, Medium thresholds
  MAX_DRAWDOWN: [0.05, 0.15], // 5%, 15%
  VOLATILITY: [0.10, 0.25], // 10%, 25%
  SHARPE_RATIO: [1.0, 2.0], // Good, Excellent
  CORRELATION: [0.3, 0.7] // Low, High correlation
};

export const PERFORMANCE_BENCHMARKS = {
  SHARPE_RATIO: {
    EXCELLENT: 2.0,
    GOOD: 1.5,
    ACCEPTABLE: 1.0,
    POOR: 0.5
  },
  WIN_RATE: {
    EXCELLENT: 0.70,
    GOOD: 0.60,
    ACCEPTABLE: 0.50,
    POOR: 0.40
  },
  PROFIT_FACTOR: {
    EXCELLENT: 2.0,
    GOOD: 1.5,
    ACCEPTABLE: 1.2,
    POOR: 1.0
  },
  MAX_DRAWDOWN: {
    EXCELLENT: 0.05,
    GOOD: 0.10,
    ACCEPTABLE: 0.15,
    POOR: 0.25
  }
};

// Chart Color Schemes
export const CHART_COLORS = {
  profit: '#4caf50',
  loss: '#f44336',
  neutral: '#ff9800',
  primary: '#2196f3',
  secondary: '#9c27b0',
  background: '#f5f5f5',
  grid: '#e0e0e0'
};

// Export Formats
export const EXPORT_FORMATS = [
  { value: 'pdf', label: 'PDF Report', icon: 'picture_as_pdf' },
  { value: 'excel', label: 'Excel Spreadsheet', icon: 'table_chart' },
  { value: 'csv', label: 'CSV Data', icon: 'description' },
  { value: 'json', label: 'JSON Data', icon: 'code' }
];

// Version Information
export const PERFORMANCE_ANALYTICS_VERSION = '1.0.0';
export const PERFORMANCE_ANALYTICS_DESCRIPTION = 'Comprehensive Performance Analytics & Reporting for Platform3 Forex Trading';

// Export all components as a bundle
export const PerformanceAnalyticsComponents = {
  PerformanceAnalyticsDashboard,
  RiskAnalytics
};

// Export utility functions as a bundle
export const PerformanceAnalyticsUtils = {
  calculateSharpeRatio,
  calculateSortinoRatio,
  calculateMaxDrawdown,
  calculateVaR,
  calculateBeta,
  calculateAlpha,
  calculateWinRate,
  calculateProfitFactor,
  formatPercentage,
  formatCurrency,
  formatNumber,
  getRiskLevel,
  getPerformanceGrade
};

// Export constants as a bundle
export const PerformanceAnalyticsConstants = {
  DEFAULT_PERFORMANCE_CONFIG,
  PERFORMANCE_PERIODS,
  RISK_THRESHOLDS,
  PERFORMANCE_BENCHMARKS,
  CHART_COLORS,
  EXPORT_FORMATS,
  PERFORMANCE_ANALYTICS_VERSION,
  PERFORMANCE_ANALYTICS_DESCRIPTION
};
