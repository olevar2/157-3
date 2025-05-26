/**
 * AI Insights & Predictions Visualization Module
 * Centralized exports for all AI insights components
 * 
 * This module provides comprehensive AI analytics and prediction visualization:
 * - Real-time AI predictions and confidence scores
 * - Pattern recognition visualization
 * - Sentiment analysis displays
 * - Model performance monitoring
 * - Interactive prediction charts
 * - Risk assessment visualization
 */

// Main Dashboard Component
export { default as AIInsightsDashboard } from './AIInsightsDashboard';

// Prediction Components
export { default as PredictionChart } from './PredictionChart';

// Performance Monitoring
export { default as ModelPerformanceMonitor } from './ModelPerformanceMonitor';

// Type Definitions
export interface AIPrediction {
  symbol: string;
  direction: 'buy' | 'sell' | 'hold';
  confidence: number;
  targetPrice: number;
  stopLoss: number;
  timeframe: string;
  reasoning: string[];
  modelUsed: string;
  timestamp: Date;
  accuracy: number;
}

export interface PatternRecognition {
  pattern: string;
  confidence: number;
  symbol: string;
  timeframe: string;
  completion: number;
  expectedMove: number;
  historicalAccuracy: number;
}

export interface SentimentData {
  symbol: string;
  overall: number; // -1 to 1
  news: number;
  social: number;
  technical: number;
  sources: number;
  lastUpdate: Date;
}

export interface ModelPerformance {
  modelName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  winRate: number;
  lastUpdate: Date;
}

export interface ModelMetrics {
  modelId: string;
  modelName: string;
  modelType: 'LSTM' | 'CNN' | 'Random Forest' | 'Ensemble' | 'Transformer';
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalPredictions: number;
  correctPredictions: number;
  avgConfidence: number;
  latency: number; // ms
  lastUpdate: Date;
  status: 'healthy' | 'degraded' | 'critical' | 'offline';
  trend: 'improving' | 'stable' | 'declining';
}

export interface PredictionPoint {
  timestamp: Date;
  actualPrice: number;
  predictedPrice: number;
  confidence: number;
  direction: 'buy' | 'sell' | 'hold';
  accuracy?: number; // For historical predictions
}

export interface PerformanceHistory {
  timestamp: Date;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  latency: number;
}

// Configuration Types
export interface AIInsightsConfig {
  symbols: string[];
  autoRefresh: boolean;
  refreshInterval: number;
  showAdvancedMetrics: boolean;
  predictionTimeframes: string[];
  confidenceThreshold: number;
  accuracyThreshold: number;
}

// API Response Types
export interface AIInsightsResponse {
  predictions: AIPrediction[];
  patterns: PatternRecognition[];
  sentiment: SentimentData[];
  performance: ModelPerformance[];
  timestamp: Date;
  status: 'success' | 'error' | 'partial';
}

// Utility Functions
export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return '#4caf50';
  if (confidence >= 0.6) return '#ff9800';
  return '#f44336';
};

export const getSentimentColor = (sentiment: number): string => {
  if (sentiment > 0.3) return '#4caf50';
  if (sentiment > -0.3) return '#ff9800';
  return '#f44336';
};

export const getModelStatusColor = (status: string): 'success' | 'warning' | 'error' | 'default' => {
  switch (status) {
    case 'healthy': return 'success';
    case 'degraded': return 'warning';
    case 'critical': return 'error';
    case 'offline': return 'default';
    default: return 'default';
  }
};

export const calculatePredictionAccuracy = (predictions: PredictionPoint[]): number => {
  const accuratePredictions = predictions.filter(p => p.accuracy && p.accuracy > 0.7).length;
  return predictions.length > 0 ? accuratePredictions / predictions.length : 0;
};

export const calculateAverageConfidence = (predictions: AIPrediction[]): number => {
  return predictions.length > 0 
    ? predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length 
    : 0;
};

export const formatPredictionDirection = (direction: string): string => {
  return direction.toUpperCase();
};

export const formatConfidencePercentage = (confidence: number): string => {
  return `${(confidence * 100).toFixed(1)}%`;
};

export const formatSentimentPercentage = (sentiment: number): string => {
  const sign = sentiment > 0 ? '+' : '';
  return `${sign}${(sentiment * 100).toFixed(0)}%`;
};

// Constants
export const DEFAULT_AI_INSIGHTS_CONFIG: AIInsightsConfig = {
  symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'],
  autoRefresh: true,
  refreshInterval: 30000,
  showAdvancedMetrics: true,
  predictionTimeframes: ['M1', 'M5', 'M15', 'H1', 'H4'],
  confidenceThreshold: 0.7,
  accuracyThreshold: 0.65
};

export const PREDICTION_TIMEFRAMES = [
  { value: 'M1', label: '1 Minute' },
  { value: 'M5', label: '5 Minutes' },
  { value: 'M15', label: '15 Minutes' },
  { value: 'H1', label: '1 Hour' },
  { value: 'H4', label: '4 Hours' },
  { value: 'D1', label: '1 Day' }
];

export const MODEL_TYPES = [
  'LSTM',
  'CNN', 
  'Random Forest',
  'Ensemble',
  'Transformer'
] as const;

export const PATTERN_TYPES = [
  'Head & Shoulders',
  'Double Top',
  'Double Bottom',
  'Triangle',
  'Flag',
  'Wedge',
  'Channel',
  'Support/Resistance'
] as const;

export const SENTIMENT_SOURCES = [
  'News Articles',
  'Social Media',
  'Economic Calendar',
  'Central Bank Communications',
  'Market Analysis Reports'
] as const;

// Performance Thresholds
export const PERFORMANCE_THRESHOLDS = {
  ACCURACY: {
    EXCELLENT: 0.8,
    GOOD: 0.7,
    ACCEPTABLE: 0.6,
    POOR: 0.5
  },
  CONFIDENCE: {
    HIGH: 0.8,
    MEDIUM: 0.6,
    LOW: 0.4
  },
  LATENCY: {
    EXCELLENT: 50,  // ms
    GOOD: 100,
    ACCEPTABLE: 200,
    POOR: 500
  }
} as const;

// Version Information
export const AI_INSIGHTS_VERSION = '1.0.0';
export const AI_INSIGHTS_DESCRIPTION = 'Professional AI Insights & Predictions Visualization for Platform3 Forex Trading';

// Export all components as a bundle
export const AIInsightsComponents = {
  AIInsightsDashboard,
  PredictionChart,
  ModelPerformanceMonitor
};

// Export utility functions as a bundle
export const AIInsightsUtils = {
  getConfidenceColor,
  getSentimentColor,
  getModelStatusColor,
  calculatePredictionAccuracy,
  calculateAverageConfidence,
  formatPredictionDirection,
  formatConfidencePercentage,
  formatSentimentPercentage
};

// Export constants as a bundle
export const AIInsightsConstants = {
  DEFAULT_AI_INSIGHTS_CONFIG,
  PREDICTION_TIMEFRAMES,
  MODEL_TYPES,
  PATTERN_TYPES,
  SENTIMENT_SOURCES,
  PERFORMANCE_THRESHOLDS,
  AI_INSIGHTS_VERSION,
  AI_INSIGHTS_DESCRIPTION
};
