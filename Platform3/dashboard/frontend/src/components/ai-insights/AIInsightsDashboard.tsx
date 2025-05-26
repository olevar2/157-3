/**
 * AI Insights & Predictions Visualization Dashboard
 * Professional-grade AI analytics and prediction visualization for forex trading
 * 
 * Features:
 * - Real-time AI predictions and confidence scores
 * - Pattern recognition visualization
 * - Sentiment analysis displays
 * - Model performance metrics
 * - Interactive prediction charts
 * - Risk assessment visualization
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Psychology,
  Analytics,
  Speed,
  Warning,
  CheckCircle,
  Error,
  Refresh,
  Settings,
  Visibility,
  VisibilityOff
} from '@mui/icons-material';
import { Line, Bar, Radar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

interface AIPrediction {
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

interface PatternRecognition {
  pattern: string;
  confidence: number;
  symbol: string;
  timeframe: string;
  completion: number;
  expectedMove: number;
  historicalAccuracy: number;
}

interface SentimentData {
  symbol: string;
  overall: number; // -1 to 1
  news: number;
  social: number;
  technical: number;
  sources: number;
  lastUpdate: Date;
}

interface ModelPerformance {
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

interface AIInsightsDashboardProps {
  symbols?: string[];
  autoRefresh?: boolean;
  refreshInterval?: number;
  showAdvancedMetrics?: boolean;
}

const AIInsightsDashboard: React.FC<AIInsightsDashboardProps> = ({
  symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
  autoRefresh = true,
  refreshInterval = 30000,
  showAdvancedMetrics = true
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [predictions, setPredictions] = useState<AIPrediction[]>([]);
  const [patterns, setPatterns] = useState<PatternRecognition[]>([]);
  const [sentiment, setSentiment] = useState<SentimentData[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [realTimeMode, setRealTimeMode] = useState(true);

  // Fetch AI insights data
  const fetchAIInsights = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Simulate API calls - replace with actual API endpoints
      const [predictionsRes, patternsRes, sentimentRes, performanceRes] = await Promise.all([
        fetch('/api/v1/ai/predictions').then(res => res.json()),
        fetch('/api/v1/ai/patterns').then(res => res.json()),
        fetch('/api/v1/ai/sentiment').then(res => res.json()),
        fetch('/api/v1/ai/performance').then(res => res.json())
      ]);
      
      setPredictions(predictionsRes.data || generateMockPredictions());
      setPatterns(patternsRes.data || generateMockPatterns());
      setSentiment(sentimentRes.data || generateMockSentiment());
      setModelPerformance(performanceRes.data || generateMockPerformance());
      
    } catch (err) {
      setError('Failed to fetch AI insights data');
      console.error('AI Insights fetch error:', err);
      
      // Use mock data on error
      setPredictions(generateMockPredictions());
      setPatterns(generateMockPatterns());
      setSentiment(generateMockSentiment());
      setModelPerformance(generateMockPerformance());
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh effect
  useEffect(() => {
    fetchAIInsights();
    
    if (autoRefresh && realTimeMode) {
      const interval = setInterval(fetchAIInsights, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchAIInsights, autoRefresh, realTimeMode, refreshInterval]);

  // Mock data generators (replace with actual API calls)
  const generateMockPredictions = (): AIPrediction[] => {
    return symbols.map(symbol => ({
      symbol,
      direction: Math.random() > 0.5 ? 'buy' : 'sell',
      confidence: 0.6 + Math.random() * 0.35,
      targetPrice: 1.1000 + Math.random() * 0.1,
      stopLoss: 1.0900 + Math.random() * 0.05,
      timeframe: ['M15', 'H1', 'H4'][Math.floor(Math.random() * 3)],
      reasoning: [
        'Strong bullish momentum detected',
        'Support level holding',
        'Positive sentiment confluence'
      ],
      modelUsed: ['LSTM', 'Random Forest', 'Ensemble'][Math.floor(Math.random() * 3)],
      timestamp: new Date(),
      accuracy: 0.65 + Math.random() * 0.25
    }));
  };

  const generateMockPatterns = (): PatternRecognition[] => {
    const patternTypes = ['Head & Shoulders', 'Double Top', 'Triangle', 'Flag', 'Wedge'];
    return symbols.map(symbol => ({
      pattern: patternTypes[Math.floor(Math.random() * patternTypes.length)],
      confidence: 0.7 + Math.random() * 0.25,
      symbol,
      timeframe: 'H4',
      completion: 0.6 + Math.random() * 0.35,
      expectedMove: 50 + Math.random() * 100,
      historicalAccuracy: 0.65 + Math.random() * 0.25
    }));
  };

  const generateMockSentiment = (): SentimentData[] => {
    return symbols.map(symbol => ({
      symbol,
      overall: -0.5 + Math.random(),
      news: -0.5 + Math.random(),
      social: -0.5 + Math.random(),
      technical: -0.5 + Math.random(),
      sources: Math.floor(10 + Math.random() * 50),
      lastUpdate: new Date()
    }));
  };

  const generateMockPerformance = (): ModelPerformance[] => {
    const models = ['LSTM Price Predictor', 'Pattern Recognition CNN', 'Sentiment Analyzer', 'Ensemble Model'];
    return models.map(modelName => ({
      modelName,
      accuracy: 0.6 + Math.random() * 0.3,
      precision: 0.65 + Math.random() * 0.25,
      recall: 0.6 + Math.random() * 0.3,
      f1Score: 0.62 + Math.random() * 0.28,
      sharpeRatio: 1.2 + Math.random() * 1.5,
      maxDrawdown: 0.05 + Math.random() * 0.15,
      totalTrades: Math.floor(100 + Math.random() * 500),
      winRate: 0.55 + Math.random() * 0.35,
      lastUpdate: new Date()
    }));
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.6) return '#ff9800';
    return '#f44336';
  };

  const getSentimentColor = (sentiment: number): string => {
    if (sentiment > 0.3) return '#4caf50';
    if (sentiment > -0.3) return '#ff9800';
    return '#f44336';
  };

  const renderPredictionsTab = () => (
    <Grid container spacing={3}>
      {predictions.map((prediction, index) => (
        <Grid item xs={12} md={6} lg={4} key={index}>
          <Card elevation={3}>
            <CardHeader
              title={prediction.symbol}
              subheader={`${prediction.timeframe} • ${prediction.modelUsed}`}
              action={
                <Chip
                  icon={prediction.direction === 'buy' ? <TrendingUp /> : <TrendingDown />}
                  label={prediction.direction.toUpperCase()}
                  color={prediction.direction === 'buy' ? 'success' : 'error'}
                  size="small"
                />
              }
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Confidence
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={prediction.confidence * 100}
                    sx={{
                      flexGrow: 1,
                      mr: 1,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: getConfidenceColor(prediction.confidence)
                      }
                    }}
                  />
                  <Typography variant="body2" sx={{ minWidth: 35 }}>
                    {(prediction.confidence * 100).toFixed(0)}%
                  </Typography>
                </Box>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Target: {prediction.targetPrice.toFixed(5)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Stop Loss: {prediction.stopLoss.toFixed(5)}
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  AI Reasoning:
                </Typography>
                {prediction.reasoning.map((reason, idx) => (
                  <Typography key={idx} variant="caption" display="block">
                    • {reason}
                  </Typography>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderPatternsTab = () => (
    <Grid container spacing={3}>
      {patterns.map((pattern, index) => (
        <Grid item xs={12} md={6} lg={4} key={index}>
          <Card elevation={3}>
            <CardHeader
              title={pattern.pattern}
              subheader={`${pattern.symbol} • ${pattern.timeframe}`}
              action={
                <Chip
                  label={`${(pattern.confidence * 100).toFixed(0)}%`}
                  color={pattern.confidence > 0.8 ? 'success' : pattern.confidence > 0.6 ? 'warning' : 'error'}
                  size="small"
                />
              }
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Pattern Completion
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={pattern.completion * 100}
                  sx={{ mt: 1 }}
                />
                <Typography variant="caption">
                  {(pattern.completion * 100).toFixed(0)}% Complete
                </Typography>
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Expected Move: {pattern.expectedMove.toFixed(0)} pips
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Historical Accuracy: {(pattern.historicalAccuracy * 100).toFixed(0)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderSentimentTab = () => (
    <Grid container spacing={3}>
      {sentiment.map((data, index) => (
        <Grid item xs={12} md={6} lg={4} key={index}>
          <Card elevation={3}>
            <CardHeader
              title={data.symbol}
              subheader={`${data.sources} sources analyzed`}
              action={
                <Chip
                  icon={<Psychology />}
                  label={data.overall > 0 ? 'Bullish' : data.overall < 0 ? 'Bearish' : 'Neutral'}
                  sx={{ backgroundColor: getSentimentColor(data.overall), color: 'white' }}
                  size="small"
                />
              }
            />
            <CardContent>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Overall Sentiment
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(data.overall + 1) * 50}
                  sx={{
                    mt: 1,
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getSentimentColor(data.overall)
                    }
                  }}
                />
                <Typography variant="caption">
                  {data.overall > 0 ? '+' : ''}{(data.overall * 100).toFixed(0)}%
                </Typography>
              </Box>
              
              <Grid container spacing={1}>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">News</Typography>
                  <Typography variant="body2">{(data.news * 100).toFixed(0)}%</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Social</Typography>
                  <Typography variant="body2">{(data.social * 100).toFixed(0)}%</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="caption" color="text.secondary">Technical</Typography>
                  <Typography variant="body2">{(data.technical * 100).toFixed(0)}%</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  const renderPerformanceTab = () => (
    <Grid container spacing={3}>
      {modelPerformance.map((model, index) => (
        <Grid item xs={12} md={6} key={index}>
          <Card elevation={3}>
            <CardHeader
              title={model.modelName}
              subheader={`${model.totalTrades} trades • Updated ${model.lastUpdate.toLocaleTimeString()}`}
              action={
                <Chip
                  icon={model.accuracy > 0.7 ? <CheckCircle /> : model.accuracy > 0.6 ? <Warning /> : <Error />}
                  label={`${(model.accuracy * 100).toFixed(1)}%`}
                  color={model.accuracy > 0.7 ? 'success' : model.accuracy > 0.6 ? 'warning' : 'error'}
                  size="small"
                />
              }
            />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Precision</Typography>
                  <Typography variant="h6">{(model.precision * 100).toFixed(1)}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Recall</Typography>
                  <Typography variant="h6">{(model.recall * 100).toFixed(1)}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Win Rate</Typography>
                  <Typography variant="h6">{(model.winRate * 100).toFixed(1)}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Sharpe Ratio</Typography>
                  <Typography variant="h6">{model.sharpeRatio.toFixed(2)}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">Max Drawdown</Typography>
                  <LinearProgress
                    variant="determinate"
                    value={model.maxDrawdown * 100}
                    color="error"
                    sx={{ mt: 1 }}
                  />
                  <Typography variant="caption">
                    {(model.maxDrawdown * 100).toFixed(1)}%
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          AI Insights & Predictions
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={realTimeMode}
                onChange={(e) => setRealTimeMode(e.target.checked)}
                color="primary"
              />
            }
            label="Real-time"
          />
          <Tooltip title="Refresh Data">
            <IconButton onClick={fetchAIInsights} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading Indicator */}
      {loading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Tabs */}
      <Paper elevation={2} sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="fullWidth"
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab icon={<TrendingUp />} label="Predictions" />
          <Tab icon={<Analytics />} label="Patterns" />
          <Tab icon={<Psychology />} label="Sentiment" />
          <Tab icon={<Speed />} label="Performance" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Box sx={{ mt: 3 }}>
        {activeTab === 0 && renderPredictionsTab()}
        {activeTab === 1 && renderPatternsTab()}
        {activeTab === 2 && renderSentimentTab()}
        {activeTab === 3 && renderPerformanceTab()}
      </Box>
    </Box>
  );
};

export default AIInsightsDashboard;
