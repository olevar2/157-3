/**
 * AI Prediction Chart Component
 * Interactive chart displaying AI predictions with confidence intervals
 * 
 * Features:
 * - Real-time price data with AI predictions overlay
 * - Confidence intervals visualization
 * - Multiple timeframe support
 * - Interactive prediction markers
 * - Historical accuracy tracking
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Grid,
  Card,
  CardContent,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { TrendingUp, TrendingDown, Timeline } from '@mui/icons-material';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface PredictionPoint {
  timestamp: Date;
  actualPrice: number;
  predictedPrice: number;
  confidence: number;
  direction: 'buy' | 'sell' | 'hold';
  accuracy?: number; // For historical predictions
}

interface PredictionChartProps {
  symbol: string;
  timeframe: string;
  predictions: PredictionPoint[];
  height?: number;
  showConfidenceInterval?: boolean;
  showAccuracy?: boolean;
}

const PredictionChart: React.FC<PredictionChartProps> = ({
  symbol,
  timeframe,
  predictions,
  height = 400,
  showConfidenceInterval = true,
  showAccuracy = true
}) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [showPredictions, setShowPredictions] = useState(true);
  const [showActual, setShowActual] = useState(true);

  // Generate chart data
  const chartData = useMemo(() => {
    const labels = predictions.map(p => p.timestamp.toLocaleTimeString());
    
    const datasets = [];
    
    // Actual price line
    if (showActual) {
      datasets.push({
        label: 'Actual Price',
        data: predictions.map(p => p.actualPrice),
        borderColor: '#2196f3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1
      });
    }
    
    // Predicted price line
    if (showPredictions) {
      datasets.push({
        label: 'AI Prediction',
        data: predictions.map(p => p.predictedPrice),
        borderColor: '#ff9800',
        backgroundColor: 'rgba(255, 152, 0, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        fill: false,
        tension: 0.1
      });
    }
    
    // Confidence interval
    if (showConfidenceInterval && showPredictions) {
      const upperBound = predictions.map(p => 
        p.predictedPrice + (p.predictedPrice * 0.001 * (1 - p.confidence))
      );
      const lowerBound = predictions.map(p => 
        p.predictedPrice - (p.predictedPrice * 0.001 * (1 - p.confidence))
      );
      
      datasets.push({
        label: 'Confidence Upper',
        data: upperBound,
        borderColor: 'rgba(255, 152, 0, 0.3)',
        backgroundColor: 'rgba(255, 152, 0, 0.1)',
        borderWidth: 1,
        fill: '+1',
        tension: 0.1,
        pointRadius: 0
      });
      
      datasets.push({
        label: 'Confidence Lower',
        data: lowerBound,
        borderColor: 'rgba(255, 152, 0, 0.3)',
        backgroundColor: 'rgba(255, 152, 0, 0.1)',
        borderWidth: 1,
        fill: false,
        tension: 0.1,
        pointRadius: 0
      });
    }
    
    return {
      labels,
      datasets
    };
  }, [predictions, showActual, showPredictions, showConfidenceInterval]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          filter: (legendItem: any) => {
            return !legendItem.text.includes('Confidence');
          }
        }
      },
      title: {
        display: true,
        text: `${symbol} - AI Predictions vs Actual (${selectedTimeframe})`
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          afterBody: (context: any) => {
            const index = context[0].dataIndex;
            const prediction = predictions[index];
            if (prediction) {
              return [
                `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`,
                `Direction: ${prediction.direction.toUpperCase()}`,
                prediction.accuracy ? `Accuracy: ${(prediction.accuracy * 100).toFixed(1)}%` : ''
              ].filter(Boolean);
            }
            return [];
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Price'
        },
        beginAtZero: false
      }
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false
    }
  };

  // Calculate prediction statistics
  const stats = useMemo(() => {
    const accuratePredictions = predictions.filter(p => p.accuracy && p.accuracy > 0.7).length;
    const totalPredictions = predictions.filter(p => p.accuracy !== undefined).length;
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
    const avgAccuracy = predictions
      .filter(p => p.accuracy !== undefined)
      .reduce((sum, p) => sum + (p.accuracy || 0), 0) / totalPredictions;
    
    return {
      accuracy: totalPredictions > 0 ? avgAccuracy : 0,
      confidence: avgConfidence,
      totalPredictions,
      accuratePredictions
    };
  }, [predictions]);

  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6" component="h2">
          AI Prediction Analysis
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={selectedTimeframe}
              label="Timeframe"
              onChange={(e) => setSelectedTimeframe(e.target.value)}
            >
              <MenuItem value="M1">M1</MenuItem>
              <MenuItem value="M5">M5</MenuItem>
              <MenuItem value="M15">M15</MenuItem>
              <MenuItem value="H1">H1</MenuItem>
              <MenuItem value="H4">H4</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="primary">
                {(stats.accuracy * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Accuracy
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="secondary">
                {(stats.confidence * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Confidence
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="success.main">
                {stats.accuratePredictions}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Accurate Predictions
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4">
                {stats.totalPredictions}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Predictions
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Chart Controls */}
      <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <FormControlLabel
          control={
            <Switch
              checked={showActual}
              onChange={(e) => setShowActual(e.target.checked)}
              color="primary"
            />
          }
          label="Show Actual Price"
        />
        <FormControlLabel
          control={
            <Switch
              checked={showPredictions}
              onChange={(e) => setShowPredictions(e.target.checked)}
              color="secondary"
            />
          }
          label="Show Predictions"
        />
        <FormControlLabel
          control={
            <Switch
              checked={showConfidenceInterval}
              onChange={(e) => setShowConfidenceInterval(e.target.checked)}
              color="warning"
            />
          }
          label="Show Confidence Interval"
        />
      </Box>

      {/* Chart */}
      <Box sx={{ height: height }}>
        <Line data={chartData} options={chartOptions} />
      </Box>

      {/* Recent Predictions Summary */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Recent Predictions
        </Typography>
        <Grid container spacing={1}>
          {predictions.slice(-5).map((prediction, index) => (
            <Grid item xs={12} sm={6} md={4} lg={2.4} key={index}>
              <Card variant="outlined" sx={{ p: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      {prediction.timestamp.toLocaleTimeString()}
                    </Typography>
                    <Typography variant="body2">
                      {prediction.predictedPrice.toFixed(5)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <Chip
                      icon={prediction.direction === 'buy' ? <TrendingUp /> : <TrendingDown />}
                      label={prediction.direction.toUpperCase()}
                      size="small"
                      color={prediction.direction === 'buy' ? 'success' : 'error'}
                      sx={{ mb: 0.5 }}
                    />
                    <Typography variant="caption">
                      {(prediction.confidence * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Paper>
  );
};

export default PredictionChart;
