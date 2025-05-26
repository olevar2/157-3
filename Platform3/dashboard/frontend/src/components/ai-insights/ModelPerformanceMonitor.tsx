/**
 * Model Performance Monitor Component
 * Real-time monitoring and visualization of AI model performance metrics
 * 
 * Features:
 * - Real-time performance metrics tracking
 * - Model comparison and ranking
 * - Performance trend analysis
 * - Alert system for performance degradation
 * - Model health status indicators
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Speed,
  Analytics,
  Timeline,
  Refresh,
  CompareArrows
} from '@mui/icons-material';
import { Line, Radar, Bar } from 'react-chartjs-2';
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

interface ModelMetrics {
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

interface PerformanceHistory {
  timestamp: Date;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  latency: number;
}

interface ModelPerformanceMonitorProps {
  models: ModelMetrics[];
  performanceHistory: Record<string, PerformanceHistory[]>;
  autoRefresh?: boolean;
  refreshInterval?: number;
  showComparison?: boolean;
}

const ModelPerformanceMonitor: React.FC<ModelPerformanceMonitorProps> = ({
  models,
  performanceHistory,
  autoRefresh = true,
  refreshInterval = 30000,
  showComparison = true
}) => {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<keyof ModelMetrics>('accuracy');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showTrends, setShowTrends] = useState(true);

  // Sort models
  const sortedModels = useMemo(() => {
    return [...models].sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'desc' ? bValue - aValue : aValue - bValue;
      }
      
      return sortOrder === 'desc' 
        ? String(bValue).localeCompare(String(aValue))
        : String(aValue).localeCompare(String(bValue));
    });
  }, [models, sortBy, sortOrder]);

  // Get status color
  const getStatusColor = (status: string): 'success' | 'warning' | 'error' | 'default' => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'critical': return 'error';
      case 'offline': return 'default';
      default: return 'default';
    }
  };

  // Get trend icon
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp color="success" />;
      case 'declining': return <TrendingDown color="error" />;
      case 'stable': return <Timeline color="action" />;
      default: return <Timeline color="action" />;
    }
  };

  // Generate performance comparison chart
  const generateComparisonChart = () => {
    if (selectedModels.length === 0) return null;

    const selectedModelData = models.filter(m => selectedModels.includes(m.modelId));
    
    const data = {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Win Rate'],
      datasets: selectedModelData.map((model, index) => ({
        label: model.modelName,
        data: [
          model.accuracy * 100,
          model.precision * 100,
          model.recall * 100,
          model.f1Score * 100,
          model.winRate * 100
        ],
        borderColor: `hsl(${index * 60}, 70%, 50%)`,
        backgroundColor: `hsla(${index * 60}, 70%, 50%, 0.2)`,
        borderWidth: 2,
        pointBackgroundColor: `hsl(${index * 60}, 70%, 50%)`,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: `hsl(${index * 60}, 70%, 50%)`
      }))
    };

    const options = {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Model Performance Comparison'
        },
        legend: {
          position: 'top' as const
        }
      },
      scales: {
        r: {
          angleLines: {
            display: false
          },
          suggestedMin: 0,
          suggestedMax: 100
        }
      }
    };

    return <Radar data={data} options={options} />;
  };

  // Generate performance trend chart
  const generateTrendChart = (modelId: string) => {
    const history = performanceHistory[modelId] || [];
    
    if (history.length === 0) return null;

    const data = {
      labels: history.map(h => h.timestamp.toLocaleTimeString()),
      datasets: [
        {
          label: 'Accuracy',
          data: history.map(h => h.accuracy * 100),
          borderColor: '#2196f3',
          backgroundColor: 'rgba(33, 150, 243, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1
        },
        {
          label: 'Precision',
          data: history.map(h => h.precision * 100),
          borderColor: '#4caf50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1
        },
        {
          label: 'Recall',
          data: history.map(h => h.recall * 100),
          borderColor: '#ff9800',
          backgroundColor: 'rgba(255, 152, 0, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.1
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const
        },
        title: {
          display: true,
          text: 'Performance Trends'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: 'Percentage (%)'
          }
        }
      }
    };

    return (
      <Box sx={{ height: 300 }}>
        <Line data={data} options={options} />
      </Box>
    );
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2" sx={{ fontWeight: 600 }}>
          Model Performance Monitor
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={showTrends}
                onChange={(e) => setShowTrends(e.target.checked)}
                color="primary"
              />
            }
            label="Show Trends"
          />
          <FormControlLabel
            control={
              <Switch
                checked={showComparison}
                onChange={(e) => setSelectedModels([])}
                color="secondary"
              />
            }
            label="Comparison Mode"
          />
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {models.filter(m => m.status === 'healthy').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Healthy Models
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {models.filter(m => m.status === 'degraded').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Degraded Models
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {(models.reduce((sum, m) => sum + m.accuracy, 0) / models.length * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Accuracy
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4">
                {models.reduce((sum, m) => sum + m.latency, 0) / models.length}ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg Latency
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Model Comparison Chart */}
      {showComparison && selectedModels.length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Model Comparison
          </Typography>
          <Box sx={{ height: 400 }}>
            {generateComparisonChart()}
          </Box>
        </Paper>
      )}

      {/* Models Table */}
      <Paper elevation={3}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                {showComparison && (
                  <TableCell padding="checkbox">
                    Compare
                  </TableCell>
                )}
                <TableCell>Model</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Accuracy</TableCell>
                <TableCell>Precision</TableCell>
                <TableCell>Recall</TableCell>
                <TableCell>Win Rate</TableCell>
                <TableCell>Latency</TableCell>
                <TableCell>Trend</TableCell>
                <TableCell>Last Update</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedModels.map((model) => (
                <TableRow key={model.modelId} hover>
                  {showComparison && (
                    <TableCell padding="checkbox">
                      <Switch
                        checked={selectedModels.includes(model.modelId)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedModels([...selectedModels, model.modelId]);
                          } else {
                            setSelectedModels(selectedModels.filter(id => id !== model.modelId));
                          }
                        }}
                        size="small"
                      />
                    </TableCell>
                  )}
                  <TableCell>
                    <Box>
                      <Typography variant="body2" fontWeight="medium">
                        {model.modelName}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {model.modelType}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      icon={
                        model.status === 'healthy' ? <CheckCircle /> :
                        model.status === 'degraded' ? <Warning /> :
                        model.status === 'critical' ? <Error /> :
                        <Error />
                      }
                      label={model.status.toUpperCase()}
                      color={getStatusColor(model.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={model.accuracy * 100}
                        sx={{ width: 60, height: 6 }}
                        color={model.accuracy > 0.8 ? 'success' : model.accuracy > 0.6 ? 'warning' : 'error'}
                      />
                      <Typography variant="body2">
                        {(model.accuracy * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>{(model.precision * 100).toFixed(1)}%</TableCell>
                  <TableCell>{(model.recall * 100).toFixed(1)}%</TableCell>
                  <TableCell>{(model.winRate * 100).toFixed(1)}%</TableCell>
                  <TableCell>
                    <Chip
                      label={`${model.latency}ms`}
                      color={model.latency < 50 ? 'success' : model.latency < 100 ? 'warning' : 'error'}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip title={`Performance is ${model.trend}`}>
                      {getTrendIcon(model.trend)}
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption">
                      {model.lastUpdate.toLocaleString()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Tooltip title="View Details">
                      <IconButton size="small">
                        <Analytics />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Performance Trends */}
      {showTrends && (
        <Grid container spacing={3} sx={{ mt: 3 }}>
          {models.slice(0, 4).map((model) => (
            <Grid item xs={12} md={6} key={model.modelId}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  {model.modelName} - Performance Trends
                </Typography>
                {generateTrendChart(model.modelId)}
              </Paper>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default ModelPerformanceMonitor;
