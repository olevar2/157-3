import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Tab,
  Tabs,
  Alert,
  LinearProgress,
  Chip,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress
} from '@mui/material';
import {
  Timeline,
  Assessment,
  TrendingUp,
  TrendingDown,
  Speed,
  Refresh,
  Download,
  Settings,
  Notifications,
  CheckCircle,
  Error,
  Warning
} from '@mui/icons-material';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface RealtimeMetric {
  metric_name: string;
  value: number;
  timestamp: string;
  context: Record<string, any>;
  alert_threshold?: number;
}

interface AnalyticsReport {
  report_id: string;
  report_type: string;
  generated_at: string;
  data: Record<string, any>;
  summary: string;
  recommendations: string[];
  confidence_score: number;
}

interface AnalyticsEngine {
  name: string;
  status: 'active' | 'inactive' | 'error';
  last_update: string;
  performance_score: number;
  processed_items: number;
}

const AdvancedAnalyticsDashboard: React.FC = () => {
  // State management
  const [currentTab, setCurrentTab] = useState(0);
  const [realtimeMetrics, setRealtimeMetrics] = useState<Record<string, RealtimeMetric>>({});
  const [analyticsEngines, setAnalyticsEngines] = useState<AnalyticsEngine[]>([]);
  const [reports, setReports] = useState<AnalyticsReport[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // WebSocket connection for real-time data
  const [ws, setWs] = useState<WebSocket | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const websocket = new WebSocket('ws://localhost:8001/analytics');
        
        websocket.onopen = () => {
          console.log('Connected to Advanced Analytics Framework');
          setIsConnected(true);
          setLoading(false);
        };

        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'metrics_update') {
              setRealtimeMetrics(data.metrics);
            } else if (data.type === 'engines_status') {
              setAnalyticsEngines(data.engines);
            } else if (data.type === 'report_generated') {
              setReports(prev => [data.report, ...prev.slice(0, 9)]); // Keep last 10 reports
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        websocket.onclose = () => {
          console.log('WebSocket connection closed');
          setIsConnected(false);
          // Attempt to reconnect after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };

        websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
        };

        setWs(websocket);
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        setLoading(false);
      }
    };

    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Auto-refresh functionality
  useEffect(() => {
    if (autoRefresh && isConnected) {
      const interval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'refresh_metrics' }));
        }
      }, refreshInterval * 1000);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, isConnected, ws]);

  // Manual refresh
  const handleManualRefresh = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'refresh_all' }));
    }
  }, [ws]);

  // Generate new report
  const handleGenerateReport = useCallback(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ 
        action: 'generate_report', 
        timeframe: selectedTimeframe 
      }));
    }
  }, [ws, selectedTimeframe]);

  // Download report
  const handleDownloadReport = useCallback((reportId: string) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ 
        action: 'download_report', 
        report_id: reportId 
      }));
    }
  }, [ws]);

  // Prepare chart data for metrics
  const prepareMetricsChartData = () => {
    const performanceMetrics = Object.values(realtimeMetrics)
      .filter(metric => metric.metric_name.includes('performance'))
      .slice(-20); // Last 20 data points

    return {
      labels: performanceMetrics.map(metric => 
        new Date(metric.timestamp).toLocaleTimeString()
      ),
      datasets: [{
        label: 'Performance Score',
        data: performanceMetrics.map(metric => metric.value),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      }]
    };
  };

  // Prepare engines status chart
  const prepareEnginesChartData = () => {
    const activeEngines = analyticsEngines.filter(engine => engine.status === 'active').length;
    const inactiveEngines = analyticsEngines.filter(engine => engine.status === 'inactive').length;
    const errorEngines = analyticsEngines.filter(engine => engine.status === 'error').length;

    return {
      labels: ['Active', 'Inactive', 'Error'],
      datasets: [{
        data: [activeEngines, inactiveEngines, errorEngines],
        backgroundColor: ['#4caf50', '#ff9800', '#f44336'],
        borderWidth: 2
      }]
    };
  };

  // Get metric status color
  const getMetricStatusColor = (metric: RealtimeMetric) => {
    if (metric.alert_threshold && metric.value < metric.alert_threshold) {
      return 'error';
    }
    if (metric.value > 0.8) return 'success';
    if (metric.value > 0.6) return 'warning';
    return 'error';
  };

  // Render connection status
  const renderConnectionStatus = () => (
    <Alert 
      severity={isConnected ? 'success' : 'error'} 
      sx={{ mb: 2 }}
    >
      {isConnected ? 'Connected to Advanced Analytics Framework' : 'Disconnected - Attempting to reconnect...'}
    </Alert>
  );

  // Render metrics overview
  const renderMetricsOverview = () => (
    <Grid container spacing={3}>
      {Object.entries(realtimeMetrics).map(([key, metric]) => (
        <Grid item xs={12} sm={6} md={4} key={key}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="h6" component="h3">
                  {metric.metric_name.replace(/_/g, ' ').toUpperCase()}
                </Typography>
                <Chip
                  size="small"
                  color={getMetricStatusColor(metric)}
                  label={metric.value.toFixed(2)}
                />
              </Box>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Last updated: {new Date(metric.timestamp).toLocaleTimeString()}
              </Typography>
              {metric.alert_threshold && (
                <LinearProgress 
                  variant="determinate" 
                  value={(metric.value / metric.alert_threshold) * 100}
                  sx={{ mt: 2 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  // Render analytics engines status
  const renderEnginesStatus = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Engine Name</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Performance Score</TableCell>
            <TableCell>Processed Items</TableCell>
            <TableCell>Last Update</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {analyticsEngines.map((engine, index) => (
            <TableRow key={index}>
              <TableCell>{engine.name}</TableCell>
              <TableCell>
                <Chip
                  size="small"
                  color={
                    engine.status === 'active' ? 'success' :
                    engine.status === 'inactive' ? 'warning' : 'error'
                  }
                  icon={
                    engine.status === 'active' ? <CheckCircle /> :
                    engine.status === 'inactive' ? <Warning /> : <Error />
                  }
                  label={engine.status.toUpperCase()}
                />
              </TableCell>
              <TableCell>{engine.performance_score.toFixed(2)}</TableCell>
              <TableCell>{engine.processed_items.toLocaleString()}</TableCell>
              <TableCell>{new Date(engine.last_update).toLocaleString()}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  // Render reports section
  const renderReports = () => (
    <Box>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Typography variant="h6">Analytics Reports</Typography>
        <Box display="flex" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              label="Timeframe"
            >
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="6h">6 Hours</MenuItem>
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
            </Select>
          </FormControl>
          <Button 
            variant="contained" 
            onClick={handleGenerateReport}
            startIcon={<Assessment />}
          >
            Generate Report
          </Button>
        </Box>
      </Box>
      
      <Grid container spacing={3}>
        {reports.map((report, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                  <Typography variant="h6">{report.report_type.toUpperCase()}</Typography>
                  <Chip 
                    size="small" 
                    color="primary" 
                    label={`${report.confidence_score.toFixed(0)}%`}
                  />
                </Box>
                <Typography variant="body2" paragraph>
                  {report.summary}
                </Typography>
                <Box mb={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Recommendations:
                  </Typography>
                  {report.recommendations.map((rec, recIndex) => (
                    <Typography key={recIndex} variant="body2" sx={{ fontSize: '0.875rem' }}>
                      • {rec}
                    </Typography>
                  ))}
                </Box>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="caption" color="textSecondary">
                    {new Date(report.generated_at).toLocaleString()}
                  </Typography>
                  <Button 
                    size="small" 
                    onClick={() => handleDownloadReport(report.report_id)}
                    startIcon={<Download />}
                  >
                    Download
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  // Render charts section
  const renderCharts = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Performance Metrics Trend
          </Typography>
          {Object.keys(realtimeMetrics).length > 0 ? (
            <Line 
              data={prepareMetricsChartData()} 
              options={{
                responsive: true,
                plugins: {
                  legend: { position: 'top' as const },
                  title: { display: true, text: 'Real-time Performance Metrics' }
                }
              }}
            />
          ) : (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          )}
        </Paper>
      </Grid>
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Engines Status
          </Typography>
          {analyticsEngines.length > 0 ? (
            <Doughnut 
              data={prepareEnginesChartData()}
              options={{
                responsive: true,
                plugins: {
                  legend: { position: 'bottom' as const }
                }
              }}
            />
          ) : (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          )}
        </Paper>
      </Grid>
    </Grid>
  );

  // Render settings dialog
  const renderSettingsDialog = () => (
    <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
      <DialogTitle>Analytics Dashboard Settings</DialogTitle>
      <DialogContent>
        <Box display="flex" flexDirection="column" gap={3} pt={2}>
          <FormControlLabel
            control={
              <Switch 
                checked={autoRefresh} 
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          
          <FormControl fullWidth>
            <InputLabel>Refresh Interval (seconds)</InputLabel>
            <Select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              label="Refresh Interval (seconds)"
              disabled={!autoRefresh}
            >
              <MenuItem value={10}>10 seconds</MenuItem>
              <MenuItem value={30}>30 seconds</MenuItem>
              <MenuItem value={60}>1 minute</MenuItem>
              <MenuItem value={300}>5 minutes</MenuItem>
            </Select>
          </FormControl>

          <FormControlLabel
            control={
              <Switch 
                checked={alertsEnabled} 
                onChange={(e) => setAlertsEnabled(e.target.checked)}
              />
            }
            label="Enable Alerts"
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setSettingsOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Connecting to Advanced Analytics Framework...
        </Typography>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Advanced Analytics Dashboard
        </Typography>
        <Box display="flex" gap={2}>
          <Button 
            variant="outlined" 
            onClick={handleManualRefresh}
            startIcon={<Refresh />}
            disabled={!isConnected}
          >
            Refresh
          </Button>
          <Button 
            variant="outlined" 
            onClick={() => setSettingsOpen(true)}
            startIcon={<Settings />}
          >
            Settings
          </Button>
        </Box>
      </Box>

      {/* Connection Status */}
      {renderConnectionStatus()}

      {/* Navigation Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={(e, newValue) => setCurrentTab(newValue)}>
          <Tab label="Overview" icon={<Assessment />} />
          <Tab label="Real-time Metrics" icon={<Speed />} />
          <Tab label="Analytics Engines" icon={<Timeline />} />
          <Tab label="Reports" icon={<Download />} />
          <Tab label="Charts" icon={<TrendingUp />} />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {currentTab === 0 && (
        <Box>
          <Typography variant="h5" gutterBottom>
            System Overview
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Active Engines
                  </Typography>
                  <Typography variant="h4">
                    {analyticsEngines.filter(e => e.status === 'active').length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Real-time Metrics
                  </Typography>
                  <Typography variant="h4">
                    {Object.keys(realtimeMetrics).length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography color="textSecondary" gutterBottom>
                    Generated Reports
                  </Typography>
                  <Typography variant="h4">
                    {reports.length}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      )}

      {currentTab === 1 && renderMetricsOverview()}
      {currentTab === 2 && renderEnginesStatus()}
      {currentTab === 3 && renderReports()}
      {currentTab === 4 && renderCharts()}

      {/* Settings Dialog */}
      {renderSettingsDialog()}
    </Box>
  );
};

export default AdvancedAnalyticsDashboard;
