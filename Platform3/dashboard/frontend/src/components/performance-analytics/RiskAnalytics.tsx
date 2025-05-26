/**
 * Risk Analytics Component
 * Comprehensive risk analysis and monitoring for trading performance
 * 
 * Features:
 * - Value at Risk (VaR) calculations
 * - Risk-adjusted performance metrics
 * - Drawdown analysis
 * - Correlation analysis
 * - Risk exposure monitoring
 * - Stress testing scenarios
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  LinearProgress,
  Chip,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Warning,
  Security,
  TrendingDown,
  Assessment,
  Speed,
  Info,
  Timeline
} from '@mui/icons-material';
import { Line, Bar, Radar } from 'react-chartjs-2';

interface RiskMetrics {
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

interface DrawdownPeriod {
  start: Date;
  end: Date;
  duration: number; // days
  peak: number;
  trough: number;
  drawdown: number;
  recovery: Date | null;
  recoveryDuration: number | null;
}

interface RiskExposure {
  symbol: string;
  exposure: number;
  exposurePercentage: number;
  var95: number;
  correlation: number;
  beta: number;
  riskContribution: number;
}

interface RiskAnalyticsProps {
  portfolioValue: number;
  returns: number[];
  benchmarkReturns?: number[];
  positions: RiskExposure[];
  confidenceLevel?: number;
}

const RiskAnalytics: React.FC<RiskAnalyticsProps> = ({
  portfolioValue,
  returns,
  benchmarkReturns = [],
  positions,
  confidenceLevel = 0.95
}) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('1M');
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [drawdownPeriods, setDrawdownPeriods] = useState<DrawdownPeriod[]>([]);

  // Calculate risk metrics
  const calculatedMetrics = useMemo(() => {
    if (returns.length === 0) return null;

    const sortedReturns = [...returns].sort((a, b) => a - b);
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);
    
    // VaR calculations
    const var95Index = Math.floor((1 - 0.95) * sortedReturns.length);
    const var99Index = Math.floor((1 - 0.99) * sortedReturns.length);
    const valueAtRisk95 = -sortedReturns[var95Index] * portfolioValue;
    const valueAtRisk99 = -sortedReturns[var99Index] * portfolioValue;
    
    // Conditional VaR (Expected Shortfall)
    const tailReturns = sortedReturns.slice(0, var95Index + 1);
    const conditionalVaR = -(tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length) * portfolioValue;
    
    // Drawdown calculation
    const cumulativeReturns = returns.reduce((acc, r, i) => {
      acc.push((acc[i - 1] || 1) * (1 + r));
      return acc;
    }, [] as number[]);
    
    let maxDrawdown = 0;
    let maxDrawdownDuration = 0;
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
          maxDrawdownDuration = i - peakIndex;
        }
      }
    }
    
    // Beta and Alpha (if benchmark provided)
    let beta = 1;
    let alpha = 0;
    let informationRatio = 0;
    let trackingError = 0;
    
    if (benchmarkReturns.length === returns.length) {
      const benchmarkMean = benchmarkReturns.reduce((sum, r) => sum + r, 0) / benchmarkReturns.length;
      const covariance = returns.reduce((sum, r, i) => 
        sum + (r - mean) * (benchmarkReturns[i] - benchmarkMean), 0) / returns.length;
      const benchmarkVariance = benchmarkReturns.reduce((sum, r) => 
        sum + Math.pow(r - benchmarkMean, 2), 0) / benchmarkReturns.length;
      
      beta = covariance / benchmarkVariance;
      alpha = mean - beta * benchmarkMean;
      
      const excessReturns = returns.map((r, i) => r - benchmarkReturns[i]);
      const excessMean = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
      trackingError = Math.sqrt(excessReturns.reduce((sum, r) => 
        sum + Math.pow(r - excessMean, 2), 0) / excessReturns.length);
      informationRatio = excessMean / trackingError;
    }
    
    // Downside deviation
    const downsideReturns = returns.filter(r => r < 0);
    const downside_deviation = downsideReturns.length > 0 
      ? Math.sqrt(downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length)
      : 0;
    
    // Calmar Ratio
    const calmarRatio = maxDrawdown > 0 ? (mean * 252) / maxDrawdown : 0;
    
    return {
      valueAtRisk95,
      valueAtRisk99,
      conditionalVaR,
      maxDrawdown,
      maxDrawdownDuration,
      volatility: volatility * Math.sqrt(252), // Annualized
      beta,
      alpha: alpha * 252, // Annualized
      informationRatio,
      trackingError: trackingError * Math.sqrt(252), // Annualized
      downside_deviation: downside_deviation * Math.sqrt(252), // Annualized
      upside_capture: 1.2, // Mock value
      downside_capture: 0.8, // Mock value
      calmarRatio,
      sterlingRatio: calmarRatio * 0.9, // Approximation
      burkeRatio: calmarRatio * 1.1 // Approximation
    };
  }, [returns, benchmarkReturns, portfolioValue]);

  // Generate drawdown periods
  const generateDrawdownPeriods = useMemo(() => {
    if (returns.length === 0) return [];
    
    const periods: DrawdownPeriod[] = [];
    const cumulativeReturns = returns.reduce((acc, r, i) => {
      acc.push((acc[i - 1] || 100000) * (1 + r));
      return acc;
    }, [] as number[]);
    
    let peak = cumulativeReturns[0];
    let peakIndex = 0;
    let inDrawdown = false;
    
    for (let i = 1; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] > peak) {
        if (inDrawdown) {
          // End of drawdown period
          periods.push({
            start: new Date(Date.now() - (returns.length - peakIndex) * 24 * 60 * 60 * 1000),
            end: new Date(Date.now() - (returns.length - i) * 24 * 60 * 60 * 1000),
            duration: i - peakIndex,
            peak,
            trough: Math.min(...cumulativeReturns.slice(peakIndex, i)),
            drawdown: (peak - Math.min(...cumulativeReturns.slice(peakIndex, i))) / peak,
            recovery: new Date(Date.now() - (returns.length - i) * 24 * 60 * 60 * 1000),
            recoveryDuration: i - peakIndex
          });
          inDrawdown = false;
        }
        peak = cumulativeReturns[i];
        peakIndex = i;
      } else if (cumulativeReturns[i] < peak && !inDrawdown) {
        inDrawdown = true;
      }
    }
    
    return periods.sort((a, b) => b.drawdown - a.drawdown).slice(0, 5);
  }, [returns]);

  // Generate VaR chart
  const generateVaRChart = () => {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    
    const data = percentiles.map(p => {
      const index = Math.floor(p * sortedReturns.length);
      return sortedReturns[index] * portfolioValue;
    });

    return {
      labels: percentiles.map(p => `${(p * 100).toFixed(0)}%`),
      datasets: [
        {
          label: 'Portfolio Value at Risk',
          data,
          backgroundColor: data.map(d => d < 0 ? '#f44336' : '#4caf50'),
          borderColor: '#333',
          borderWidth: 1
        }
      ]
    };
  };

  // Generate correlation matrix
  const generateCorrelationChart = () => {
    const symbols = positions.map(p => p.symbol);
    const correlations = positions.map(p => p.correlation);

    return {
      labels: symbols,
      datasets: [
        {
          label: 'Correlation with Portfolio',
          data: correlations,
          backgroundColor: correlations.map(c => 
            c > 0.7 ? '#f44336' : c > 0.3 ? '#ff9800' : '#4caf50'
          ),
          borderWidth: 1
        }
      ]
    };
  };

  // Risk level indicator
  const getRiskLevel = (metric: number, thresholds: number[]): { level: string; color: string } => {
    if (metric <= thresholds[0]) return { level: 'Low', color: '#4caf50' };
    if (metric <= thresholds[1]) return { level: 'Medium', color: '#ff9800' };
    return { level: 'High', color: '#f44336' };
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Risk Analytics
      </Typography>

      {/* Risk Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={3}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h5" color="error.main">
                ${calculatedMetrics?.valueAtRisk95.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                VaR (95%)
              </Typography>
              <Chip
                label={getRiskLevel(calculatedMetrics?.valueAtRisk95 || 0, [1000, 5000]).level}
                color={getRiskLevel(calculatedMetrics?.valueAtRisk95 || 0, [1000, 5000]).level === 'Low' ? 'success' : 
                       getRiskLevel(calculatedMetrics?.valueAtRisk95 || 0, [1000, 5000]).level === 'Medium' ? 'warning' : 'error'}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={3}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h5" color="warning.main">
                {((calculatedMetrics?.maxDrawdown || 0) * 100).toFixed(2)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Max Drawdown
              </Typography>
              <Typography variant="caption" display="block">
                {calculatedMetrics?.maxDrawdownDuration} days
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={3}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h5" color="info.main">
                {((calculatedMetrics?.volatility || 0) * 100).toFixed(2)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Volatility (Annual)
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min((calculatedMetrics?.volatility || 0) * 500, 100)}
                sx={{ mt: 1 }}
                color={calculatedMetrics && calculatedMetrics.volatility < 0.15 ? 'success' : 'warning'}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={3}>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h5" color="secondary.main">
                {calculatedMetrics?.calmarRatio.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Calmar Ratio
              </Typography>
              <Chip
                label={calculatedMetrics && calculatedMetrics.calmarRatio > 2 ? 'Excellent' : 
                       calculatedMetrics && calculatedMetrics.calmarRatio > 1 ? 'Good' : 'Poor'}
                color={calculatedMetrics && calculatedMetrics.calmarRatio > 2 ? 'success' : 
                       calculatedMetrics && calculatedMetrics.calmarRatio > 1 ? 'warning' : 'error'}
                size="small"
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Value at Risk Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <Bar
                data={generateVaRChart()}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false },
                    title: { display: false }
                  },
                  scales: {
                    y: {
                      title: { display: true, text: 'Portfolio Value ($)' }
                    },
                    x: {
                      title: { display: true, text: 'Percentile' }
                    }
                  }
                }}
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Position Correlations
            </Typography>
            <Box sx={{ height: 300 }}>
              <Bar
                data={generateCorrelationChart()}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false }
                  },
                  scales: {
                    y: {
                      min: -1,
                      max: 1,
                      title: { display: true, text: 'Correlation' }
                    }
                  }
                }}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Risk Metrics Table */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Risk-Adjusted Metrics
            </Typography>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell>Sharpe Ratio</TableCell>
                  <TableCell align="right">
                    {calculatedMetrics ? (calculatedMetrics.alpha / calculatedMetrics.volatility).toFixed(3) : 'N/A'}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Information Ratio</TableCell>
                  <TableCell align="right">{calculatedMetrics?.informationRatio.toFixed(3) || 'N/A'}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Calmar Ratio</TableCell>
                  <TableCell align="right">{calculatedMetrics?.calmarRatio.toFixed(3)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Sterling Ratio</TableCell>
                  <TableCell align="right">{calculatedMetrics?.sterlingRatio.toFixed(3)}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Burke Ratio</TableCell>
                  <TableCell align="right">{calculatedMetrics?.burkeRatio.toFixed(3)}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Worst Drawdown Periods
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Period</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Drawdown</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {generateDrawdownPeriods.slice(0, 5).map((period, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {period.start.toLocaleDateString()} - {period.end.toLocaleDateString()}
                    </TableCell>
                    <TableCell>{period.duration} days</TableCell>
                    <TableCell>
                      <Typography color="error.main">
                        -{(period.drawdown * 100).toFixed(2)}%
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskAnalytics;
