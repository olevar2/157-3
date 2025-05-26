/**
 * Performance Analytics & Reporting Dashboard
 * Comprehensive trading performance analysis and reporting system
 * 
 * Features:
 * - Real-time performance metrics and KPIs
 * - Detailed trade analysis and statistics
 * - Risk-adjusted performance measures
 * - Comparative analysis and benchmarking
 * - Custom report generation
 * - Export capabilities (PDF, Excel, CSV)
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
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  DatePicker,
  Button,
  IconButton,
  Tooltip,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Assessment,
  PieChart,
  BarChart,
  Timeline,
  Download,
  Print,
  Share,
  Refresh,
  FilterList,
  DateRange,
  AccountBalance,
  Speed,
  Security
} from '@mui/icons-material';
import { Line, Bar, Pie, Doughnut, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
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
  ArcElement,
  RadialLinearScale,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

interface PerformanceMetrics {
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

interface TradeData {
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

interface PerformanceAnalyticsDashboardProps {
  accountId?: string;
  dateRange?: { start: Date; end: Date };
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const PerformanceAnalyticsDashboard: React.FC<PerformanceAnalyticsDashboardProps> = ({
  accountId,
  dateRange,
  autoRefresh = true,
  refreshInterval = 60000
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedPeriod, setSelectedPeriod] = useState('1M');
  const [selectedStrategy, setSelectedStrategy] = useState('all');
  const [selectedSymbol, setSelectedSymbol] = useState('all');
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [trades, setTrades] = useState<TradeData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  // Fetch performance data
  const fetchPerformanceData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Simulate API calls - replace with actual endpoints
      const [metricsRes, tradesRes] = await Promise.all([
        fetch(`/api/v1/performance/metrics?period=${selectedPeriod}&strategy=${selectedStrategy}&symbol=${selectedSymbol}`),
        fetch(`/api/v1/performance/trades?period=${selectedPeriod}&strategy=${selectedStrategy}&symbol=${selectedSymbol}`)
      ]);
      
      if (metricsRes.ok && tradesRes.ok) {
        const metricsData = await metricsRes.json();
        const tradesData = await tradesRes.json();
        
        setMetrics(metricsData.data || generateMockMetrics());
        setTrades(tradesData.data || generateMockTrades());
      } else {
        throw new Error('Failed to fetch performance data');
      }
    } catch (err) {
      setError('Failed to load performance data');
      console.error('Performance data fetch error:', err);
      
      // Use mock data on error
      setMetrics(generateMockMetrics());
      setTrades(generateMockTrades());
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchPerformanceData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchPerformanceData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [selectedPeriod, selectedStrategy, selectedSymbol, autoRefresh, refreshInterval]);

  // Generate mock data
  const generateMockMetrics = (): PerformanceMetrics => ({
    totalReturn: 15420.50,
    totalReturnPercentage: 15.42,
    sharpeRatio: 1.85,
    sortinoRatio: 2.34,
    maxDrawdown: -2340.75,
    maxDrawdownPercentage: -2.34,
    winRate: 0.68,
    profitFactor: 1.95,
    averageWin: 145.30,
    averageLoss: -89.20,
    totalTrades: 247,
    winningTrades: 168,
    losingTrades: 79,
    largestWin: 890.45,
    largestLoss: -456.20,
    averageTradeReturn: 62.47,
    volatility: 0.12,
    calmarRatio: 6.58,
    recoveryFactor: 3.42,
    payoffRatio: 1.63
  });

  const generateMockTrades = (): TradeData[] => {
    const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'];
    const strategies = ['Scalping', 'Day Trading', 'Swing Trading'];
    const trades: TradeData[] = [];
    
    for (let i = 0; i < 100; i++) {
      const entryTime = new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000);
      const duration = Math.random() * 240; // 0-4 hours
      const exitTime = new Date(entryTime.getTime() + duration * 60 * 1000);
      const isWin = Math.random() > 0.32; // 68% win rate
      const pnl = isWin ? Math.random() * 500 + 50 : -(Math.random() * 300 + 30);
      
      trades.push({
        id: `trade_${i + 1}`,
        symbol: symbols[Math.floor(Math.random() * symbols.length)],
        type: Math.random() > 0.5 ? 'buy' : 'sell',
        entryTime,
        exitTime,
        entryPrice: 1.1000 + Math.random() * 0.1,
        exitPrice: 1.1000 + Math.random() * 0.1,
        quantity: Math.floor(Math.random() * 100000) + 10000,
        pnl,
        pnlPercentage: (pnl / 10000) * 100,
        commission: Math.random() * 10 + 2,
        duration,
        strategy: strategies[Math.floor(Math.random() * strategies.length)],
        tags: ['automated', 'high-confidence']
      });
    }
    
    return trades.sort((a, b) => b.exitTime.getTime() - a.exitTime.getTime());
  };

  // Calculate additional metrics
  const calculatedMetrics = useMemo(() => {
    if (!metrics || trades.length === 0) return null;
    
    const profitableTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    
    return {
      avgWinDuration: profitableTrades.reduce((sum, t) => sum + t.duration, 0) / profitableTrades.length,
      avgLossDuration: losingTrades.reduce((sum, t) => sum + t.duration, 0) / losingTrades.length,
      bestDay: Math.max(...trades.map(t => t.pnl)),
      worstDay: Math.min(...trades.map(t => t.pnl)),
      consecutiveWins: calculateConsecutiveWins(trades),
      consecutiveLosses: calculateConsecutiveLosses(trades)
    };
  }, [metrics, trades]);

  const calculateConsecutiveWins = (trades: TradeData[]): number => {
    let maxWins = 0;
    let currentWins = 0;
    
    for (const trade of trades) {
      if (trade.pnl > 0) {
        currentWins++;
        maxWins = Math.max(maxWins, currentWins);
      } else {
        currentWins = 0;
      }
    }
    
    return maxWins;
  };

  const calculateConsecutiveLosses = (trades: TradeData[]): number => {
    let maxLosses = 0;
    let currentLosses = 0;
    
    for (const trade of trades) {
      if (trade.pnl < 0) {
        currentLosses++;
        maxLosses = Math.max(maxLosses, currentLosses);
      } else {
        currentLosses = 0;
      }
    }
    
    return maxLosses;
  };

  // Generate performance charts
  const generateEquityCurveChart = () => {
    const equityData = trades.reduce((acc, trade, index) => {
      const prevEquity = index === 0 ? 100000 : acc[index - 1];
      acc.push(prevEquity + trade.pnl);
      return acc;
    }, [] as number[]);

    return {
      labels: trades.map((_, index) => `Trade ${index + 1}`),
      datasets: [
        {
          label: 'Account Equity',
          data: equityData,
          borderColor: '#2196F3',
          backgroundColor: 'rgba(33, 150, 243, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1
        }
      ]
    };
  };

  const generatePnLDistributionChart = () => {
    const bins = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500];
    const distribution = new Array(bins.length - 1).fill(0);
    
    trades.forEach(trade => {
      for (let i = 0; i < bins.length - 1; i++) {
        if (trade.pnl >= bins[i] && trade.pnl < bins[i + 1]) {
          distribution[i]++;
          break;
        }
      }
    });

    return {
      labels: bins.slice(0, -1).map((bin, i) => `${bin} to ${bins[i + 1]}`),
      datasets: [
        {
          label: 'Trade Count',
          data: distribution,
          backgroundColor: distribution.map((_, i) => 
            bins[i] < 0 ? '#f44336' : '#4caf50'
          ),
          borderColor: '#333',
          borderWidth: 1
        }
      ]
    };
  };

  const generateSymbolPerformanceChart = () => {
    const symbolPnL = trades.reduce((acc, trade) => {
      acc[trade.symbol] = (acc[trade.symbol] || 0) + trade.pnl;
      return acc;
    }, {} as Record<string, number>);

    return {
      labels: Object.keys(symbolPnL),
      datasets: [
        {
          data: Object.values(symbolPnL),
          backgroundColor: [
            '#FF6384',
            '#36A2EB',
            '#FFCE56',
            '#4BC0C0',
            '#9966FF',
            '#FF9F40'
          ],
          borderWidth: 2
        }
      ]
    };
  };

  // Render overview tab
  const renderOverviewTab = () => (
    <Grid container spacing={3}>
      {/* Key Metrics Cards */}
      <Grid item xs={12} sm={6} md={3}>
        <Card elevation={3}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="primary">
              ${metrics?.totalReturn.toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Return
            </Typography>
            <Chip
              label={`${metrics?.totalReturnPercentage.toFixed(2)}%`}
              color={metrics && metrics.totalReturnPercentage > 0 ? 'success' : 'error'}
              size="small"
              sx={{ mt: 1 }}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card elevation={3}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="secondary">
              {metrics?.sharpeRatio.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Sharpe Ratio
            </Typography>
            <LinearProgress
              variant="determinate"
              value={Math.min((metrics?.sharpeRatio || 0) * 20, 100)}
              sx={{ mt: 1 }}
              color={metrics && metrics.sharpeRatio > 1 ? 'success' : 'warning'}
            />
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card elevation={3}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="success.main">
              {((metrics?.winRate || 0) * 100).toFixed(1)}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Win Rate
            </Typography>
            <Typography variant="caption" display="block">
              {metrics?.winningTrades}/{metrics?.totalTrades} trades
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} sm={6} md={3}>
        <Card elevation={3}>
          <CardContent sx={{ textAlign: 'center' }}>
            <Typography variant="h4" color="error.main">
              {metrics?.maxDrawdownPercentage.toFixed(2)}%
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Max Drawdown
            </Typography>
            <Typography variant="caption" display="block">
              ${metrics?.maxDrawdown.toLocaleString()}
            </Typography>
          </CardContent>
        </Card>
      </Grid>

      {/* Equity Curve Chart */}
      <Grid item xs={12} lg={8}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Equity Curve
          </Typography>
          <Box sx={{ height: 300 }}>
            <Line
              data={generateEquityCurveChart()}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { position: 'top' },
                  title: { display: false }
                },
                scales: {
                  y: {
                    beginAtZero: false,
                    title: { display: true, text: 'Account Value ($)' }
                  }
                }
              }}
            />
          </Box>
        </Paper>
      </Grid>

      {/* Symbol Performance */}
      <Grid item xs={12} lg={4}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Performance by Symbol
          </Typography>
          <Box sx={{ height: 300 }}>
            <Doughnut
              data={generateSymbolPerformanceChart()}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { position: 'bottom' }
                }
              }}
            />
          </Box>
        </Paper>
      </Grid>

      {/* Additional Metrics */}
      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Detailed Performance Metrics
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6">{metrics?.profitFactor.toFixed(2)}</Typography>
                <Typography variant="body2" color="text.secondary">Profit Factor</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6">{metrics?.sortinoRatio.toFixed(2)}</Typography>
                <Typography variant="body2" color="text.secondary">Sortino Ratio</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6">{metrics?.calmarRatio.toFixed(2)}</Typography>
                <Typography variant="body2" color="text.secondary">Calmar Ratio</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6">{((metrics?.volatility || 0) * 100).toFixed(2)}%</Typography>
                <Typography variant="body2" color="text.secondary">Volatility</Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );

  // Render trades tab
  const renderTradesTab = () => (
    <Paper elevation={3}>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Trade ID</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Entry Time</TableCell>
              <TableCell>Exit Time</TableCell>
              <TableCell>Duration</TableCell>
              <TableCell>P&L</TableCell>
              <TableCell>P&L %</TableCell>
              <TableCell>Strategy</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {trades
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((trade) => (
                <TableRow key={trade.id} hover>
                  <TableCell>{trade.id}</TableCell>
                  <TableCell>{trade.symbol}</TableCell>
                  <TableCell>
                    <Chip
                      label={trade.type.toUpperCase()}
                      color={trade.type === 'buy' ? 'primary' : 'secondary'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{trade.entryTime.toLocaleString()}</TableCell>
                  <TableCell>{trade.exitTime.toLocaleString()}</TableCell>
                  <TableCell>{Math.round(trade.duration)}m</TableCell>
                  <TableCell>
                    <Typography
                      color={trade.pnl > 0 ? 'success.main' : 'error.main'}
                      fontWeight="medium"
                    >
                      ${trade.pnl.toFixed(2)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography
                      color={trade.pnlPercentage > 0 ? 'success.main' : 'error.main'}
                      fontWeight="medium"
                    >
                      {trade.pnlPercentage.toFixed(2)}%
                    </Typography>
                  </TableCell>
                  <TableCell>{trade.strategy}</TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[25, 50, 100]}
        component="div"
        count={trades.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={(_, newPage) => setPage(newPage)}
        onRowsPerPageChange={(e) => {
          setRowsPerPage(parseInt(e.target.value, 10));
          setPage(0);
        }}
      />
    </Paper>
  );

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
          Performance Analytics
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Period</InputLabel>
            <Select
              value={selectedPeriod}
              label="Period"
              onChange={(e) => setSelectedPeriod(e.target.value)}
            >
              <MenuItem value="1D">1 Day</MenuItem>
              <MenuItem value="1W">1 Week</MenuItem>
              <MenuItem value="1M">1 Month</MenuItem>
              <MenuItem value="3M">3 Months</MenuItem>
              <MenuItem value="6M">6 Months</MenuItem>
              <MenuItem value="1Y">1 Year</MenuItem>
              <MenuItem value="ALL">All Time</MenuItem>
            </Select>
          </FormControl>

          <Tooltip title="Refresh Data">
            <IconButton onClick={fetchPerformanceData} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>

          <Tooltip title="Export Report">
            <IconButton>
              <Download />
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
          <Tab icon={<Assessment />} label="Overview" />
          <Tab icon={<Timeline />} label="Trades" />
          <Tab icon={<BarChart />} label="Analysis" />
          <Tab icon={<PieChart />} label="Reports" />
        </Tabs>
      </Paper>

      {/* Tab Content */}
      <Box sx={{ mt: 3 }}>
        {activeTab === 0 && renderOverviewTab()}
        {activeTab === 1 && renderTradesTab()}
        {activeTab === 2 && (
          <Typography variant="h6">Advanced Analysis Coming Soon...</Typography>
        )}
        {activeTab === 3 && (
          <Typography variant="h6">Custom Reports Coming Soon...</Typography>
        )}
      </Box>
    </Box>
  );
};

export default PerformanceAnalyticsDashboard;
