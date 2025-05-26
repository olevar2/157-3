import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  Tab,
  Tabs,
  Switch,
  FormControlLabel,
  Alert,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Assessment,
  Notifications,
  Settings,
  Fullscreen,
  Dashboard as DashboardIcon,
  Timeline,
  Speed,
  AttachMoney,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import RealTimeChart from '../components/RealTimeChart';
import SignalBoard from '../components/SignalBoard';
import AIAnalyticsDashboard from '../components/AIAnalyticsDashboard';
import { useWebSocket } from '../contexts/WebSocketContext';

interface PortfolioSummary {
  totalBalance: number;
  totalPL: number;
  totalPLPercentage: number;
  openPositions: number;
  todaysPL: number;
  todaysPLPercentage: number;
  dailyVolume: number;
  winRate: number;
  avgRiskReward: number;
}

interface MarketStatus {
  session: string;
  volatility: 'low' | 'medium' | 'high';
  sentiment: 'bullish' | 'bearish' | 'neutral';
  activeSignals: number;
  majorPairs: {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
  }[];
}

const DashboardPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [selectedSymbol, setSelectedSymbol] = useState('EUR/USD');
  
  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary>({
    totalBalance: 125450.75,
    totalPL: 12347.50,
    totalPLPercentage: 10.87,
    openPositions: 5,
    todaysPL: 1234.50,
    todaysPLPercentage: 0.99,
    dailyVolume: 2500000,
    winRate: 68.5,
    avgRiskReward: 2.3,
  });

  const [marketStatus, setMarketStatus] = useState<MarketStatus>({
    session: 'London/New York Overlap',
    volatility: 'high',
    sentiment: 'bullish',
    activeSignals: 8,
    majorPairs: [
      { symbol: 'EUR/USD', price: 1.0780, change: 0.0012, changePercent: 0.11 },
      { symbol: 'GBP/USD', price: 1.2520, change: -0.0025, changePercent: -0.20 },
      { symbol: 'USD/JPY', price: 149.85, change: 0.45, changePercent: 0.30 },
      { symbol: 'AUD/USD', price: 0.6580, change: 0.0008, changePercent: 0.12 },
    ],
  });

  const { sendMessage, lastMessage } = useWebSocket();

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  };

  // Quick stats for overview cards
  const quickStats = [
    {
      title: 'Account Balance',
      value: `$${portfolioSummary.totalBalance.toLocaleString()}`,
      change: `+${portfolioSummary.totalPLPercentage}%`,
      icon: <AccountBalance />,
      color: 'primary',
      trend: 'up',
    },
    {
      title: 'Today\'s P&L',
      value: `$${portfolioSummary.todaysPL.toLocaleString()}`,
      change: `+${portfolioSummary.todaysPLPercentage}%`,
      icon: <ShowChart />,
      color: 'success',
      trend: 'up',
    },
    {
      title: 'Open Positions',
      value: portfolioSummary.openPositions.toString(),
      change: '3 profitable',
      icon: <Assessment />,
      color: 'info',
      trend: 'neutral',
    },
    {
      title: 'Active Signals',
      value: marketStatus.activeSignals.toString(),
      change: '2 new',
      icon: <Notifications />,
      color: 'warning',
      trend: 'neutral',
    },
    {
      title: 'Win Rate',
      value: `${portfolioSummary.winRate}%`,
      change: '+2.3%',
      icon: <Speed />,
      color: 'success',
      trend: 'up',
    },
    {
      title: 'Avg R/R',
      value: portfolioSummary.avgRiskReward.toString(),
      change: 'Excellent',
      icon: <Timeline />,
      color: 'success',
      trend: 'up',
    },
  ];

  // Update data periodically
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Simulate real-time updates
      setPortfolioSummary(prev => ({
        ...prev,
        todaysPL: prev.todaysPL + (Math.random() - 0.5) * 100,
        todaysPLPercentage: prev.todaysPLPercentage + (Math.random() - 0.5) * 0.1,
      }));

      setMarketStatus(prev => ({
        ...prev,
        majorPairs: prev.majorPairs.map(pair => ({
          ...pair,
          price: pair.price + (Math.random() - 0.5) * 0.001,
          change: pair.change + (Math.random() - 0.5) * 0.0005,
        })),
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const getVolatilityColor = (volatility: string) => {
    switch (volatility) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'default';
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'success';
      case 'bearish': return 'error';
      case 'neutral': return 'warning';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3, minHeight: '100vh' }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div variants={itemVariants}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <DashboardIcon />
              Professional Trading Dashboard
            </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                }
                label="Auto Refresh"
              />
              <Tooltip title="Dashboard Settings">
                <IconButton>
                  <Settings />
                </IconButton>
              </Tooltip>
              <Tooltip title="Fullscreen Mode">
                <IconButton onClick={() => setIsFullscreen(!isFullscreen)}>
                  <Fullscreen />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </motion.div>

        {/* Quick Stats Cards */}
        <motion.div variants={itemVariants}>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {quickStats.map((stat, index) => (
              <Grid item xs={12} sm={6} md={4} lg={2} key={index}>
                <Card
                  elevation={3}
                  sx={{
                    background: 'linear-gradient(145deg, #1a1b23 0%, #141520 100%)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: '0 12px 40px rgba(0, 0, 0, 0.4)',
                    },
                  }}
                >
                  <CardContent sx={{ p: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 40,
                          height: 40,
                          borderRadius: '8px',
                          bgcolor: `${stat.color}.main`,
                          color: 'white',
                        }}
                      >
                        {stat.icon}
                      </Box>
                      {stat.trend === 'up' && <TrendingUp color="success" fontSize="small" />}
                      {stat.trend === 'down' && <TrendingDown color="error" fontSize="small" />}
                    </Box>

                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                      {stat.value}
                    </Typography>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5, fontSize: '0.75rem' }}>
                      {stat.title}
                    </Typography>

                    <Chip
                      label={stat.change}
                      size="small"
                      color={stat.trend === 'up' ? 'success' : stat.trend === 'down' ? 'error' : 'default'}
                      sx={{ fontSize: '0.7rem', height: 20 }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </motion.div>

        {/* Market Overview */}
        <motion.div variants={itemVariants}>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={8}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Market Overview
                </Typography>
                <Grid container spacing={2}>
                  {marketStatus.majorPairs.map((pair, index) => (
                    <Grid item xs={6} md={3} key={index}>
                      <Box
                        sx={{
                          p: 2,
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: 1,
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          bgcolor: selectedSymbol === pair.symbol ? 'primary.dark' : 'transparent',
                          '&:hover': {
                            bgcolor: 'primary.dark',
                          },
                        }}
                        onClick={() => setSelectedSymbol(pair.symbol)}
                      >
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          {pair.symbol}
                        </Typography>
                        <Typography variant="h6" color={pair.change >= 0 ? 'success.main' : 'error.main'}>
                          {pair.price.toFixed(pair.symbol.includes('JPY') ? 2 : 5)}
                        </Typography>
                        <Typography variant="caption" color={pair.change >= 0 ? 'success.main' : 'error.main'}>
                          {pair.change >= 0 ? '+' : ''}{pair.change.toFixed(4)} ({pair.changePercent >= 0 ? '+' : ''}{pair.changePercent.toFixed(2)}%)
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Market Status
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Trading Session
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    {marketStatus.session}
                  </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Chip
                    label={`${marketStatus.volatility.toUpperCase()} Volatility`}
                    color={getVolatilityColor(marketStatus.volatility) as any}
                    size="small"
                  />
                  <Chip
                    label={`${marketStatus.sentiment.toUpperCase()} Sentiment`}
                    color={getSentimentColor(marketStatus.sentiment) as any}
                    size="small"
                  />
                  <Chip
                    label="Active Trading"
                    color="success"
                    size="small"
                  />
                </Box>

                <Divider sx={{ my: 2 }} />
                
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    High volatility period detected. Monitor risk levels closely and consider reducing position sizes.
                  </Typography>
                </Alert>
              </Paper>
            </Grid>
          </Grid>
        </motion.div>

        {/* Main Dashboard Tabs */}
        <motion.div variants={itemVariants}>
          <Paper elevation={3} sx={{ mb: 3 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Trading Chart" icon={<ShowChart />} />
              <Tab label="Signal Board" icon={<Timeline />} />
              <Tab label="AI Analytics" icon={<Assessment />} />
            </Tabs>
          </Paper>
        </motion.div>

        {/* Tab Content */}
        <motion.div variants={itemVariants}>
          {activeTab === 0 && (
            <RealTimeChart
              symbol={selectedSymbol}
              height={700}
              showControls={true}
            />
          )}

          {activeTab === 1 && (
            <SignalBoard
              height={700}
              maxSignals={15}
              autoRefresh={autoRefresh}
              showFilters={true}
            />
          )}

          {activeTab === 2 && (
            <AIAnalyticsDashboard />
          )}
        </motion.div>
      </motion.div>
    </Box>
  );
};

export default DashboardPage;
