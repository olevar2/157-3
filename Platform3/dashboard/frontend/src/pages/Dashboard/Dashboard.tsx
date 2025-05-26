import React from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Assessment,
  Notifications,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import AIAnalyticsDashboard from '../../components/AIAnalyticsDashboard';

const Dashboard: React.FC = () => {
  // Mock data for the dashboard overview
  const portfolioSummary = {
    totalBalance: 125450.75,
    totalPL: 12347.50,
    totalPLPercentage: 10.87,
    openPositions: 5,
    todaysPL: 1234.50,
    todaysPLPercentage: 0.99,
  };

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
      change: '2 profitable',
      icon: <Assessment />,
      color: 'info',
      trend: 'neutral',
    },
    {
      title: 'Market Alerts',
      value: '3',
      change: '1 new',
      icon: <Notifications />,
      color: 'warning',
      trend: 'neutral',
    },
  ];

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

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header */}
        <motion.div variants={itemVariants}>
          <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
            Trading Dashboard
          </Typography>
        </motion.div>

        {/* Quick Stats Cards */}
        <motion.div variants={itemVariants}>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {quickStats.map((stat, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
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
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 48,
                          height: 48,
                          borderRadius: '12px',
                          bgcolor: `${stat.color}.main`,
                          color: 'white',
                        }}
                      >
                        {stat.icon}
                      </Box>
                      {stat.trend === 'up' && <TrendingUp color="success" />}
                      {stat.trend === 'down' && <TrendingDown color="error" />}
                    </Box>

                    <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
                      {stat.value}
                    </Typography>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {stat.title}
                    </Typography>

                    <Chip
                      label={stat.change}
                      size="small"
                      color={stat.trend === 'up' ? 'success' : stat.trend === 'down' ? 'error' : 'default'}
                      sx={{ fontSize: '0.75rem' }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </motion.div>

        {/* Portfolio Overview */}
        <motion.div variants={itemVariants}>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Portfolio Performance
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Profit/Loss
                  </Typography>
                  <Typography variant="h5" color="success.main" sx={{ fontWeight: 600 }}>
                    +${portfolioSummary.totalPL.toLocaleString()} ({portfolioSummary.totalPLPercentage}%)
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(portfolioSummary.totalPLPercentage * 5, 100)}
                  color="success"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Market Status
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Trading Session
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    London/New York Overlap
                  </Typography>
                </Box>
                <Chip
                  label="High Volatility"
                  color="warning"
                  size="small"
                />
                <Chip
                  label="Active Trading"
                  color="success"
                  size="small"
                  sx={{ ml: 1 }}
                />
              </Paper>
            </Grid>
          </Grid>
        </motion.div>

        {/* AI Analytics Dashboard - Main Feature */}
        <motion.div variants={itemVariants}>
          <AIAnalyticsDashboard />
        </motion.div>
      </motion.div>
    </Box>
  );
};

export default Dashboard;
