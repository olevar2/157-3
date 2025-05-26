import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Badge,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  PlayArrow,
  Pause,
  Stop,
  CheckCircle,
  Warning,
  Error,
  Notifications,
  FilterList,
  Refresh,
  Settings,
  Timeline,
  AttachMoney,
  Schedule,
  Speed,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useWebSocket } from '../contexts/WebSocketContext';

interface TradingSignal {
  id: string;
  symbol: string;
  action: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop';
  price: number;
  targetPrice?: number;
  stopLoss?: number;
  confidence: number;
  strength: number;
  timeframe: string;
  strategy: string;
  reasoning: string;
  timestamp: Date;
  status: 'pending' | 'active' | 'executed' | 'cancelled' | 'expired';
  session: 'asian' | 'london' | 'newyork' | 'overlap';
  riskReward?: number;
  expectedDuration?: number;
  volume?: number;
}

interface SignalBoardProps {
  height?: number;
  maxSignals?: number;
  autoRefresh?: boolean;
  showFilters?: boolean;
}

const SignalBoard: React.FC<SignalBoardProps> = ({
  height = 600,
  maxSignals = 10,
  autoRefresh = true,
  showFilters = true,
}) => {
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [filteredSignals, setFilteredSignals] = useState<TradingSignal[]>([]);
  const [selectedSignal, setSelectedSignal] = useState<TradingSignal | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterSymbol, setFilterSymbol] = useState<string>('all');
  const [filterAction, setFilterAction] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const { sendMessage, lastMessage } = useWebSocket();

  // Generate mock signals for demonstration
  const generateMockSignals = (): TradingSignal[] => {
    const symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF', 'NZD/USD'];
    const strategies = ['Scalping VWAP', 'Momentum Breakout', 'Support/Resistance', 'MA Crossover', 'RSI Divergence'];
    const timeframes = ['M1', 'M5', 'M15', 'H1'];
    const sessions = ['asian', 'london', 'newyork', 'overlap'] as const;
    const statuses = ['pending', 'active', 'executed', 'cancelled'] as const;

    return Array.from({ length: 15 }, (_, i) => {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const action = Math.random() > 0.5 ? 'buy' : 'sell';
      const basePrice = 1.0780 + (Math.random() - 0.5) * 0.02;
      const confidence = Math.floor(Math.random() * 40) + 60;
      const strength = Math.floor(Math.random() * 30) + 70;

      return {
        id: `signal_${i + 1}`,
        symbol,
        action,
        type: Math.random() > 0.7 ? 'limit' : 'market',
        price: basePrice,
        targetPrice: action === 'buy' ? basePrice + 0.002 : basePrice - 0.002,
        stopLoss: action === 'buy' ? basePrice - 0.001 : basePrice + 0.001,
        confidence,
        strength,
        timeframe: timeframes[Math.floor(Math.random() * timeframes.length)],
        strategy: strategies[Math.floor(Math.random() * strategies.length)],
        reasoning: `Strong ${action} signal based on ${strategies[Math.floor(Math.random() * strategies.length)]} analysis. Market conditions favorable with ${confidence}% confidence.`,
        timestamp: new Date(Date.now() - Math.random() * 3600000),
        status: statuses[Math.floor(Math.random() * statuses.length)],
        session: sessions[Math.floor(Math.random() * sessions.length)],
        riskReward: Math.round((Math.random() * 2 + 1) * 10) / 10,
        expectedDuration: Math.floor(Math.random() * 120) + 30,
        volume: Math.floor(Math.random() * 10000) + 1000,
      };
    });
  };

  // Initialize signals
  useEffect(() => {
    const mockSignals = generateMockSignals();
    setSignals(mockSignals);
    setLastUpdate(new Date());
  }, []);

  // Filter signals
  useEffect(() => {
    let filtered = signals;

    if (filterStatus !== 'all') {
      filtered = filtered.filter(signal => signal.status === filterStatus);
    }

    if (filterSymbol !== 'all') {
      filtered = filtered.filter(signal => signal.symbol === filterSymbol);
    }

    if (filterAction !== 'all') {
      filtered = filtered.filter(signal => signal.action === filterAction);
    }

    // Sort by timestamp (newest first)
    filtered.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    // Limit to maxSignals
    filtered = filtered.slice(0, maxSignals);

    setFilteredSignals(filtered);
  }, [signals, filterStatus, filterSymbol, filterAction, maxSignals]);

  // Auto refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Simulate new signals
      if (Math.random() > 0.7) {
        const newSignal = generateMockSignals()[0];
        newSignal.id = `signal_${Date.now()}`;
        newSignal.timestamp = new Date();
        setSignals(prev => [newSignal, ...prev.slice(0, 19)]);
        setLastUpdate(new Date());
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleExecuteSignal = (signalId: string) => {
    setSignals(prev =>
      prev.map(signal =>
        signal.id === signalId
          ? { ...signal, status: 'executed' as const }
          : signal
      )
    );
  };

  const handleCancelSignal = (signalId: string) => {
    setSignals(prev =>
      prev.map(signal =>
        signal.id === signalId
          ? { ...signal, status: 'cancelled' as const }
          : signal
      )
    );
  };

  const handleSignalClick = (signal: TradingSignal) => {
    setSelectedSignal(signal);
    setIsDialogOpen(true);
  };

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => {
      const newSignals = generateMockSignals();
      setSignals(newSignals);
      setLastUpdate(new Date());
      setIsLoading(false);
    }, 1000);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Schedule color="warning" />;
      case 'active': return <PlayArrow color="info" />;
      case 'executed': return <CheckCircle color="success" />;
      case 'cancelled': return <Stop color="error" />;
      case 'expired': return <Error color="disabled" />;
      default: return <Warning />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'warning';
      case 'active': return 'info';
      case 'executed': return 'success';
      case 'cancelled': return 'error';
      case 'expired': return 'default';
      default: return 'default';
    }
  };

  const getSessionColor = (session: string) => {
    switch (session) {
      case 'asian': return '#ff9800';
      case 'london': return '#2196f3';
      case 'newyork': return '#4caf50';
      case 'overlap': return '#9c27b0';
      default: return '#757575';
    }
  };

  const uniqueSymbols = [...new Set(signals.map(s => s.symbol))];

  return (
    <Paper elevation={3} sx={{ p: 2, height }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline />
            Trading Signals
          </Typography>
          <Badge badgeContent={filteredSignals.filter(s => s.status === 'pending').length} color="warning">
            <Notifications />
          </Badge>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Last update: {lastUpdate.toLocaleTimeString()}
          </Typography>
          <Tooltip title="Refresh Signals">
            <IconButton onClick={handleRefresh} size="small" disabled={isLoading}>
              <Refresh />
            </IconButton>
          </Tooltip>
          <Tooltip title="Signal Settings">
            <IconButton size="small">
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {showFilters && (
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Status</InputLabel>
              <Select
                value={filterStatus}
                label="Status"
                onChange={(e) => setFilterStatus(e.target.value)}
              >
                <MenuItem value="all">All Status</MenuItem>
                <MenuItem value="pending">Pending</MenuItem>
                <MenuItem value="active">Active</MenuItem>
                <MenuItem value="executed">Executed</MenuItem>
                <MenuItem value="cancelled">Cancelled</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Symbol</InputLabel>
              <Select
                value={filterSymbol}
                label="Symbol"
                onChange={(e) => setFilterSymbol(e.target.value)}
              >
                <MenuItem value="all">All Symbols</MenuItem>
                {uniqueSymbols.map(symbol => (
                  <MenuItem key={symbol} value={symbol}>{symbol}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Action</InputLabel>
              <Select
                value={filterAction}
                label="Action"
                onChange={(e) => setFilterAction(e.target.value)}
              >
                <MenuItem value="all">All Actions</MenuItem>
                <MenuItem value="buy">Buy</MenuItem>
                <MenuItem value="sell">Sell</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                label={`${filteredSignals.length} signals`}
                color="primary"
                size="small"
              />
              <Chip
                label={`${filteredSignals.filter(s => s.status === 'pending').length} pending`}
                color="warning"
                size="small"
              />
            </Box>
          </Grid>
        </Grid>
      )}

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Signals List */}
      <Box sx={{ height: height - 200, overflow: 'auto' }}>
        <AnimatePresence>
          {filteredSignals.map((signal, index) => (
            <motion.div
              key={signal.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card
                elevation={2}
                sx={{
                  mb: 2,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: 4,
                  },
                  border: signal.status === 'pending' ? '2px solid #ff9800' : '1px solid rgba(255, 255, 255, 0.1)',
                }}
                onClick={() => handleSignalClick(signal)}
              >
                <CardContent sx={{ pb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="h6">{signal.symbol}</Typography>
                      <Chip
                        icon={signal.action === 'buy' ? <TrendingUp /> : <TrendingDown />}
                        label={signal.action.toUpperCase()}
                        color={signal.action === 'buy' ? 'success' : 'error'}
                        size="small"
                      />
                      <Chip
                        label={signal.timeframe}
                        size="small"
                        variant="outlined"
                      />
                    </Box>

                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip
                        icon={getStatusIcon(signal.status)}
                        label={signal.status.toUpperCase()}
                        color={getStatusColor(signal.status) as any}
                        size="small"
                      />
                    </Box>
                  </Box>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    {signal.strategy} • {signal.reasoning.substring(0, 80)}...
                  </Typography>

                  <Grid container spacing={2} sx={{ mb: 1 }}>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        <strong>Price:</strong> {signal.price.toFixed(5)}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Target:</strong> {signal.targetPrice?.toFixed(5)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        <strong>Stop Loss:</strong> {signal.stopLoss?.toFixed(5)}
                      </Typography>
                      <Typography variant="body2">
                        <strong>R/R:</strong> {signal.riskReward}
                      </Typography>
                    </Grid>
                  </Grid>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Chip
                        label={`${signal.confidence}% confidence`}
                        size="small"
                        color={signal.confidence >= 80 ? 'success' : signal.confidence >= 60 ? 'warning' : 'default'}
                      />
                      <Chip
                        label={signal.session}
                        size="small"
                        sx={{ bgcolor: getSessionColor(signal.session), color: 'white' }}
                      />
                    </Box>

                    <Typography variant="caption" color="text.secondary">
                      {signal.timestamp.toLocaleTimeString()}
                    </Typography>
                  </Box>
                </CardContent>

                {signal.status === 'pending' && (
                  <CardActions sx={{ pt: 0 }}>
                    <Button
                      size="small"
                      variant="contained"
                      color={signal.action === 'buy' ? 'success' : 'error'}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleExecuteSignal(signal.id);
                      }}
                      startIcon={<PlayArrow />}
                    >
                      Execute
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      color="error"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleCancelSignal(signal.id);
                      }}
                      startIcon={<Stop />}
                    >
                      Cancel
                    </Button>
                  </CardActions>
                )}
              </Card>
            </motion.div>
          ))}
        </AnimatePresence>

        {filteredSignals.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" color="text.secondary">
              No signals found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Try adjusting your filters or wait for new signals
            </Typography>
          </Box>
        )}
      </Box>

      {/* Signal Detail Dialog */}
      <Dialog
        open={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedSignal && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">
                  {selectedSignal.symbol} - {selectedSignal.action.toUpperCase()} Signal
                </Typography>
                <Chip
                  icon={getStatusIcon(selectedSignal.status)}
                  label={selectedSignal.status.toUpperCase()}
                  color={getStatusColor(selectedSignal.status) as any}
                />
              </Box>
            </DialogTitle>

            <DialogContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Signal Details
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemIcon><AttachMoney /></ListItemIcon>
                      <ListItemText
                        primary="Entry Price"
                        secondary={selectedSignal.price.toFixed(5)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><TrendingUp /></ListItemIcon>
                      <ListItemText
                        primary="Target Price"
                        secondary={selectedSignal.targetPrice?.toFixed(5)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><TrendingDown /></ListItemIcon>
                      <ListItemText
                        primary="Stop Loss"
                        secondary={selectedSignal.stopLoss?.toFixed(5)}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><Speed /></ListItemIcon>
                      <ListItemText
                        primary="Risk/Reward Ratio"
                        secondary={selectedSignal.riskReward}
                      />
                    </ListItem>
                  </List>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Analysis
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemIcon><Timeline /></ListItemIcon>
                      <ListItemText
                        primary="Strategy"
                        secondary={selectedSignal.strategy}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><CheckCircle /></ListItemIcon>
                      <ListItemText
                        primary="Confidence"
                        secondary={`${selectedSignal.confidence}%`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><Schedule /></ListItemIcon>
                      <ListItemText
                        primary="Expected Duration"
                        secondary={`${selectedSignal.expectedDuration} minutes`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><Notifications /></ListItemIcon>
                      <ListItemText
                        primary="Trading Session"
                        secondary={selectedSignal.session.toUpperCase()}
                      />
                    </ListItem>
                  </List>
                </Grid>

                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" gutterBottom>
                    Reasoning
                  </Typography>
                  <Alert severity="info" sx={{ mt: 1 }}>
                    {selectedSignal.reasoning}
                  </Alert>
                </Grid>
              </Grid>
            </DialogContent>

            <DialogActions>
              {selectedSignal.status === 'pending' && (
                <>
                  <Button
                    variant="contained"
                    color={selectedSignal.action === 'buy' ? 'success' : 'error'}
                    onClick={() => {
                      handleExecuteSignal(selectedSignal.id);
                      setIsDialogOpen(false);
                    }}
                    startIcon={<PlayArrow />}
                  >
                    Execute Signal
                  </Button>
                  <Button
                    variant="outlined"
                    color="error"
                    onClick={() => {
                      handleCancelSignal(selectedSignal.id);
                      setIsDialogOpen(false);
                    }}
                    startIcon={<Stop />}
                  >
                    Cancel Signal
                  </Button>
                </>
              )}
              <Button onClick={() => setIsDialogOpen(false)}>
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Paper>
  );
};

export default SignalBoard;
