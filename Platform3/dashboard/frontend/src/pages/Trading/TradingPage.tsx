import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  AccountBalance,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface MarketPrice {
  symbol: string;
  bid: string;
  ask: string;
  change: string;
  changePercent: string;
  volume: number;
  timestamp: string;
}

interface Position {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
}

const TradingPage: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [orderType, setOrderType] = useState('market');
  const [orderSide, setOrderSide] = useState('buy');
  const [orderSize, setOrderSize] = useState('10000');
  const [limitPrice, setLimitPrice] = useState('');
  const [marketPrices, setMarketPrices] = useState<MarketPrice[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Currency pairs available for trading
  const currencyPairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP'
  ];

  // Fetch market prices from our Market Data Service
  useEffect(() => {
    const fetchMarketPrices = async () => {
      try {
        const symbols = currencyPairs.join(',');
        const marketDataUrl = import.meta.env.VITE_MARKET_DATA_URL || 'http://localhost:3004';
        const response = await fetch(`${marketDataUrl}/api/market-data/prices?symbols=${symbols}`);
        
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data) {
            setMarketPrices(data.data);
          }
        }
      } catch (error) {
        console.error('Error fetching market prices:', error);
      }
    };

    // Initial fetch
    fetchMarketPrices();

    // Update prices every 5 seconds
    const interval = setInterval(fetchMarketPrices, 5000);

    return () => clearInterval(interval);
  }, []);

  // Mock positions data
  useEffect(() => {
    setPositions([
      {
        id: '1',
        symbol: 'EURUSD',
        side: 'buy',
        size: 10000,
        entryPrice: 1.0750,
        currentPrice: 1.0785,
        unrealizedPL: 350.00,
        unrealizedPLPercent: 0.33,
      },
      {
        id: '2',
        symbol: 'GBPUSD',
        side: 'sell',
        size: 15000,
        entryPrice: 1.2550,
        currentPrice: 1.2520,
        unrealizedPL: 450.00,
        unrealizedPLPercent: 0.24,
      },
    ]);
  }, []);

  const handlePlaceOrder = async () => {
    setIsLoading(true);
    try {
      // Simulate order placement
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // In a real implementation, you would call the Trading Service API
      console.log('Order placed:', {
        symbol: selectedSymbol,
        type: orderType,
        side: orderSide,
        size: orderSize,
        price: limitPrice || 'market',
      });

      // Reset form
      setOrderSize('10000');
      setLimitPrice('');
      
      alert('Order placed successfully!');
    } catch (error) {
      console.error('Error placing order:', error);
      alert('Failed to place order');
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentPrice = (symbol: string) => {
    const price = marketPrices.find(p => p.symbol === symbol);
    return price ? { bid: parseFloat(price.bid), ask: parseFloat(price.ask) } : null;
  };

  const selectedPrice = getCurrentPrice(selectedSymbol);

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
          <ShowChart sx={{ mr: 2, verticalAlign: 'middle' }} />
          Trading Platform
        </Typography>

        <Grid container spacing={3}>
          {/* Order Placement Panel */}
          <Grid item xs={12} md={4}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Place Order
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Currency Pair</InputLabel>
                <Select
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                  label="Currency Pair"
                >
                  {currencyPairs.map((pair) => (
                    <MenuItem key={pair} value={pair}>
                      {pair}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedPrice && (
                <Box sx={{ mb: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Current Price
                  </Typography>
                  <Box display="flex" justifyContent="space-between">
                    <Typography variant="body1">
                      Bid: <strong>{selectedPrice.bid.toFixed(5)}</strong>
                    </Typography>
                    <Typography variant="body1">
                      Ask: <strong>{selectedPrice.ask.toFixed(5)}</strong>
                    </Typography>
                  </Box>
                </Box>
              )}

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Order Type</InputLabel>
                <Select
                  value={orderType}
                  onChange={(e) => setOrderType(e.target.value)}
                  label="Order Type"
                >
                  <MenuItem value="market">Market Order</MenuItem>
                  <MenuItem value="limit">Limit Order</MenuItem>
                </Select>
              </FormControl>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Side</InputLabel>
                <Select
                  value={orderSide}
                  onChange={(e) => setOrderSide(e.target.value)}
                  label="Side"
                >
                  <MenuItem value="buy">Buy</MenuItem>
                  <MenuItem value="sell">Sell</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Size (Units)"
                value={orderSize}
                onChange={(e) => setOrderSize(e.target.value)}
                sx={{ mb: 2 }}
                type="number"
              />

              {orderType === 'limit' && (
                <TextField
                  fullWidth
                  label="Limit Price"
                  value={limitPrice}
                  onChange={(e) => setLimitPrice(e.target.value)}
                  sx={{ mb: 2 }}
                  type="number"
                  step="0.00001"
                />
              )}

              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={handlePlaceOrder}
                disabled={isLoading}
                color={orderSide === 'buy' ? 'success' : 'error'}
                sx={{ py: 1.5 }}
              >
                {isLoading ? 'Placing Order...' : `${orderSide.toUpperCase()} ${selectedSymbol}`}
              </Button>
            </Paper>
          </Grid>

          {/* Market Prices */}
          <Grid item xs={12} md={8}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Live Market Prices
              </Typography>

              {marketPrices.length === 0 ? (
                <Alert severity="info">
                  Connecting to Market Data Service... Make sure the service is running on port 3004.
                </Alert>
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell align="right">Bid</TableCell>
                        <TableCell align="right">Ask</TableCell>
                        <TableCell align="right">Change</TableCell>
                        <TableCell align="right">Volume</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {marketPrices.map((price) => (
                        <TableRow
                          key={price.symbol}
                          sx={{
                            backgroundColor: selectedSymbol === price.symbol ? 'action.selected' : 'inherit',
                            cursor: 'pointer',
                            '&:hover': {
                              backgroundColor: 'action.hover',
                            },
                          }}
                          onClick={() => setSelectedSymbol(price.symbol)}
                        >
                          <TableCell component="th" scope="row">
                            <strong>{price.symbol}</strong>
                          </TableCell>
                          <TableCell align="right">{parseFloat(price.bid).toFixed(5)}</TableCell>
                          <TableCell align="right">{parseFloat(price.ask).toFixed(5)}</TableCell>
                          <TableCell align="right">
                            <Box display="flex" alignItems="center" justifyContent="flex-end">
                              {parseFloat(price.changePercent) >= 0 ? (
                                <TrendingUp color="success" sx={{ mr: 1 }} />
                              ) : (
                                <TrendingDown color="error" sx={{ mr: 1 }} />
                              )}
                              <Chip
                                label={`${price.changePercent}%`}
                                size="small"
                                color={parseFloat(price.changePercent) >= 0 ? 'success' : 'error'}
                              />
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            {price.volume.toLocaleString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>
          </Grid>

          {/* Open Positions */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                <AccountBalance sx={{ mr: 1, verticalAlign: 'middle' }} />
                Open Positions
              </Typography>

              {positions.length === 0 ? (
                <Alert severity="info">
                  No open positions. Place your first trade to get started!
                </Alert>
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Side</TableCell>
                        <TableCell align="right">Size</TableCell>
                        <TableCell align="right">Entry Price</TableCell>
                        <TableCell align="right">Current Price</TableCell>
                        <TableCell align="right">Unrealized P&L</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {positions.map((position) => (
                        <TableRow key={position.id}>
                          <TableCell component="th" scope="row">
                            <strong>{position.symbol}</strong>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={position.side.toUpperCase()}
                              size="small"
                              color={position.side === 'buy' ? 'success' : 'error'}
                            />
                          </TableCell>
                          <TableCell align="right">
                            {position.size.toLocaleString()}
                          </TableCell>
                          <TableCell align="right">
                            {position.entryPrice.toFixed(5)}
                          </TableCell>
                          <TableCell align="right">
                            {position.currentPrice.toFixed(5)}
                          </TableCell>
                          <TableCell align="right">
                            <Box
                              sx={{
                                color: position.unrealizedPL >= 0 ? 'success.main' : 'error.main',
                                fontWeight: 'bold',
                              }}
                            >
                              ${position.unrealizedPL.toFixed(2)}
                              <br />
                              <Typography variant="caption">
                                ({position.unrealizedPLPercent >= 0 ? '+' : ''}{position.unrealizedPLPercent.toFixed(2)}%)
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Button
                              variant="outlined"
                              size="small"
                              color="error"
                              onClick={() => {
                                // Close position logic
                                console.log('Close position:', position.id);
                              }}
                            >
                              Close
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>
          </Grid>
        </Grid>
      </motion.div>
    </Box>
  );
};

export default TradingPage;
