require('dotenv').config();

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');

const app = express();
const PORT = process.env.PORT || 3004;

// Enhanced Market Data Provider
class MarketDataProvider {
  constructor() {
    this.pairs = [
      'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
      'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD', 'GBPAUD', 'EURCHF'
    ];
    this.lastPrices = {};
    this.priceMomentum = {};
  }

  async generateMockPrices() {
    const baseRates = {
      'EURUSD': 1.0950, 'GBPUSD': 1.2750, 'USDJPY': 149.50, 'USDCHF': 0.8850,
      'AUDUSD': 0.6650, 'USDCAD': 1.3550, 'NZDUSD': 0.6100, 'EURGBP': 0.8600,
      'EURJPY': 163.80, 'GBPJPY': 190.50, 'AUDJPY': 99.40, 'EURAUD': 1.6475,
      'GBPAUD': 1.9180, 'EURCHF': 0.9695
    };

    const prices = [];
    const now = new Date();
    const isMarketOpen = this.isMarketOpen(now);
    const volatilityMultiplier = isMarketOpen ? 1.0 : 0.3;
    
    for (const [symbol, basePrice] of Object.entries(baseRates)) {
      const previousPrice = this.lastPrices?.[symbol] || basePrice;
      const momentum = this.priceMomentum?.[symbol] || 0;
      const shouldContinueTrend = Math.random() > 0.3;
      
      let variation = (Math.random() - 0.5) * 0.002 * volatilityMultiplier;
      if (shouldContinueTrend && Math.abs(momentum) > 0.0001) {
        variation += momentum * 0.5;
      }
      
      let currentPrice = previousPrice * (1 + variation);
      const maxDeviation = basePrice * 0.02;
      currentPrice = Math.max(basePrice - maxDeviation, 
                             Math.min(basePrice + maxDeviation, currentPrice));
      
      const baseSpread = this.getSpreadForPair(symbol, currentPrice);
      const volatilitySpread = Math.abs(variation) * currentPrice * 2;
      const totalSpread = baseSpread + volatilitySpread;
      
      const bid = currentPrice - totalSpread / 2;
      const ask = currentPrice + totalSpread / 2;

      const priceData = {
        symbol,
        bid: bid.toFixed(symbol.includes('JPY') ? 3 : 5),
        ask: ask.toFixed(symbol.includes('JPY') ? 3 : 5),
        change: ((currentPrice - basePrice) / basePrice * 100).toFixed(4),
        changePercent: ((currentPrice - basePrice) / basePrice * 100).toFixed(2),
        volume: Math.floor(Math.random() * 10000000) + 1000000,
        timestamp: now
      };

      prices.push(priceData);
      this.lastPrices[symbol] = currentPrice;
      this.priceMomentum[symbol] = variation;
    }

    return prices;
  }

  getSpreadForPair(symbol) {
    const spreads = {
      'EURUSD': 0.00015, 'GBPUSD': 0.00020, 'USDJPY': 0.015, 'USDCHF': 0.00020,
      'AUDUSD': 0.00025, 'USDCAD': 0.00025, 'NZDUSD': 0.00030, 'EURGBP': 0.00025,
      'EURJPY': 0.020, 'GBPJPY': 0.030, 'AUDJPY': 0.025, 'EURAUD': 0.00035,
      'GBPAUD': 0.00040, 'EURCHF': 0.00030
    };
    return spreads[symbol] || 0.00030;
  }

  isMarketOpen(date) {
    return true; // For demo, always open
  }

  calculateSMA(prices, period) {
    if (prices.length < period) return null;
    const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
    return (sum / period).toFixed(5);
  }

  calculateRSI(prices, period) {
    if (prices.length < period + 1) return null;
    
    const gains = [];
    const losses = [];
    
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    const avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    
    return rsi.toFixed(2);
  }

  getSessionStatus(session, date) {
    const hour = date.getUTCHours();
    const sessions = {
      sydney: { start: 21, end: 6 },
      tokyo: { start: 23, end: 8 },
      london: { start: 7, end: 16 },
      newYork: { start: 12, end: 21 }
    };

    const sessionTimes = sessions[session];
    if (!sessionTimes) return { active: false };

    let isActive;
    if (sessionTimes.start > sessionTimes.end) {
      isActive = hour >= sessionTimes.start || hour < sessionTimes.end;
    } else {
      isActive = hour >= sessionTimes.start && hour < sessionTimes.end;
    }

    return { active: isActive };
  }
}

// Initialize provider
const marketDataProvider = new MarketDataProvider();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001', 'http://localhost:5173'],
  credentials: true
}));

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    success: true,
    service: 'Market Data Service',
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    features: ['enhanced_pricing', 'technical_indicators', 'market_sessions']
  });
});

// Current prices endpoint
app.get('/api/market-data/prices', async (req, res) => {
  try {
    const prices = await marketDataProvider.generateMockPrices();
    res.json({
      success: true,
      data: prices,
      count: prices.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch current prices',
      message: error.message
    });
  }
});

// Specific price endpoint
app.get('/api/market-data/prices/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const prices = await marketDataProvider.generateMockPrices();
    const price = prices.find(p => p.symbol === symbol.toUpperCase());
    
    if (!price) {
      return res.status(404).json({
        success: false,
        error: 'Currency pair not found'
      });
    }

    res.json({
      success: true,
      data: price,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price data',
      message: error.message
    });
  }
});

// Market status endpoint
app.get('/api/market-data/market-status', (req, res) => {
  try {
    const now = new Date();
    const marketStatus = {
      timestamp: now.toISOString(),
      utcTime: now.toUTCString(),
      sessions: {
        sydney: marketDataProvider.getSessionStatus('sydney', now),
        tokyo: marketDataProvider.getSessionStatus('tokyo', now),
        london: marketDataProvider.getSessionStatus('london', now),
        newYork: marketDataProvider.getSessionStatus('newYork', now)
      },
      isWeekend: now.getUTCDay() === 0 || now.getUTCDay() === 6,
      serverTime: now.toISOString(),
      marketVolatility: 'normal'
    };

    res.json({
      success: true,
      data: marketStatus,
      timestamp: now.toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to get market status',
      message: error.message
    });
  }
});

// Technical indicators endpoint
app.get('/api/market-data/indicators/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { period = 14 } = req.query;
    
    // Generate sample historical data for calculation
    const samplePrices = Array.from({ length: 50 }, (_, i) => {
      const basePrice = 1.0950; // EURUSD base
      const variation = (Math.random() - 0.5) * 0.01;
      return basePrice * (1 + variation);
    });

    const indicators = {
      symbol: symbol.toUpperCase(),
      timeframe: '1h',
      timestamp: new Date().toISOString(),
      sma: marketDataProvider.calculateSMA(samplePrices, parseInt(period)),
      rsi: marketDataProvider.calculateRSI(samplePrices, parseInt(period)),
      period: parseInt(period)
    };

    res.json({
      success: true,
      data: indicators,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to calculate technical indicators',
      message: error.message
    });
  }
});

// Watchlist endpoint
app.get('/api/market-data/watchlist', async (req, res) => {
  try {
    const { symbols } = req.query;
    let targetSymbols = symbols ? symbols.split(',').map(s => s.toUpperCase()) : null;
    
    const prices = await marketDataProvider.generateMockPrices();
    let watchlist = prices;
    
    if (targetSymbols) {
      watchlist = prices.filter(price => targetSymbols.includes(price.symbol));
    }
    
    // Enhance with basic indicators
    const enhancedWatchlist = watchlist.map(price => ({
      ...price,
      trend: parseFloat(price.changePercent) > 0 ? 'bullish' : 'bearish',
      volatility: 'normal'
    }));

    res.json({
      success: true,
      data: enhancedWatchlist,
      count: enhancedWatchlist.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to get watchlist data',
      message: error.message
    });
  }
});

// API info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    service: 'Market Data Service',
    version: '1.0.0',
    status: 'running',
    features: [
      'Real-time price feeds',
      'Technical indicators',
      'Market session tracking',
      'Enhanced price simulation',
      'WebSocket streaming (planned)'
    ],
    endpoints: [
      'GET /health',
      'GET /api/market-data/prices',
      'GET /api/market-data/prices/:symbol',
      'GET /api/market-data/market-status',
      'GET /api/market-data/indicators/:symbol',
      'GET /api/market-data/watchlist'
    ],
    timestamp: new Date().toISOString()
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Market Data Service Error:', error);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    availableEndpoints: [
      'GET /health',
      'GET /api/market-data/prices',
      'GET /api/market-data/prices/:symbol',
      'GET /api/market-data/market-status',
      'GET /api/market-data/indicators/:symbol',
      'GET /api/market-data/watchlist',
      'GET /api/info'
    ]
  });
});

// Start server
app.listen(PORT, () => {
  console.log('🚀 Market Data Service started successfully!');
  console.log(`📡 Server running on port ${PORT}`);
  console.log(`🌐 Health check: http://localhost:${PORT}/health`);
  console.log(`📊 API endpoints: http://localhost:${PORT}/api/info`);
  console.log(`📈 Sample prices: http://localhost:${PORT}/api/market-data/prices`);
  console.log(`🎯 Market status: http://localhost:${PORT}/api/market-data/market-status`);
  console.log('✅ Ready to serve your AI Dashboard!');
});
