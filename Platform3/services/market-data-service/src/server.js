require('dotenv').config();

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const WebSocket = require('ws');
const cron = require('node-cron');

// Import custom modules
const { pool, testConnection, closePool } = require('./config/database');
const marketDataRoutes = require('./routes/marketData');
const { requestLogger, errorHandler } = require('./middleware');
const MarketDataProvider = require('./services/MarketDataProvider');
const WebSocketServer = require('./services/WebSocketServer');

const app = express();
const PORT = process.env.PORT || 3004;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 200, // Higher limit for market data requests
  message: 'Too many requests from this IP, please try again later'
});
app.use(limiter);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use(requestLogger);

// Routes
app.use('/api/market-data', marketDataRoutes);

// Initialize market data provider
const marketDataProvider = new MarketDataProvider(pool);

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const dbStatus = await testConnection();
    const marketDataStatus = await marketDataProvider.checkConnection();
    
    res.status(dbStatus ? 200 : 503).json({
      status: (dbStatus && marketDataStatus) ? 'healthy' : 'unhealthy',
      service: 'market-data-service',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      database: dbStatus ? 'connected' : 'disconnected',
      market_data: marketDataStatus ? 'connected' : 'disconnected'
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      service: 'market-data-service',
      error: error.message
    });
  }
});

// Service info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    service: 'market-data-service',
    version: '1.0.0',
    description: 'Provides real-time and historical forex market data',
    endpoints: {
      'market-data': '/api/market-data',
      prices: '/api/market-data/prices',
      history: '/api/market-data/history',
      stats: '/api/market-data/stats',
      instruments: '/api/market-data/instruments',
      websocket: '/ws',
      health: '/health'
    },
    features: [
      'Real-time price feeds',
      'Historical data storage',
      'WebSocket streaming',
      'Technical indicators',
      'Multiple data sources'
    ]
  });
});

// Error handling middleware
app.use(errorHandler);

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Endpoint not found'
  });
});

// Initialize WebSocket server
let wsServer;

// Schedule periodic data updates
cron.schedule('*/10 * * * * *', async () => {
  // Update prices every 10 seconds
  try {
    await marketDataProvider.fetchAndStorePrices();
  } catch (error) {
    console.error('Error updating market data:', error.message);
  }
});

// Graceful shutdown
const gracefulShutdown = async (signal) => {
  console.log(`\n📶 Market Data Service: Received ${signal}, shutting down gracefully...`);
  
  // Close WebSocket server
  if (wsServer) {
    wsServer.close();
  }
  
  // Close database pool
  await closePool();
  
  server.close(() => {
    console.log('🔌 Market Data Service: HTTP server closed');
    process.exit(0);
  });
  
  // Force shutdown after 10 seconds
  setTimeout(() => {
    console.error('⚠️ Market Data Service: Forced shutdown after timeout');
    process.exit(1);
  }, 10000);
};

// Initialize and start server
const startServer = async () => {
  try {
    // Test database connection
    const dbConnected = await testConnection();
    if (!dbConnected) {
      console.warn('⚠️ Market Data Service: Starting without database connection');
    }

    // Initialize market data provider
    await marketDataProvider.initialize();

    const server = app.listen(PORT, () => {
      console.log(`🚀 Market Data Service running on port ${PORT}`);
      console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`🔗 Database: ${dbConnected ? 'Connected' : 'Disconnected'}`);
    });

    // Initialize WebSocket server
    wsServer = new WebSocketServer(server, marketDataProvider);
    console.log('🔄 WebSocket server initialized for real-time data streaming');

    // Signal handlers
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));

    return server;
  } catch (error) {
    console.error('❌ Market Data Service: Failed to start server:', error.message);
    process.exit(1);
  }
};

// Start the server
startServer();

module.exports = app;
