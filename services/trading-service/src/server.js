require('dotenv').config();

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

// Import custom modules
const { pool, testConnection, closePool } = require('./config/database');
const { router: tradesRouter, initializeTradeRoutes } = require('./routes/trades');
const { router: portfolioRouter, initializePortfolioRoutes } = require('./routes/portfolio');
const { requestLogger, errorHandler } = require('./middleware/auth');

const app = express();
const PORT = process.env.PORT || 3003;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later'
});
app.use(limiter);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use(requestLogger);

// Initialize database-dependent routes
initializeTradeRoutes(pool);
initializePortfolioRoutes(pool);

// Health check endpoint
app.get('/health', async (req, res) => {
  const dbStatus = await testConnection();
  
  res.status(dbStatus ? 200 : 503).json({
    status: dbStatus ? 'healthy' : 'unhealthy',
    service: 'trading-service',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    database: dbStatus ? 'connected' : 'disconnected'
  });
});

// API routes
app.use('/api/trades', tradesRouter);
app.use('/api/portfolio', portfolioRouter);

// Service info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    service: 'trading-service',
    version: '1.0.0',
    description: 'Handles trading operations and portfolio management',
    endpoints: {
      trades: '/api/trades',
      portfolio: '/api/portfolio',
      health: '/health'
    },
    features: [
      'Trade creation and management',
      'Portfolio balance tracking',
      'Trading statistics',
      'Real-time trade execution'
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

// Graceful shutdown
const gracefulShutdown = async (signal) => {
  console.log(`\n📶 Trading Service: Received ${signal}, shutting down gracefully...`);
  
  // Close database pool
  await closePool();
  
  server.close(() => {
    console.log('🔌 Trading Service: HTTP server closed');
    process.exit(0);
  });
  
  // Force shutdown after 10 seconds
  setTimeout(() => {
    console.error('⚠️ Trading Service: Forced shutdown after timeout');
    process.exit(1);
  }, 10000);
};

// Initialize and start server
const startServer = async () => {
  try {
    // Test database connection
    const dbConnected = await testConnection();
    if (!dbConnected) {
      console.warn('⚠️ Trading Service: Starting without database connection');
    }

    const server = app.listen(PORT, () => {
      console.log(`🚀 Trading Service running on port ${PORT}`);
      console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`🔗 Database: ${dbConnected ? 'Connected' : 'Disconnected'}`);
    });

    // Signal handlers
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));

    return server;
  } catch (error) {
    console.error('❌ Trading Service: Failed to start server:', error.message);
    process.exit(1);
  }
};

// Start the server
startServer();

module.exports = app;
