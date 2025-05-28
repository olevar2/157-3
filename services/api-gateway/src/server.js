require('dotenv').config();

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const { createProxyMiddleware } = require('http-proxy-middleware');

// Import custom middleware
const authMiddleware = require('./middleware/auth');
const logger = require('./utils/logger');
const healthCheck = require('./middleware/healthCheck');
const ServiceDiscoveryMiddleware = require('./middleware/serviceDiscovery');
const mTLSMiddleware = require('./middleware/mtls');

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize service discovery and mTLS
const serviceDiscovery = new ServiceDiscoveryMiddleware();
const mtlsHandler = new mTLSMiddleware({
  caCertPath: process.env.CA_CERT_PATH || './certs/ca-cert.pem',
  serverCertPath: process.env.SERVER_CERT_PATH || './certs/api-gateway-cert.pem',
  serverKeyPath: process.env.SERVER_KEY_PATH || './certs/api-gateway-key.pem',
  requireClientCert: process.env.REQUIRE_CLIENT_CERT !== 'false'
});

// Trust proxy (important for rate limiting behind reverse proxy)
app.set('trust proxy', 1);

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:"]
    }
  }
}));

// CORS configuration
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || [
    'http://localhost:3000',
    'http://localhost:3001',
    'http://localhost:3002'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

// Compression and logging
app.use(compression());
app.use(morgan('combined', { stream: { write: (msg) => logger.info(msg.trim()) } }));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting configurations
const createRateLimiter = (windowMs, max, message) => rateLimit({
  windowMs,
  max,
  message: { error: message },
  standardHeaders: true,
  legacyHeaders: false,
  handler: (req, res) => {
    logger.warn(`Rate limit exceeded for IP: ${req.ip}`);
    res.status(429).json({ error: message });
  }
});

// Different rate limits for different endpoints
const generalLimiter = createRateLimiter(15 * 60 * 1000, 1000, 'Too many requests, please try again later');
const authLimiter = createRateLimiter(15 * 60 * 1000, 50, 'Too many authentication attempts');
const tradingLimiter = createRateLimiter(60 * 1000, 100, 'Trading API rate limit exceeded');
const marketDataLimiter = createRateLimiter(60 * 1000, 200, 'Market data API rate limit exceeded');

// Apply general rate limiting
app.use(generalLimiter);

// Apply service discovery and mTLS middleware
app.use(serviceDiscovery.resolveService());
app.use(mtlsHandler.validateClientCertificate());

// Health check endpoint (no auth required)
app.get('/health', async (req, res) => {
  const serviceDiscoveryHealth = await serviceDiscovery.healthCheck();
  const mtlsHealth = mtlsHandler.healthCheck();
  const health = await healthCheck(req, res, () => {});

  // If health check passed, also check service discovery and mTLS
  if (res.statusCode === 200) {
    const overallHealthy = serviceDiscoveryHealth && mtlsHealth.status === 'healthy';

    const healthData = {
      status: overallHealthy ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      service: 'api-gateway',
      serviceDiscovery: {
        status: serviceDiscoveryHealth ? 'connected' : 'disconnected',
        cacheStats: serviceDiscovery.getCacheStats()
      },
      mTLS: mtlsHealth
    };

    res.status(overallHealthy ? 200 : 200).json(healthData);
  }
});

// API Gateway info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    service: 'api-gateway',
    version: '1.0.0',
    description: 'API Gateway for Forex Trading Platform',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    services: {
      'auth-service': process.env.AUTH_SERVICE_URL || 'http://localhost:3001',
      'user-service': process.env.USER_SERVICE_URL || 'http://localhost:3002',
      'trading-service': process.env.TRADING_SERVICE_URL || 'http://localhost:3003',
      'market-data-service': process.env.MARKET_DATA_SERVICE_URL || 'http://localhost:3004'
    },
    features: [
      'Request routing',
      'Authentication & authorization',
      'Rate limiting',
      'CORS handling',
      'Security headers',
      'Request/response logging',
      'Health monitoring'
    ]
  });
});

// Authentication endpoints (high security)
app.use('/api/auth', authLimiter, createProxyMiddleware({
  target: 'http://placeholder', // Will be replaced by router
  changeOrigin: true,
  pathRewrite: {
    '^/api/auth': '/api/auth'
  },
  router: async (req) => {
    return req.serviceUrl || process.env.AUTH_SERVICE_URL || 'http://localhost:3001';
  },
  onProxyReq: (proxyReq, req, res) => {
    logger.info(`Proxying auth request: ${req.method} ${req.path} to ${req.serviceUrl}`);
  },
  onError: (err, req, res) => {
    logger.error(`Auth service proxy error: ${err.message}`);
    res.status(502).json({
      error: 'Authentication service unavailable',
      timestamp: new Date().toISOString()
    });
  }
}));

// User management endpoints (authentication required)
app.use('/api/users',
  authMiddleware.validateToken,
  createProxyMiddleware({
    target: 'http://placeholder', // Will be replaced by router
    changeOrigin: true,
    pathRewrite: {
      '^/api/users': '/api/users'
    },
    router: async (req) => {
      return req.serviceUrl || process.env.USER_SERVICE_URL || 'http://localhost:3002';
    },
    onProxyReq: (proxyReq, req, res) => {
      // Forward user info from JWT
      if (req.user) {
        proxyReq.setHeader('X-User-ID', req.user.id);
        proxyReq.setHeader('X-User-Role', req.user.role);
      }
      logger.info(`Proxying user request: ${req.method} ${req.path} to ${req.serviceUrl}`);
    },
    onError: (err, req, res) => {
      logger.error(`User service proxy error: ${err.message}`);
      res.status(502).json({
        error: 'User service unavailable',
        timestamp: new Date().toISOString()
      });
    }
  })
);

// Trading endpoints (authentication + rate limiting)
app.use('/api/trading',
  tradingLimiter,
  authMiddleware.validateToken,
  createProxyMiddleware({
    target: 'http://placeholder', // Will be replaced by router
    changeOrigin: true,
    pathRewrite: {
      '^/api/trading': '/api'
    },
    router: async (req) => {
      return req.serviceUrl || process.env.TRADING_SERVICE_URL || 'http://localhost:3003';
    },
    onProxyReq: (proxyReq, req, res) => {
      // Forward user info from JWT
      if (req.user) {
        proxyReq.setHeader('X-User-ID', req.user.id);
        proxyReq.setHeader('X-User-Role', req.user.role);
      }
      logger.info(`Proxying trading request: ${req.method} ${req.path} to ${req.serviceUrl}`);
    },
    onError: (err, req, res) => {
      logger.error(`Trading service proxy error: ${err.message}`);
      res.status(502).json({
        error: 'Trading service unavailable',
        timestamp: new Date().toISOString()
      });
    }
  })
);

// Market data endpoints (authentication + higher rate limits)
app.use('/api/market-data',
  marketDataLimiter,
  authMiddleware.validateToken,
  createProxyMiddleware({
    target: 'http://placeholder', // Will be replaced by router
    changeOrigin: true,
    pathRewrite: {
      '^/api/market-data': '/api/market-data'
    },
    router: async (req) => {
      return req.serviceUrl || process.env.MARKET_DATA_SERVICE_URL || 'http://localhost:3004';
    },
    onProxyReq: (proxyReq, req, res) => {
      if (req.user) {
        proxyReq.setHeader('X-User-ID', req.user.id);
        proxyReq.setHeader('X-User-Role', req.user.role);
      }
      logger.info(`Proxying market data request: ${req.method} ${req.path} to ${req.serviceUrl}`);
    },
    onError: (err, req, res) => {
      logger.error(`Market data service proxy error: ${err.message}`);
      res.status(502).json({
        error: 'Market data service unavailable',
        timestamp: new Date().toISOString()
      });
    }
  })
);

// WebSocket proxy for real-time data
app.use('/ws', createProxyMiddleware({
  target: process.env.MARKET_DATA_SERVICE_URL || 'http://localhost:3004',
  ws: true,
  changeOrigin: true,
  onError: (err, req, socket) => {
    logger.error(`WebSocket proxy error: ${err.message}`);
  }
}));

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error(`Unhandled error: ${err.message}`, {
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip
  });

  res.status(err.status || 500).json({
    error: process.env.NODE_ENV === 'production'
      ? 'Internal server error'
      : err.message,
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  logger.warn(`404 - Route not found: ${req.method} ${req.originalUrl}`);
  res.status(404).json({
    error: 'Route not found',
    timestamp: new Date().toISOString(),
    path: req.originalUrl
  });
});

// Graceful shutdown
const gracefulShutdown = (signal) => {
  logger.info(`Received ${signal}, shutting down gracefully...`);

  server.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });

  // Force shutdown after 10 seconds
  setTimeout(() => {
    logger.error('Forced shutdown after timeout');
    process.exit(1);
  }, 10000);
};

// Start server with HTTPS if certificates are available
const https = require('https');
const httpsOptions = mtlsHandler.getHttpsOptions();

let server;
if (httpsOptions && process.env.ENABLE_HTTPS !== 'false') {
  // Start HTTPS server with mTLS
  server = https.createServer(httpsOptions, app).listen(PORT, async () => {
    logger.info(`🚀 API Gateway running on HTTPS port ${PORT} with mTLS`);
    logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
    logger.info(`🔐 Authentication: ${process.env.JWT_SECRET ? 'Enabled' : 'Disabled'}`);
    logger.info(`🔒 mTLS: Enabled with client certificate validation`);

    // Register with service discovery
    try {
      await serviceDiscovery.registerSelf();
      logger.info('✅ API Gateway registered with service discovery');
    } catch (error) {
      logger.warn('⚠️ Failed to register with service discovery, continuing without it', {
        error: error.message
      });
    }
  });
} else {
  // Fallback to HTTP server
  server = app.listen(PORT, async () => {
    logger.info(`🚀 API Gateway running on HTTP port ${PORT}`);
    logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
    logger.info(`🔐 Authentication: ${process.env.JWT_SECRET ? 'Enabled' : 'Disabled'}`);
    logger.warn('⚠️ mTLS: Disabled - certificates not found or HTTPS disabled');

    // Register with service discovery
    try {
      await serviceDiscovery.registerSelf();
      logger.info('✅ API Gateway registered with service discovery');
    } catch (error) {
      logger.warn('⚠️ Failed to register with service discovery, continuing without it', {
        error: error.message
      });
    }
  });
}

// Signal handlers
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

module.exports = app;
