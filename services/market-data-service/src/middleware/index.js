const rateLimit = require('express-rate-limit');

/**
 * Rate limiting middleware for market data endpoints
 * More permissive than trading endpoints since market data is frequently accessed
 */
const marketDataLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 300, // Limit each IP to 300 requests per windowMs
  message: {
    success: false,
    error: 'Too many requests, please try again later.',
    retryAfter: '1 minute'
  },
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
});

/**
 * Rate limiting for WebSocket connections
 */
const websocketLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 10, // Limit each IP to 10 WebSocket connection attempts per minute
  message: {
    success: false,
    error: 'Too many WebSocket connection attempts, please try again later.',
    retryAfter: '1 minute'
  },
  skip: (req) => {
    // Skip rate limiting for existing connections (upgrade requests)
    return req.headers.upgrade === 'websocket';
  }
});

/**
 * CORS configuration for market data service
 */
const corsOptions = {
  origin: function (origin, callback) {
    // Allow requests with no origin (mobile apps, curl, etc.)
    if (!origin) return callback(null, true);
    
    const allowedOrigins = [
      'http://localhost:3000',  // AI Dashboard
      'http://localhost:3001',  // Frontend
      'http://localhost:3002',  // Any other frontend
      'http://localhost:5173',  // Vite dev server
      'https://platform3-frontend.vercel.app', // Production frontend
    ];
    
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};

/**
 * Security headers middleware
 */
const securityHeaders = (req, res, next) => {
  // Set security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  
  // Remove X-Powered-By header
  res.removeHeader('X-Powered-By');
  
  next();
};

/**
 * Request logging middleware
 */
const requestLogger = (req, res, next) => {
  const start = Date.now();
  const timestamp = new Date().toISOString();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const logData = {
      timestamp,
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('User-Agent'),
      ip: req.ip || req.connection.remoteAddress
    };
    
    console.log(`[${timestamp}] ${req.method} ${req.url} - ${res.statusCode} (${duration}ms)`);
  });
  
  next();
};

/**
 * Error handling middleware
 */
const errorHandler = (err, req, res, next) => {
  console.error('Market Data Service Error:', {
    error: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
    timestamp: new Date().toISOString()
  });

  // CORS error
  if (err.message === 'Not allowed by CORS') {
    return res.status(403).json({
      success: false,
      error: 'CORS policy violation'
    });
  }

  // Rate limiting error
  if (err.status === 429) {
    return res.status(429).json({
      success: false,
      error: 'Rate limit exceeded'
    });
  }

  // Default error response
  res.status(err.status || 500).json({
    success: false,
    error: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : err.message
  });
};

/**
 * 404 handler for unknown routes
 */
const notFoundHandler = (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    availableEndpoints: [
      'GET /health',
      'GET /api/info',
      'GET /api/market-data/prices',
      'GET /api/market-data/prices/:symbol',
      'GET /api/market-data/history/:symbol',
      'GET /api/market-data/stats/:symbol',
      'GET /api/market-data/instruments',
      'WebSocket /ws'
    ]
  });
};

module.exports = {
  marketDataLimiter,
  websocketLimiter,
  corsOptions,
  securityHeaders,
  requestLogger,
  errorHandler,
  notFoundHandler
};
