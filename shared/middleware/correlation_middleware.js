// Platform3 Correlation Middleware
const { v4: uuidv4 } = require('uuid');
const logger = require('../logging/platform3_logger');

/**
 * Correlation ID middleware for request tracking
 */
function correlationMiddleware(req, res, next) {
    // Generate or extract correlation ID
    const correlationId = req.headers['x-correlation-id'] || uuidv4();
    
    // Set correlation ID in request and response
    req.correlationId = correlationId;
    res.setHeader('X-Correlation-ID', correlationId);
    
    // Add to response locals for access in other middleware
    res.locals.correlationId = correlationId;
    
    // Log request start with correlation ID
    logger.info(`Request started: ${req.method} ${req.path}`, {
        correlationId,
        method: req.method,
        path: req.path,
        userAgent: req.get('User-Agent'),
        ip: req.ip
    });
    
    // Override res.end to log completion
    const originalEnd = res.end;
    res.end = function(...args) {
        logger.info(`Request completed: ${req.method} ${req.path}`, {
            correlationId,
            statusCode: res.statusCode,
            duration: Date.now() - req.startTime
        });
        originalEnd.apply(this, args);
    };
    
    req.startTime = Date.now();
    next();
}

module.exports = correlationMiddleware;
