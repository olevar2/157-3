// Platform3 Circuit Breaker Middleware
const logger = require('../logging/platform3_logger');

class CircuitBreaker {
    constructor(options = {}) {
        this.failureThreshold = options.failureThreshold || 5;
        this.recoveryTimeout = options.recoveryTimeout || 60000; // 60 seconds
        this.monitoringPeriod = options.monitoringPeriod || 10000; // 10 seconds
        
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        this.failureCount = 0;
        this.lastFailureTime = null;
        this.nextAttempt = null;
    }
    
    async execute(operation, fallback = null) {
        if (this.state === 'OPEN') {
            if (Date.now() >= this.nextAttempt) {
                this.state = 'HALF_OPEN';
                logger.info('Circuit breaker: Attempting recovery (HALF_OPEN)');
            } else {
                logger.warn('Circuit breaker: Request rejected (OPEN)');
                return fallback ? fallback() : Promise.reject(new Error('Circuit breaker is OPEN'));
            }
        }
        
        try {
            const result = await operation();
            this.onSuccess();
            return result;
        } catch (error) {
            this.onFailure();
            throw error;
        }
    }
    
    onSuccess() {
        this.failureCount = 0;
        this.state = 'CLOSED';
        logger.info('Circuit breaker: Reset to CLOSED state');
    }
    
    onFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
            this.nextAttempt = Date.now() + this.recoveryTimeout;
            logger.error(`Circuit breaker: Opened due to ${this.failureCount} failures`);
        }
    }
    
    getStatus() {
        return {
            state: this.state,
            failureCount: this.failureCount,
            lastFailureTime: this.lastFailureTime,
            nextAttempt: this.nextAttempt
        };
    }
}

// Global circuit breakers for different services
const circuitBreakers = new Map();

function getCircuitBreaker(serviceName, options = {}) {
    if (!circuitBreakers.has(serviceName)) {
        circuitBreakers.set(serviceName, new CircuitBreaker(options));
    }
    return circuitBreakers.get(serviceName);
}

function circuitBreakerMiddleware(serviceName, options = {}) {
    return (req, res, next) => {
        const circuitBreaker = getCircuitBreaker(serviceName, options);
        req.circuitBreaker = circuitBreaker;
        
        // Add circuit breaker status to response headers
        const status = circuitBreaker.getStatus();
        res.setHeader('X-Circuit-Breaker-State', status.state);
        
        next();
    };
}

module.exports = {
    CircuitBreaker,
    circuitBreakerMiddleware,
    getCircuitBreaker
};
