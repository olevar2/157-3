const axios = require('axios');
const { createLogger } = require('../utils/logger');

const logger = createLogger('ServiceDiscoveryMiddleware');

class ServiceDiscoveryMiddleware {
  constructor(serviceDiscoveryUrl = process.env.SERVICE_DISCOVERY_URL || 'http://localhost:3010') {
    this.serviceDiscoveryUrl = serviceDiscoveryUrl;
    this.serviceCache = new Map();
    this.cacheTimeout = 30000; // 30 seconds
    
    // Periodically refresh cache
    setInterval(() => this.refreshCache(), this.cacheTimeout);
  }

  /**
   * Middleware to resolve service URLs dynamically
   */
  resolveService() {
    return async (req, res, next) => {
      try {
        // Extract service name from the request path
        const pathSegments = req.path.split('/').filter(segment => segment);
        if (pathSegments.length === 0) {
          return next();
        }

        const serviceName = this.mapPathToService(pathSegments[0]);
        if (!serviceName) {
          return next();
        }

        // Get service URL from discovery
        const serviceUrl = await this.getServiceUrl(serviceName);
        if (!serviceUrl) {
          logger.warn('No healthy instances found for service', { serviceName, path: req.path });
          return res.status(503).json({
            error: 'Service temporarily unavailable',
            serviceName,
            timestamp: new Date().toISOString()
          });
        }

        // Add service URL to request for proxy middleware
        req.serviceUrl = serviceUrl;
        req.serviceName = serviceName;
        
        logger.debug('Service resolved', {
          serviceName,
          serviceUrl,
          originalPath: req.path
        });

        next();
      } catch (error) {
        logger.error('Service discovery failed', {
          error: error.message,
          path: req.path
        });
        
        res.status(503).json({
          error: 'Service discovery failed',
          message: error.message,
          timestamp: new Date().toISOString()
        });
      }
    };
  }

  /**
   * Get service URL with caching
   */
  async getServiceUrl(serviceName) {
    // Check cache first
    const cached = this.serviceCache.get(serviceName);
    if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
      return cached.url;
    }

    try {
      const response = await axios.get(
        `${this.serviceDiscoveryUrl}/services/${serviceName}/url`,
        { timeout: 5000 }
      );

      if (response.data.success) {
        const serviceUrl = response.data.url;
        
        // Update cache
        this.serviceCache.set(serviceName, {
          url: serviceUrl,
          timestamp: Date.now()
        });

        return serviceUrl;
      }
      
      return null;
    } catch (error) {
      logger.error('Failed to get service URL from discovery', {
        serviceName,
        error: error.message
      });
      
      // Return cached URL if available, even if expired
      const cached = this.serviceCache.get(serviceName);
      if (cached) {
        logger.warn('Using expired cached service URL', { serviceName });
        return cached.url;
      }
      
      return null;
    }
  }

  /**
   * Map request path to service name
   */
  mapPathToService(pathSegment) {
    const serviceMap = {
      'api/trading': 'trading-service',
      'api/analytics': 'analytics-service',
      'api/users': 'user-service',
      'api/notifications': 'notification-service',
      'api/compliance': 'compliance-service',
      'api/risk': 'risk-service',
      'api/ml': 'ml-service',
      'api/backtest': 'backtest-service',
      'api/data': 'data-quality-service',
      'api/order': 'order-execution-service',
      'api/qa': 'qa-service',
      'trading': 'trading-service',
      'analytics': 'analytics-service',
      'users': 'user-service',
      'notifications': 'notification-service',
      'compliance': 'compliance-service',
      'risk': 'risk-service',
      'ml': 'ml-service',
      'backtest': 'backtest-service',
      'data': 'data-quality-service',
      'order': 'order-execution-service',
      'qa': 'qa-service'
    };

    return serviceMap[pathSegment] || null;
  }

  /**
   * Register this API Gateway with service discovery
   */
  async registerSelf() {
    try {
      const registration = {
        name: 'api-gateway',
        address: process.env.HOST || 'localhost',
        port: parseInt(process.env.PORT || '3000'),
        tags: ['api', 'gateway', 'http'],
        meta: {
          version: process.env.VERSION || '1.0.0',
          environment: process.env.NODE_ENV || 'development'
        },
        check: {
          http: `http://${process.env.HOST || 'localhost'}:${process.env.PORT || '3000'}/health`,
          interval: '10s',
          timeout: '5s',
          deregisterCriticalServiceAfter: '30s'
        }
      };

      const response = await axios.post(
        `${this.serviceDiscoveryUrl}/services/register`,
        registration,
        { timeout: 10000 }
      );

      if (response.data.success) {
        logger.info('API Gateway registered with service discovery', {
          serviceId: response.data.serviceId
        });
      }
    } catch (error) {
      logger.error('Failed to register API Gateway with service discovery', {
        error: error.message
      });
    }
  }

  /**
   * Refresh service cache
   */
  async refreshCache() {
    const servicesToRefresh = Array.from(this.serviceCache.keys());
    
    for (const serviceName of servicesToRefresh) {
      try {
        await this.getServiceUrl(serviceName);
      } catch (error) {
        logger.debug('Failed to refresh cache for service', {
          serviceName,
          error: error.message
        });
      }
    }
  }

  /**
   * Health check for service discovery connectivity
   */
  async healthCheck() {
    try {
      const response = await axios.get(
        `${this.serviceDiscoveryUrl}/health`,
        { timeout: 5000 }
      );
      
      return response.status === 200 && response.data.status === 'healthy';
    } catch (error) {
      logger.error('Service discovery health check failed', {
        error: error.message
      });
      return false;
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    const stats = [];
    const now = Date.now();
    
    for (const [serviceName, cached] of this.serviceCache) {
      stats.push({
        serviceName,
        url: cached.url,
        age: now - cached.timestamp,
        expired: (now - cached.timestamp) > this.cacheTimeout
      });
    }
    
    return stats;
  }
}

module.exports = ServiceDiscoveryMiddleware;
