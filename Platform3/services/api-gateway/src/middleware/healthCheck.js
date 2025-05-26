const axios = require('axios');
const logger = require('../utils/logger');

// Health check for the API Gateway and downstream services
const healthCheck = async (req, res) => {
  const startTime = Date.now();
  const services = {};
  
  // Define service URLs
  const serviceUrls = {
    'auth-service': process.env.AUTH_SERVICE_URL || 'http://localhost:3001',
    'user-service': process.env.USER_SERVICE_URL || 'http://localhost:3002',
    'trading-service': process.env.TRADING_SERVICE_URL || 'http://localhost:3003',
    'market-data-service': process.env.MARKET_DATA_SERVICE_URL || 'http://localhost:3004'
  };

  // Check each service health
  const healthChecks = Object.entries(serviceUrls).map(async ([serviceName, url]) => {
    try {
      const healthUrl = `${url}/health`;
      const response = await axios.get(healthUrl, { 
        timeout: 3000,
        validateStatus: (status) => status < 500 // Accept 4xx as "reachable"
      });
      
      services[serviceName] = {
        status: response.status < 400 ? 'healthy' : 'degraded',
        responseTime: Date.now() - startTime,
        url: healthUrl,
        lastCheck: new Date().toISOString()
      };
      
    } catch (error) {
      services[serviceName] = {
        status: 'unhealthy',
        error: error.message,
        url: `${url}/health`,
        lastCheck: new Date().toISOString()
      };
    }
  });

  // Wait for all health checks to complete
  await Promise.all(healthChecks);

  // Determine overall health
  const healthyServices = Object.values(services).filter(s => s.status === 'healthy').length;
  const totalServices = Object.keys(services).length;
  const overallStatus = healthyServices === totalServices ? 'healthy' : 
                       healthyServices > 0 ? 'degraded' : 'unhealthy';

  const healthResponse = {
    status: overallStatus,
    service: 'api-gateway',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    responseTime: Date.now() - startTime,
    services,
    summary: {
      total: totalServices,
      healthy: healthyServices,
      degraded: Object.values(services).filter(s => s.status === 'degraded').length,
      unhealthy: Object.values(services).filter(s => s.status === 'unhealthy').length
    },
    environment: process.env.NODE_ENV || 'development',
    memory: {
      used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) + ' MB',
      total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024) + ' MB'
    }
  };

  // Log health check results
  if (overallStatus !== 'healthy') {
    logger.warn(`Health check - Overall status: ${overallStatus}`, {
      services: Object.entries(services)
        .filter(([_, service]) => service.status !== 'healthy')
        .map(([name, service]) => ({ name, status: service.status, error: service.error }))
    });
  }

  // Return appropriate HTTP status
  const httpStatus = overallStatus === 'healthy' ? 200 : 
                    overallStatus === 'degraded' ? 200 : 503;

  res.status(httpStatus).json(healthResponse);
};

module.exports = healthCheck;
