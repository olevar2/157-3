
const consul = require('consul')();
const logger = require('../../../shared/logging/platform3_logger');

class ServiceDiscoveryMiddleware {
    constructor(serviceName, port) {
        this.serviceName = serviceName;
        this.port = port;
        this.serviceId = `${serviceName}-${process.env.NODE_ENV || 'development'}-${port}`;
        this.healthCheckInterval = null;
    }

    async registerService() {
        try {
            const serviceConfig = {
                name: this.serviceName,
                id: this.serviceId,
                port: this.port,
                check: {
                    http: `http://localhost:${this.port}/health`,
                    interval: '10s',
                    timeout: '5s'
                },
                tags: ['platform3', 'microservice']
            };

            await consul.agent.service.register(serviceConfig);
            logger.info(`Service ${this.serviceName} registered with Consul`, {
                serviceId: this.serviceId,
                port: this.port
            });

            // Start health check monitoring
            this.startHealthCheckMonitoring();
        } catch (error) {
            logger.error('Failed to register service with Consul', {
                serviceName: this.serviceName,
                error: error.message
            });
        }
    }

    async deregisterService() {
        try {
            await consul.agent.service.deregister(this.serviceId);
            if (this.healthCheckInterval) {
                clearInterval(this.healthCheckInterval);
            }
            logger.info(`Service ${this.serviceName} deregistered from Consul`);
        } catch (error) {
            logger.error('Failed to deregister service', {
                serviceName: this.serviceName,
                error: error.message
            });
        }
    }

    startHealthCheckMonitoring() {
        this.healthCheckInterval = setInterval(async () => {
            try {
                const healthStatus = await this.checkServiceHealth();
                if (!healthStatus.healthy) {
                    logger.warn('Service health check failed', {
                        serviceName: this.serviceName,
                        status: healthStatus
                    });
                }
            } catch (error) {
                logger.error('Health check monitoring error', {
                    serviceName: this.serviceName,
                    error: error.message
                });
            }
        }, 30000); // Check every 30 seconds
    }

    async checkServiceHealth() {
        // Override this method in specific services
        return {
            healthy: true,
            timestamp: new Date().toISOString(),
            service: this.serviceName
        };
    }

    middleware() {
        return (req, res, next) => {
            // Add correlation ID for request tracking
            if (!req.headers['x-correlation-id']) {
                req.headers['x-correlation-id'] = this.generateCorrelationId();
            }
            
            req.correlationId = req.headers['x-correlation-id'];
            res.setHeader('X-Correlation-ID', req.correlationId);
            
            // Add service identification
            res.setHeader('X-Service-Name', this.serviceName);
            res.setHeader('X-Service-ID', this.serviceId);
            
            next();
        };
    }

    generateCorrelationId() {
        return `${this.serviceName}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
}

module.exports = ServiceDiscoveryMiddleware;
