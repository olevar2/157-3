
const express = require('express');
const router = express.Router();
const logger = require('../../../shared/logging/platform3_logger');

class HealthCheckEndpoint {
    constructor(serviceName, dependencies = []) {
        this.serviceName = serviceName;
        this.dependencies = dependencies;
        this.startTime = new Date();
    }

    async checkDependencies() {
        const dependencyChecks = await Promise.allSettled(
            this.dependencies.map(async (dep) => {
                try {
                    const result = await dep.check();
                    return {
                        name: dep.name,
                        status: 'healthy',
                        responseTime: result.responseTime || 0,
                        details: result.details || {}
                    };
                } catch (error) {
                    return {
                        name: dep.name,
                        status: 'unhealthy',
                        error: error.message,
                        responseTime: -1
                    };
                }
            })
        );

        return dependencyChecks.map((result, index) => {
            if (result.status === 'fulfilled') {
                return result.value;
            } else {
                return {
                    name: this.dependencies[index].name,
                    status: 'error',
                    error: result.reason.message
                };
            }
        });
    }

    getRouter() {
        router.get('/health', async (req, res) => {
            try {
                const startTime = Date.now();
                const dependencyResults = await this.checkDependencies();
                const responseTime = Date.now() - startTime;

                const allDependenciesHealthy = dependencyResults.every(
                    dep => dep.status === 'healthy'
                );

                const healthStatus = {
                    service: this.serviceName,
                    status: allDependenciesHealthy ? 'healthy' : 'degraded',
                    timestamp: new Date().toISOString(),
                    uptime: Date.now() - this.startTime.getTime(),
                    responseTime: responseTime,
                    dependencies: dependencyResults,
                    version: process.env.SERVICE_VERSION || '1.0.0',
                    environment: process.env.NODE_ENV || 'development'
                };

                const statusCode = allDependenciesHealthy ? 200 : 503;
                
                if (statusCode === 503) {
                    logger.warn('Health check failed', {
                        service: this.serviceName,
                        dependencies: dependencyResults.filter(dep => dep.status !== 'healthy')
                    });
                }

                res.status(statusCode).json(healthStatus);
            } catch (error) {
                logger.error('Health check endpoint error', {
                    service: this.serviceName,
                    error: error.message
                });

                res.status(500).json({
                    service: this.serviceName,
                    status: 'error',
                    timestamp: new Date().toISOString(),
                    error: error.message
                });
            }
        });

        router.get('/ready', async (req, res) => {
            try {
                // Check if service is ready to accept traffic
                const dependencyResults = await this.checkDependencies();
                const criticalDependenciesHealthy = dependencyResults
                    .filter(dep => dep.critical !== false)
                    .every(dep => dep.status === 'healthy');

                const readinessStatus = {
                    service: this.serviceName,
                    ready: criticalDependenciesHealthy,
                    timestamp: new Date().toISOString(),
                    dependencies: dependencyResults
                };

                const statusCode = criticalDependenciesHealthy ? 200 : 503;
                res.status(statusCode).json(readinessStatus);
            } catch (error) {
                logger.error('Readiness check error', {
                    service: this.serviceName,
                    error: error.message
                });

                res.status(500).json({
                    service: this.serviceName,
                    ready: false,
                    timestamp: new Date().toISOString(),
                    error: error.message
                });
            }
        });

        return router;
    }
}

module.exports = HealthCheckEndpoint;
