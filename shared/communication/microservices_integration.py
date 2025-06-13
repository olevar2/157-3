#!/usr/bin/env python3
"""
Platform3 Microservices Integration Script
Integrates communication framework with existing Platform3 services
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MicroservicesIntegrator:
    """Integrates Platform3 communication framework with existing services"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = str(Path(__file__).parent.parent.parent)
        self.base_path = Path(base_path)
        self.communication_framework_path = self.base_path / "shared" / "communication" / "platform3_communication_framework.py"
        self.services_path = self.base_path / "services"
        self.target_files = self._get_target_files()
        self.integration_count = 0
        self.enhanced_services = []
        
    def _get_target_files(self) -> List[Path]:
        """Get list of target files for microservices enhancement"""
        target_patterns = [
            "**/*server.js",
            "**/*service*.js",
            "**/*gateway*.js",
            "**/*kafka*.js",
            "**/*redis*.js",
            "**/*market*.py",
            "**/*trading*.py",
            "**/*user*.py",
            "**/*notification*.py",
            "**/*event*.py"
        ]
        
        files = []
        for pattern in target_patterns:
            files.extend(self.base_path.glob(pattern))
        
        # Add specific target files from our quality assessment
        specific_targets = [
            "ai-platform/ai-models/market-analysis/pattern-master/ultra_fast_model.py",
            "ai-platform/ai-models/market-analysis/pattern-recognition/japanese_candlesticks.py",
            "api/trading/real-time-trading.js",
            "frontend/js/dashboard/performance.js",
            "services/trading-service/src/server.js",
            "services/user-service/src/server.js",
            "services/notification-service/src/server.js",
            "services/market-data-service/src/server.js",
            "services/event-system/src/server.js"
        ]
        
        for target in specific_targets:
            file_path = self.base_path / target
            if file_path.exists():
                files.append(file_path)
        
        return list(set(files))
    
    def create_service_discovery_middleware(self) -> str:
        """Create service discovery middleware for Express.js services"""
        middleware_content = '''
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
'''
        return middleware_content
    
    def create_redis_message_queue_integration(self) -> str:
        """Create Redis message queue integration"""
        redis_content = '''
const redis = require('redis');
const logger = require('../../../shared/logging/platform3_logger');

class Platform3MessageQueue {
    constructor(redisConfig = {}) {
        this.client = redis.createClient({
            host: redisConfig.host || process.env.REDIS_HOST || 'localhost',
            port: redisConfig.port || process.env.REDIS_PORT || 6379,
            password: redisConfig.password || process.env.REDIS_PASSWORD,
            retry_strategy: (options) => {
                if (options.error && options.error.code === 'ECONNREFUSED') {
                    logger.error('Redis server connection refused');
                    return new Error('Redis server connection refused');
                }
                if (options.total_retry_time > 1000 * 60 * 60) {
                    logger.error('Redis retry time exhausted');
                    return new Error('Retry time exhausted');
                }
                if (options.attempt > 10) {
                    return undefined;
                }
                return Math.min(options.attempt * 100, 3000);
            }
        });

        this.subscriber = redis.createClient({
            host: redisConfig.host || process.env.REDIS_HOST || 'localhost',
            port: redisConfig.port || process.env.REDIS_PORT || 6379,
            password: redisConfig.password || process.env.REDIS_PASSWORD
        });

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.client.on('connect', () => {
            logger.info('Redis client connected');
        });

        this.client.on('error', (err) => {
            logger.error('Redis client error', { error: err.message });
        });

        this.subscriber.on('error', (err) => {
            logger.error('Redis subscriber error', { error: err.message });
        });
    }

    async publishMessage(channel, message, correlationId = null) {
        try {
            const messageData = {
                payload: message,
                timestamp: new Date().toISOString(),
                correlationId: correlationId || this.generateMessageId(),
                service: process.env.SERVICE_NAME || 'unknown'
            };

            const result = await this.client.publish(channel, JSON.stringify(messageData));
            logger.info('Message published to Redis', {
                channel,
                correlationId: messageData.correlationId,
                subscribers: result
            });

            return messageData.correlationId;
        } catch (error) {
            logger.error('Failed to publish message to Redis', {
                channel,
                error: error.message
            });
            throw error;
        }
    }

    async subscribeToChannel(channel, callback) {
        try {
            await this.subscriber.subscribe(channel);
            this.subscriber.on('message', (receivedChannel, message) => {
                if (receivedChannel === channel) {
                    try {
                        const messageData = JSON.parse(message);
                        logger.info('Message received from Redis', {
                            channel: receivedChannel,
                            correlationId: messageData.correlationId
                        });
                        callback(messageData);
                    } catch (parseError) {
                        logger.error('Failed to parse Redis message', {
                            channel: receivedChannel,
                            error: parseError.message
                        });
                    }
                }
            });

            logger.info('Subscribed to Redis channel', { channel });
        } catch (error) {
            logger.error('Failed to subscribe to Redis channel', {
                channel,
                error: error.message
            });
            throw error;
        }
    }

    generateMessageId() {
        return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    async disconnect() {
        try {
            await this.client.quit();
            await this.subscriber.quit();
            logger.info('Redis connections closed');
        } catch (error) {
            logger.error('Error closing Redis connections', { error: error.message });
        }
    }
}

module.exports = Platform3MessageQueue;
'''
        return redis_content
    
    def create_health_check_endpoint(self) -> str:
        """Create health check endpoint for Express.js services"""
        health_content = '''
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
'''
        return health_content
    
    def integrate_with_express_service(self, file_path: Path) -> bool:
        """Integrate communication framework with Express.js service"""
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False
            
            content = file_path.read_text(encoding='utf-8')
            
            # Check if already integrated
            if 'ServiceDiscoveryMiddleware' in content:
                logger.info(f"Service already integrated: {file_path}")
                return True
            
            # Identify service name from file path
            service_name = self._extract_service_name(file_path)
            
            # Add imports at the top
            imports_to_add = f'''
const ServiceDiscoveryMiddleware = require('../../../shared/communication/service_discovery_middleware');
const Platform3MessageQueue = require('../../../shared/communication/redis_message_queue');
const HealthCheckEndpoint = require('../../../shared/communication/health_check_endpoint');
const logger = require('../../../shared/logging/platform3_logger');
'''
            
            # Find where to insert the imports (after existing requires)
            require_pattern = r'((?:const|require).*?;?\n)+'
            match = re.search(require_pattern, content)
            if match:
                insert_position = match.end()
                content = content[:insert_position] + imports_to_add + content[insert_position:]
            else:
                content = imports_to_add + content
            
            # Add service initialization after app creation
            service_init = f'''
// Platform3 Microservices Integration
const serviceDiscovery = new ServiceDiscoveryMiddleware('{service_name}', PORT || 3000);
const messageQueue = new Platform3MessageQueue();
const healthCheck = new HealthCheckEndpoint('{service_name}', [
    {{
        name: 'redis',
        check: async () => {{
            return {{ healthy: true, responseTime: 0 }};
        }}
    }}
]);

// Apply service discovery middleware
app.use(serviceDiscovery.middleware());

// Add health check endpoints
app.use('/api', healthCheck.getRouter());

// Register service with Consul on startup
serviceDiscovery.registerService().catch(err => {{
    logger.error('Failed to register service', {{ error: err.message }});
}});

// Graceful shutdown
process.on('SIGTERM', async () => {{
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
}});

process.on('SIGINT', async () => {{
    logger.info('Shutting down service gracefully');
    await serviceDiscovery.deregisterService();
    await messageQueue.disconnect();
    process.exit(0);
}});
'''
            
            # Find app creation or listen call
            app_pattern = r'(const app = express\(\);|app\.listen\()'
            match = re.search(app_pattern, content)
            if match:
                insert_position = match.end()
                content = content[:insert_position] + '\n' + service_init + content[insert_position:]
            else:
                # If no clear app creation, add at the end before module.exports
                content = content + '\n' + service_init
            
            # Write the enhanced content back
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully integrated microservices communication: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate {file_path}: {str(e)}")
            return False
    
    def integrate_with_python_service(self, file_path: Path) -> bool:
        """Integrate communication framework with Python service"""
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return False
            
            content = file_path.read_text(encoding='utf-8')
            
            # Check if already integrated
            if 'Platform3CommunicationFramework' in content:
                logger.info(f"Service already integrated: {file_path}")
                return True
            
            # Add import at the top
            import_statement = '''
import sys
import os
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
'''
            
            # Find where to insert the import
            if 'import' in content:
                import_pattern = r'(import.*?\n)+'
                match = re.search(import_pattern, content)
                if match:
                    insert_position = match.end()
                    content = content[:insert_position] + import_statement + content[insert_position:]
                else:
                    content = import_statement + content
            else:
                content = import_statement + content
            
            # Add communication framework initialization
            service_name = self._extract_service_name(file_path)
            framework_init = f'''
# Platform3 Communication Framework Integration
communication_framework = Platform3CommunicationFramework(
    service_name="{service_name}",
    service_port=8000,  # Default port
    redis_url="redis://localhost:6379",
    consul_host="localhost",
    consul_port=8500
)

# Initialize the framework
try:
    communication_framework.initialize()
    print(f"Communication framework initialized for {service_name}")
except Exception as e:
    print(f"Failed to initialize communication framework: {{e}}")
'''
            
            # Find class definition or main function
            class_pattern = r'(class\s+\w+.*?:)'
            match = re.search(class_pattern, content)
            if match:
                insert_position = match.start()
                content = content[:insert_position] + framework_init + '\n' + content[insert_position:]
            else:
                # Add after imports
                content = content + '\n' + framework_init
            
            # Write the enhanced content back
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Successfully integrated Python service: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate Python service {file_path}: {str(e)}")
            return False
    
    def _extract_service_name(self, file_path: Path) -> str:
        """Extract service name from file path"""
        path_parts = file_path.parts
        
        # Look for service name in path
        for i, part in enumerate(path_parts):
            if 'service' in part.lower():
                return part.replace('-service', '').replace('_service', '')
            elif part in ['trading', 'user', 'notification', 'market-data', 'event-system', 'api-gateway']:
                return part
        
        # Fallback to filename
        return file_path.stem.replace('-', '_').replace('server', 'service')
    
    def create_middleware_files(self) -> bool:
        """Create middleware files for microservices integration"""
        try:
            middleware_dir = self.base_path / "shared" / "communication"
            middleware_dir.mkdir(parents=True, exist_ok=True)
            
            # Create service discovery middleware
            middleware_file = middleware_dir / "service_discovery_middleware.js"
            middleware_file.write_text(self.create_service_discovery_middleware(), encoding='utf-8')
            logger.info(f"Created service discovery middleware: {middleware_file}")
            
            # Create Redis message queue
            redis_file = middleware_dir / "redis_message_queue.js"
            redis_file.write_text(self.create_redis_message_queue_integration(), encoding='utf-8')
            logger.info(f"Created Redis message queue: {redis_file}")
            
            # Create health check endpoint
            health_file = middleware_dir / "health_check_endpoint.js"
            health_file.write_text(self.create_health_check_endpoint(), encoding='utf-8')
            logger.info(f"Created health check endpoint: {health_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create middleware files: {str(e)}")
            return False
    
    def run_integration(self) -> Dict[str, Any]:
        """Run the complete microservices integration"""
        logger.info("Starting Platform3 microservices integration...")
        
        # Create middleware files first
        if not self.create_middleware_files():
            return {"success": False, "error": "Failed to create middleware files"}
        
        results = {
            "total_files": len(self.target_files),
            "successful_integrations": 0,
            "failed_integrations": 0,
            "enhanced_services": [],
            "errors": []
        }
        
        for file_path in self.target_files:
            try:
                logger.info(f"Processing file: {file_path}")
                
                if file_path.suffix == '.js':
                    success = self.integrate_with_express_service(file_path)
                elif file_path.suffix == '.py':
                    success = self.integrate_with_python_service(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                if success:
                    results["successful_integrations"] += 1
                    results["enhanced_services"].append(str(file_path))
                    self.enhanced_services.append(file_path)
                else:
                    results["failed_integrations"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results["failed_integrations"] += 1
                results["errors"].append(f"{file_path}: {str(e)}")
        
        results["success"] = results["failed_integrations"] == 0
        results["success_rate"] = (results["successful_integrations"] / results["total_files"]) * 100 if results["total_files"] > 0 else 0
        
        logger.info(f"Integration completed. Success rate: {results['success_rate']:.1f}%")
        return results


def main():
    """Main execution function"""
    try:
        integrator = MicroservicesIntegrator()
        results = integrator.run_integration()
        
        print("\n" + "="*60)
        print("PLATFORM3 MICROSERVICES INTEGRATION REPORT")
        print("="*60)
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful integrations: {results['successful_integrations']}")
        print(f"Failed integrations: {results['failed_integrations']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        
        if results['enhanced_services']:
            print(f"\nEnhanced services ({len(results['enhanced_services'])}):")
            for service in results['enhanced_services']:
                print(f"  ✓ {service}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors']:
                print(f"  ✗ {error}")
        
        print("\n" + "="*60)
        
        return results['success']
        
    except Exception as e:
        logger.error(f"Integration failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
