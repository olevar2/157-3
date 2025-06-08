#!/usr/bin/env python3
"""
Platform3 Simple Correlation and Circuit Breaker System
Creates essential middleware components for microservices communication
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCorrelationSystem:
    """Simple correlation and circuit breaker system creator"""
    
    def __init__(self):
        self.platform_root = Path("d:/MD/Platform3")
        self.communication_dir = self.platform_root / "shared" / "communication"
        self.middleware_dir = self.platform_root / "shared" / "middleware"
        self.monitoring_dir = self.platform_root / "shared" / "monitoring"
        
        # Ensure directories exist
        self.communication_dir.mkdir(parents=True, exist_ok=True)
        self.middleware_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    def create_correlation_middleware(self):
        """Create correlation ID middleware"""
        middleware_content = '''// Platform3 Correlation Middleware
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
'''
        
        middleware_file = self.middleware_dir / "correlation_middleware.js"
        with open(middleware_file, 'w', encoding='utf-8') as f:
            f.write(middleware_content)
        
        logger.info(f"Created correlation middleware: {middleware_file}")
        return True
    
    def create_circuit_breaker_middleware(self):
        """Create circuit breaker middleware"""
        circuit_breaker_content = '''// Platform3 Circuit Breaker Middleware
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
'''
        
        circuit_file = self.middleware_dir / "circuit_breaker_middleware.js"
        with open(circuit_file, 'w', encoding='utf-8') as f:
            f.write(circuit_breaker_content)
        
        logger.info(f"Created circuit breaker middleware: {circuit_file}")
        return True
    
    def create_service_mesh_integrator(self):
        """Create service mesh integration script"""
        integrator_content = '''#!/usr/bin/env node
/**
 * Platform3 Service Mesh Integrator
 * Enhances existing services with correlation and circuit breaker patterns
 */

const fs = require('fs').promises;
const path = require('path');

class ServiceMeshIntegrator {
    constructor() {
        this.servicesDir = path.join(__dirname, '../../services');
        this.enhancements = 0;
    }
    
    async integrateServices() {
        console.log('Starting service mesh integration...');
        
        try {
            const services = await this.findServices();
            
            for (const servicePath of services) {
                await this.enhanceService(servicePath);
            }
            
            console.log(`Enhanced ${this.enhancements} services with mesh patterns`);
            return true;
        } catch (error) {
            console.error('Service mesh integration failed:', error);
            return false;
        }
    }
    
    async findServices() {
        const services = [];
        try {
            const entries = await fs.readdir(this.servicesDir, { withFileTypes: true });
            
            for (const entry of entries) {
                if (entry.isDirectory()) {
                    const serverFile = path.join(this.servicesDir, entry.name, 'src', 'server.js');
                    try {
                        await fs.access(serverFile);
                        services.push(serverFile);
                    } catch {
                        // Server file doesn't exist, skip
                    }
                }
            }
        } catch (error) {
            console.log('Services directory not found, checking individual files');
        }
        
        return services;
    }
    
    async enhanceService(servicePath) {
        try {
            const content = await fs.readFile(servicePath, 'utf-8');
            
            // Check if already enhanced
            if (content.includes('correlationMiddleware') || content.includes('X-Correlation-ID')) {
                console.log(`Service already enhanced: ${servicePath}`);
                return;
            }
            
            // Add middleware imports
            const middlewareImports = `
// Platform3 Service Mesh Integration
const correlationMiddleware = require('../../shared/middleware/correlation_middleware');
const { circuitBreakerMiddleware } = require('../../shared/middleware/circuit_breaker_middleware');
`;
            
            // Add middleware usage
            const middlewareUsage = `
// Apply service mesh middleware
app.use(correlationMiddleware);
app.use(circuitBreakerMiddleware('${path.basename(path.dirname(path.dirname(servicePath)))}'));
`;
            
            // Insert imports after existing requires
            let enhancedContent = content.replace(
                /(const.*require.*[\\n\\r]+)+/,
                `$&${middlewareImports}`
            );
            
            // Insert middleware usage after app creation
            enhancedContent = enhancedContent.replace(
                /(const app = express\\(\\);)/,
                `$1${middlewareUsage}`
            );
            
            await fs.writeFile(servicePath, enhancedContent, 'utf-8');
            console.log(`Enhanced service: ${servicePath}`);
            this.enhancements++;
            
        } catch (error) {
            console.error(`Failed to enhance service ${servicePath}:`, error);
        }
    }
}

// Main execution
async function main() {
    const integrator = new ServiceMeshIntegrator();
    const success = await integrator.integrateServices();
    process.exit(success ? 0 : 1);
}

if (require.main === module) {
    main();
}

module.exports = ServiceMeshIntegrator;
'''
        
        integrator_file = self.communication_dir / "service_mesh_integrator.js"
        with open(integrator_file, 'w', encoding='utf-8') as f:
            f.write(integrator_content)
        
        logger.info(f"Created service mesh integrator: {integrator_file}")
        return True
    
    def create_monitoring_endpoints(self):
        """Create monitoring endpoints for health checks"""
        monitoring_content = '''// Platform3 Service Monitoring Endpoints
const express = require('express');
const { getCircuitBreaker } = require('../middleware/circuit_breaker_middleware');

function createMonitoringRoutes() {
    const router = express.Router();
    
    // Health check endpoint
    router.get('/health', (req, res) => {
        res.status(200).json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            correlationId: req.correlationId,
            service: process.env.SERVICE_NAME || 'unknown'
        });
    });
    
    // Readiness check endpoint
    router.get('/ready', (req, res) => {
        // Add your readiness checks here
        const isReady = true; // Implement actual readiness logic
        
        res.status(isReady ? 200 : 503).json({
            ready: isReady,
            timestamp: new Date().toISOString(),
            correlationId: req.correlationId
        });
    });
    
    // Circuit breaker status endpoint
    router.get('/circuit-breaker/status', (req, res) => {
        const serviceName = req.query.service || 'default';
        const circuitBreaker = getCircuitBreaker(serviceName);
        const status = circuitBreaker.getStatus();
        
        res.json({
            service: serviceName,
            circuitBreaker: status,
            timestamp: new Date().toISOString()
        });
    });
    
    // Correlation tracking endpoint
    router.get('/correlation/trace/:id', (req, res) => {
        const correlationId = req.params.id;
        
        // In a real implementation, you would query logs or tracing system
        res.json({
            correlationId,
            message: 'Correlation tracking endpoint - implement with your logging system',
            timestamp: new Date().toISOString()
        });
    });
    
    return router;
}

module.exports = createMonitoringRoutes;
'''
        
        monitoring_file = self.monitoring_dir / "monitoring_endpoints.js"
        with open(monitoring_file, 'w', encoding='utf-8') as f:
            f.write(monitoring_content)
        
        logger.info(f"Created monitoring endpoints: {monitoring_file}")
        return True
    
    def create_package_json_update(self):
        """Create package.json update script for dependencies"""
        update_script = '''#!/usr/bin/env node
/**
 * Platform3 Package Dependencies Updater
 * Updates package.json files with required dependencies
 */

const fs = require('fs').promises;
const path = require('path');

const requiredDependencies = {
    "uuid": "^9.0.0",
    "express": "^4.18.2"
};

async function updatePackageJsonFiles() {
    const servicesDir = path.join(__dirname, '../../services');
    
    try {
        const entries = await fs.readdir(servicesDir, { withFileTypes: true });
        
        for (const entry of entries) {
            if (entry.isDirectory()) {
                const packagePath = path.join(servicesDir, entry.name, 'package.json');
                await updatePackageJson(packagePath);
            }
        }
        
        console.log('Updated package.json files with required dependencies');
    } catch (error) {
        console.error('Failed to update package.json files:', error);
    }
}

async function updatePackageJson(packagePath) {
    try {
        const content = await fs.readFile(packagePath, 'utf-8');
        const packageData = JSON.parse(content);
        
        if (!packageData.dependencies) {
            packageData.dependencies = {};
        }
        
        let updated = false;
        for (const [dep, version] of Object.entries(requiredDependencies)) {
            if (!packageData.dependencies[dep]) {
                packageData.dependencies[dep] = version;
                updated = true;
            }
        }
        
        if (updated) {
            await fs.writeFile(packagePath, JSON.stringify(packageData, null, 2), 'utf-8');
            console.log(`Updated: ${packagePath}`);
        }
        
    } catch (error) {
        console.log(`Skipping ${packagePath} - file not found or invalid`);
    }
}

if (require.main === module) {
    updatePackageJsonFiles();
}
'''
        
        update_file = self.communication_dir / "update_dependencies.js"
        with open(update_file, 'w', encoding='utf-8') as f:
            f.write(update_script)
        
        logger.info(f"Created dependency updater: {update_file}")
        return True

def main():
    """Main execution function"""
    system = SimpleCorrelationSystem()
    
    success = True
    success &= system.create_correlation_middleware()
    success &= system.create_circuit_breaker_middleware()
    success &= system.create_service_mesh_integrator()
    success &= system.create_monitoring_endpoints()
    success &= system.create_package_json_update()
    
    if success:
        print("\n" + "="*60)
        print("PLATFORM3 CORRELATION & CIRCUIT BREAKER SYSTEM")
        print("="*60)
        print("✓ Created correlation middleware")
        print("✓ Created circuit breaker middleware") 
        print("✓ Created service mesh integrator")
        print("✓ Created monitoring endpoints")
        print("✓ Created dependency updater")
        print("\nFiles created:")
        print("- shared/middleware/correlation_middleware.js")
        print("- shared/middleware/circuit_breaker_middleware.js")
        print("- shared/communication/service_mesh_integrator.js")
        print("- shared/monitoring/monitoring_endpoints.js")
        print("- shared/communication/update_dependencies.js")
        print("\nNext steps:")
        print("1. Run: node shared/communication/service_mesh_integrator.js")
        print("2. Run: node shared/communication/update_dependencies.js")
        print("3. Install dependencies: npm install")
        print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
