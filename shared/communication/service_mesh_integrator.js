#!/usr/bin/env node
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
                /(const.*require.*[\n\r]+)+/,
                `$&${middlewareImports}`
            );
            
            // Insert middleware usage after app creation
            enhancedContent = enhancedContent.replace(
                /(const app = express\(\);)/,
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
