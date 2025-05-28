import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { ConsulServiceRegistry } from './ConsulServiceRegistry';
import { ServiceDiscoveryClient, RoundRobinStrategy, LeastConnectionsStrategy, RandomStrategy } from './ServiceDiscoveryClient';
import { createLogger } from './utils/logger';

const app = express();
const logger = createLogger('ServiceDiscoveryService');
const port = process.env.PORT || 3010;

// Initialize Consul registry and discovery client
const registry = new ConsulServiceRegistry();
const discoveryClient = new ServiceDiscoveryClient(registry, {
  loadBalancerStrategy: new RoundRobinStrategy(),
  cacheTimeout: 30000,
  retryAttempts: 3,
  retryDelay: 1000
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const isHealthy = await registry.healthCheck();
    res.status(isHealthy ? 200 : 503).json({
      status: isHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      service: 'service-discovery'
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString(),
      service: 'service-discovery'
    });
  }
});

// Register a service
app.post('/services/register', async (req, res) => {
  try {
    const registration = req.body;
    await registry.registerService(registration);
    
    logger.info('Service registered via API', {
      serviceName: registration.name,
      serviceId: registration.id
    });
    
    res.status(201).json({
      success: true,
      message: 'Service registered successfully',
      serviceId: registration.id || `${registration.name}-${registration.address}-${registration.port}`
    });
  } catch (error) {
    logger.error('Failed to register service via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Deregister a service
app.delete('/services/:serviceId', async (req, res) => {
  try {
    const { serviceId } = req.params;
    await registry.deregisterService(serviceId);
    
    logger.info('Service deregistered via API', { serviceId });
    
    res.json({
      success: true,
      message: 'Service deregistered successfully'
    });
  } catch (error) {
    logger.error('Failed to deregister service via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Discover service instances
app.get('/services/:serviceName/instances', async (req, res) => {
  try {
    const { serviceName } = req.params;
    const instances = await discoveryClient.getServiceInstances(serviceName);
    
    res.json({
      success: true,
      serviceName,
      instances,
      count: instances.length
    });
  } catch (error) {
    logger.error('Failed to discover service instances via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get a service instance using load balancing
app.get('/services/:serviceName/instance', async (req, res) => {
  try {
    const { serviceName } = req.params;
    const instance = await discoveryClient.discoverService(serviceName);
    
    if (!instance) {
      return res.status(404).json({
        success: false,
        message: 'No healthy instances found for service',
        serviceName
      });
    }
    
    res.json({
      success: true,
      serviceName,
      instance
    });
  } catch (error) {
    logger.error('Failed to discover service instance via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get service URL
app.get('/services/:serviceName/url', async (req, res) => {
  try {
    const { serviceName } = req.params;
    const { path = '' } = req.query;
    const url = await discoveryClient.getServiceUrl(serviceName, path as string);
    
    if (!url) {
      return res.status(404).json({
        success: false,
        message: 'No healthy instances found for service',
        serviceName
      });
    }
    
    res.json({
      success: true,
      serviceName,
      url
    });
  } catch (error) {
    logger.error('Failed to get service URL via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// List all services
app.get('/services', async (req, res) => {
  try {
    const services = await registry.getAllServices();
    
    res.json({
      success: true,
      services
    });
  } catch (error) {
    logger.error('Failed to list services via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get cache statistics
app.get('/cache/stats', (req, res) => {
  try {
    const stats = discoveryClient.getCacheStats();
    
    res.json({
      success: true,
      cacheStats: stats
    });
  } catch (error) {
    logger.error('Failed to get cache stats via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Clear cache
app.delete('/cache', (req, res) => {
  try {
    discoveryClient.clearCache();
    
    res.json({
      success: true,
      message: 'Cache cleared successfully'
    });
  } catch (error) {
    logger.error('Failed to clear cache via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Set load balancer strategy
app.post('/loadbalancer/strategy', (req, res) => {
  try {
    const { strategy } = req.body;
    
    let loadBalancerStrategy;
    switch (strategy) {
      case 'round-robin':
        loadBalancerStrategy = new RoundRobinStrategy();
        break;
      case 'least-connections':
        loadBalancerStrategy = new LeastConnectionsStrategy();
        break;
      case 'random':
        loadBalancerStrategy = new RandomStrategy();
        break;
      default:
        return res.status(400).json({
          success: false,
          error: 'Invalid strategy. Supported: round-robin, least-connections, random'
        });
    }
    
    discoveryClient.setLoadBalancerStrategy(loadBalancerStrategy);
    
    res.json({
      success: true,
      message: `Load balancer strategy set to ${strategy}`
    });
  } catch (error) {
    logger.error('Failed to set load balancer strategy via API', { error: error.message });
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Start the service
app.listen(port, () => {
  logger.info('Service Discovery service started', {
    port,
    consulHost: process.env.CONSUL_HOST || 'localhost',
    consulPort: process.env.CONSUL_PORT || '8500'
  });
});

// Export for testing
export { app, registry, discoveryClient };
