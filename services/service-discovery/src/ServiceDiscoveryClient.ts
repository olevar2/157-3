import { ConsulServiceRegistry, ServiceInstance } from './ConsulServiceRegistry';
import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { createLogger } from './utils/logger';

export interface LoadBalancerStrategy {
  selectInstance(instances: ServiceInstance[]): ServiceInstance | null;
}

export class RoundRobinStrategy implements LoadBalancerStrategy {
  private counters: Map<string, number> = new Map();

  selectInstance(instances: ServiceInstance[]): ServiceInstance | null {
    if (instances.length === 0) return null;
    
    const serviceName = instances[0].name;
    const currentCount = this.counters.get(serviceName) || 0;
    const selectedIndex = currentCount % instances.length;
    
    this.counters.set(serviceName, currentCount + 1);
    
    return instances[selectedIndex];
  }
}

export class LeastConnectionsStrategy implements LoadBalancerStrategy {
  private connections: Map<string, number> = new Map();

  selectInstance(instances: ServiceInstance[]): ServiceInstance | null {
    if (instances.length === 0) return null;
    
    let selectedInstance = instances[0];
    let minConnections = this.connections.get(selectedInstance.id) || 0;
    
    for (const instance of instances) {
      const connections = this.connections.get(instance.id) || 0;
      if (connections < minConnections) {
        selectedInstance = instance;
        minConnections = connections;
      }
    }
    
    return selectedInstance;
  }

  incrementConnections(instanceId: string): void {
    const current = this.connections.get(instanceId) || 0;
    this.connections.set(instanceId, current + 1);
  }

  decrementConnections(instanceId: string): void {
    const current = this.connections.get(instanceId) || 0;
    this.connections.set(instanceId, Math.max(0, current - 1));
  }
}

export class RandomStrategy implements LoadBalancerStrategy {
  selectInstance(instances: ServiceInstance[]): ServiceInstance | null {
    if (instances.length === 0) return null;
    
    const randomIndex = Math.floor(Math.random() * instances.length);
    return instances[randomIndex];
  }
}

export interface ServiceDiscoveryOptions {
  loadBalancerStrategy?: LoadBalancerStrategy;
  cacheTimeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

export class ServiceDiscoveryClient extends EventEmitter {
  private registry: ConsulServiceRegistry;
  private logger: Logger;
  private loadBalancer: LoadBalancerStrategy;
  private serviceCache: Map<string, { instances: ServiceInstance[]; timestamp: number }> = new Map();
  private cacheTimeout: number;
  private retryAttempts: number;
  private retryDelay: number;

  constructor(registry: ConsulServiceRegistry, options: ServiceDiscoveryOptions = {}) {
    super();
    
    this.registry = registry;
    this.logger = createLogger('ServiceDiscoveryClient');
    this.loadBalancer = options.loadBalancerStrategy || new RoundRobinStrategy();
    this.cacheTimeout = options.cacheTimeout || 30000; // 30 seconds
    this.retryAttempts = options.retryAttempts || 3;
    this.retryDelay = options.retryDelay || 1000; // 1 second
    
    // Listen for service changes to invalidate cache
    this.registry.on('serviceChanged', ({ serviceName }) => {
      this.invalidateCache(serviceName);
    });
  }

  /**
   * Discover and select a service instance using load balancing
   */
  async discoverService(serviceName: string): Promise<ServiceInstance | null> {
    try {
      const instances = await this.getServiceInstances(serviceName);
      
      if (instances.length === 0) {
        this.logger.warn('No healthy instances found for service', { serviceName });
        return null;
      }
      
      const selectedInstance = this.loadBalancer.selectInstance(instances);
      
      if (selectedInstance) {
        this.logger.debug('Service instance selected', {
          serviceName,
          instanceId: selectedInstance.id,
          address: selectedInstance.address,
          port: selectedInstance.port
        });
      }
      
      return selectedInstance;
    } catch (error) {
      this.logger.error('Failed to discover service', {
        serviceName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Get all healthy instances of a service (with caching)
   */
  async getServiceInstances(serviceName: string): Promise<ServiceInstance[]> {
    // Check cache first
    const cached = this.serviceCache.get(serviceName);
    if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
      this.logger.debug('Returning cached service instances', {
        serviceName,
        instanceCount: cached.instances.length
      });
      return cached.instances;
    }
    
    // Fetch from registry with retry logic
    const instances = await this.retryOperation(
      () => this.registry.discoverService(serviceName),
      this.retryAttempts,
      this.retryDelay
    );
    
    // Update cache
    this.serviceCache.set(serviceName, {
      instances,
      timestamp: Date.now()
    });
    
    this.logger.debug('Service instances fetched and cached', {
      serviceName,
      instanceCount: instances.length
    });
    
    return instances;
  }

  /**
   * Get service URL for HTTP requests
   */
  async getServiceUrl(serviceName: string, path: string = ''): Promise<string | null> {
    const instance = await this.discoverService(serviceName);
    
    if (!instance) {
      return null;
    }
    
    const protocol = instance.tags.includes('https') ? 'https' : 'http';
    const url = `${protocol}://${instance.address}:${instance.port}${path}`;
    
    this.logger.debug('Service URL generated', {
      serviceName,
      instanceId: instance.id,
      url
    });
    
    return url;
  }

  /**
   * Watch for service changes
   */
  watchService(serviceName: string): void {
    this.registry.watchService(serviceName);
    this.logger.info('Started watching service for changes', { serviceName });
  }

  /**
   * Stop watching service changes
   */
  unwatchService(serviceName: string): void {
    this.registry.unwatchService(serviceName);
    this.invalidateCache(serviceName);
    this.logger.info('Stopped watching service', { serviceName });
  }

  /**
   * Invalidate cache for a specific service
   */
  invalidateCache(serviceName: string): void {
    this.serviceCache.delete(serviceName);
    this.logger.debug('Cache invalidated for service', { serviceName });
    this.emit('cacheInvalidated', { serviceName });
  }

  /**
   * Clear all cached service instances
   */
  clearCache(): void {
    this.serviceCache.clear();
    this.logger.info('All service cache cleared');
    this.emit('cacheCleared');
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { serviceName: string; instanceCount: number; age: number }[] {
    const stats: { serviceName: string; instanceCount: number; age: number }[] = [];
    const now = Date.now();
    
    for (const [serviceName, cached] of this.serviceCache) {
      stats.push({
        serviceName,
        instanceCount: cached.instances.length,
        age: now - cached.timestamp
      });
    }
    
    return stats;
  }

  /**
   * Health check for the discovery client
   */
  async healthCheck(): Promise<boolean> {
    try {
      return await this.registry.healthCheck();
    } catch (error) {
      this.logger.error('Service discovery client health check failed', {
        error: error.message
      });
      return false;
    }
  }

  /**
   * Set load balancer strategy
   */
  setLoadBalancerStrategy(strategy: LoadBalancerStrategy): void {
    this.loadBalancer = strategy;
    this.logger.info('Load balancer strategy updated', {
      strategy: strategy.constructor.name
    });
  }

  /**
   * Retry operation with exponential backoff
   */
  private async retryOperation<T>(
    operation: () => Promise<T>,
    maxAttempts: number,
    baseDelay: number
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        if (attempt === maxAttempts) {
          break;
        }
        
        const delay = baseDelay * Math.pow(2, attempt - 1);
        this.logger.warn('Operation failed, retrying', {
          attempt,
          maxAttempts,
          delay,
          error: error.message
        });
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }
}
