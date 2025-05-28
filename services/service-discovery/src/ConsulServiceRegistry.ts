import Consul from 'consul';
import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { createLogger } from './utils/logger';

export interface ServiceInstance {
  id: string;
  name: string;
  address: string;
  port: number;
  tags: string[];
  meta: Record<string, string>;
  health: 'passing' | 'warning' | 'critical';
}

export interface ServiceRegistration {
  name: string;
  id?: string;
  address: string;
  port: number;
  tags?: string[];
  meta?: Record<string, string>;
  check?: {
    http?: string;
    tcp?: string;
    interval: string;
    timeout?: string;
    deregisterCriticalServiceAfter?: string;
  };
}

export class ConsulServiceRegistry extends EventEmitter {
  private consul: Consul.Consul;
  private logger: Logger;
  private registeredServices: Map<string, ServiceRegistration> = new Map();
  private watchedServices: Map<string, Consul.Watch> = new Map();

  constructor(consulOptions: Consul.ConsulOptions = {}) {
    super();

    this.consul = new Consul({
      host: process.env.CONSUL_HOST || 'localhost',
      port: process.env.CONSUL_PORT || '8500',
      secure: process.env.CONSUL_SECURE === 'true',
      ...consulOptions
    });

    this.logger = createLogger('ConsulServiceRegistry');

    // Handle graceful shutdown
    process.on('SIGTERM', () => this.shutdown());
    process.on('SIGINT', () => this.shutdown());
  }

  /**
   * Register a service with Consul
   */
  async registerService(registration: ServiceRegistration): Promise<void> {
    try {
      const serviceId = registration.id || `${registration.name}-${registration.address}-${registration.port}`;

      const consulRegistration: Consul.Agent.Service.RegisterOptions = {
        name: registration.name,
        id: serviceId,
        address: registration.address,
        port: registration.port,
        tags: registration.tags || [],
        meta: registration.meta || {},
        check: registration.check ? {
          http: registration.check.http,
          interval: registration.check.interval,
          timeout: registration.check.timeout || '5s',
          deregisterCriticalServiceAfter: registration.check.deregisterCriticalServiceAfter || '30s'
        } : undefined
      };

      await this.consul.agent.service.register(consulRegistration);

      this.registeredServices.set(serviceId, registration);

      this.logger.info('Service registered successfully', {
        serviceId,
        serviceName: registration.name,
        address: registration.address,
        port: registration.port
      });

      this.emit('serviceRegistered', { serviceId, registration });
    } catch (error) {
      this.logger.error('Failed to register service', {
        serviceName: registration.name,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Deregister a service from Consul
   */
  async deregisterService(serviceId: string): Promise<void> {
    try {
      await this.consul.agent.service.deregister(serviceId);

      this.registeredServices.delete(serviceId);

      this.logger.info('Service deregistered successfully', { serviceId });

      this.emit('serviceDeregistered', { serviceId });
    } catch (error) {
      this.logger.error('Failed to deregister service', {
        serviceId,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Discover healthy instances of a service
   */
  async discoverService(serviceName: string): Promise<ServiceInstance[]> {
    try {
      const result = await this.consul.health.service({
        service: serviceName,
        passing: true
      });

      const instances: ServiceInstance[] = result.map((entry: any) => ({
        id: entry.Service.ID,
        name: entry.Service.Service,
        address: entry.Service.Address,
        port: entry.Service.Port,
        tags: entry.Service.Tags || [],
        meta: entry.Service.Meta || {},
        health: this.determineHealthStatus(entry.Checks)
      }));

      this.logger.debug('Service discovery completed', {
        serviceName,
        instanceCount: instances.length
      });

      return instances;
    } catch (error) {
      this.logger.error('Failed to discover service', {
        serviceName,
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Watch for service changes
   */
  watchService(serviceName: string): void {
    if (this.watchedServices.has(serviceName)) {
      this.logger.warn('Service is already being watched', { serviceName });
      return;
    }

    const watch = this.consul.watch({
      method: this.consul.health.service,
      options: {
        passing: true
      }
    } as any);

    watch.on('change', (data: any) => {
      const instances: ServiceInstance[] = data.map((entry: any) => ({
        id: entry.Service.ID,
        name: entry.Service.Service,
        address: entry.Service.Address,
        port: entry.Service.Port,
        tags: entry.Service.Tags || [],
        meta: entry.Service.Meta || {},
        health: this.determineHealthStatus(entry.Checks)
      }));

      this.logger.info('Service instances changed', {
        serviceName,
        instanceCount: instances.length
      });

      this.emit('serviceChanged', { serviceName, instances });
    });

    watch.on('error', (error: Error) => {
      this.logger.error('Service watch error', {
        serviceName,
        error: error.message
      });
      this.emit('watchError', { serviceName, error });
    });

    this.watchedServices.set(serviceName, watch);

    this.logger.info('Started watching service', { serviceName });
  }

  /**
   * Stop watching a service
   */
  unwatchService(serviceName: string): void {
    const watch = this.watchedServices.get(serviceName);
    if (watch) {
      watch.end();
      this.watchedServices.delete(serviceName);
      this.logger.info('Stopped watching service', { serviceName });
    }
  }

  /**
   * Get all registered services
   */
  async getAllServices(): Promise<Record<string, string[]>> {
    try {
      return await this.consul.catalog.service.list();
    } catch (error) {
      this.logger.error('Failed to get all services', { error: error.message });
      throw error;
    }
  }

  /**
   * Health check for the registry itself
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.consul.status.leader();
      return true;
    } catch (error) {
      this.logger.error('Consul health check failed', { error: error.message });
      return false;
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown(): Promise<void> {
    this.logger.info('Shutting down Consul Service Registry');

    // Stop all watches
    for (const [serviceName, watch] of this.watchedServices) {
      watch.end();
      this.logger.info('Stopped watch for service', { serviceName });
    }
    this.watchedServices.clear();

    // Deregister all services
    for (const [serviceId] of this.registeredServices) {
      try {
        await this.deregisterService(serviceId);
      } catch (error) {
        this.logger.error('Failed to deregister service during shutdown', {
          serviceId,
          error: error.message
        });
      }
    }

    this.logger.info('Consul Service Registry shutdown completed');
  }

  private determineHealthStatus(checks: any[]): 'passing' | 'warning' | 'critical' {
    if (!checks || checks.length === 0) return 'passing';

    const hasCritical = checks.some(check => check.Status === 'critical');
    const hasWarning = checks.some(check => check.Status === 'warning');

    if (hasCritical) return 'critical';
    if (hasWarning) return 'warning';
    return 'passing';
  }
}
