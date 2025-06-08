/**
 * Platform3 Service Discovery and Health Management System
 * Comprehensive microservices communication enhancement framework
 */

import { EventEmitter } from 'events';
import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { createHash } from 'crypto';

// Core interfaces for service communication
export interface ServiceInstance {
  id: string;
  name: string;
  version: string;
  host: string;
  port: number;
  status: ServiceStatus;
  healthEndpoint: string;
  metadata: Record<string, any>;
  lastHeartbeat: Date;
  registrationTime: Date;
}

export interface ServiceMessage {
  correlationId: string;
  service: string;
  timestamp: Date;
  payload: any;
  retryCount?: number;
  priority?: MessagePriority;
  ttl?: number;
}

export interface HealthCheckResult {
  status: ServiceStatus;
  timestamp: Date;
  latency: number;
  dependencies: DependencyHealth[];
  metrics: ServiceMetrics;
}

export interface ServiceMetrics {
  cpu: number;
  memory: number;
  activeConnections: number;
  requestsPerSecond: number;
  errorRate: number;
}

export interface DependencyHealth {
  service: string;
  status: ServiceStatus;
  latency: number;
}

export enum ServiceStatus {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy',
  UNKNOWN = 'unknown'
}

export enum MessagePriority {
  LOW = 1,
  NORMAL = 2,
  HIGH = 3,
  CRITICAL = 4
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  recoveryTimeout: number;
  monitoringPeriod: number;
  halfOpenMaxCalls: number;
}

export enum CircuitBreakerState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open'
}

export class ServiceRegistry extends EventEmitter {
  private services: Map<string, ServiceInstance[]> = new Map();
  private healthChecks: Map<string, NodeJS.Timeout> = new Map();
  private httpClient: AxiosInstance;
  private registryConfig: {
    healthCheckInterval: number;
    serviceTimeout: number;
    maxRetries: number;
  };

  constructor(config?: any) {
    super();
    this.registryConfig = {
      healthCheckInterval: config?.healthCheckInterval || 30000,
      serviceTimeout: config?.serviceTimeout || 5000,
      maxRetries: config?.maxRetries || 3,
      ...config
    };

    this.httpClient = axios.create({
      timeout: this.registryConfig.serviceTimeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Platform3-ServiceRegistry/1.0'
      }
    });
  }

  /**
   * Register a new service instance
   */
  registerService(service: Omit<ServiceInstance, 'id' | 'registrationTime' | 'lastHeartbeat'>): string {
    const serviceId = this.generateServiceId(service.name, service.host, service.port);
    
    const serviceInstance: ServiceInstance = {
      ...service,
      id: serviceId,
      registrationTime: new Date(),
      lastHeartbeat: new Date()
    };

    if (!this.services.has(service.name)) {
      this.services.set(service.name, []);
    }

    const instances = this.services.get(service.name)!;
    const existingIndex = instances.findIndex(inst => inst.id === serviceId);
    
    if (existingIndex >= 0) {
      instances[existingIndex] = serviceInstance;
    } else {
      instances.push(serviceInstance);
    }

    // Start health monitoring
    this.startHealthMonitoring(serviceInstance);
    
    this.emit('serviceRegistered', serviceInstance);
    return serviceId;
  }

  /**
   * Deregister a service instance
   */
  deregisterService(serviceName: string, serviceId: string): boolean {
    const instances = this.services.get(serviceName);
    if (!instances) return false;

    const index = instances.findIndex(inst => inst.id === serviceId);
    if (index >= 0) {
      const removedInstance = instances.splice(index, 1)[0];
      
      // Stop health monitoring
      this.stopHealthMonitoring(serviceId);
      
      this.emit('serviceDeregistered', removedInstance);
      return true;
    }

    return false;
  }

  /**
   * Get healthy service instances
   */
  getHealthyInstances(serviceName: string): ServiceInstance[] {
    const instances = this.services.get(serviceName) || [];
    return instances.filter(inst => inst.status === ServiceStatus.HEALTHY);
  }

  /**
   * Get service instance using load balancing
   */
  getServiceInstance(serviceName: string, strategy: 'round-robin' | 'random' | 'least-connections' = 'round-robin'): ServiceInstance | null {
    const healthyInstances = this.getHealthyInstances(serviceName);
    
    if (healthyInstances.length === 0) {
      return null;
    }

    switch (strategy) {
      case 'random':
        return healthyInstances[Math.floor(Math.random() * healthyInstances.length)];
      
      case 'least-connections':
        return healthyInstances.reduce((least, current) => 
          current.metadata.activeConnections < least.metadata.activeConnections ? current : least
        );
      
      case 'round-robin':
      default:
        // Simple round-robin based on timestamp
        const timestamp = Date.now();
        const index = timestamp % healthyInstances.length;
        return healthyInstances[index];
    }
  }

  /**
   * Perform health check on service instance
   */
  private async performHealthCheck(service: ServiceInstance): Promise<HealthCheckResult> {
    const startTime = Date.now();
    
    try {
      const response: AxiosResponse = await this.httpClient.get(
        `http://${service.host}:${service.port}${service.healthEndpoint}`
      );
      
      const latency = Date.now() - startTime;
      
      if (response.status === 200) {
        const healthData = response.data || {};
        
        return {
          status: this.determineHealthStatus(healthData, latency),
          timestamp: new Date(),
          latency,
          dependencies: healthData.dependencies || [],
          metrics: {
            cpu: healthData.cpu || 0,
            memory: healthData.memory || 0,
            activeConnections: healthData.activeConnections || 0,
            requestsPerSecond: healthData.requestsPerSecond || 0,
            errorRate: healthData.errorRate || 0
          }
        };
      }
    } catch (error) {
      return {
        status: ServiceStatus.UNHEALTHY,
        timestamp: new Date(),
        latency: Date.now() - startTime,
        dependencies: [],
        metrics: {
          cpu: 0,
          memory: 0,
          activeConnections: 0,
          requestsPerSecond: 0,
          errorRate: 100
        }
      };
    }

    return {
      status: ServiceStatus.UNKNOWN,
      timestamp: new Date(),
      latency: Date.now() - startTime,
      dependencies: [],
      metrics: {
        cpu: 0,
        memory: 0,
        activeConnections: 0,
        requestsPerSecond: 0,
        errorRate: 0
      }
    };
  }

  /**
   * Determine health status based on response data and latency
   */
  private determineHealthStatus(healthData: any, latency: number): ServiceStatus {
    const errorRate = healthData.errorRate || 0;
    const cpuUsage = healthData.cpu || 0;
    const memoryUsage = healthData.memory || 0;

    // Critical thresholds
    if (errorRate > 50 || latency > 5000 || cpuUsage > 90 || memoryUsage > 90) {
      return ServiceStatus.UNHEALTHY;
    }

    // Degraded thresholds
    if (errorRate > 10 || latency > 2000 || cpuUsage > 70 || memoryUsage > 70) {
      return ServiceStatus.DEGRADED;
    }

    return ServiceStatus.HEALTHY;
  }

  /**
   * Start health monitoring for a service
   */
  private startHealthMonitoring(service: ServiceInstance): void {
    const checkHealth = async () => {
      const healthResult = await this.performHealthCheck(service);
      
      // Update service status
      service.status = healthResult.status;
      service.lastHeartbeat = healthResult.timestamp;
      service.metadata = {
        ...service.metadata,
        ...healthResult.metrics
      };

      this.emit('healthCheck', service, healthResult);
      
      // If service is unhealthy for too long, remove it
      if (healthResult.status === ServiceStatus.UNHEALTHY) {
        const timeSinceLastHealthy = Date.now() - service.lastHeartbeat.getTime();
        if (timeSinceLastHealthy > 300000) { // 5 minutes
          this.deregisterService(service.name, service.id);
        }
      }
    };

    const interval = setInterval(checkHealth, this.registryConfig.healthCheckInterval);
    this.healthChecks.set(service.id, interval);
    
    // Perform initial health check
    checkHealth();
  }

  /**
   * Stop health monitoring for a service
   */
  private stopHealthMonitoring(serviceId: string): void {
    const interval = this.healthChecks.get(serviceId);
    if (interval) {
      clearInterval(interval);
      this.healthChecks.delete(serviceId);
    }
  }

  /**
   * Generate unique service ID
   */
  private generateServiceId(name: string, host: string, port: number): string {
    const data = `${name}:${host}:${port}:${Date.now()}`;
    return createHash('md5').update(data).digest('hex').substring(0, 16);
  }

  /**
   * Get all registered services
   */
  getAllServices(): Map<string, ServiceInstance[]> {
    return new Map(this.services);
  }

  /**
   * Get service statistics
   */
  getServiceStats(serviceName: string): any {
    const instances = this.services.get(serviceName) || [];
    
    const stats = {
      totalInstances: instances.length,
      healthyInstances: instances.filter(i => i.status === ServiceStatus.HEALTHY).length,
      degradedInstances: instances.filter(i => i.status === ServiceStatus.DEGRADED).length,
      unhealthyInstances: instances.filter(i => i.status === ServiceStatus.UNHEALTHY).length,
      averageLatency: 0,
      totalRequests: 0,
      errorRate: 0
    };

    if (instances.length > 0) {
      const metrics = instances.map(i => i.metadata);
      stats.averageLatency = metrics.reduce((sum, m) => sum + (m.latency || 0), 0) / instances.length;
      stats.totalRequests = metrics.reduce((sum, m) => sum + (m.requestsPerSecond || 0), 0);
      stats.errorRate = metrics.reduce((sum, m) => sum + (m.errorRate || 0), 0) / instances.length;
    }

    return stats;
  }
}

export class CircuitBreaker {
  private state: CircuitBreakerState = CircuitBreakerState.CLOSED;
  private failureCount: number = 0;
  private lastFailureTime: number = 0;
  private halfOpenCalls: number = 0;
  private config: CircuitBreakerConfig;

  constructor(config: CircuitBreakerConfig) {
    this.config = config;
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitBreakerState.OPEN) {
      if (Date.now() - this.lastFailureTime < this.config.recoveryTimeout) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.state = CircuitBreakerState.HALF_OPEN;
      this.halfOpenCalls = 0;
    }

    if (this.state === CircuitBreakerState.HALF_OPEN && this.halfOpenCalls >= this.config.halfOpenMaxCalls) {
      throw new Error('Circuit breaker is HALF_OPEN with max calls reached');
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

  private onSuccess(): void {
    this.failureCount = 0;
    if (this.state === CircuitBreakerState.HALF_OPEN) {
      this.state = CircuitBreakerState.CLOSED;
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.state === CircuitBreakerState.HALF_OPEN) {
      this.state = CircuitBreakerState.OPEN;
    } else if (this.failureCount >= this.config.failureThreshold) {
      this.state = CircuitBreakerState.OPEN;
    }
  }

  getState(): CircuitBreakerState {
    return this.state;
  }

  getMetrics(): any {
    return {
      state: this.state,
      failureCount: this.failureCount,
      lastFailureTime: this.lastFailureTime
    };
  }
}

export default {
  ServiceRegistry,
  CircuitBreaker,
  ServiceStatus,
  MessagePriority,
  CircuitBreakerState
};
