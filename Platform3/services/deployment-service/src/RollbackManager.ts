/**
 * Rollback Manager for Platform3
 * Manages automated rollback mechanisms for microservices
 * Ensures system stability and rapid recovery from issues
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { RedisClient } from 'redis';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

interface RollbackConfig {
  enabled: boolean;
  healthCheckInterval: number;
  rollbackThresholds: {
    latencyMs: number;
    errorRate: number;
    accuracyThreshold: number;
    resourceUsage: number;
  };
  services: ServiceConfig[];
}

interface ServiceConfig {
  name: string;
  version: string;
  previousVersion: string;
  healthEndpoint: string;
  rollbackCommand: string;
  dependencies: string[];
}

interface HealthMetrics {
  serviceId: string;
  timestamp: number;
  latency: number;
  errorRate: number;
  accuracy: number;
  cpuUsage: number;
  memoryUsage: number;
  isHealthy: boolean;
}

interface RollbackEvent {
  serviceId: string;
  reason: string;
  timestamp: number;
  fromVersion: string;
  toVersion: string;
  status: 'initiated' | 'in-progress' | 'completed' | 'failed';
  duration?: number;
}

export class RollbackManager extends EventEmitter {
  private config: RollbackConfig;
  private logger: Logger;
  private redis: RedisClient;
  private healthMetrics: Map<string, HealthMetrics[]>;
  private rollbackHistory: RollbackEvent[];
  private isMonitoring: boolean = false;
  private healthCheckTimer?: NodeJS.Timeout;

  constructor(config: RollbackConfig, logger: Logger, redis: RedisClient) {
    super();
    this.config = config;
    this.logger = logger;
    this.redis = redis;
    this.healthMetrics = new Map();
    this.rollbackHistory = [];
  }

  /**
   * Start rollback monitoring
   */
  async start(): Promise<void> {
    if (!this.config.enabled) {
      this.logger.info('Rollback manager is disabled');
      return;
    }

    this.logger.info('Starting Rollback Manager');
    this.isMonitoring = true;

    // Start health check monitoring
    this.healthCheckTimer = setInterval(
      () => this.performHealthChecks(),
      this.config.healthCheckInterval
    );

    // Load rollback history from Redis
    await this.loadRollbackHistory();

    this.logger.info('Rollback Manager started successfully');
  }

  /**
   * Stop rollback monitoring
   */
  async stop(): Promise<void> {
    this.logger.info('Stopping Rollback Manager');
    this.isMonitoring = false;

    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    // Save rollback history to Redis
    await this.saveRollbackHistory();

    this.logger.info('Rollback Manager stopped');
  }

  /**
   * Perform health checks on all services
   */
  private async performHealthChecks(): Promise<void> {
    if (!this.isMonitoring) return;

    try {
      const healthCheckPromises = this.config.services.map(service =>
        this.checkServiceHealth(service)
      );

      const healthResults = await Promise.allSettled(healthCheckPromises);

      healthResults.forEach((result, index) => {
        const service = this.config.services[index];
        
        if (result.status === 'fulfilled') {
          this.processHealthMetrics(service.name, result.value);
        } else {
          this.logger.error(`Health check failed for ${service.name}:`, result.reason);
          this.triggerRollback(service.name, 'health-check-failure');
        }
      });

    } catch (error) {
      this.logger.error('Error performing health checks:', error);
    }
  }

  /**
   * Check health of a specific service
   */
  private async checkServiceHealth(service: ServiceConfig): Promise<HealthMetrics> {
    const startTime = Date.now();

    try {
      // Call service health endpoint
      const response = await fetch(service.healthEndpoint, {
        method: 'GET',
        timeout: 5000 // 5 second timeout
      });

      const latency = Date.now() - startTime;
      const healthData = await response.json();

      // Extract metrics from health response
      const metrics: HealthMetrics = {
        serviceId: service.name,
        timestamp: Date.now(),
        latency,
        errorRate: healthData.errorRate || 0,
        accuracy: healthData.accuracy || 1.0,
        cpuUsage: healthData.cpuUsage || 0,
        memoryUsage: healthData.memoryUsage || 0,
        isHealthy: response.ok && this.evaluateHealth(healthData, latency)
      };

      return metrics;

    } catch (error) {
      this.logger.error(`Health check failed for ${service.name}:`, error);
      
      return {
        serviceId: service.name,
        timestamp: Date.now(),
        latency: Date.now() - startTime,
        errorRate: 1.0, // 100% error rate on failure
        accuracy: 0,
        cpuUsage: 0,
        memoryUsage: 0,
        isHealthy: false
      };
    }
  }

  /**
   * Evaluate if service is healthy based on metrics
   */
  private evaluateHealth(healthData: any, latency: number): boolean {
    const thresholds = this.config.rollbackThresholds;

    return (
      latency <= thresholds.latencyMs &&
      (healthData.errorRate || 0) <= thresholds.errorRate &&
      (healthData.accuracy || 1.0) >= thresholds.accuracyThreshold &&
      (healthData.cpuUsage || 0) <= thresholds.resourceUsage &&
      (healthData.memoryUsage || 0) <= thresholds.resourceUsage
    );
  }

  /**
   * Process health metrics and trigger rollback if needed
   */
  private processHealthMetrics(serviceId: string, metrics: HealthMetrics): void {
    // Store metrics
    if (!this.healthMetrics.has(serviceId)) {
      this.healthMetrics.set(serviceId, []);
    }

    const serviceMetrics = this.healthMetrics.get(serviceId)!;
    serviceMetrics.push(metrics);

    // Keep only last 100 metrics per service
    if (serviceMetrics.length > 100) {
      serviceMetrics.splice(0, serviceMetrics.length - 100);
    }

    // Check if rollback is needed
    if (!metrics.isHealthy) {
      this.evaluateRollbackNeed(serviceId, metrics);
    }

    // Emit metrics for monitoring
    this.emit('health-metrics', metrics);
  }

  /**
   * Evaluate if rollback is needed based on recent metrics
   */
  private evaluateRollbackNeed(serviceId: string, currentMetrics: HealthMetrics): void {
    const serviceMetrics = this.healthMetrics.get(serviceId) || [];
    const recentMetrics = serviceMetrics.slice(-5); // Last 5 metrics

    // Check for consistent failures
    const unhealthyCount = recentMetrics.filter(m => !m.isHealthy).length;
    
    if (unhealthyCount >= 3) { // 3 out of 5 recent checks failed
      let reason = 'multiple-health-failures';

      // Determine specific reason
      if (currentMetrics.latency > this.config.rollbackThresholds.latencyMs) {
        reason = 'high-latency';
      } else if (currentMetrics.errorRate > this.config.rollbackThresholds.errorRate) {
        reason = 'high-error-rate';
      } else if (currentMetrics.accuracy < this.config.rollbackThresholds.accuracyThreshold) {
        reason = 'low-accuracy';
      } else if (currentMetrics.cpuUsage > this.config.rollbackThresholds.resourceUsage ||
                 currentMetrics.memoryUsage > this.config.rollbackThresholds.resourceUsage) {
        reason = 'resource-exhaustion';
      }

      this.triggerRollback(serviceId, reason);
    }
  }

  /**
   * Trigger rollback for a service
   */
  async triggerRollback(serviceId: string, reason: string): Promise<void> {
    try {
      const service = this.config.services.find(s => s.name === serviceId);
      if (!service) {
        this.logger.error(`Service configuration not found for ${serviceId}`);
        return;
      }

      // Check if rollback is already in progress
      const ongoingRollback = this.rollbackHistory.find(
        r => r.serviceId === serviceId && r.status === 'in-progress'
      );

      if (ongoingRollback) {
        this.logger.warn(`Rollback already in progress for ${serviceId}`);
        return;
      }

      this.logger.warn(`Triggering rollback for ${serviceId}, reason: ${reason}`);

      const rollbackEvent: RollbackEvent = {
        serviceId,
        reason,
        timestamp: Date.now(),
        fromVersion: service.version,
        toVersion: service.previousVersion,
        status: 'initiated'
      };

      this.rollbackHistory.push(rollbackEvent);
      this.emit('rollback-initiated', rollbackEvent);

      // Execute rollback
      await this.executeRollback(service, rollbackEvent);

    } catch (error) {
      this.logger.error(`Error triggering rollback for ${serviceId}:`, error);
    }
  }

  /**
   * Execute the actual rollback
   */
  private async executeRollback(service: ServiceConfig, rollbackEvent: RollbackEvent): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Update status
      rollbackEvent.status = 'in-progress';
      this.emit('rollback-progress', rollbackEvent);

      this.logger.info(`Executing rollback for ${service.name} from ${service.version} to ${service.previousVersion}`);

      // Execute rollback command
      const { stdout, stderr } = await execAsync(service.rollbackCommand);
      
      if (stderr) {
        this.logger.warn(`Rollback stderr for ${service.name}:`, stderr);
      }

      this.logger.info(`Rollback stdout for ${service.name}:`, stdout);

      // Wait for service to stabilize
      await this.waitForServiceStability(service);

      // Update status
      rollbackEvent.status = 'completed';
      rollbackEvent.duration = Date.now() - startTime;
      
      this.logger.info(`Rollback completed for ${service.name} in ${rollbackEvent.duration}ms`);
      this.emit('rollback-completed', rollbackEvent);

    } catch (error) {
      rollbackEvent.status = 'failed';
      rollbackEvent.duration = Date.now() - startTime;
      
      this.logger.error(`Rollback failed for ${service.name}:`, error);
      this.emit('rollback-failed', rollbackEvent);
      
      // Trigger emergency procedures
      await this.triggerEmergencyProcedures(service.name);
    }
  }

  /**
   * Wait for service to stabilize after rollback
   */
  private async waitForServiceStability(service: ServiceConfig): Promise<void> {
    const maxWaitTime = 60000; // 1 minute
    const checkInterval = 5000; // 5 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      try {
        const metrics = await this.checkServiceHealth(service);
        
        if (metrics.isHealthy) {
          this.logger.info(`Service ${service.name} is stable after rollback`);
          return;
        }

        await new Promise(resolve => setTimeout(resolve, checkInterval));
      } catch (error) {
        this.logger.warn(`Health check failed during stability wait for ${service.name}:`, error);
        await new Promise(resolve => setTimeout(resolve, checkInterval));
      }
    }

    throw new Error(`Service ${service.name} did not stabilize within ${maxWaitTime}ms`);
  }

  /**
   * Trigger emergency procedures when rollback fails
   */
  private async triggerEmergencyProcedures(serviceId: string): Promise<void> {
    this.logger.error(`Triggering emergency procedures for ${serviceId}`);
    
    // Emit emergency event for external systems
    this.emit('emergency-rollback-failure', {
      serviceId,
      timestamp: Date.now(),
      action: 'manual-intervention-required'
    });

    // Could trigger additional emergency actions like:
    // - Alerting on-call engineers
    // - Disabling traffic to the service
    // - Activating backup systems
  }

  /**
   * Manual rollback trigger
   */
  async manualRollback(serviceId: string, reason: string = 'manual-trigger'): Promise<void> {
    this.logger.info(`Manual rollback triggered for ${serviceId}`);
    await this.triggerRollback(serviceId, reason);
  }

  /**
   * Get rollback statistics
   */
  getRollbackStatistics(): any {
    const last24Hours = Date.now() - (24 * 60 * 60 * 1000);
    const recentRollbacks = this.rollbackHistory.filter(r => r.timestamp > last24Hours);

    const stats = {
      total: this.rollbackHistory.length,
      last24Hours: recentRollbacks.length,
      byService: {} as any,
      byReason: {} as any,
      successRate: 0
    };

    // Calculate statistics
    this.rollbackHistory.forEach(rollback => {
      // By service
      if (!stats.byService[rollback.serviceId]) {
        stats.byService[rollback.serviceId] = 0;
      }
      stats.byService[rollback.serviceId]++;

      // By reason
      if (!stats.byReason[rollback.reason]) {
        stats.byReason[rollback.reason] = 0;
      }
      stats.byReason[rollback.reason]++;
    });

    // Success rate
    const completedRollbacks = this.rollbackHistory.filter(r => r.status === 'completed');
    stats.successRate = completedRollbacks.length / this.rollbackHistory.length;

    return stats;
  }

  /**
   * Load rollback history from Redis
   */
  private async loadRollbackHistory(): Promise<void> {
    try {
      const historyData = await this.redis.get('platform3:rollback:history');
      if (historyData) {
        this.rollbackHistory = JSON.parse(historyData);
      }
    } catch (error) {
      this.logger.error('Error loading rollback history:', error);
    }
  }

  /**
   * Save rollback history to Redis
   */
  private async saveRollbackHistory(): Promise<void> {
    try {
      await this.redis.set(
        'platform3:rollback:history',
        JSON.stringify(this.rollbackHistory),
        'EX',
        7 * 24 * 60 * 60 // 7 days TTL
      );
    } catch (error) {
      this.logger.error('Error saving rollback history:', error);
    }
  }
}
