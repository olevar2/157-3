/**
 * Performance Monitor for Platform3
 * Real-time monitoring and alerting for enterprise-grade forex trading platform
 * Tracks KPIs, SLAs, and business metrics across all microservices
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { RedisClient } from 'redis';
import { PrometheusRegistry, Counter, Histogram, Gauge } from 'prom-client';

interface MonitoringConfig {
  enabled: boolean;
  metricsInterval: number;
  alertThresholds: {
    latencyP99: number;
    errorRate: number;
    throughput: number;
    accuracy: number;
    availability: number;
  };
  services: string[];
  businessMetrics: boolean;
}

interface PerformanceMetrics {
  serviceId: string;
  timestamp: number;
  latency: {
    p50: number;
    p95: number;
    p99: number;
    avg: number;
  };
  throughput: {
    requestsPerSecond: number;
    tradesPerSecond: number;
    indicatorsPerSecond: number;
  };
  errors: {
    rate: number;
    count: number;
    types: Record<string, number>;
  };
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkIO: number;
  };
  business: {
    tradingAccuracy: number;
    profitLoss: number;
    riskMetrics: number;
    clientSatisfaction: number;
  };
}

interface Alert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  service: string;
  metric: string;
  threshold: number;
  currentValue: number;
  timestamp: number;
  description: string;
  resolved: boolean;
}

export class PerformanceMonitor extends EventEmitter {
  private config: MonitoringConfig;
  private logger: Logger;
  private redis: RedisClient;
  private registry: PrometheusRegistry;
  private metrics: Map<string, PerformanceMetrics[]>;
  private alerts: Alert[];
  private isMonitoring: boolean = false;

  // Prometheus metrics
  private latencyHistogram: Histogram<string>;
  private throughputCounter: Counter<string>;
  private errorCounter: Counter<string>;
  private resourceGauge: Gauge<string>;
  private businessGauge: Gauge<string>;

  constructor(config: MonitoringConfig, logger: Logger, redis: RedisClient) {
    super();
    this.config = config;
    this.logger = logger;
    this.redis = redis;
    this.registry = new PrometheusRegistry();
    this.metrics = new Map();
    this.alerts = [];

    this.initializePrometheusMetrics();
  }

  /**
   * Initialize Prometheus metrics
   */
  private initializePrometheusMetrics(): void {
    this.latencyHistogram = new Histogram({
      name: 'platform3_request_duration_seconds',
      help: 'Request duration in seconds',
      labelNames: ['service', 'endpoint', 'method'],
      buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    });

    this.throughputCounter = new Counter({
      name: 'platform3_requests_total',
      help: 'Total number of requests',
      labelNames: ['service', 'endpoint', 'status']
    });

    this.errorCounter = new Counter({
      name: 'platform3_errors_total',
      help: 'Total number of errors',
      labelNames: ['service', 'type', 'severity']
    });

    this.resourceGauge = new Gauge({
      name: 'platform3_resource_usage',
      help: 'Resource usage metrics',
      labelNames: ['service', 'resource']
    });

    this.businessGauge = new Gauge({
      name: 'platform3_business_metrics',
      help: 'Business performance metrics',
      labelNames: ['metric', 'timeframe']
    });

    // Register metrics
    this.registry.registerMetric(this.latencyHistogram);
    this.registry.registerMetric(this.throughputCounter);
    this.registry.registerMetric(this.errorCounter);
    this.registry.registerMetric(this.resourceGauge);
    this.registry.registerMetric(this.businessGauge);
  }

  /**
   * Start performance monitoring
   */
  async start(): Promise<void> {
    if (!this.config.enabled) {
      this.logger.info('Performance monitoring is disabled');
      return;
    }

    this.logger.info('Starting Performance Monitor');
    this.isMonitoring = true;

    // Start metrics collection
    setInterval(() => this.collectMetrics(), this.config.metricsInterval);
    
    // Start alert evaluation
    setInterval(() => this.evaluateAlerts(), 10000); // Every 10 seconds
    
    // Start cleanup
    setInterval(() => this.cleanupOldMetrics(), 300000); // Every 5 minutes

    this.logger.info('Performance Monitor started successfully');
  }

  /**
   * Stop performance monitoring
   */
  async stop(): Promise<void> {
    this.logger.info('Stopping Performance Monitor');
    this.isMonitoring = false;
    this.logger.info('Performance Monitor stopped');
  }

  /**
   * Collect metrics from all services
   */
  private async collectMetrics(): Promise<void> {
    if (!this.isMonitoring) return;

    try {
      const metricsPromises = this.config.services.map(service =>
        this.collectServiceMetrics(service)
      );

      const metricsResults = await Promise.allSettled(metricsPromises);

      metricsResults.forEach((result, index) => {
        const service = this.config.services[index];
        
        if (result.status === 'fulfilled') {
          this.storeMetrics(service, result.value);
          this.updatePrometheusMetrics(service, result.value);
        } else {
          this.logger.error(`Failed to collect metrics for ${service}:`, result.reason);
        }
      });

      // Collect business metrics if enabled
      if (this.config.businessMetrics) {
        await this.collectBusinessMetrics();
      }

    } catch (error) {
      this.logger.error('Error collecting metrics:', error);
    }
  }

  /**
   * Collect metrics for a specific service
   */
  private async collectServiceMetrics(service: string): Promise<PerformanceMetrics> {
    try {
      // Call service metrics endpoint
      const response = await fetch(`http://${service}:3000/metrics`, {
        method: 'GET',
        timeout: 5000
      });

      if (!response.ok) {
        throw new Error(`Metrics endpoint failed: ${response.statusText}`);
      }

      const rawMetrics = await response.json();

      // Transform to standardized format
      const metrics: PerformanceMetrics = {
        serviceId: service,
        timestamp: Date.now(),
        latency: {
          p50: rawMetrics.latency?.p50 || 0,
          p95: rawMetrics.latency?.p95 || 0,
          p99: rawMetrics.latency?.p99 || 0,
          avg: rawMetrics.latency?.avg || 0
        },
        throughput: {
          requestsPerSecond: rawMetrics.throughput?.requests || 0,
          tradesPerSecond: rawMetrics.throughput?.trades || 0,
          indicatorsPerSecond: rawMetrics.throughput?.indicators || 0
        },
        errors: {
          rate: rawMetrics.errors?.rate || 0,
          count: rawMetrics.errors?.count || 0,
          types: rawMetrics.errors?.types || {}
        },
        resources: {
          cpuUsage: rawMetrics.resources?.cpu || 0,
          memoryUsage: rawMetrics.resources?.memory || 0,
          diskUsage: rawMetrics.resources?.disk || 0,
          networkIO: rawMetrics.resources?.network || 0
        },
        business: {
          tradingAccuracy: rawMetrics.business?.accuracy || 0,
          profitLoss: rawMetrics.business?.pnl || 0,
          riskMetrics: rawMetrics.business?.risk || 0,
          clientSatisfaction: rawMetrics.business?.satisfaction || 0
        }
      };

      return metrics;

    } catch (error) {
      this.logger.error(`Error collecting metrics for ${service}:`, error);
      throw error;
    }
  }

  /**
   * Collect business metrics
   */
  private async collectBusinessMetrics(): Promise<void> {
    try {
      // Collect trading performance metrics
      const tradingMetrics = await this.collectTradingMetrics();
      
      // Update business gauges
      this.businessGauge.set(
        { metric: 'trading_accuracy', timeframe: '1h' },
        tradingMetrics.accuracy
      );
      
      this.businessGauge.set(
        { metric: 'profit_loss', timeframe: '1h' },
        tradingMetrics.profitLoss
      );
      
      this.businessGauge.set(
        { metric: 'risk_score', timeframe: '1h' },
        tradingMetrics.riskScore
      );

    } catch (error) {
      this.logger.error('Error collecting business metrics:', error);
    }
  }

  /**
   * Collect trading performance metrics
   */
  private async collectTradingMetrics(): Promise<any> {
    // This would integrate with your trading service to get real metrics
    // For now, we'll simulate realistic trading metrics
    
    return {
      accuracy: 0.78 + Math.random() * 0.1, // 78-88% accuracy
      profitLoss: (Math.random() - 0.5) * 1000, // -500 to +500
      riskScore: Math.random() * 0.3 // 0-30% risk
    };
  }

  /**
   * Store metrics in memory and Redis
   */
  private storeMetrics(service: string, metrics: PerformanceMetrics): void {
    // Store in memory
    if (!this.metrics.has(service)) {
      this.metrics.set(service, []);
    }
    
    const serviceMetrics = this.metrics.get(service)!;
    serviceMetrics.push(metrics);
    
    // Keep only last 1000 metrics per service
    if (serviceMetrics.length > 1000) {
      serviceMetrics.splice(0, serviceMetrics.length - 1000);
    }

    // Store in Redis for persistence
    this.redis.lpush(
      `platform3:metrics:${service}`,
      JSON.stringify(metrics)
    );
    
    // Keep only last 24 hours in Redis
    this.redis.ltrim(`platform3:metrics:${service}`, 0, 1440); // 1440 minutes = 24 hours
  }

  /**
   * Update Prometheus metrics
   */
  private updatePrometheusMetrics(service: string, metrics: PerformanceMetrics): void {
    // Update latency histogram
    this.latencyHistogram.observe(
      { service, endpoint: 'all', method: 'all' },
      metrics.latency.avg / 1000 // Convert to seconds
    );

    // Update throughput counter
    this.throughputCounter.inc(
      { service, endpoint: 'all', status: 'success' },
      metrics.throughput.requestsPerSecond
    );

    // Update error counter
    this.errorCounter.inc(
      { service, type: 'all', severity: 'all' },
      metrics.errors.count
    );

    // Update resource gauges
    this.resourceGauge.set(
      { service, resource: 'cpu' },
      metrics.resources.cpuUsage
    );
    
    this.resourceGauge.set(
      { service, resource: 'memory' },
      metrics.resources.memoryUsage
    );
  }

  /**
   * Evaluate alerts based on thresholds
   */
  private async evaluateAlerts(): Promise<void> {
    try {
      for (const [service, serviceMetrics] of this.metrics) {
        if (serviceMetrics.length === 0) continue;

        const latestMetrics = serviceMetrics[serviceMetrics.length - 1];
        
        // Check latency threshold
        if (latestMetrics.latency.p99 > this.config.alertThresholds.latencyP99) {
          this.createAlert(
            'high',
            service,
            'latency_p99',
            this.config.alertThresholds.latencyP99,
            latestMetrics.latency.p99,
            `High P99 latency detected: ${latestMetrics.latency.p99}ms`
          );
        }

        // Check error rate threshold
        if (latestMetrics.errors.rate > this.config.alertThresholds.errorRate) {
          this.createAlert(
            'high',
            service,
            'error_rate',
            this.config.alertThresholds.errorRate,
            latestMetrics.errors.rate,
            `High error rate detected: ${(latestMetrics.errors.rate * 100).toFixed(2)}%`
          );
        }

        // Check throughput threshold
        if (latestMetrics.throughput.requestsPerSecond < this.config.alertThresholds.throughput) {
          this.createAlert(
            'medium',
            service,
            'throughput',
            this.config.alertThresholds.throughput,
            latestMetrics.throughput.requestsPerSecond,
            `Low throughput detected: ${latestMetrics.throughput.requestsPerSecond} req/s`
          );
        }

        // Check business metrics
        if (this.config.businessMetrics && latestMetrics.business.tradingAccuracy < this.config.alertThresholds.accuracy) {
          this.createAlert(
            'critical',
            service,
            'trading_accuracy',
            this.config.alertThresholds.accuracy,
            latestMetrics.business.tradingAccuracy,
            `Low trading accuracy: ${(latestMetrics.business.tradingAccuracy * 100).toFixed(2)}%`
          );
        }
      }
    } catch (error) {
      this.logger.error('Error evaluating alerts:', error);
    }
  }

  /**
   * Create an alert
   */
  private createAlert(
    severity: Alert['severity'],
    service: string,
    metric: string,
    threshold: number,
    currentValue: number,
    description: string
  ): void {
    const alertId = `${service}-${metric}-${Date.now()}`;
    
    const alert: Alert = {
      id: alertId,
      severity,
      service,
      metric,
      threshold,
      currentValue,
      timestamp: Date.now(),
      description,
      resolved: false
    };

    this.alerts.push(alert);
    
    this.logger.warn(`Alert created: ${description}`, alert);
    this.emit('alert-created', alert);

    // Store alert in Redis
    this.redis.lpush('platform3:alerts', JSON.stringify(alert));
  }

  /**
   * Get current performance dashboard
   */
  getPerformanceDashboard(): any {
    const dashboard: any = {
      timestamp: Date.now(),
      services: {},
      summary: {
        totalServices: this.config.services.length,
        healthyServices: 0,
        avgLatency: 0,
        totalThroughput: 0,
        totalErrors: 0
      },
      alerts: {
        active: this.alerts.filter(a => !a.resolved).length,
        critical: this.alerts.filter(a => a.severity === 'critical' && !a.resolved).length,
        high: this.alerts.filter(a => a.severity === 'high' && !a.resolved).length
      }
    };

    // Calculate service-level metrics
    for (const [service, serviceMetrics] of this.metrics) {
      if (serviceMetrics.length === 0) continue;

      const latest = serviceMetrics[serviceMetrics.length - 1];
      const last10 = serviceMetrics.slice(-10);
      
      dashboard.services[service] = {
        status: latest.errors.rate < 0.01 ? 'healthy' : 'degraded',
        latency: latest.latency,
        throughput: latest.throughput,
        errors: latest.errors,
        resources: latest.resources,
        trends: {
          latencyTrend: this.calculateTrend(last10.map(m => m.latency.avg)),
          throughputTrend: this.calculateTrend(last10.map(m => m.throughput.requestsPerSecond)),
          errorTrend: this.calculateTrend(last10.map(m => m.errors.rate))
        }
      };

      // Update summary
      if (dashboard.services[service].status === 'healthy') {
        dashboard.summary.healthyServices++;
      }
      dashboard.summary.avgLatency += latest.latency.avg;
      dashboard.summary.totalThroughput += latest.throughput.requestsPerSecond;
      dashboard.summary.totalErrors += latest.errors.count;
    }

    // Finalize summary calculations
    dashboard.summary.avgLatency /= Object.keys(dashboard.services).length;

    return dashboard;
  }

  /**
   * Calculate trend direction
   */
  private calculateTrend(values: number[]): 'up' | 'down' | 'stable' {
    if (values.length < 2) return 'stable';
    
    const first = values[0];
    const last = values[values.length - 1];
    const change = (last - first) / first;
    
    if (change > 0.1) return 'up';
    if (change < -0.1) return 'down';
    return 'stable';
  }

  /**
   * Clean up old metrics
   */
  private cleanupOldMetrics(): void {
    const cutoffTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours ago
    
    for (const [service, serviceMetrics] of this.metrics) {
      const filteredMetrics = serviceMetrics.filter(m => m.timestamp > cutoffTime);
      this.metrics.set(service, filteredMetrics);
    }

    // Clean up old alerts
    this.alerts = this.alerts.filter(a => a.timestamp > cutoffTime);
  }

  /**
   * Get Prometheus metrics
   */
  async getPrometheusMetrics(): Promise<string> {
    return this.registry.metrics();
  }
}
