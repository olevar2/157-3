import { EventEmitter } from 'events';
import { LatencyMeasurer, LatencyStats } from './latency-measurement';

export interface ProfileConfig {
    sampleInterval: number; // milliseconds
    alertThresholds: {
        latencyMs: number;
        errorRate: number;
        p95Ms: number;
        p99Ms: number;
    };
    retentionPeriod: number; // milliseconds
}

export interface PerformanceAlert {
    timestamp: number;
    operation: string;
    metric: string;
    value: number;
    threshold: number;
    severity: 'warning' | 'critical';
}

export class PerformanceProfiler extends EventEmitter {
    private latencyMeasurer: LatencyMeasurer;
    private config: ProfileConfig;
    private profileTimer: NodeJS.Timeout | null = null;
    private alerts: PerformanceAlert[] = [];
    private isRunning: boolean = false;

    constructor(config: ProfileConfig) {
        super();
        this.config = config;
        this.latencyMeasurer = new LatencyMeasurer();
        
        // Forward latency events
        this.latencyMeasurer.on('measurement:completed', (measurement) => {
            this.emit('measurement', measurement);
            this.checkAlerts(measurement.operation);
        });
    }

    /**
     * Start continuous profiling
     */
    start(): void {
        if (this.isRunning) {
            return;
        }

        this.isRunning = true;
        this.profileTimer = setInterval(() => {
            this.performProfileCheck();
        }, this.config.sampleInterval);

        this.emit('profiler:started');
    }    /**
     * Stop profiling
     */
    stop(): void {
        if (!this.isRunning) {
            return;
        }

        this.isRunning = false;
        if (this.profileTimer) {
            clearInterval(this.profileTimer);
            this.profileTimer = null;
        }

        this.emit('profiler:stopped');
    }

    /**
     * Get latency measurer instance for direct access
     */
    getLatencyMeasurer(): LatencyMeasurer {
        return this.latencyMeasurer;
    }

    /**
     * Perform periodic profile check
     */
    private performProfileCheck(): void {
        const operations = this.latencyMeasurer.getOperations();
        
        for (const operation of operations) {
            this.checkAlerts(operation);
        }

        // Clean old alerts
        this.cleanOldAlerts();
    }

    /**
     * Check for performance alerts on an operation
     */
    private checkAlerts(operation: string): void {
        const stats = this.latencyMeasurer.getStats(operation, this.config.retentionPeriod);
        
        if (stats.count === 0) {
            return;
        }

        const thresholds = this.config.alertThresholds;
        const now = Date.now();

        // Check latency alert
        if (stats.mean > thresholds.latencyMs) {
            this.createAlert(now, operation, 'mean_latency', stats.mean, thresholds.latencyMs, 'warning');
        }

        // Check error rate alert
        if (stats.errorRate > thresholds.errorRate) {
            this.createAlert(now, operation, 'error_rate', stats.errorRate, thresholds.errorRate, 'critical');
        }        // Check P95 alert
        if (stats.p95 > thresholds.p95Ms) {
            this.createAlert(now, operation, 'p95_latency', stats.p95, thresholds.p95Ms, 'warning');
        }

        // Check P99 alert
        if (stats.p99 > thresholds.p99Ms) {
            this.createAlert(now, operation, 'p99_latency', stats.p99, thresholds.p99Ms, 'critical');
        }
    }

    /**
     * Create and emit an alert
     */
    private createAlert(timestamp: number, operation: string, metric: string, value: number, threshold: number, severity: 'warning' | 'critical'): void {
        const alert: PerformanceAlert = {
            timestamp,
            operation,
            metric,
            value,
            threshold,
            severity
        };

        this.alerts.push(alert);
        this.emit('alert', alert);
    }

    /**
     * Clean old alerts beyond retention period
     */
    private cleanOldAlerts(): void {
        const cutoff = Date.now() - this.config.retentionPeriod;
        this.alerts = this.alerts.filter(alert => alert.timestamp >= cutoff);
    }

    /**
     * Get recent alerts
     */
    getAlerts(timeWindowMs?: number): PerformanceAlert[] {
        if (!timeWindowMs) {
            return [...this.alerts];
        }

        const cutoff = Date.now() - timeWindowMs;
        return this.alerts.filter(alert => alert.timestamp >= cutoff);
    }

    /**
     * Get performance statistics for an operation
     */
    getStats(operation: string, timeWindowMs?: number): LatencyStats {
        return this.latencyMeasurer.getStats(operation, timeWindowMs);
    }

    /**
     * Get all monitored operations
     */
    getOperations(): string[] {
        return this.latencyMeasurer.getOperations();
    }
}