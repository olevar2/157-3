import { LatencyStats } from './latency-measurement';
import { EventEmitter } from 'events';

export interface RegressionConfig {
    thresholds: {
        meanIncrease: number; // percentage
        p95Increase: number; // percentage  
        p99Increase: number; // percentage
        errorRateIncrease: number; // percentage
    };
    minimumSamples: number;
    baselineRetentionDays: number;
}

export interface RegressionAlert {
    timestamp: number;
    operation: string;
    metric: string;
    baseline: number;
    current: number;
    changePercent: number;
    severity: 'minor' | 'major' | 'critical';
}

export class RegressionDetector extends EventEmitter {
    private config: RegressionConfig;
    private baselines: Map<string, LatencyStats> = new Map();
    private alerts: RegressionAlert[] = [];

    constructor(config: RegressionConfig) {
        super();
        this.config = config;
    }

    /**
     * Set baseline for an operation
     */
    setBaseline(operation: string, stats: LatencyStats): void {
        if (stats.count < this.config.minimumSamples) {
            throw new Error(`Insufficient samples for baseline. Required: ${this.config.minimumSamples}, got: ${stats.count}`);
        }
        
        this.baselines.set(operation, { ...stats });
        this.emit('baseline:set', { operation, stats });
    }

    /**
     * Check for performance regression
     */
    checkRegression(operation: string, currentStats: LatencyStats): RegressionAlert[] {
        const baseline = this.baselines.get(operation);
        if (!baseline) {
            throw new Error(`No baseline found for operation: ${operation}`);
        }

        if (currentStats.count < this.config.minimumSamples) {
            return [];
        }        const alerts: RegressionAlert[] = [];
        const timestamp = Date.now();

        // Check mean latency regression
        const meanChange = this.calculatePercentageChange(baseline.mean, currentStats.mean);
        if (meanChange > this.config.thresholds.meanIncrease) {
            alerts.push(this.createAlert(timestamp, operation, 'mean', baseline.mean, currentStats.mean, meanChange));
        }

        // Check P95 regression
        const p95Change = this.calculatePercentageChange(baseline.p95, currentStats.p95);
        if (p95Change > this.config.thresholds.p95Increase) {
            alerts.push(this.createAlert(timestamp, operation, 'p95', baseline.p95, currentStats.p95, p95Change));
        }

        // Check P99 regression
        const p99Change = this.calculatePercentageChange(baseline.p99, currentStats.p99);
        if (p99Change > this.config.thresholds.p99Increase) {
            alerts.push(this.createAlert(timestamp, operation, 'p99', baseline.p99, currentStats.p99, p99Change));
        }

        // Check error rate regression
        const errorRateChange = this.calculatePercentageChange(baseline.errorRate, currentStats.errorRate);
        if (errorRateChange > this.config.thresholds.errorRateIncrease) {
            alerts.push(this.createAlert(timestamp, operation, 'errorRate', baseline.errorRate, currentStats.errorRate, errorRateChange));
        }

        // Store and emit alerts
        alerts.forEach(alert => {
            this.alerts.push(alert);
            this.emit('regression:detected', alert);
        });

        return alerts;
    }

    /**
     * Create a regression alert
     */
    private createAlert(timestamp: number, operation: string, metric: string, baseline: number, current: number, changePercent: number): RegressionAlert {
        let severity: 'minor' | 'major' | 'critical' = 'minor';
        
        if (changePercent > 50) {
            severity = 'critical';
        } else if (changePercent > 25) {
            severity = 'major';
        }

        return {
            timestamp,
            operation,
            metric,
            baseline,
            current,
            changePercent,
            severity
        };
    }

    /**
     * Calculate percentage change between two values
     */
    private calculatePercentageChange(baseline: number, current: number): number {
        if (baseline === 0) return current > 0 ? 100 : 0;
        return ((current - baseline) / baseline) * 100;
    }    /**
     * Get all baselines
     */
    getBaselines(): Map<string, LatencyStats> {
        return new Map(this.baselines);
    }

    /**
     * Get recent alerts
     */
    getAlerts(timeWindowMs?: number): RegressionAlert[] {
        if (!timeWindowMs) {
            return [...this.alerts];
        }

        const cutoff = Date.now() - timeWindowMs;
        return this.alerts.filter(alert => alert.timestamp >= cutoff);
    }

    /**
     * Clear alerts older than retention period
     */
    cleanOldAlerts(): void {
        const cutoff = Date.now() - (this.config.baselineRetentionDays * 24 * 60 * 60 * 1000);
        this.alerts = this.alerts.filter(alert => alert.timestamp >= cutoff);
    }

    /**
     * Clear all alerts
     */
    clearAlerts(): void {
        this.alerts = [];
    }

    /**
     * Remove baseline for an operation
     */
    removeBaseline(operation: string): boolean {
        return this.baselines.delete(operation);
    }

    /**
     * Update regression thresholds
     */
    updateThresholds(thresholds: Partial<RegressionConfig['thresholds']>): void {
        this.config.thresholds = { ...this.config.thresholds, ...thresholds };
        this.emit('thresholds:updated', this.config.thresholds);
    }
}