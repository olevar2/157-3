export { LatencyMeasurer, LatencyMeasurement, LatencyStats } from './latency-measurement';
export { PerformanceProfiler, ProfileConfig, PerformanceAlert } from './performance-profiler';
export { BenchmarkSuite, BenchmarkConfig, BenchmarkResult } from './benchmark-suite';
export { RegressionDetector, RegressionConfig, RegressionAlert } from './regression-detector';
export { LatencyDashboard, DashboardConfig } from './latency-dashboard';

import { LatencyMeasurer } from './latency-measurement';
import { PerformanceProfiler, ProfileConfig } from './performance-profiler';
import { BenchmarkSuite, BenchmarkConfig } from './benchmark-suite';
import { RegressionDetector, RegressionConfig } from './regression-detector';
import { LatencyDashboard, DashboardConfig } from './latency-dashboard';

/**
 * Comprehensive Latency Testing Framework for Platform3
 * 
 * This framework provides:
 * - High-precision latency measurement
 * - Continuous performance profiling
 * - Automated benchmark testing
 * - Performance regression detection
 * - Real-time monitoring dashboard
 */
export class Platform3LatencyFramework {
    private latencyMeasurer: LatencyMeasurer;
    private profiler: PerformanceProfiler;
    private benchmarkSuite: BenchmarkSuite;
    private regressionDetector: RegressionDetector;
    private dashboard: LatencyDashboard;

    constructor(
        profileConfig: ProfileConfig,
        regressionConfig: RegressionConfig,
        dashboardConfig: DashboardConfig
    ) {
        this.latencyMeasurer = new LatencyMeasurer();
        this.profiler = new PerformanceProfiler(profileConfig);
        this.benchmarkSuite = new BenchmarkSuite();
        this.regressionDetector = new RegressionDetector(regressionConfig);
        this.dashboard = new LatencyDashboard(
            dashboardConfig,
            this.profiler,
            this.benchmarkSuite,
            this.regressionDetector
        );
    }

    /**
     * Initialize and start the framework
     */
    async start(): Promise<void> {
        this.profiler.start();
        await this.dashboard.start();
        console.log('Platform3 Latency Framework started successfully');
    }

    /**
     * Stop the framework
     */
    async stop(): Promise<void> {
        this.profiler.stop();
        await this.dashboard.stop();
        console.log('Platform3 Latency Framework stopped');
    }    /**
     * Get the latency measurer instance
     */
    getLatencyMeasurer(): LatencyMeasurer {
        return this.latencyMeasurer;
    }

    /**
     * Get the performance profiler instance
     */
    getProfiler(): PerformanceProfiler {
        return this.profiler;
    }

    /**
     * Get the benchmark suite instance
     */
    getBenchmarkSuite(): BenchmarkSuite {
        return this.benchmarkSuite;
    }

    /**
     * Get the regression detector instance
     */
    getRegressionDetector(): RegressionDetector {
        return this.regressionDetector;
    }

    /**
     * Get the dashboard instance
     */
    getDashboard(): LatencyDashboard {
        return this.dashboard;
    }

    /**
     * Run a complete latency test cycle
     */
    async runCompleteTest(benchmarkConfig: BenchmarkConfig): Promise<{
        benchmarks: any[];
        alerts: any[];
        regressions: any[];
    }> {
        // Run benchmarks
        const benchmarks = await this.benchmarkSuite.runAllBenchmarks(benchmarkConfig);
        
        // Check for alerts
        const alerts = this.profiler.getAlerts();
        
        // Check for regressions (if baselines exist)
        const regressions = [];
        const operations = this.profiler.getOperations();
        
        for (const operation of operations) {
            try {
                const stats = this.profiler.getStats(operation);
                const regressionAlerts = this.regressionDetector.checkRegression(operation, stats);
                regressions.push(...regressionAlerts);
            } catch (error) {
                // No baseline set for this operation
            }
        }

        return { benchmarks, alerts, regressions };
    }
}