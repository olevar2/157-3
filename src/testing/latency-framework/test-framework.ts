import { Platform3LatencyFramework } from './index';

/**
 * Test the comprehensive latency framework
 */
async function testLatencyFramework() {
    console.log('🚀 Starting Platform3 Latency Framework Test...');

    // Configuration
    const profileConfig = {
        sampleInterval: 1000, // 1 second
        alertThresholds: {
            latencyMs: 1.0, // Alert if mean latency > 1ms
            errorRate: 0.01, // Alert if error rate > 1%
            p95Ms: 2.0, // Alert if P95 > 2ms
            p99Ms: 5.0  // Alert if P99 > 5ms
        },
        retentionPeriod: 60000 // 1 minute
    };

    const regressionConfig = {
        thresholds: {
            meanIncrease: 20, // 20% increase
            p95Increase: 25,  // 25% increase
            p99Increase: 30,  // 30% increase
            errorRateIncrease: 50 // 50% increase
        },
        minimumSamples: 100,
        baselineRetentionDays: 7
    };

    const dashboardConfig = {
        port: 3001,
        updateInterval: 5000,
        enableRealTimeUpdates: true
    };

    // Initialize framework
    const framework = new Platform3LatencyFramework(
        profileConfig,
        regressionConfig,
        dashboardConfig
    );

    try {
        // Start framework
        await framework.start();
        console.log('✅ Framework started successfully');

        // Get components for direct testing
        const latencyMeasurer = framework.getLatencyMeasurer();
        const profiler = framework.getProfiler();

        // Test basic latency measurement
        console.log('\n📊 Testing basic latency measurement...');
        
        const { result, measurement } = latencyMeasurer.measureSync('test-operation', () => {
            // Simulate some work
            const start = Date.now();
            while (Date.now() - start < 0.5) {} // ~0.5ms of work
            return 'test-result';
        });

        console.log(`Result: ${result}`);
        console.log(`Latency: ${measurement.duration.toFixed(4)}ms`);

        // Test async measurement
        console.log('\n⚡ Testing async latency measurement...');
        
        const { result: asyncResult, measurement: asyncMeasurement } = await latencyMeasurer.measureAsync('async-test', async () => {
            await new Promise(resolve => setTimeout(resolve, 1));
            return 'async-result';
        });

        console.log(`Async Result: ${asyncResult}`);
        console.log(`Async Latency: ${asyncMeasurement.duration.toFixed(4)}ms`);

        // Run benchmark test
        console.log('\n🏃 Running benchmark tests...');
        
        const benchmarkConfig = {
            iterations: 100,
            warmupIterations: 10,
            concurrency: 1,
            timeout: 5000
        };

        const testResults = await framework.runCompleteTest(benchmarkConfig);
        console.log(`Benchmarks completed: ${testResults.benchmarks.length} tests`);
        console.log(`Alerts generated: ${testResults.alerts.length}`);
        console.log(`Regressions detected: ${testResults.regressions.length}`);

        // Display results
        testResults.benchmarks.forEach(benchmark => {
            console.log(`\n📈 ${benchmark.name}:`);
            console.log(`  Mean: ${benchmark.stats.mean.toFixed(4)}ms`);
            console.log(`  P95: ${benchmark.stats.p95.toFixed(4)}ms`);
            console.log(`  P99: ${benchmark.stats.p99.toFixed(4)}ms`);
            console.log(`  Success Rate: ${((1 - benchmark.stats.errorRate) * 100).toFixed(2)}%`);
        });

        console.log('\n🎯 Framework test completed successfully!');
        console.log(`Dashboard available at: http://localhost:${dashboardConfig.port}`);

        // Keep running for a bit to allow manual testing
        console.log('\n⏳ Keeping framework running for 30 seconds for manual testing...');
        await new Promise(resolve => setTimeout(resolve, 30000));

    } catch (error) {
        console.error('❌ Framework test failed:', error);
    } finally {
        // Clean shutdown
        await framework.stop();
        console.log('🔄 Framework stopped');
    }
}

// Run the test
if (require.main === module) {
    testLatencyFramework().catch(console.error);
}

export { testLatencyFramework };