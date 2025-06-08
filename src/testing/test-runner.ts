import { LatencyTestFramework } from './latency-framework/latency.test';

async function runTests() {
    console.log('🚀 Platform3 Bridge Testing Suite\n');
    console.log('================================\n');
    
    const framework = new LatencyTestFramework();
    
    try {
        console.log('📊 Running latency benchmarks...\n');
        const results = await framework.runLatencyTests(100);
        
        // Display results
        console.log('📈 Serialization Performance:');
        console.log(`   Average: ${results.serialization.avg.toFixed(3)}ms`);
        console.log(`   Min: ${results.serialization.min.toFixed(3)}ms`);
        console.log(`   Max: ${results.serialization.max.toFixed(3)}ms`);
        console.log(`   P95: ${results.serialization.p95.toFixed(3)}ms`);
        console.log(`   P99: ${results.serialization.p99.toFixed(3)}ms`);
        
        console.log('\n📊 Throughput:');
        console.log(`   Messages/sec: ${results.throughput.messagesPerSecond.toFixed(0)}`);
        console.log(`   Bytes/sec: ${(results.throughput.bytesPerSecond / 1024).toFixed(2)} KB/s`);
        
        console.log('\n🔄 Load Test (1 minute):');
        console.log(`   Success Rate: ${(results.loadTest.successRate * 100).toFixed(2)}%`);
        console.log(`   Errors: ${results.loadTest.errors}`);
        
        // Final verdict
        console.log('\n================================');
        const passed = results.serialization.avg < 1;
        console.log(`\n🎯 Target: <1ms average latency`);
        console.log(`📋 Result: ${results.serialization.avg.toFixed(3)}ms`);
        console.log(`\n${passed ? '✅ TEST PASSED!' : '❌ TEST FAILED!'}`);
        
        process.exit(passed ? 0 : 1);
        
    } catch (error) {
        console.error('\n❌ Test execution failed:', error);
        process.exit(1);
    }
}

// Run the tests
runTests();
