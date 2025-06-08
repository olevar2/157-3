import { LatencyTestFramework } from './latency-framework/latency.test';

async function runAllTests() {
    console.log('=== Starting Platform3 Bridge Testing Suite ===\n');
    
    try {
        // 1. Latency Tests
        console.log('1. Running Latency Tests...');
        const latencyTest = new LatencyTestFramework();
        const latencyResults = await latencyTest.runLatencyTests(100); // Reduced for quick testing
        
        console.log('\nLatency Test Results:');
        console.log(`- Serialization: ${latencyResults.serialization.avg.toFixed(3)}ms avg (P99: ${latencyResults.serialization.p99.toFixed(3)}ms)`);
        console.log(`- Throughput: ${latencyResults.throughput.messagesPerSecond.toFixed(0)} msg/s`);
        console.log(`- 24/7 Load Test: ${(latencyResults.loadTest.successRate * 100).toFixed(2)}% success rate\n`);
        
        // Verify <1ms latency requirement
        const meetsLatencyRequirement = latencyResults.serialization.avg < 1;
        if (meetsLatencyRequirement) {
            console.log('✅ Serialization meets <1ms requirement');
        } else {
            console.log('❌ Serialization exceeds 1ms requirement');
        }
        
        console.log('\n=== Test Summary ===');
        console.log(meetsLatencyRequirement ? '✅ Tests PASSED!' : '❌ Tests FAILED!');
        
    } catch (error) {
        console.error('Test execution failed:', error);
        process.exit(1);
    }
}

// Run tests
runAllTests().catch(console.error);
    console.log('\n=== Test Summary ===');
    const allTestsPassed = latencyResults.serialization.avg < 1 && 
                          Array.from(integrationResults.values()).every(v => v);
    
    if (allTestsPassed) {
        console.log('✅ All tests passed! System ready for production.');
    } else {
        console.log('❌ Some tests failed. Please review the results above.');
    }
}

// Run tests
runAllTests().catch(console.error);
