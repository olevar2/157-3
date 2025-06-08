import { LatencyTestFramework } from './latency-framework/latency.test';

async function main() {
    console.log('🚀 MessagePack Latency Test\n');
    console.log('================================\n');
    
    try {
        const framework = new LatencyTestFramework();
        console.log('Running 100 iterations...\n');
        
        const results = await framework.runLatencyTests(100);
        
        console.log('📊 Serialization Performance:');
        console.log(`   Average: ${results.serialization.avg.toFixed(3)}ms`);
        console.log(`   Min: ${results.serialization.min.toFixed(3)}ms`);
        console.log(`   Max: ${results.serialization.max.toFixed(3)}ms`);
        console.log(`   P95: ${results.serialization.p95.toFixed(3)}ms`);
        console.log(`   P99: ${results.serialization.p99.toFixed(3)}ms`);
        
        console.log('\n📈 Throughput:');
        console.log(`   Messages/sec: ${results.throughput.messagesPerSecond.toFixed(0)}`);
        console.log(`   Bytes/sec: ${(results.throughput.bytesPerSecond / 1024).toFixed(2)} KB/s`);
        
        console.log('\n================================');
        console.log(`\n🎯 Target: <1ms average latency`);
        console.log(`📋 Result: ${results.serialization.avg.toFixed(3)}ms`);
        console.log(`\n${results.serialization.avg < 1 ? '✅ TEST PASSED!' : '❌ TEST FAILED!'}`);
        
    } catch (error) {
        console.error('\n❌ Test failed:', error);
        process.exit(1);
    }
}

main();
