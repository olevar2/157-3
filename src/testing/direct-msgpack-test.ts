import { performance } from 'perf_hooks';
import { encode, decode } from '@msgpack/msgpack';

async function testMessagePackPerformance() {
    console.log('🚀 Direct MessagePack Performance Test\n');
    console.log('=====================================\n');
    
    const testData = {
        id: 'test-123',
        type: 'signal',
        payload: {
            indicator: 'RSI',
            value: 70.5,
            timestamp: Date.now(),
            metadata: { confidence: 0.95 }
        }
    };
    
    const iterations = 1000;
    const latencies: number[] = [];
    
    // Warm up
    for (let i = 0; i < 10; i++) {
        const encoded = encode(testData);
        decode(encoded);
    }
    
    // Actual test
    console.log(`Running ${iterations} encode/decode cycles...\n`);
    
    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        const encoded = encode(testData);
        const decoded = decode(encoded);
        const latency = performance.now() - start;
        latencies.push(latency);
    }
    
    // Calculate stats
    latencies.sort((a, b) => a - b);
    const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const p99 = latencies[Math.floor(latencies.length * 0.99)];
    const p95 = latencies[Math.floor(latencies.length * 0.95)];
    const min = latencies[0];
    const max = latencies[latencies.length - 1];
    
    console.log('📊 Results:');
    console.log(`   Average: ${avg.toFixed(3)}ms`);
    console.log(`   Min: ${min.toFixed(3)}ms`);
    console.log(`   Max: ${max.toFixed(3)}ms`);
    console.log(`   P95: ${p95.toFixed(3)}ms`);
    console.log(`   P99: ${p99.toFixed(3)}ms`);
    
    console.log('\n📈 Throughput Test (1 second)...');
    let count = 0;
    let bytes = 0;
    const startTime = performance.now();
    
    while (performance.now() - startTime < 1000) {
        const encoded = encode(testData);
        bytes += encoded.length;
        count++;
    }
    
    console.log(`   Messages/sec: ${count}`);
    console.log(`   KB/sec: ${(bytes / 1024).toFixed(2)}`);
    
    console.log('\n=====================================');
    console.log(`\n🎯 Target: <1ms average latency`);
    console.log(`📋 Result: ${avg.toFixed(3)}ms`);
    console.log(`\n${avg < 1 ? '✅ TEST PASSED!' : '❌ TEST FAILED!'}`);
}

testMessagePackPerformance().catch(console.error);
