import { LatencyTestFramework } from './latency-framework/latency.test';

async function runSimpleTest() {
    console.log('=== Running Simple MessagePack Latency Test ===\n');
    
    try {
        const framework = new LatencyTestFramework();
        const results = await framework.runLatencyTests(100);
        
        console.log('Results:');
        console.log(`- Avg latency: ${results.serialization.avg.toFixed(3)}ms`);
        console.log(`- P99 latency: ${results.serialization.p99.toFixed(3)}ms`);
        console.log(`- Status: ${results.serialization.avg < 1 ? '✅ PASS' : '❌ FAIL'}`);
        
    } catch (error) {
        console.error('Test failed:', error);
    }
}

runSimpleTest();
    // Test decoding
    const decodeStart = process.hrtime.bigint();
    const decoded = msgpack.decode(encoded);
    const decodeTime = Number(process.hrtime.bigint() - decodeStart) / 1e6;
    
    console.log('Original:', testData);
    console.log('Encoded size:', encoded.length, 'bytes');
    console.log('Decoded:', decoded);
    console.log(`Encode time: ${encodeTime.toFixed(3)}ms`);
    console.log(`Decode time: ${decodeTime.toFixed(3)}ms`);
    console.log(`Total time: ${(encodeTime + decodeTime).toFixed(3)}ms`);
    
    if (encodeTime + decodeTime < 1) {
        console.log('✅ MessagePack meets <1ms requirement!');
    } else {
        console.log('❌ MessagePack exceeds 1ms requirement');
    }
}

testMessagePack().catch(console.error);
