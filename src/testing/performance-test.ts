import { performance } from 'perf_hooks';
import { MessagePackProtocol } from '../bridge/protocols/messagepack.protocol';
import { BridgeMessage } from '../bridge/types';

async function runPerformanceTest() {
    console.log('=== MessagePack Performance Test ===\n');
    
    const protocol = new MessagePackProtocol({ type: 'messagepack' });
    
    // Test messages of different sizes
    const testCases = [
        {
            name: 'Small Message',
            message: {
                id: 'test-1',
                type: 'signal' as const,
                payload: { value: 42 }
            }
        },
        {
            name: 'Medium Message',
            message: {
                id: 'test-2',
                type: 'data' as const,
                payload: {
                    indicator: 'RSI',
                    value: 70.5,
                    timestamp: Date.now(),
                    metadata: { confidence: 0.95 }
                }
            }
        },
        {
            name: 'Large Message',
            message: {
                id: 'test-3',
                type: 'data' as const,
                payload: {
                    data: Array(100).fill(0).map((_, i) => ({
                        index: i,
                        value: Math.random(),
                        timestamp: Date.now()
                    }))
                }
            }
        }
    ];    // Warm up
    console.log('Warming up...');
    for (let i = 0; i < 1000; i++) {
        const encoded = await protocol.encode(testCases[0].message as BridgeMessage);
        await protocol.decode(encoded);
    }

    // Run tests
    for (const testCase of testCases) {
        console.log(`\n${testCase.name}:`);
        
        const iterations = 10000;
        const encodeTimes: number[] = [];
        const decodeTimes: number[] = [];
        let totalSize = 0;

        for (let i = 0; i < iterations; i++) {
            // Test encode
            const encodeStart = performance.now();
            const encoded = await protocol.encode(testCase.message as BridgeMessage);
            const encodeTime = performance.now() - encodeStart;
            encodeTimes.push(encodeTime);
            
            if (i === 0) totalSize = encoded.length;

            // Test decode
            const decodeStart = performance.now();
            await protocol.decode(encoded);
            const decodeTime = performance.now() - decodeStart;
            decodeTimes.push(decodeTime);
        }

        // Calculate statistics
        encodeTimes.sort((a, b) => a - b);
        decodeTimes.sort((a, b) => a - b);

        const encodeAvg = encodeTimes.reduce((a, b) => a + b) / encodeTimes.length;
        const decodeAvg = decodeTimes.reduce((a, b) => a + b) / decodeTimes.length;        const encodeP99 = encodeTimes[Math.floor(encodeTimes.length * 0.99)];
        const decodeP99 = decodeTimes[Math.floor(decodeTimes.length * 0.99)];

        console.log(`  Size: ${totalSize} bytes`);
        console.log(`  Encode: avg=${encodeAvg.toFixed(3)}ms, p99=${encodeP99.toFixed(3)}ms`);
        console.log(`  Decode: avg=${decodeAvg.toFixed(3)}ms, p99=${decodeP99.toFixed(3)}ms`);
        console.log(`  Total: avg=${(encodeAvg + decodeAvg).toFixed(3)}ms`);
        
        if (encodeAvg < 1 && decodeAvg < 1) {
            console.log(`  ✅ Meets <1ms requirement`);
        } else {
            console.log(`  ❌ Exceeds 1ms requirement`);
        }
    }

    // Get protocol statistics
    const stats = (protocol as any).metrics.getStats();
    console.log('\n=== Overall Statistics ===');
    console.log(`Total encode operations: ${stats.encode.length}`);
    console.log(`Total decode operations: ${stats.decode.length}`);
}

runPerformanceTest().catch(console.error);
