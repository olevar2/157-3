import { DualChannelWebSocket } from '../bridge/channels/dual-channel.websocket';
import { ChannelConfig } from '../bridge/types';
import { createMockWebSocketServer } from './mock/mock-websocket-server';
import { performance } from 'perf_hooks';

async function testFullBridge() {
    console.log('🌉 Full Bridge Integration Test\n');
    console.log('================================\n');
    
    // Start mock servers
    const server1 = createMockWebSocketServer(8001);
    const server2 = createMockWebSocketServer(8002);
    
    console.log('✅ Mock servers started on ports 8001 and 8002\n');
    
    // Wait for servers to initialize
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const config: ChannelConfig = {
        primaryUrl: 'ws://localhost:8001',
        secondaryUrl: 'ws://localhost:8002',
        protocol: { type: 'messagepack' },
        reconnectDelay: 1000,
        heartbeatInterval: 5000
    };
    
    const bridge = new DualChannelWebSocket(config);
    
    // Wait for connections
    await new Promise(resolve => {
        let connected = 0;
        bridge.on('channel:connected', (channel) => {
            console.log(`✅ ${channel} channel connected`);
            connected++;
            if (connected === 2) resolve(undefined);
        });
    });
    
    console.log('\n📊 Testing message round-trip...\n');
    
    // Test round-trip latency
    const latencies: number[] = [];
    
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        
        await bridge.send({
            id: `test-${i}`,
            type: 'signal',
            payload: { test: true, index: i }
        });
        
        const latency = performance.now() - start;
        latencies.push(latency);
    }
    
    // Calculate stats
    const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const max = Math.max(...latencies);
    const min = Math.min(...latencies);
    
    console.log('📈 Round-trip Results:');
    console.log(`   Average: ${avg.toFixed(3)}ms`);
    console.log(`   Min: ${min.toFixed(3)}ms`);
    console.log(`   Max: ${max.toFixed(3)}ms`);
    
    console.log('\n✅ Bridge test completed successfully!');
    
    // Cleanup
    server1.close();
    server2.close();
    process.exit(0);
}

testFullBridge().catch(error => {
    console.error('❌ Test failed:', error);
    process.exit(1);
});
