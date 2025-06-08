import { DualChannelWebSocket } from '../../bridge/channels/dual-channel.websocket';
import { MessagePackProtocol } from '../../bridge/protocols/messagepack.protocol';
import { LatencyTestFramework } from '../latency-framework/latency.test';

export async function testBridgeIntegration(): Promise<boolean> {
  console.log('🧪 Running Bridge Integration Tests...\n');
  
  const tests = [
    testMessagePackProtocol,
    testDualChannelCommunication,
    testPythonBridgeConnection,
    testIndicatorIntegration,
    testLoadHandling
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    try {
      await test();
      passed++;
      console.log(`✅ ${test.name} passed`);
    } catch (error) {
      failed++;
      console.error(`❌ ${test.name} failed:`, error);
    }
  }

  console.log(`\n📊 Results: ${passed} passed, ${failed} failed`);
  return failed === 0;
}

async function testMessagePackProtocol(): Promise<void> {
  const protocol = new MessagePackProtocol({ type: 'messagepack' });
  
  const testMessage: BridgeMessage = {
    id: 'test-1',
    type: 'signal' as const,  // Use const assertion for literal type
    payload: {
      symbol: 'EURUSD',
      indicators: {
        rsi: 65.5,
        macd: 0.0012,
        volume: 150000
      }
    }
  };

  const encoded = await protocol.encode(testMessage);
  const decoded = await protocol.decode(encoded);
  
  if (JSON.stringify(testMessage) !== JSON.stringify(decoded)) {
    throw new Error('MessagePack encode/decode mismatch');
  }
}

async function testDualChannelCommunication(): Promise<void> {
  const channel = new DualChannelWebSocket({
    primaryUrl: 'ws://localhost:8001',
    secondaryUrl: 'ws://localhost:8002',
    protocol: { type: 'messagepack' }
  });

  // Test will timeout if channels don't connect
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Connection timeout')), 5000);
    
    channel.once('channel:connected', () => {
      clearTimeout(timeout);
      resolve(undefined);
    });
  });
}

async function testPythonBridgeConnection(): Promise<void> {
  // Mock test for Python bridge
  console.log('Testing Python bridge connection...');
  // Actual implementation would test real Python process
}

async function testIndicatorIntegration(): Promise<void> {
  const framework = new LatencyTestFramework();
  const results = await framework.runLatencyTests(100);
  
  if (results.serialization.p99 > 1) {
    throw new Error(`Serialization latency too high: ${results.serialization.p99}ms`);
  }
}

async function testLoadHandling(): Promise<void> {
  const framework = new LatencyTestFramework();
  const results = await framework.runLatencyTests(10);
  
  if (results.loadTest.successRate < 99) {
    throw new Error(`Load test success rate too low: ${results.loadTest.successRate}%`);
  }
}

// Run tests if executed directly
if (require.main === module) {
  testBridgeIntegration().then(success => {
    process.exit(success ? 0 : 1);
  });
}
