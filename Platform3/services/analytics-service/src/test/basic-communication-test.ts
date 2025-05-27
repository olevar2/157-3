/**
 * Simple test to diagnose TypeScript-Python communication issue
 */

import { ComprehensiveIndicatorEngine } from '../engines/ComprehensiveIndicatorEngine';
import winston from 'winston';

// Setup logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.simple(),
  transports: [new winston.transports.Console()]
});

async function testBasicCommunication() {
  console.log('🔍 Testing basic TypeScript-Python communication...\n');

  try {
    // 1. Create engine
    console.log('1. Creating ComprehensiveIndicatorEngine...');
    const engine = new ComprehensiveIndicatorEngine(logger);

    // 2. Initialize
    console.log('2. Initializing engine...');
    await engine.initialize();
    console.log('✅ Engine initialized successfully');

    // 3. Check if ready
    console.log('3. Checking if engine is ready...');
    const isReady = engine.isReady();
    console.log(`   Engine ready: ${isReady}`);

    // 4. Test simple calculation
    console.log('4. Testing simple SMA calculation...');    const testData = {
      timestamps: [Date.now() - 4000, Date.now() - 3000, Date.now() - 2000, Date.now() - 1000, Date.now()],
      open: [100, 102, 106, 108, 110],
      high: [105, 108, 110, 112, 115],
      low: [95, 98, 104, 106, 109],
      close: [102, 106, 108, 110, 113],
      volume: [1000, 1200, 900, 1100, 1300]
    };

    const result = await engine.calculateIndicator('sma', testData, 'TEST_SYMBOL', '1h');
    console.log('✅ SMA calculation result:', result);

    console.log('\n🎉 Basic communication test PASSED!');

  } catch (error) {
    console.error('\n❌ Basic communication test FAILED:', error);
    console.error('\nError details:', {
      message: String(error),
      stack: (error as any)?.stack
    });
    process.exit(1);
  }
}

// Run the test
testBasicCommunication();
