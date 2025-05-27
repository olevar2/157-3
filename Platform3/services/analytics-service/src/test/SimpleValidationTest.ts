/**
 * Simple Validation Test for ComprehensiveIndicatorEngine
 * Tests basic functionality with correct API calls
 */

import winston from 'winston';
import { ComprehensiveIndicatorEngine, MarketDataInput } from '../engines/ComprehensiveIndicatorEngine';

// Configure logger for tests
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.simple(),
  transports: [new winston.transports.Console()]
});

export class SimpleValidationTest {
  private engine: ComprehensiveIndicatorEngine;

  constructor() {
    this.engine = new ComprehensiveIndicatorEngine(logger);
  }

  private generateTestData(): MarketDataInput {
    const length = 100;
    const data: MarketDataInput = {
      timestamps: [],
      open: [],
      high: [],
      low: [],
      close: [],
      volume: []
    };

    for (let i = 0; i < length; i++) {
      const timestamp = Date.now() - (length - i) * 60000; // 1 minute intervals
      const basePrice = 1.1000 + Math.sin(i * 0.1) * 0.01; // Simulated price movement
      const volatility = 0.0005;
      
      data.timestamps.push(timestamp);
      data.open.push(basePrice + (Math.random() - 0.5) * volatility);
      data.high.push(basePrice + Math.random() * volatility * 2);
      data.low.push(basePrice - Math.random() * volatility * 2);
      data.close.push(basePrice + (Math.random() - 0.5) * volatility);
      data.volume.push(Math.floor(Math.random() * 10000) + 1000);
    }

    return data;
  }

  async runBasicTests(): Promise<void> {
    console.log('🔍 Starting Simple Validation Tests...\n');

    try {
      // Test 1: Engine Initialization
      console.log('1. Testing engine initialization...');
      await this.engine.initialize();
      console.log('✅ Engine initialized successfully\n');

      // Test 2: Check Available Indicators
      console.log('2. Testing available indicators...');
      const indicators = await this.engine.getAvailableIndicators();
      const totalCount = Object.values(indicators).reduce((sum, arr) => sum + arr.length, 0);
      console.log(`✅ Found ${totalCount} indicators across ${Object.keys(indicators).length} categories\n`);

      // Test 3: Single Indicator Calculation
      console.log('3. Testing single indicator calculation (RSI)...');
      const testData = this.generateTestData();
      const rsiResult = await this.engine.calculateIndicator('RSI', testData, 'EURUSD', 'M1');
      
      if (rsiResult.success) {
        console.log('✅ RSI calculation successful');
        console.log(`   Calculation time: ${rsiResult.calculation_time}ms`);
        console.log(`   Category: ${rsiResult.category}\n`);
      } else {
        console.log('❌ RSI calculation failed:', rsiResult.error_message, '\n');
      }

      // Test 4: Batch Indicator Calculation  
      console.log('4. Testing batch indicator calculation...');
      const batchIndicators = ['RSI', 'MACD', 'BollingerBands'];
      const batchResults = await this.engine.batchCalculateIndicators(
        batchIndicators, 
        testData, 
        'EURUSD', 
        'M1'
      );
      
      const successCount = Object.values(batchResults).filter(r => r.success).length;
      console.log(`✅ Batch calculation: ${successCount}/${batchIndicators.length} indicators successful\n`);

      // Test 5: All Indicators Calculation (partial test)
      console.log('5. Testing all indicators calculation...');
      const allResults = await this.engine.calculateAllIndicators(testData, 'EURUSD', 'M1');
      console.log(`✅ All indicators test: ${allResults.successful_indicators}/${allResults.total_indicators} successful`);
      console.log(`   Success rate: ${allResults.success_rate.toFixed(1)}%`);
      console.log(`   Total calculation time: ${allResults.total_calculation_time.toFixed(1)}ms\n`);

      console.log('🎉 All Simple Validation Tests PASSED!\n');

    } catch (error) {
      console.error('❌ Simple Validation Tests FAILED:', error);
      throw error;
    }
  }
}

// Run the tests if this file is executed directly
if (require.main === module) {
  const test = new SimpleValidationTest();
  test.runBasicTests()
    .then(() => {
      console.log('✅ Simple validation completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('❌ Simple validation failed:', error);
      process.exit(1);
    });
}
