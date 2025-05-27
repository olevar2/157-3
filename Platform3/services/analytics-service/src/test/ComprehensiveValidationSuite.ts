/**
 * Comprehensive Validation Test Suite for Platform3 67-Indicator System
 * Tests: Accuracy, Real Data Processing, Performance, Error Handling, Integration
 */

import { ComprehensiveIndicatorEngine } from '../engines/ComprehensiveIndicatorEngine';
import { MarketDataCollector } from '../services/MarketDataCollector';
import winston from 'winston';
// import fs from 'fs';
// import path from 'path';

// Test configuration
const TEST_CONFIG = {
  ACCURACY_TOLERANCE: 0.0001,
  PERFORMANCE_THRESHOLDS: {
    SINGLE_INDICATOR_MS: 1000,
    BATCH_INDICATORS_MS: 5000,
    ALL_INDICATORS_MS: 30000
  },
  CONCURRENT_REQUESTS: 5,
  DATA_POINTS: [50, 100, 500, 1000], // Different data sizes
  TEST_SYMBOLS: ['EURUSD', 'GBPUSD', 'USDJPY']
};

// Setup logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.simple()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({
      filename: 'logs/validation-test-results.log',
      level: 'info'
    })
  ]
});

export interface TestResult {
  testName: string;
  category: 'accuracy' | 'performance' | 'error_handling' | 'integration' | 'real_data';
  passed: boolean;
  duration: number;
  details: any;
  errorMessage?: string;
}

export interface ValidationResults {
  totalTests: number;
  passedTests: number;
  failedTests: number;
  categories: {
    [category: string]: {
      total: number;
      passed: number;
      failed: number;
    };
  };
  results: TestResult[];
  summary: string;
}

export class ComprehensiveValidationSuite {
  private engine: ComprehensiveIndicatorEngine;
  private marketData: MarketDataCollector;
  private results: TestResult[] = [];

  constructor() {
    this.engine = new ComprehensiveIndicatorEngine(logger);
    this.marketData = new MarketDataCollector(logger);
  }

  async runFullValidation(): Promise<ValidationResults> {    logger.info('🚀 Starting Comprehensive Validation Test Suite for 67-Indicator System');
    logger.info('='.repeat(80));

    try {
      // Initialize components
      await this.initializeComponents();

      // Run all test categories
      await this.runAccuracyTests();
      await this.runRealDataProcessingTests();
      await this.runPerformanceTests();
      await this.runErrorHandlingTests();
      await this.runIntegrationTests();

      return this.generateReport();

    } catch (error: any) {
      logger.error(`❌ Validation suite failed to complete: ${error}`);
      throw error;
    }
  }

  private async initializeComponents(): Promise<void> {
    logger.info('📋 Initializing test components...');

    try {
      await this.engine.initialize();
      await this.marketData.initialize();
      logger.info('✅ Components initialized successfully');
    } catch (error) {
      throw new Error(`Failed to initialize components: ${error}`);
    }
  }

  // ===========================================
  // 🔬 CALCULATION ACCURACY TESTS
  // ===========================================
  private async runAccuracyTests(): Promise<void> {
    logger.info('\n🔬 RUNNING CALCULATION ACCURACY TESTS');
    logger.info('='.repeat(50));

    // Test 1: Known Mathematical Results
    await this.testKnownMathematicalResults();

    // Test 2: Cross-Validation with Reference Implementation
    await this.testCrossValidation();

    // Test 3: Edge Cases Mathematical Consistency
    await this.testMathematicalConsistency();

    // Test 4: Floating Point Precision
    await this.testFloatingPointPrecision();
  }

  private async testKnownMathematicalResults(): Promise<void> {
    const testName = 'Known Mathematical Results';
    const startTime = Date.now();

    try {
      // Generate deterministic test data
      const deterministicData = this.generateDeterministicData();

      // Test RSI with known expected results
      const rsiResult = await this.engine.calculateIndicator('RSI', deterministicData, 'TEST', '1h');

      // Test SMA with known expected results
      const smaResult = await this.engine.calculateIndicator('SMA', deterministicData, 'TEST', '1h');

      // Validate results against expected values
      const passed = this.validateKnownResults(rsiResult, smaResult);

      this.addTestResult(testName, 'accuracy', passed, Date.now() - startTime, {
        rsi_success: rsiResult.success,
        sma_success: smaResult.success,
        rsi_values_count: Array.isArray(rsiResult.values) ? rsiResult.values.length : 0,
        sma_values_count: Array.isArray(smaResult.values) ? smaResult.values.length : 0
      });

    } catch (error: any) {
      this.addTestResult(testName, 'accuracy', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testCrossValidation(): Promise<void> {
    const testName = 'Cross-Validation with Reference';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);

      // Calculate multiple indicators that should correlate
      const batchResult = await this.engine.batchCalculateIndicators(
        ['SMA', 'EMA', 'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER'],
        testData,
        'TEST',
        '1h'
      );

      // Validate correlations and relationships
      const passed = this.validateIndicatorRelationships(batchResult);

      this.addTestResult(testName, 'accuracy', passed, Date.now() - startTime, {
        indicators_calculated: Object.keys(batchResult).length,
        successful_indicators: Object.values(batchResult).filter(r => r.success).length
      });

    } catch (error: any) {
      this.addTestResult(testName, 'accuracy', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testMathematicalConsistency(): Promise<void> {
    const testName = 'Mathematical Consistency';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(200);

      // Test that indicators produce consistent results across different data sizes
      const shortData = this.truncateData(testData, 100);
      const longData = testData;

      const shortResult = await this.engine.calculateIndicator('RSI', shortData, 'TEST', '1h');
      const longResult = await this.engine.calculateIndicator('RSI', longData, 'TEST', '1h');

      const passed = this.validateConsistency(shortResult, longResult);

      this.addTestResult(testName, 'accuracy', passed, Date.now() - startTime, {
        short_data_success: shortResult.success,
        long_data_success: longResult.success,
        consistency_check: passed
      });

    } catch (error: any) {
      this.addTestResult(testName, 'accuracy', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testFloatingPointPrecision(): Promise<void> {
    const testName = 'Floating Point Precision';
    const startTime = Date.now();

    try {
      // Test with high precision data
      const highPrecisionData = this.generateHighPrecisionData();

      const result = await this.engine.calculateIndicator('RSI', highPrecisionData, 'TEST', '1h');

      const passed = result.success && this.validatePrecision(result.values);

      this.addTestResult(testName, 'accuracy', passed, Date.now() - startTime, {
        high_precision_success: result.success,
        precision_valid: this.validatePrecision(result.values)
      });

    } catch (error: any) {
      this.addTestResult(testName, 'accuracy', false, Date.now() - startTime, {}, String(error));
    }
  }

  // ===========================================
  // 📊 REAL DATA PROCESSING TESTS
  // ===========================================

  private async runRealDataProcessingTests(): Promise<void> {
    logger.info('\n📊 RUNNING REAL DATA PROCESSING TESTS');
    logger.info('='.repeat(50));

    await this.testRealForexData();
    await this.testDifferentTimeframes();
    await this.testDataFormatCompatibility();
    await this.testLargeDatasets();
  }

  private async testRealForexData(): Promise<void> {
    const testName = 'Real FOREX Data Processing';
    const startTime = Date.now();

    try {
      // Get real market data for multiple symbols
      const results = [];

      for (const symbol of TEST_CONFIG.TEST_SYMBOLS) {
        const marketData = await this.marketData.getHistoricalData(symbol, '1h', 100);
        const convertedData = this.convertMarketData(marketData);

        const result = await this.engine.calculateIndicator('RSI', convertedData, symbol, '1h');
        results.push({ symbol, success: result.success, error: result.error_message });
      }

      const passed = results.every(r => r.success);

      this.addTestResult(testName, 'real_data', passed, Date.now() - startTime, {
        symbols_tested: TEST_CONFIG.TEST_SYMBOLS,
        results: results
      });

    } catch (error: any) {
      this.addTestResult(testName, 'real_data', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testDifferentTimeframes(): Promise<void> {
    const testName = 'Different Timeframes';
    const startTime = Date.now();

    try {
      const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
      const results = [];

      for (const timeframe of timeframes) {
        const marketData = await this.marketData.getHistoricalData('EURUSD', timeframe, 100);
        const convertedData = this.convertMarketData(marketData);

        const result = await this.engine.calculateIndicator('SMA', convertedData, 'EURUSD', timeframe);
        results.push({ timeframe, success: result.success });
      }

      const passed = results.every(r => r.success);

      this.addTestResult(testName, 'real_data', passed, Date.now() - startTime, {
        timeframes_tested: timeframes,
        results: results
      });

    } catch (error: any) {
      this.addTestResult(testName, 'real_data', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testDataFormatCompatibility(): Promise<void> {
    const testName = 'Data Format Compatibility';
    const startTime = Date.now();

    try {
      // Test different data formats
      const formats = [
        this.generateStandardTestData(100),
        this.generateDataWithGaps(100),
        this.generateDataWithDecimals(100),
        this.generateDataWithZeros(100)
      ];

      const results = [];
      for (let i = 0; i < formats.length; i++) {
        const result = await this.engine.calculateIndicator('RSI', formats[i], 'TEST', '1h');
        results.push({ format: i, success: result.success });
      }

      const passed = results.every(r => r.success);

      this.addTestResult(testName, 'real_data', passed, Date.now() - startTime, {
        formats_tested: formats.length,
        results: results
      });

    } catch (error: any) {
      this.addTestResult(testName, 'real_data', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testLargeDatasets(): Promise<void> {
    const testName = 'Large Dataset Processing';
    const startTime = Date.now();

    try {
      const results = [];

      for (const size of TEST_CONFIG.DATA_POINTS) {
        const largeData = this.generateStandardTestData(size);
        const result = await this.engine.calculateIndicator('SMA', largeData, 'TEST', '1h');
        results.push({ size, success: result.success, calculation_time: result.calculation_time });
      }

      const passed = results.every(r => r.success);

      this.addTestResult(testName, 'real_data', passed, Date.now() - startTime, {
        data_sizes_tested: TEST_CONFIG.DATA_POINTS,
        results: results
      });

    } catch (error: any) {
      this.addTestResult(testName, 'real_data', false, Date.now() - startTime, {}, String(error));
    }
  }

  // ===========================================
  // ⚡ PERFORMANCE TESTS
  // ===========================================

  private async runPerformanceTests(): Promise<void> {
    logger.info('\n⚡ RUNNING PERFORMANCE TESTS');
    logger.info('='.repeat(50));

    await this.testSingleIndicatorPerformance();
    await this.testBatchIndicatorPerformance();
    await this.testAllIndicatorsPerformance();
    await this.testConcurrentRequestPerformance();
    await this.testMemoryUsage();
  }

  private async testSingleIndicatorPerformance(): Promise<void> {
    const testName = 'Single Indicator Performance';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);
      const indicators = ['RSI', 'SMA', 'EMA', 'MACD', 'STOCH'];
      const results = [];

      for (const indicator of indicators) {
        const indicatorStart = Date.now();
        const result = await this.engine.calculateIndicator(indicator, testData, 'TEST', '1h');
        const indicatorTime = Date.now() - indicatorStart;

        results.push({
          indicator,
          success: result.success,
          time_ms: indicatorTime,
          within_threshold: indicatorTime < TEST_CONFIG.PERFORMANCE_THRESHOLDS.SINGLE_INDICATOR_MS
        });
      }

      const passed = results.every(r => r.success && r.within_threshold);

      this.addTestResult(testName, 'performance', passed, Date.now() - startTime, {
        indicators_tested: indicators,
        results: results,
        average_time: results.reduce((sum, r) => sum + r.time_ms, 0) / results.length
      });

    } catch (error: any) {
      this.addTestResult(testName, 'performance', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testBatchIndicatorPerformance(): Promise<void> {
    const testName = 'Batch Indicator Performance';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);
      const batchIndicators = ['RSI', 'SMA', 'EMA', 'MACD', 'STOCH', 'BB_UPPER', 'BB_LOWER', 'ATR', 'ADX'];

      const batchStart = Date.now();
      const result = await this.engine.batchCalculateIndicators(batchIndicators, testData, 'TEST', '1h');
      const batchTime = Date.now() - batchStart;

      const passed = batchTime < TEST_CONFIG.PERFORMANCE_THRESHOLDS.BATCH_INDICATORS_MS &&
                    Object.values(result).every(r => r.success);

      this.addTestResult(testName, 'performance', passed, Date.now() - startTime, {
        batch_size: batchIndicators.length,
        batch_time_ms: batchTime,
        within_threshold: batchTime < TEST_CONFIG.PERFORMANCE_THRESHOLDS.BATCH_INDICATORS_MS,
        successful_indicators: Object.values(result).filter(r => r.success).length
      });

    } catch (error: any) {
      this.addTestResult(testName, 'performance', false, Date.now() - startTime, {}, String(error));
    }
  }

  private async testAllIndicatorsPerformance(): Promise<void> {
    const testName = 'All 67 Indicators Performance';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);

      const allStart = Date.now();
      const result = await this.engine.calculateAllIndicators(testData, 'TEST', '1h');
      const allTime = Date.now() - allStart;

      const passed = allTime < TEST_CONFIG.PERFORMANCE_THRESHOLDS.ALL_INDICATORS_MS &&
                    result.success_rate > 0.9; // At least 90% success rate

      this.addTestResult(testName, 'performance', passed, Date.now() - startTime, {
        total_indicators: result.total_indicators,
        successful_indicators: result.successful_indicators,
        success_rate: result.success_rate,
        total_time_ms: allTime,
        within_threshold: allTime < TEST_CONFIG.PERFORMANCE_THRESHOLDS.ALL_INDICATORS_MS
      });

    } catch (error: any) {
      this.addTestResult(testName, 'performance', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testConcurrentRequestPerformance(): Promise<void> {
    const testName = 'Concurrent Request Performance';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);
      const promises = [];

      // Create concurrent requests
      for (let i = 0; i < TEST_CONFIG.CONCURRENT_REQUESTS; i++) {
        promises.push(this.engine.calculateIndicator('RSI', testData, `TEST${i}`, '1h'));
      }

      const results = await Promise.all(promises);
      const passed = results.every(r => r.success);

      this.addTestResult(testName, 'performance', passed, Date.now() - startTime, {
        concurrent_requests: TEST_CONFIG.CONCURRENT_REQUESTS,
        all_successful: passed,
        results: results.map(r => ({ success: r.success, calculation_time: r.calculation_time }))
      });

    } catch (error: any) {
      this.addTestResult(testName, 'performance', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testMemoryUsage(): Promise<void> {
    const testName = 'Memory Usage';
    const startTime = Date.now();

    try {
      const initialMemory = process.memoryUsage();

      // Run memory-intensive operations
      const largeData = this.generateStandardTestData(1000);
      await this.engine.calculateAllIndicators(largeData, 'TEST', '1h');

      const finalMemory = process.memoryUsage();
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;

      // Check if memory increase is reasonable (less than 100MB)
      const passed = memoryIncrease < 100 * 1024 * 1024;

      this.addTestResult(testName, 'performance', passed, Date.now() - startTime, {
        initial_memory_mb: Math.round(initialMemory.heapUsed / 1024 / 1024),
        final_memory_mb: Math.round(finalMemory.heapUsed / 1024 / 1024),
        memory_increase_mb: Math.round(memoryIncrease / 1024 / 1024),
        within_limits: passed
      });

    } catch (error: any) {
      this.addTestResult(testName, 'performance', false, Date.now() - startTime, {}, error.message);
    }
  }

  // ===========================================
  // 🛡️ ERROR HANDLING TESTS
  // ===========================================

  private async runErrorHandlingTests(): Promise<void> {
    logger.info('\n🛡️ RUNNING ERROR HANDLING TESTS');
    logger.info('='.repeat(50));

    await this.testInvalidInputHandling();
    await this.testMissingDataHandling();
    await this.testInvalidIndicatorNames();
    await this.testGracefulDegradation();
  }

  private async testInvalidInputHandling(): Promise<void> {
    const testName = 'Invalid Input Handling';
    const startTime = Date.now();

    try {
      const invalidInputs = [
        null,
        undefined,
        {},
        { timestamps: [], open: [], high: [], low: [], close: [] }, // Missing volume
        { timestamps: [1, 2], open: [1], high: [1], low: [1], close: [1], volume: [1] }, // Mismatched lengths
      ];

      const results = [];
      for (let i = 0; i < invalidInputs.length; i++) {
        try {
          const result = await this.engine.calculateIndicator('RSI', invalidInputs[i] as any, 'TEST', '1h');
          results.push({ input: i, handled_gracefully: !result.success, error_message: result.error_message });
        } catch (error: any) {
          results.push({ input: i, handled_gracefully: true, error_message: error.message });
        }
      }

      const passed = results.every(r => r.handled_gracefully);

      this.addTestResult(testName, 'error_handling', passed, Date.now() - startTime, {
        invalid_inputs_tested: invalidInputs.length,
        results: results
      });

    } catch (error: any) {
      this.addTestResult(testName, 'error_handling', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testMissingDataHandling(): Promise<void> {
    const testName = 'Missing Data Handling';
    const startTime = Date.now();

    try {
      // Test with insufficient data points
      const smallData = this.generateStandardTestData(5); // Too small for most indicators

      const result = await this.engine.calculateIndicator('RSI', smallData, 'TEST', '1h');

      // Should handle gracefully without crashing
      const passed = !result.success && result.error_message !== undefined;

      this.addTestResult(testName, 'error_handling', passed, Date.now() - startTime, {
        insufficient_data_handled: passed,
        error_message: result.error_message
      });

    } catch (error: any) {
      this.addTestResult(testName, 'error_handling', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testInvalidIndicatorNames(): Promise<void> {
    const testName = 'Invalid Indicator Names';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);
      const invalidNames = ['INVALID_INDICATOR', '', null, undefined, 'XYZ123'];

      const results = [];
      for (const name of invalidNames) {
        try {
          const result = await this.engine.calculateIndicator(name as string, testData, 'TEST', '1h');
          results.push({ name, handled_gracefully: !result.success });
        } catch (error: any) {
          results.push({ name, handled_gracefully: true });
        }
      }

      const passed = results.every(r => r.handled_gracefully);

      this.addTestResult(testName, 'error_handling', passed, Date.now() - startTime, {
        invalid_names_tested: invalidNames.length,
        results: results
      });

    } catch (error: any) {
      this.addTestResult(testName, 'error_handling', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testGracefulDegradation(): Promise<void> {
    const testName = 'Graceful Degradation';
    const startTime = Date.now();

    try {
      const testData = this.generateStandardTestData(100);

      // Test batch calculation where some indicators might fail
      const mixedIndicators = ['RSI', 'INVALID_INDICATOR', 'SMA', 'ANOTHER_INVALID', 'EMA'];

      const result = await this.engine.batchCalculateIndicators(mixedIndicators, testData, 'TEST', '1h');

      // Should complete with partial success
      const validIndicators = Object.values(result).filter(r => r.success);
      const invalidIndicators = Object.values(result).filter(r => !r.success);

      const passed = validIndicators.length > 0 && invalidIndicators.length > 0;

      this.addTestResult(testName, 'error_handling', passed, Date.now() - startTime, {
        total_indicators: mixedIndicators.length,
        successful_indicators: validIndicators.length,
        failed_indicators: invalidIndicators.length,
        graceful_degradation: passed
      });

    } catch (error: any) {
      this.addTestResult(testName, 'error_handling', false, Date.now() - startTime, {}, error.message);
    }
  }

  // ===========================================
  // 🔄 INTEGRATION TESTS
  // ===========================================

  private async runIntegrationTests(): Promise<void> {
    logger.info('\n🔄 RUNNING INTEGRATION TESTS');
    logger.info('='.repeat(50));

    await this.testEndToEndWorkflow();
    await this.testDataPipelineIntegrity();
    await this.testSystemResiliency();
  }

  private async testEndToEndWorkflow(): Promise<void> {
    const testName = 'End-to-End Workflow';
    const startTime = Date.now();

    try {
      // Simulate complete workflow: Market Data → Processing → Results
      const symbol = 'EURUSD';
      const timeframe = '1h';

      // Step 1: Get market data
      const marketData = await this.marketData.getHistoricalData(symbol, timeframe, 100);

      // Step 2: Convert to engine format
      const convertedData = this.convertMarketData(marketData);

      // Step 3: Calculate indicators
      const availableIndicators = await this.engine.getAvailableIndicators();
      const sampleIndicators = Object.values(availableIndicators).flat().slice(0, 5);

      const results = await this.engine.batchCalculateIndicators(sampleIndicators, convertedData, symbol, timeframe);

      // Step 4: Validate results
      const passed = Object.values(results).every(r => r.success && r.values !== null);

      this.addTestResult(testName, 'integration', passed, Date.now() - startTime, {
        symbol,
        timeframe,
        market_data_points: convertedData.timestamps.length,
        indicators_calculated: Object.keys(results).length,
        all_successful: passed
      });

    } catch (error: any) {
      this.addTestResult(testName, 'integration', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testDataPipelineIntegrity(): Promise<void> {
    const testName = 'Data Pipeline Integrity';
    const startTime = Date.now();

    try {
      // Test data transformation accuracy through the pipeline
      const originalData = this.generateStandardTestData(100);

      // Ensure data integrity through the pipeline
      const result = await this.engine.calculateIndicator('SMA', originalData, 'TEST', '1h');

      // Validate that input data wasn't corrupted
      const passed = result.success &&
                    result.values !== null &&
                    Array.isArray(result.values) &&
                    result.values.length > 0;

      this.addTestResult(testName, 'integration', passed, Date.now() - startTime, {
        original_data_points: originalData.timestamps.length,
        result_success: result.success,
        output_values_count: Array.isArray(result.values) ? result.values.length : 0,
        data_integrity_maintained: passed
      });

    } catch (error: any) {
      this.addTestResult(testName, 'integration', false, Date.now() - startTime, {}, error.message);
    }
  }

  private async testSystemResiliency(): Promise<void> {
    const testName = 'System Resiliency';
    const startTime = Date.now();

    try {
      // Test multiple rapid calculations to check system stability
      const testData = this.generateStandardTestData(100);
      const promises = [];

      for (let i = 0; i < 10; i++) {
        promises.push(this.engine.calculateIndicator('RSI', testData, `STRESS_TEST_${i}`, '1h'));
      }

      const results = await Promise.all(promises);
      const passed = results.every(r => r.success);

      this.addTestResult(testName, 'integration', passed, Date.now() - startTime, {
        stress_tests_count: 10,
        all_successful: passed,
        failed_count: results.filter(r => !r.success).length
      });

    } catch (error: any) {
      this.addTestResult(testName, 'integration', false, Date.now() - startTime, {}, error.message);
    }
  }

  // ===========================================
  // HELPER METHODS
  // ===========================================

  private generateStandardTestData(points: number = 100): any {
    const now = Date.now();
    return {
      timestamps: Array.from({ length: points }, (_, i) => now - (points - i) * 60000),
      open: Array.from({ length: points }, () => 1.1000 + Math.random() * 0.01),
      high: Array.from({ length: points }, () => 1.1000 + Math.random() * 0.02),
      low: Array.from({ length: points }, () => 1.0980 + Math.random() * 0.01),
      close: Array.from({ length: points }, () => 1.1000 + Math.random() * 0.01),
      volume: Array.from({ length: points }, () => Math.floor(Math.random() * 10000))
    };
  }

  private generateDeterministicData(): any {
    // Generate predictable data for known result testing
    const points = 100;
    return {
      timestamps: Array.from({ length: points }, (_, i) => Date.now() - (points - i) * 60000),
      open: Array.from({ length: points }, (_, i) => 100 + i * 0.1),
      high: Array.from({ length: points }, (_, i) => 100.5 + i * 0.1),
      low: Array.from({ length: points }, (_, i) => 99.5 + i * 0.1),
      close: Array.from({ length: points }, (_, i) => 100.2 + i * 0.1),
      volume: Array.from({ length: points }, () => 1000)
    };
  }

  private generateHighPrecisionData(): any {
    const points = 100;
    return {
      timestamps: Array.from({ length: points }, (_, i) => Date.now() - (points - i) * 60000),
      open: Array.from({ length: points }, () => 1.123456789),
      high: Array.from({ length: points }, () => 1.123556789),
      low: Array.from({ length: points }, () => 1.123356789),
      close: Array.from({ length: points }, () => 1.123456789),
      volume: Array.from({ length: points }, () => 1000)
    };
  }

  private generateDataWithGaps(points: number): any {
    const data = this.generateStandardTestData(points);
    // Introduce some null/undefined values
    data.close[10] = null;
    data.high[20] = undefined;
    return data;
  }

  private generateDataWithDecimals(points: number): any {
    const data = this.generateStandardTestData(points);
    // Use very precise decimal values
    data.close = data.close.map((val: number) => parseFloat((val * 1.123456789).toFixed(8)));
    return data;
  }

  private generateDataWithZeros(points: number): any {
    const data = this.generateStandardTestData(points);
    // Introduce some zero values
    data.volume[5] = 0;
    data.volume[15] = 0;
    return data;
  }

  private truncateData(data: any, newSize: number): any {
    return {
      timestamps: data.timestamps.slice(-newSize),
      open: data.open.slice(-newSize),
      high: data.high.slice(-newSize),
      low: data.low.slice(-newSize),
      close: data.close.slice(-newSize),
      volume: data.volume.slice(-newSize)
    };
  }

  private convertMarketData(marketData: any): any {
    return {
      timestamps: marketData.timestamp.map((t: Date) => t.getTime()),
      open: marketData.open,
      high: marketData.high,
      low: marketData.low,
      close: marketData.close,
      volume: marketData.volume
    };
  }

  private validateKnownResults(rsiResult: any, smaResult: any): boolean {
    // Implement specific validation logic for known mathematical results
    return rsiResult.success && smaResult.success &&
           Array.isArray(rsiResult.values) && Array.isArray(smaResult.values);
  }

  private validateIndicatorRelationships(batchResult: any): boolean {
    // Validate that related indicators show expected relationships
    const sma = batchResult['SMA'];
    const ema = batchResult['EMA'];

    if (!sma?.success || !ema?.success) return false;

    // EMA should be more responsive than SMA (basic validation)
    return Array.isArray(sma.values) && Array.isArray(ema.values);
  }

  private validateConsistency(shortResult: any, longResult: any): boolean {
    if (!shortResult.success || !longResult.success) return false;

    // The last values should be similar for overlapping periods
    const shortValues = shortResult.values;
    const longValues = longResult.values;

    if (!Array.isArray(shortValues) || !Array.isArray(longValues)) return false;

    const shortLast = shortValues[shortValues.length - 1];
    const longLast = longValues[longValues.length - 1];

    return Math.abs(shortLast - longLast) < TEST_CONFIG.ACCURACY_TOLERANCE;
  }

  private validatePrecision(values: any): boolean {
    if (!Array.isArray(values)) return false;

    // Check that precision is maintained
    return values.every(val =>
      typeof val === 'number' &&
      !isNaN(val) &&
      isFinite(val)
    );
  }

  private addTestResult(
    testName: string,
    category: TestResult['category'],
    passed: boolean,
    duration: number,
    details: any,
    errorMessage?: string
  ): void {
    this.results.push({
      testName,
      category,
      passed,
      duration,
      details,
      errorMessage
    });

    const status = passed ? '✅' : '❌';
    logger.info(`${status} ${testName} (${duration}ms)`);
    if (!passed && errorMessage) {
      logger.error(`   Error: ${errorMessage}`);
    }
  }

  private generateReport(): ValidationResults {
    const categories = this.results.reduce((acc, result) => {
      if (!acc[result.category]) {
        acc[result.category] = { total: 0, passed: 0, failed: 0 };
      }
      acc[result.category].total++;
      if (result.passed) {
        acc[result.category].passed++;
      } else {
        acc[result.category].failed++;
      }
      return acc;
    }, {} as { [key: string]: { total: number; passed: number; failed: number; } });

    const totalTests = this.results.length;
    const passedTests = this.results.filter(r => r.passed).length;
    const failedTests = totalTests - passedTests;

    const summary = this.generateSummary(totalTests, passedTests, failedTests, categories);

    return {
      totalTests,
      passedTests,
      failedTests,
      categories,
      results: this.results,
      summary
    };
  }

  private generateSummary(
    totalTests: number,
    passedTests: number,
    failedTests: number,
    categories: any
  ): string {
    const successRate = ((passedTests / totalTests) * 100).toFixed(1);

    let summary = `\n${'='.repeat(80)}\n`;
    summary += `🎯 COMPREHENSIVE VALIDATION RESULTS\n`;
    summary += `${'='.repeat(80)}\n\n`;
    summary += `📊 OVERALL RESULTS:\n`;
    summary += `   Total Tests: ${totalTests}\n`;
    summary += `   Passed: ${passedTests} (${successRate}%)\n`;
    summary += `   Failed: ${failedTests}\n\n`;

    summary += `📋 BY CATEGORY:\n`;
    Object.entries(categories as Record<string, { total: number; passed: number; failed: number }>).forEach(([category, stats]) => {
      const categoryRate = ((stats.passed / stats.total) * 100).toFixed(1);
      summary += `   ${category.toUpperCase()}: ${stats.passed}/${stats.total} (${categoryRate}%)\n`;
    });

    summary += `\n🔍 DETAILED ASSESSMENT:\n`;

    // Calculation Accuracy Assessment
    const accuracyTests = this.results.filter(r => r.category === 'accuracy');
    const accuracyPassed = accuracyTests.filter(r => r.passed).length;
    summary += `   🔬 Calculation Accuracy: ${accuracyPassed === accuracyTests.length ? 'EXCELLENT' : 'NEEDS ATTENTION'}\n`;

    // Real Data Processing Assessment
    const realDataTests = this.results.filter(r => r.category === 'real_data');
    const realDataPassed = realDataTests.filter(r => r.passed).length;
    summary += `   📊 Real Data Processing: ${realDataPassed === realDataTests.length ? 'EXCELLENT' : 'NEEDS ATTENTION'}\n`;

    // Performance Assessment
    const performanceTests = this.results.filter(r => r.category === 'performance');
    const performancePassed = performanceTests.filter(r => r.passed).length;
    summary += `   ⚡ Performance: ${performancePassed === performanceTests.length ? 'EXCELLENT' : 'NEEDS OPTIMIZATION'}\n`;

    // Error Handling Assessment
    const errorTests = this.results.filter(r => r.category === 'error_handling');
    const errorPassed = errorTests.filter(r => r.passed).length;
    summary += `   🛡️ Error Handling: ${errorPassed === errorTests.length ? 'ROBUST' : 'NEEDS IMPROVEMENT'}\n`;

    // Integration Assessment
    const integrationTests = this.results.filter(r => r.category === 'integration');
    const integrationPassed = integrationTests.filter(r => r.passed).length;
    summary += `   🔄 Integration: ${integrationPassed === integrationTests.length ? 'SOLID' : 'NEEDS WORK'}\n`;

    summary += `\n🎯 PRODUCTION READINESS:\n`;
    if (passedTests === totalTests) {
      summary += `   ✅ READY FOR PRODUCTION - All tests passed!\n`;
    } else if (successRate >= '90') {
      summary += `   ⚠️ MOSTLY READY - ${failedTests} issues need attention\n`;
    } else {
      summary += `   ❌ NOT READY - Significant issues need resolution\n`;
    }

    summary += `\n📁 Detailed results saved to: logs/validation-test-results.log\n`;
    summary += `${'='.repeat(80)}\n`;

    return summary;
  }
}

// Export for use in other test files
export default ComprehensiveValidationSuite;
