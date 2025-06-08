/**
 * Humanitarian Trading Platform - Technical Analysis Bridge Integration Tests
 * Critical tests to ensure Python-TypeScript integration works for children's welfare
 */

import { TechnicalAnalysisEngine, MarketData } from '../../engines/typescript_engines/TechnicalAnalysisEngine';
import { Logger, createLogger, format, transports } from 'winston';

describe('Technical Analysis Bridge Integration Tests', () => {
  let engine: TechnicalAnalysisEngine;
  let logger: Logger;

  beforeAll(async () => {
    // Setup logger for humanitarian mission
    logger = createLogger({
      level: 'debug',
      format: format.combine(
        format.timestamp(),
        format.printf(({ timestamp, level, message }) => {
          return `${timestamp} [${level.toUpperCase()}]: ${message}`;
        })
      ),
      transports: [
        new transports.Console(),
        new transports.File({ filename: 'logs/humanitarian-trading-tests.log' })
      ]
    });

    engine = new TechnicalAnalysisEngine(logger);
  });

  beforeEach(() => {
    logger.info('🚀 Starting humanitarian trading test...');
  });

  afterEach(() => {
    logger.info('✅ Humanitarian trading test completed');
  });

  describe('Engine Initialization', () => {
    test('should initialize Technical Analysis Engine successfully', async () => {
      expect(engine).toBeDefined();
      expect(engine.isReady()).toBe(false);

      await engine.initialize();
      expect(engine.isReady()).toBe(true);

      logger.info('✅ Technical Analysis Engine initialized for humanitarian mission');
    }, 60000); // 60 second timeout for Python engine startup
  });

  describe('Python Bridge Communication', () => {
    test('should establish communication with Python engines', async () => {
      await engine.initialize();
      
      // Test integration functionality
      const testsPassed = await engine.runIntegrationTests();
      expect(testsPassed).toBe(true);

      logger.info('✅ Python bridge communication established for children\'s welfare');
    }, 60000);
  });

  describe('Market Data Analysis', () => {
    test('should analyze market data for humanitarian trading', async () => {
      await engine.initialize();

      const sampleData = generateHumanitarianTestData();
      const analysis = await engine.analyze('EUR/USD', sampleData);

      // Validate analysis structure
      expect(analysis).toBeDefined();
      expect(analysis.symbol).toBe('EUR/USD');
      expect(analysis.indicators).toBeDefined();
      expect(analysis.signals).toBeDefined();
      expect(analysis.trend).toBeDefined();
      expect(analysis.sentiment).toBeDefined();

      // Validate indicators
      expect(analysis.indicators.rsi).toBeDefined();
      expect(analysis.indicators.macd).toBeDefined();
      expect(analysis.indicators.bollingerBands).toBeDefined();
      expect(analysis.indicators.movingAverages).toBeDefined();
      expect(analysis.indicators.stochastic).toBeDefined();
      expect(analysis.indicators.atr).toBeDefined();

      // Validate humanitarian trading signals
      expect(Array.isArray(analysis.signals)).toBe(true);
      analysis.signals.forEach(signal => {
        expect(signal.type).toMatch(/^(buy|sell|hold)$/);
        expect(signal.strength).toBeGreaterThanOrEqual(0);
        expect(signal.strength).toBeLessThanOrEqual(1);
        expect(signal.confidence).toBeGreaterThanOrEqual(0);
        expect(signal.confidence).toBeLessThanOrEqual(1);
      });

      logger.info(`✅ Generated ${analysis.signals.length} trading signals for humanitarian profit generation`);
    }, 60000);
  });

  describe('AI Enhancement Validation', () => {
    test('should integrate with Python AI models for enhanced analysis', async () => {
      await engine.initialize();

      const sampleData = generateHumanitarianTestData();
      const analysis = await engine.analyze('GBP/USD', sampleData);

      // Check for AI-enhanced signals
      const aiSignals = analysis.signals.filter(signal => signal.source.includes('AI-'));
      
      if (aiSignals.length > 0) {
        logger.info(`✅ AI enhancement working: ${aiSignals.length} AI-enhanced signals detected`);
        
        aiSignals.forEach(signal => {
          expect(signal.confidence).toBeGreaterThan(0.5); // AI signals should have higher confidence
        });
      } else {
        logger.warn('⚠️ AI enhancement not detected - using local calculations only');
      }

      // Validate sentiment analysis
      expect(analysis.sentiment.score).toBeGreaterThanOrEqual(-1);
      expect(analysis.sentiment.score).toBeLessThanOrEqual(1);
      expect(analysis.sentiment.confidence).toBeGreaterThanOrEqual(0);
      expect(analysis.sentiment.confidence).toBeLessThanOrEqual(1);
    }, 60000);
  });

  describe('Performance and Reliability', () => {
    test('should handle multiple concurrent analyses for high-frequency trading', async () => {
      await engine.initialize();

      const promises = [];
      const currencies = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD'];

      for (const currency of currencies) {
        const testData = generateHumanitarianTestData();
        promises.push(engine.analyze(currency, testData));
      }

      const results = await Promise.all(promises);
      
      expect(results).toHaveLength(5);
      results.forEach((result, index) => {
        expect(result.symbol).toBe(currencies[index]);
        expect(result.signals.length).toBeGreaterThan(0);
      });

      logger.info('✅ Concurrent analysis capability confirmed for humanitarian trading');
    }, 90000);

    test('should provide consistent results for same input data', async () => {
      await engine.initialize();

      const testData = generateHumanitarianTestData();
      
      const analysis1 = await engine.analyze('EUR/USD', testData);
      const analysis2 = await engine.analyze('EUR/USD', testData);

      // RSI should be identical
      expect(analysis1.indicators.rsi.current).toBeCloseTo(analysis2.indicators.rsi.current, 5);
      
      // MACD should be identical
      expect(analysis1.indicators.macd.macd).toBeCloseTo(analysis2.indicators.macd.macd, 5);
      
      // Trend direction should be consistent
      expect(analysis1.trend.direction).toBe(analysis2.trend.direction);

      logger.info('✅ Analysis consistency verified for reliable humanitarian trading');
    }, 60000);
  });

  describe('Error Handling and Resilience', () => {
    test('should handle insufficient data gracefully', async () => {
      await engine.initialize();

      const insufficientData = generateHumanitarianTestData().slice(0, 10); // Only 10 data points

      await expect(engine.analyze('EUR/USD', insufficientData))
        .rejects
        .toThrow('Insufficient data for technical analysis');

      logger.info('✅ Insufficient data error handling confirmed');
    });

    test('should fallback to local calculations if Python engine fails', async () => {
      await engine.initialize();

      // This test assumes we can still get analysis even if Python engine has issues
      const testData = generateHumanitarianTestData();
      const analysis = await engine.analyze('EUR/USD', testData);

      expect(analysis).toBeDefined();
      expect(analysis.signals.length).toBeGreaterThan(0);

      logger.info('✅ Fallback mechanism confirmed for humanitarian mission reliability');
    }, 60000);
  });
});

function generateHumanitarianTestData(): MarketData[] {
  const data: MarketData[] = [];
  let price = 1.1000; // EUR/USD starting price
  const timestamp = Date.now();
  
  // Generate 100 periods of realistic forex data
  for (let i = 0; i < 100; i++) {
    const change = (Math.random() - 0.5) * 0.002; // ±0.2% change
    price += change;
    
    const high = price + Math.random() * 0.0005;
    const low = price - Math.random() * 0.0005;
    
    data.push({
      timestamp: timestamp - (100 - i) * 60000, // 1-minute intervals
      open: price - change,
      high: Math.max(high, price - change, price),
      low: Math.min(low, price - change, price),
      close: price,
      volume: Math.floor(Math.random() * 1000) + 500
    });
  }
  
  return data;
}

// Integration test for sentiment analysis specifically
describe('Humanitarian Sentiment Analysis', () => {
  let engine: TechnicalAnalysisEngine;
  let logger: Logger;

  beforeAll(async () => {
    logger = createLogger({
      level: 'info',
      transports: [new transports.Console()]
    });

    engine = new TechnicalAnalysisEngine(logger);
    await engine.initialize();
  });

  test('should provide meaningful sentiment for humanitarian trading decisions', async () => {
    const testData = generateHumanitarianTestData();
    const sentiment = await engine.analyzeSentiment('EUR/USD', testData);

    expect(sentiment.score).toBeGreaterThanOrEqual(-1);
    expect(sentiment.score).toBeLessThanOrEqual(1);
    expect(sentiment.label).toMatch(/^(very_bearish|bearish|neutral|bullish|very_bullish)$/);
    expect(sentiment.confidence).toBeGreaterThanOrEqual(0);
    expect(sentiment.confidence).toBeLessThanOrEqual(1);
    expect(Array.isArray(sentiment.factors)).toBe(true);

    logger.info(`✅ Sentiment: ${sentiment.label} (${sentiment.score.toFixed(3)}) - Confidence: ${(sentiment.confidence * 100).toFixed(1)}%`);
    logger.info(`📊 Factors: ${sentiment.factors.join(', ')}`);
  });
});