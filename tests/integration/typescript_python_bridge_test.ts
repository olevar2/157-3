/**
 * TypeScript-Python Bridge Integration Tests
 * 
 * This file tests the integration between TypeScript engines and Python AI models
 * to ensure proper communication and functioning of the entire bridge system.
 * 
 * Each test verifies that:
 * 1. TypeScript can spawn Python processes
 * 2. Commands can be sent to Python engines
 * 3. Responses are properly received and parsed
 * 4. Error handling works correctly
 * 5. All 4 engine types are functional
 */

import { expect } from 'chai';
import { TechnicalAnalysisEngine } from '../../engines/typescript_engines/TechnicalAnalysisEngine';
import { MLModelEngine } from '../../engines/typescript_engines/MLModelEngine';
import { PatternRecognitionEngine } from '../../engines/typescript_engines/PatternRecognitionEngine';
import { RiskAnalysisEngine } from '../../engines/typescript_engines/RiskAnalysisEngine';
import { Logger } from 'winston';
import * as winston from 'winston';
import { MarketData } from '../../engines/typescript_engines/TechnicalAnalysisEngine';

// Create test logger
const testLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ level, message, timestamp }) => {
      return `${timestamp} [${level}]: ${message}`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'tests/logs/bridge_tests.log' })
  ]
});

describe('TypeScript-Python Bridge Integration Tests', function() {
  this.timeout(30000); // Set timeout to 30 seconds for longer-running tests
  
  // Sample market data for testing
  const generateTestMarketData = (count = 100): MarketData[] => {
    const data: MarketData[] = [];
    let price = 1.0850;
    
    for (let i = 0; i < count; i++) {
      const change = (Math.random() - 0.5) * 0.002;
      price += change;
      
      data.push({
        timestamp: Date.now() - (count - i) * 60000,
        open: price - change,
        high: price + Math.random() * 0.001,
        low: price - Math.random() * 0.001,
        close: price,
        volume: Math.floor(Math.random() * 1000) + 500
      });
    }
    
    return data;
  };
  
  describe('TechnicalAnalysisEngine Bridge Tests', () => {
    let engine: TechnicalAnalysisEngine;
    
    before(async () => {
      engine = new TechnicalAnalysisEngine(testLogger);
      await engine.initialize();
    });
    
    after(async () => {
      if (engine['pythonInterface'] && engine['pythonInterface'].isConnected()) {
        await engine['pythonInterface'].disconnect();
      }
    });
    
    it('should initialize successfully and connect to Python engine', () => {
      expect(engine.isReady()).to.be.true;
      expect(engine['pythonEngine'].initialized).to.be.true;
      expect(engine['pythonEngine'].pythonProcess).to.not.be.undefined;
    });
    
    it('should send ping command and receive pong response', async () => {
      const pingResult = await engine['pythonInterface'].sendCommand('ping', { message: 'test' });
      expect(pingResult).to.have.property('status', 'pong');
    });
    
    it('should calculate RSI using Python engine', async () => {
      const data = generateTestMarketData(50);
      const rsiResult = await engine['pythonInterface'].sendCommand('calculate_rsi', {
        data: data.map(d => d.close),
        period: 14
      });
      
      expect(rsiResult).to.have.property('values');
      expect(Array.isArray(rsiResult.values)).to.be.true;
      expect(rsiResult.values.length).to.be.greaterThan(0);
    });
    
    it('should successfully run all integration tests', async () => {
      const result = await engine.runIntegrationTests();
      expect(result).to.be.true;
    });
  });
  
  describe('MLModelEngine Bridge Tests', () => {
    let engine: MLModelEngine;
    
    before(async () => {
      engine = new MLModelEngine(testLogger);
      await engine.initialize();
    });
    
    after(async () => {
      if (engine['pythonInterface'] && engine['pythonInterface'].isConnected()) {
        await engine['pythonInterface'].disconnect();
      }
    });
    
    it('should initialize successfully and connect to Python engine', () => {
      expect(engine.isReady()).to.be.true;
      expect(engine['pythonEngine'].initialized).to.be.true;
      expect(engine['pythonEngine'].pythonProcess).to.not.be.undefined;
    });
    
    it('should send ping command and receive pong response', async () => {
      const pingResult = await engine['pythonInterface'].sendCommand('ping', { 
        test: 'ml_integration',
        engine_type: 'ml_model'
      });
      expect(pingResult).to.have.property('status', 'pong');
    });
    
    it('should generate ML predictions using Python engine', async () => {
      const data = generateTestMarketData(150);
      const result = await engine.predict('TEST_SYMBOL', data, '1h');
      
      expect(result).to.have.property('predictions');
      expect(Array.isArray(result.predictions)).to.be.true;
      expect(result.predictions.length).to.be.greaterThan(0);
    });
    
    it('should successfully run all integration tests', async () => {
      const result = await engine.runIntegrationTests();
      expect(result).to.be.true;
    });
  });
  
  describe('PatternRecognitionEngine Bridge Tests', () => {
    let engine: PatternRecognitionEngine;
    
    before(async () => {
      engine = new PatternRecognitionEngine(testLogger);
      await engine.initialize();
    });
    
    after(async () => {
      if (engine['pythonInterface'] && engine['pythonInterface'].isConnected()) {
        await engine['pythonInterface'].disconnect();
      }
    });
    
    it('should initialize successfully and connect to Python engine', () => {
      expect(engine.isReady()).to.be.true;
      expect(engine['pythonEngine'].initialized).to.be.true;
      expect(engine['pythonEngine'].pythonProcess).to.not.be.undefined;
    });
    
    it('should send ping command and receive pong response', async () => {
      const pingResult = await engine['pythonInterface'].sendCommand('ping', { 
        test: 'pattern_integration',
        engine_type: 'pattern_recognition'
      });
      expect(pingResult).to.have.property('status', 'pong');
    });
    
    it('should detect patterns using Python engine', async () => {
      const data = generateTestMarketData(100);
      const result = await engine.detectPatterns('TEST_SYMBOL', data);
      
      expect(result).to.have.property('patterns');
      expect(Array.isArray(result.patterns)).to.be.true;
    });
    
    it('should successfully run all integration tests', async () => {
      const result = await engine.runIntegrationTests();
      expect(result).to.be.true;
    });
  });
  
  describe('RiskAnalysisEngine Bridge Tests', () => {
    let engine: RiskAnalysisEngine;
    
    before(async () => {
      engine = new RiskAnalysisEngine(testLogger);
      await engine.initialize();
    });
    
    after(async () => {
      if (engine['pythonInterface'] && engine['pythonInterface'].isConnected()) {
        await engine['pythonInterface'].disconnect();
      }
    });
    
    it('should initialize successfully and connect to Python engine', () => {
      expect(engine.isReady()).to.be.true;
      expect(engine['pythonEngine'].initialized).to.be.true;
      expect(engine['pythonEngine'].pythonProcess).to.not.be.undefined;
    });
    
    it('should send ping command and receive pong response', async () => {
      const pingResult = await engine['pythonInterface'].sendCommand('ping', { 
        test: 'risk_integration',
        engine_type: 'risk_analysis'
      });
      expect(pingResult).to.have.property('status', 'pong');
    });
    
    it('should analyze portfolio risk using Python engine', async () => {
      const positions = engine['generateTestPositions']();
      const result = await engine.analyzePortfolio(positions, 100000);
      
      expect(result).to.have.property('portfolioRisk');
      expect(result.portfolioRisk).to.have.property('riskScore');
    });
    
    it('should successfully run all integration tests', async () => {
      const result = await engine.runIntegrationTests();
      expect(result).to.be.true;
    });
  });
});