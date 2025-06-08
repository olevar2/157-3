/**
 * Jest Global Setup Configuration for Platform3 Phase 2 Testing
 * Initializes testing environment and common mocks
 */

// Global test utilities
global.testUtils = {
  // Time utilities
  sleep: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  // Mock data generators
  generateMockData: {
    user: () => ({
      id: 'test-user-123',
      username: 'testuser',
      email: 'test@platform3.com',
      role: 'trader',
      createdAt: new Date().toISOString()
    }),
    
    tradingSession: () => ({
      sessionId: 'session-123',
      userId: 'user-123',
      startTime: new Date().toISOString(),
      endTime: null,
      trades: [],
      performance: {
        totalPnL: 0,
        winRate: 0,
        sharpeRatio: 0
      }
    }),
    
    marketData: () => ({
      symbol: 'EURUSD',
      timestamp: new Date().toISOString(),
      bid: 1.0850,
      ask: 1.0852,
      volume: 1000000,
      change: 0.0012,
      changePercent: 0.11
    }),
    
    aiPrediction: () => ({
      predictionId: 'pred-123',
      symbol: 'EURUSD',
      timeframe: '1h',
      prediction: 'BUY',
      confidence: 0.85,
      targetPrice: 1.0900,
      stopLoss: 1.0800,
      timestamp: new Date().toISOString()
    })
  },
  
  // Database utilities
  database: {
    clearTestData: async () => {
      // Implementation will be added based on database setup
      console.log('Clearing test data...');
    },
    
    seedTestData: async () => {
      // Implementation will be added based on database setup
      console.log('Seeding test data...');
    }
  },
  
  // API utilities
  api: {
    createAuthHeaders: (token = 'test-token') => ({
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    }),
    
    createTestRequest: (data = {}) => ({
      headers: {
        'x-request-id': 'test-request-123',
        'x-test-mode': 'true'
      },
      body: data,
      user: global.testUtils.generateMockData.user()
    })
  }
};

// Global test configuration
global.testConfig = {
  timeout: 30000,
  retries: 2,
  environment: 'test',
  
  // Service endpoints for testing
  services: {
    authService: process.env.AUTH_SERVICE_URL || 'http://localhost:3000',
    tradingService: process.env.TRADING_SERVICE_URL || 'http://localhost:3001',
    marketDataService: process.env.MARKET_DATA_SERVICE_URL || 'http://localhost:3002',
    aiService: process.env.AI_SERVICE_URL || 'http://localhost:3003',
    configService: process.env.CONFIG_SERVICE_URL || 'http://localhost:3004'
  },
  
  // Database configuration for testing
  database: {
    host: process.env.TEST_DB_HOST || 'localhost',
    port: parseInt(process.env.TEST_DB_PORT) || 5433,
    database: process.env.TEST_DB_NAME || 'platform3_test',
    user: process.env.TEST_DB_USER || 'platform3_test',
    password: process.env.TEST_DB_PASSWORD || 'test_password'
  },
  
  // Redis configuration for testing
  redis: {
    host: process.env.TEST_REDIS_HOST || 'localhost',
    port: parseInt(process.env.TEST_REDIS_PORT) || 6380,
    db: parseInt(process.env.TEST_REDIS_DB) || 1
  }
};

// Global mocks
jest.mock('winston', () => ({
  createLogger: jest.fn(() => ({
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
    level: 'info'
  })),
  format: {
    combine: jest.fn(),
    timestamp: jest.fn(),
    errors: jest.fn(),
    json: jest.fn(),
    colorize: jest.fn(),
    simple: jest.fn(),
    printf: jest.fn()
  },
  transports: {
    Console: jest.fn(),
    File: jest.fn()
  }
}));

// Mock EventEmitter for error handling tests
jest.mock('events', () => {
  const EventEmitter = jest.requireActual('events');
  return {
    EventEmitter: class MockEventEmitter extends EventEmitter {
      constructor() {
        super();
        this.emitHistory = [];
      }
      
      emit(event, ...args) {
        this.emitHistory.push({ event, args, timestamp: new Date() });
        return super.emit(event, ...args);
      }
      
      getEmitHistory() {
        return this.emitHistory;
      }
      
      clearEmitHistory() {
        this.emitHistory = [];
      }
    }
  };
});

// Enhanced console for testing
const originalConsole = global.console;
global.console = {
  ...originalConsole,
  // Capture console output for testing
  captured: {
    logs: [],
    errors: [],
    warnings: []
  },
  
  log: (...args) => {
    global.console.captured.logs.push(args);
    if (process.env.VERBOSE_TESTS === 'true') {
      originalConsole.log(...args);
    }
  },
  
  error: (...args) => {
    global.console.captured.errors.push(args);
    if (process.env.VERBOSE_TESTS === 'true') {
      originalConsole.error(...args);
    }
  },
  
  warn: (...args) => {
    global.console.captured.warnings.push(args);
    if (process.env.VERBOSE_TESTS === 'true') {
      originalConsole.warn(...args);
    }
  },
  
  clearCaptured: () => {
    global.console.captured.logs = [];
    global.console.captured.errors = [];
    global.console.captured.warnings = [];
  }
};

// Global test hooks
beforeEach(() => {
  // Clear console captures
  global.console.clearCaptured();
  
  // Reset date mocks
  jest.clearAllTimers();
  
  // Clear any pending async operations
  jest.clearAllMocks();
});

afterEach(() => {
  // Clean up any test artifacts
  jest.restoreAllMocks();
});

// Global error handling for tests
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Promise Rejection during tests:', reason);
  throw reason;
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception during tests:', error);
  throw error;
});

console.log('📝 Platform3 Phase 2 Testing Framework initialized');
console.log(`🧪 Test environment: ${global.testConfig.environment}`);
console.log(`🎯 Coverage threshold: 90%`);
console.log(`⚡ Max workers: 50%`);
console.log(`⏱️  Test timeout: ${global.testConfig.timeout}ms`);
