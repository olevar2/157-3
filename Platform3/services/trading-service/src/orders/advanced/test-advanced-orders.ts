/**
 * Advanced Order Types Integration Test
 * Comprehensive testing for all advanced order types with performance validation
 *
 * This test suite validates:
 * - All advanced order types functionality
 * - Sub-10ms latency requirements
 * - Session-based order management
 * - Volatility-based adjustments
 * - Risk management integration
 * - Error handling and edge cases
 */

import { AdvancedOrderManager, AdvancedOrderType, AdvancedOrderRequest } from './AdvancedOrderManager';
import { OrderSide, OrderType, TimeInForce } from './ScalpingOCOOrder';
import { TradingSession } from './SessionConditionalOrder';
import { TrailType } from './FastTrailingStopOrder';

// Mock logger for testing
const mockLogger = {
  info: (msg: string, ...args: any[]) => console.log(`[INFO] ${msg}`, ...args),
  warn: (msg: string, ...args: any[]) => console.warn(`[WARN] ${msg}`, ...args),
  error: (msg: string, ...args: any[]) => console.error(`[ERROR] ${msg}`, ...args),
  debug: (msg: string, ...args: any[]) => console.debug(`[DEBUG] ${msg}`, ...args),
} as any;

interface TestResult {
  testName: string;
  passed: boolean;
  latencyMs?: number;
  error?: string;
  details?: any;
}

class AdvancedOrderTestSuite {
  private orderManager: AdvancedOrderManager;
  private testResults: TestResult[] = [];

  constructor() {
    this.orderManager = new AdvancedOrderManager(mockLogger);
  }

  /**
   * Run all tests
   */
  public async runAllTests(): Promise<void> {
    console.log('🚀 Starting Advanced Order Types Integration Test Suite\n');

    try {
      // Test 1: Scalping OCO Orders
      await this.testScalpingOCOOrders();

      // Test 2: Fast Trailing Stop Orders
      await this.testFastTrailingStopOrders();

      // Test 3: Session Conditional Orders
      await this.testSessionConditionalOrders();

      // Test 4: Volatility Based Orders
      await this.testVolatilityBasedOrders();

      // Test 5: Day Trading Bracket Orders
      await this.testDayTradingBracketOrders();

      // Test 6: Performance and Latency
      await this.testPerformanceMetrics();

      // Test 7: Error Handling
      await this.testErrorHandling();

      // Test 8: Integration and Coordination
      await this.testIntegrationScenarios();

    } catch (error: any) {
      console.error('❌ Test suite failed:', error.message);
    } finally {
      this.printTestResults();
      this.orderManager.destroy();
    }
  }

  /**
   * Test Scalping OCO Orders
   */
  private async testScalpingOCOOrders(): Promise<void> {
    console.log('📊 Testing Scalping OCO Orders...');

    const request: AdvancedOrderRequest = {
      orderType: AdvancedOrderType.SCALPING_OCO,
      input: {
        clientOrderId: 'test-scalping-oco-001',
        symbol: 'EURUSD',
        side: OrderSide.BUY,
        quantity: 10000,
        accountId: 'test-account-001',
        userId: 'test-user-001',
        takeProfitPrice: 1.0850,
        stopLossPrice: 1.0830,
        timeInForce: TimeInForce.GTC,
        activateDuringSessions: [TradingSession.LONDON, TradingSession.NEW_YORK],
      }
    };

    try {
      const startTime = performance.now();
      const response = await this.orderManager.createAdvancedOrder(request);
      const latencyMs = performance.now() - startTime;

      this.testResults.push({
        testName: 'Scalping OCO Order Creation',
        passed: response.orderId !== undefined && response.orderType === AdvancedOrderType.SCALPING_OCO,
        latencyMs,
        details: { orderId: response.orderId, status: response.status }
      });

      console.log(`✅ Scalping OCO order created: ${response.orderId} (${latencyMs.toFixed(2)}ms)`);

    } catch (error: any) {
      this.testResults.push({
        testName: 'Scalping OCO Order Creation',
        passed: false,
        error: error.message
      });
      console.log(`❌ Scalping OCO order failed: ${error.message}`);
    }
  }

  /**
   * Test Fast Trailing Stop Orders
   */
  private async testFastTrailingStopOrders(): Promise<void> {
    console.log('📈 Testing Fast Trailing Stop Orders...');

    const request: AdvancedOrderRequest = {
      orderType: AdvancedOrderType.FAST_TRAILING_STOP,
      input: {
        clientOrderId: 'test-trailing-stop-001',
        symbol: 'GBPUSD',
        side: OrderSide.SELL,
        quantity: 5000,
        accountId: 'test-account-001',
        userId: 'test-user-001',
        trailType: TrailType.POINTS,
        trailAmount: 20, // 20 pips
        initialStopPrice: 1.2650,
        timeInForce: TimeInForce.DAY,
      }
    };

    try {
      const startTime = performance.now();
      const response = await this.orderManager.createAdvancedOrder(request);
      const latencyMs = performance.now() - startTime;

      this.testResults.push({
        testName: 'Fast Trailing Stop Order Creation',
        passed: response.orderId !== undefined && response.orderType === AdvancedOrderType.FAST_TRAILING_STOP,
        latencyMs,
        details: { orderId: response.orderId, status: response.status }
      });

      console.log(`✅ Trailing stop order created: ${response.orderId} (${latencyMs.toFixed(2)}ms)`);

    } catch (error: any) {
      this.testResults.push({
        testName: 'Fast Trailing Stop Order Creation',
        passed: false,
        error: error.message
      });
      console.log(`❌ Trailing stop order failed: ${error.message}`);
    }
  }

  /**
   * Test Session Conditional Orders
   */
  private async testSessionConditionalOrders(): Promise<void> {
    console.log('🕐 Testing Session Conditional Orders...');

    const request: AdvancedOrderRequest = {
      orderType: AdvancedOrderType.SESSION_CONDITIONAL,
      input: {
        clientOrderId: 'test-session-conditional-001',
        symbol: 'USDJPY',
        side: OrderSide.BUY,
        quantity: 8000,
        accountId: 'test-account-001',
        userId: 'test-user-001',
        orderType: OrderType.LIMIT,
        price: 149.50,
        targetSessions: [TradingSession.TOKYO, TradingSession.LONDON],
        actionOutsideSession: 'HOLD',
        cancelAtSessionEnd: true,
        placeAtSessionStart: true,
      }
    };

    try {
      const startTime = performance.now();
      const response = await this.orderManager.createAdvancedOrder(request);
      const latencyMs = performance.now() - startTime;

      this.testResults.push({
        testName: 'Session Conditional Order Creation',
        passed: response.orderId !== undefined && response.orderType === AdvancedOrderType.SESSION_CONDITIONAL,
        latencyMs,
        details: { orderId: response.orderId, status: response.status }
      });

      console.log(`✅ Session conditional order created: ${response.orderId} (${latencyMs.toFixed(2)}ms)`);

    } catch (error: any) {
      this.testResults.push({
        testName: 'Session Conditional Order Creation',
        passed: false,
        error: error.message
      });
      console.log(`❌ Session conditional order failed: ${error.message}`);
    }
  }

  /**
   * Test Volatility Based Orders
   */
  private async testVolatilityBasedOrders(): Promise<void> {
    console.log('📊 Testing Volatility Based Orders...');

    const request: AdvancedOrderRequest = {
      orderType: AdvancedOrderType.VOLATILITY_BASED,
      input: {
        clientOrderId: 'test-volatility-based-001',
        symbol: 'AUDUSD',
        side: OrderSide.BUY,
        quantity: 12000,
        accountId: 'test-account-001',
        userId: 'test-user-001',
        orderType: OrderType.MARKET,
        useVolatilityAdjustment: true,
        atrMultiplierStopLoss: 1.5,
        atrMultiplierTakeProfit: 2.5,
        minStopLossPips: 10,
        maxStopLossPips: 50,
        adjustQuantity: true,
        riskPerTradePercent: 1.0,
      }
    };

    try {
      const startTime = performance.now();
      const response = await this.orderManager.createAdvancedOrder(request);
      const latencyMs = performance.now() - startTime;

      this.testResults.push({
        testName: 'Volatility Based Order Creation',
        passed: response.orderId !== undefined && response.orderType === AdvancedOrderType.VOLATILITY_BASED,
        latencyMs,
        details: { orderId: response.orderId, status: response.status }
      });

      console.log(`✅ Volatility based order created: ${response.orderId} (${latencyMs.toFixed(2)}ms)`);

    } catch (error: any) {
      this.testResults.push({
        testName: 'Volatility Based Order Creation',
        passed: false,
        error: error.message
      });
      console.log(`❌ Volatility based order failed: ${error.message}`);
    }
  }

  /**
   * Test Day Trading Bracket Orders
   */
  private async testDayTradingBracketOrders(): Promise<void> {
    console.log('🎯 Testing Day Trading Bracket Orders...');

    const request: AdvancedOrderRequest = {
      orderType: AdvancedOrderType.DAY_TRADING_BRACKET,
      input: {
        clientOrderId: 'test-bracket-order-001',
        symbol: 'USDCAD',
        side: OrderSide.SELL,
        quantity: 15000,
        accountId: 'test-account-001',
        userId: 'test-user-001',
        entryOrderType: OrderType.LIMIT,
        entryPrice: 1.3650,
        takeProfitDistance: 30, // 30 pips
        stopLossDistance: 20,   // 20 pips
        timeInForce: TimeInForce.DAY,
        activateDuringSessions: [TradingSession.NEW_YORK],
        cancelOutsideSessions: true,
      }
    };

    try {
      const startTime = performance.now();
      const response = await this.orderManager.createAdvancedOrder(request);
      const latencyMs = performance.now() - startTime;

      this.testResults.push({
        testName: 'Day Trading Bracket Order Creation',
        passed: response.orderId !== undefined && response.orderType === AdvancedOrderType.DAY_TRADING_BRACKET,
        latencyMs,
        details: { orderId: response.orderId, status: response.status }
      });

      console.log(`✅ Bracket order created: ${response.orderId} (${latencyMs.toFixed(2)}ms)`);

    } catch (error: any) {
      this.testResults.push({
        testName: 'Day Trading Bracket Order Creation',
        passed: false,
        error: error.message
      });
      console.log(`❌ Bracket order failed: ${error.message}`);
    }
  }

  /**
   * Test Performance Metrics
   */
  private async testPerformanceMetrics(): Promise<void> {
    console.log('⚡ Testing Performance Metrics...');

    const metrics = this.orderManager.getPerformanceMetrics();

    const sub10msRequirementMet = metrics.sub10msPercentage >= 90; // 90% of orders should be sub-10ms
    const averageLatencyAcceptable = metrics.averageLatencyMs <= 15; // Average should be under 15ms

    this.testResults.push({
      testName: 'Sub-10ms Latency Requirement',
      passed: sub10msRequirementMet,
      details: {
        sub10msPercentage: metrics.sub10msPercentage,
        averageLatencyMs: metrics.averageLatencyMs,
        totalOrders: metrics.totalOrders
      }
    });

    this.testResults.push({
      testName: 'Average Latency Performance',
      passed: averageLatencyAcceptable,
      details: {
        averageLatencyMs: metrics.averageLatencyMs,
        maxLatencyMs: metrics.maxLatencyMs,
        minLatencyMs: metrics.minLatencyMs
      }
    });

    console.log(`📊 Performance Metrics:`);
    console.log(`   Total Orders: ${metrics.totalOrders}`);
    console.log(`   Average Latency: ${metrics.averageLatencyMs.toFixed(2)}ms`);
    console.log(`   Sub-10ms Percentage: ${metrics.sub10msPercentage.toFixed(1)}%`);
    console.log(`   Min/Max Latency: ${metrics.minLatencyMs.toFixed(2)}ms / ${metrics.maxLatencyMs.toFixed(2)}ms`);
  }

  /**
   * Test Error Handling
   */
  private async testErrorHandling(): Promise<void> {
    console.log('🛡️ Testing Error Handling...');

    // Test invalid order type
    try {
      const invalidRequest: any = {
        orderType: 'INVALID_TYPE',
        input: {}
      };

      await this.orderManager.createAdvancedOrder(invalidRequest);
      this.testResults.push({
        testName: 'Invalid Order Type Handling',
        passed: false,
        error: 'Should have thrown error for invalid order type'
      });
    } catch (error: any) {
      this.testResults.push({
        testName: 'Invalid Order Type Handling',
        passed: true,
        details: { errorMessage: error.message }
      });
      console.log(`✅ Invalid order type properly rejected: ${error.message}`);
    }

    // Test missing required fields
    try {
      const incompleteRequest: AdvancedOrderRequest = {
        orderType: AdvancedOrderType.SCALPING_OCO,
        input: {
          symbol: 'EURUSD',
          side: OrderSide.BUY,
          quantity: 10000,
          accountId: '',
          userId: '',
          takeProfitPrice: 1.0850,
          stopLossPrice: 1.0830,
        } as any
      };

      await this.orderManager.createAdvancedOrder(incompleteRequest);
      this.testResults.push({
        testName: 'Missing Required Fields Handling',
        passed: false,
        error: 'Should have thrown error for missing fields'
      });
    } catch (error: any) {
      this.testResults.push({
        testName: 'Missing Required Fields Handling',
        passed: true,
        details: { errorMessage: error.message }
      });
      console.log(`✅ Missing fields properly rejected: ${error.message}`);
    }
  }

  /**
   * Test Integration Scenarios
   */
  private async testIntegrationScenarios(): Promise<void> {
    console.log('🔗 Testing Integration Scenarios...');

    // Test multiple order types simultaneously
    const orders: AdvancedOrderRequest[] = [
      {
        orderType: AdvancedOrderType.SCALPING_OCO,
        input: {
          clientOrderId: 'integration-oco-001',
          symbol: 'EURUSD',
          side: OrderSide.BUY,
          quantity: 5000,
          accountId: 'integration-account',
          userId: 'integration-user',
          takeProfitPrice: 1.0860,
          stopLossPrice: 1.0840,
        }
      },
      {
        orderType: AdvancedOrderType.SESSION_CONDITIONAL,
        input: {
          clientOrderId: 'integration-session-001',
          symbol: 'GBPUSD',
          side: OrderSide.SELL,
          quantity: 7500,
          accountId: 'integration-account',
          userId: 'integration-user',
          orderType: OrderType.MARKET,
          targetSessions: [TradingSession.LONDON],
          actionOutsideSession: 'HOLD',
        }
      }
    ];

    try {
      const startTime = performance.now();
      const responses = await Promise.all(
        orders.map(order => this.orderManager.createAdvancedOrder(order))
      );
      const totalLatency = performance.now() - startTime;

      const allSuccessful = responses.every(response => response.orderId !== undefined);

      this.testResults.push({
        testName: 'Multiple Order Types Integration',
        passed: allSuccessful,
        latencyMs: totalLatency,
        details: {
          orderCount: responses.length,
          orderIds: responses.map(r => r.orderId),
          averageLatencyPerOrder: totalLatency / responses.length
        }
      });

      console.log(`✅ Integration test passed: ${responses.length} orders created in ${totalLatency.toFixed(2)}ms`);

      // Test getting all active orders
      const activeOrders = this.orderManager.getAllActiveOrders();
      console.log(`📋 Active orders count: ${activeOrders.length}`);

    } catch (error: any) {
      this.testResults.push({
        testName: 'Multiple Order Types Integration',
        passed: false,
        error: error.message
      });
      console.log(`❌ Integration test failed: ${error.message}`);
    }
  }

  /**
   * Print test results summary
   */
  private printTestResults(): void {
    console.log('\n📋 Test Results Summary');
    console.log('========================');

    const passedTests = this.testResults.filter(result => result.passed);
    const failedTests = this.testResults.filter(result => !result.passed);

    console.log(`✅ Passed: ${passedTests.length}`);
    console.log(`❌ Failed: ${failedTests.length}`);
    console.log(`📊 Total: ${this.testResults.length}`);
    console.log(`🎯 Success Rate: ${((passedTests.length / this.testResults.length) * 100).toFixed(1)}%`);

    if (failedTests.length > 0) {
      console.log('\n❌ Failed Tests:');
      failedTests.forEach(test => {
        console.log(`   - ${test.testName}: ${test.error || 'Unknown error'}`);
      });
    }

    // Latency analysis
    const testsWithLatency = this.testResults.filter(result => result.latencyMs !== undefined);
    if (testsWithLatency.length > 0) {
      const avgLatency = testsWithLatency.reduce((sum, test) => sum + (test.latencyMs || 0), 0) / testsWithLatency.length;
      const sub10msTests = testsWithLatency.filter(test => (test.latencyMs || 0) < 10);

      console.log('\n⚡ Latency Analysis:');
      console.log(`   Average Latency: ${avgLatency.toFixed(2)}ms`);
      console.log(`   Sub-10ms Tests: ${sub10msTests.length}/${testsWithLatency.length} (${((sub10msTests.length / testsWithLatency.length) * 100).toFixed(1)}%)`);
    }

    console.log('\n🏁 Test Suite Complete');
  }
}

// Run the test suite
async function runTests() {
  const testSuite = new AdvancedOrderTestSuite();
  await testSuite.runAllTests();
}

// Export for use in other test files
export { AdvancedOrderTestSuite, runTests };

// Run tests if this file is executed directly
if (require.main === module) {
  runTests().catch(console.error);
}
