/**
 * Advanced Order Types - Main Export Module
 * Centralized exports for all advanced order types in Platform3 trading system
 * 
 * This module provides unified access to:
 * - All advanced order type managers
 * - Common types and interfaces
 * - Unified order management
 * - Performance monitoring
 * - Integration utilities
 */

// Core Advanced Order Manager
export { 
  AdvancedOrderManager,
  AdvancedOrderType,
  AdvancedOrderRequest,
  AdvancedOrderResponse,
  PerformanceMetrics
} from './AdvancedOrderManager';

// Scalping OCO Orders
export {
  ScalpingOCOOrderManager,
  ScalpingOCOOrderInput,
  ScalpingOCOOrderState,
  OCOLeg
} from './ScalpingOCOOrder';

// Fast Trailing Stop Orders
export {
  FastTrailingStopOrderManager,
  FastTrailingStopOrderInput,
  FastTrailingStopOrderState,
  TrailType
} from './FastTrailingStopOrder';

// Session Conditional Orders
export {
  SessionConditionalOrderManager,
  SessionConditionalOrderInput,
  SessionConditionalOrderState,
  SessionConditionalStatus,
  TradingSession
} from './SessionConditionalOrder';

// Volatility Based Orders
export {
  VolatilityBasedOrderManager,
  VolatilityBasedOrderInput,
  VolatilityBasedOrderState,
  VolatilityOrderStatus,
  VolatilityData
} from './VolatilityBasedOrders';

// Day Trading Bracket Orders
export {
  DayTradingBracketOrderManager,
  DayTradingBracketOrderInput,
  DayTradingBracketOrderState,
  BracketStatus,
  BracketLeg
} from './DayTradingBracketOrder';

// Common Types (re-exported for convenience)
export {
  OrderSide,
  OrderType,
  OrderStatus,
  TimeInForce,
  BaseOrderParams,
  Order,
  OrderFillEvent
} from './ScalpingOCOOrder';

// Test Suite
export {
  AdvancedOrderTestSuite,
  runTests
} from './test-advanced-orders';

/**
 * Factory function to create a fully configured AdvancedOrderManager
 */
import { Logger } from 'winston';
import { AdvancedOrderManager } from './AdvancedOrderManager';

export function createAdvancedOrderManager(logger: Logger): AdvancedOrderManager {
  return new AdvancedOrderManager(logger);
}

/**
 * Version information
 */
export const VERSION = '1.0.0';
export const DESCRIPTION = 'Advanced Order Types for Platform3 Forex Trading System';

/**
 * Supported order types list
 */
export const SUPPORTED_ORDER_TYPES = [
  'SCALPING_OCO',
  'FAST_TRAILING_STOP', 
  'SESSION_CONDITIONAL',
  'VOLATILITY_BASED',
  'DAY_TRADING_BRACKET'
] as const;

/**
 * Performance requirements
 */
export const PERFORMANCE_REQUIREMENTS = {
  MAX_LATENCY_MS: 10,
  TARGET_SUB_10MS_PERCENTAGE: 90,
  MAX_AVERAGE_LATENCY_MS: 15
} as const;

/**
 * Default configuration
 */
export const DEFAULT_CONFIG = {
  sessionCheckIntervalMs: 60000, // 1 minute
  volatilityUpdateIntervalMs: 30000, // 30 seconds
  performanceMonitoringEnabled: true,
  riskManagementEnabled: true
} as const;
