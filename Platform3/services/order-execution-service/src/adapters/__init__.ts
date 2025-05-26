/**
 * Order Execution Service - Broker Adapters Module
 * Centralized exports for all broker adapters and management components
 * 
 * This module provides unified access to:
 * - All broker adapter implementations
 * - Centralized broker management
 * - Common types and interfaces
 * - Configuration utilities
 * - Performance monitoring
 */

// Base Broker Adapter
export { 
  BrokerAdapter,
  BrokerConfig,
  OrderRequest,
  OrderResponse,
  MarketDataRequest,
  MarketDataResponse,
  AccountInfo,
  Position,
  Trade,
  RateLimiter
} from './BrokerAdapter';

// Broker Manager
export {
  BrokerManager,
  BrokerManagerConfig,
  BrokerConfiguration,
  BrokerFeatures,
  BrokerStatus,
  RoutingDecision
} from './BrokerManager';

// MetaTrader Adapter
export {
  MetaTraderAdapter,
  MetaTraderConfig,
  MetaTraderSymbolInfo,
  MetaTraderOrderInfo
} from './MetaTraderAdapter';

// OANDA Adapter
export {
  OANDAAdapter,
  OANDAConfig,
  OANDAInstrument,
  OANDAPrice,
  OANDAOrder
} from './OANDAAdapter';

// Interactive Brokers Adapter
export {
  InteractiveBrokersAdapter,
  IBKRConfig,
  IBKRContract,
  IBKROrder,
  IBKRPosition
} from './InteractiveBrokersAdapter';

// cTrader Adapter
export {
  cTraderAdapter,
  cTraderConfig,
  cTraderSymbol,
  cTraderPosition,
  cTraderOrder
} from './cTraderAdapter';

/**
 * Version information
 */
export const VERSION = '1.0.0';
export const DESCRIPTION = 'Multi-Broker API Integration for Platform3 Forex Trading System';

/**
 * Supported broker types
 */
export const SUPPORTED_BROKERS = [
  'metatrader',
  'oanda', 
  'interactive_brokers',
  'ctrader'
] as const;

/**
 * Default configuration templates
 */
export const DEFAULT_BROKER_CONFIGS = {
  metatrader: {
    timeout: 30000,
    maxRetries: 3,
    rateLimitPerSecond: 10,
    platform: 'MT5',
    enableEAIntegration: false,
    hedgingEnabled: true,
    symbolMapping: {
      'EURUSD': 'EURUSD',
      'GBPUSD': 'GBPUSD',
      'USDJPY': 'USDJPY',
      'USDCHF': 'USDCHF',
      'AUDUSD': 'AUDUSD',
      'USDCAD': 'USDCAD',
      'NZDUSD': 'NZDUSD'
    }
  },
  oanda: {
    timeout: 30000,
    maxRetries: 3,
    rateLimitPerSecond: 100,
    environment: 'practice',
    enableStreaming: true,
    instruments: ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD'],
    maxConcurrentOrders: 20,
    defaultUnits: 10000
  },
  interactive_brokers: {
    timeout: 30000,
    maxRetries: 3,
    rateLimitPerSecond: 50,
    paperTrading: true,
    enableTWS: true,
    enableClientPortal: true,
    marketDataSubscriptions: ['EURUSD', 'GBPUSD', 'USDJPY'],
    requestTimeout: 30000
  },
  ctrader: {
    timeout: 30000,
    maxRetries: 3,
    rateLimitPerSecond: 20,
    environment: 'demo',
    enableCopyTrading: false,
    enableSocialTrading: false,
    symbolGroups: ['Major Currencies', 'Minor Currencies'],
    maxPositions: 100
  }
};

/**
 * Performance requirements for all brokers
 */
export const PERFORMANCE_REQUIREMENTS = {
  MAX_LATENCY_MS: 100,
  TARGET_SUB_50MS_PERCENTAGE: 95,
  MAX_ERROR_RATE: 0.01,
  MIN_UPTIME_PERCENTAGE: 99.9
} as const;

/**
 * Routing strategies available
 */
export const ROUTING_STRATEGIES = [
  'best_spread',
  'lowest_latency', 
  'round_robin',
  'load_balanced'
] as const;

/**
 * Utility function to create broker manager configuration
 */
export function createBrokerManagerConfig(
  brokers: BrokerConfiguration[],
  options: Partial<BrokerManagerConfig> = {}
): BrokerManagerConfig {
  return {
    brokers,
    routingStrategy: options.routingStrategy || 'best_spread',
    enableFailover: options.enableFailover ?? true,
    maxRetries: options.maxRetries || 3,
    healthCheckInterval: options.healthCheckInterval || 30000,
    performanceWindow: options.performanceWindow || 60000,
    ...options
  };
}

/**
 * Utility function to validate broker configuration
 */
export function validateBrokerConfig(config: BrokerConfiguration): boolean {
  if (!config.id || !config.type || !config.config) {
    return false;
  }
  
  if (!SUPPORTED_BROKERS.includes(config.type as any)) {
    return false;
  }
  
  if (config.priority < 1 || config.priority > 10) {
    return false;
  }
  
  if (!Array.isArray(config.supportedSymbols) || config.supportedSymbols.length === 0) {
    return false;
  }
  
  return true;
}

/**
 * Utility function to get default features for broker type
 */
export function getDefaultBrokerFeatures(brokerType: string): BrokerFeatures {
  const baseFeatures: BrokerFeatures = {
    supportsMarketOrders: true,
    supportsLimitOrders: true,
    supportsStopOrders: true,
    supportsOCOOrders: false,
    supportsBracketOrders: false,
    supportsTrailingStops: false,
    supportsPartialFills: true,
    supportsHedging: false,
    minOrderSize: 0.01,
    maxOrderSize: 100,
    maxPositions: 50
  };
  
  switch (brokerType) {
    case 'metatrader':
      return {
        ...baseFeatures,
        supportsOCOOrders: true,
        supportsBracketOrders: true,
        supportsTrailingStops: true,
        supportsHedging: true,
        maxOrderSize: 1000,
        maxPositions: 200
      };
      
    case 'oanda':
      return {
        ...baseFeatures,
        supportsOCOOrders: true,
        supportsBracketOrders: true,
        supportsTrailingStops: true,
        maxOrderSize: 500,
        maxPositions: 100
      };
      
    case 'interactive_brokers':
      return {
        ...baseFeatures,
        supportsOCOOrders: true,
        supportsBracketOrders: true,
        supportsTrailingStops: true,
        supportsHedging: true,
        maxOrderSize: 10000,
        maxPositions: 1000
      };
      
    case 'ctrader':
      return {
        ...baseFeatures,
        supportsOCOOrders: true,
        supportsBracketOrders: true,
        supportsTrailingStops: true,
        maxOrderSize: 500,
        maxPositions: 100
      };
      
    default:
      return baseFeatures;
  }
}

/**
 * Common symbol mappings for major forex pairs
 */
export const COMMON_SYMBOL_MAPPINGS = {
  'EURUSD': {
    metatrader: 'EURUSD',
    oanda: 'EUR_USD',
    interactive_brokers: 'EUR.USD',
    ctrader: 'EURUSD'
  },
  'GBPUSD': {
    metatrader: 'GBPUSD',
    oanda: 'GBP_USD',
    interactive_brokers: 'GBP.USD',
    ctrader: 'GBPUSD'
  },
  'USDJPY': {
    metatrader: 'USDJPY',
    oanda: 'USD_JPY',
    interactive_brokers: 'USD.JPY',
    ctrader: 'USDJPY'
  },
  'USDCHF': {
    metatrader: 'USDCHF',
    oanda: 'USD_CHF',
    interactive_brokers: 'USD.CHF',
    ctrader: 'USDCHF'
  },
  'AUDUSD': {
    metatrader: 'AUDUSD',
    oanda: 'AUD_USD',
    interactive_brokers: 'AUD.USD',
    ctrader: 'AUDUSD'
  },
  'USDCAD': {
    metatrader: 'USDCAD',
    oanda: 'USD_CAD',
    interactive_brokers: 'USD.CAD',
    ctrader: 'USDCAD'
  },
  'NZDUSD': {
    metatrader: 'NZDUSD',
    oanda: 'NZD_USD',
    interactive_brokers: 'NZD.USD',
    ctrader: 'NZDUSD'
  }
};

/**
 * Export types for TypeScript support
 */
export type SupportedBrokerType = typeof SUPPORTED_BROKERS[number];
export type RoutingStrategy = typeof ROUTING_STRATEGIES[number];
