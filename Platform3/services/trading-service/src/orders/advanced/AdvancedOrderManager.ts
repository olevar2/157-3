/**
 * Advanced Order Manager
 * Centralized manager for all advanced order types in the Platform3 trading system
 *
 * This module provides:
 * - Unified interface for all advanced order types
 * - Coordinated order lifecycle management
 * - Performance monitoring and latency tracking
 * - Risk management integration
 * - Session-aware order coordination
 *
 * Expected Benefits:
 * - Simplified integration for trading strategies
 * - Consistent order management across all types
 * - Centralized performance monitoring
 * - Unified risk management enforcement
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';
import { ScalpingOCOOrderManager, ScalpingOCOOrderInput, ScalpingOCOOrderState } from './ScalpingOCOOrder';
import { FastTrailingStopOrderManager, FastTrailingStopOrderInput, FastTrailingStopOrderState } from './FastTrailingStopOrder';
import { SessionConditionalOrderManager, SessionConditionalOrderInput, SessionConditionalOrderState } from './SessionConditionalOrder';
import { VolatilityBasedOrderManager, VolatilityBasedOrderInput, VolatilityBasedOrderState } from './VolatilityBasedOrders';
import { DayTradingBracketOrderManager, DayTradingBracketOrderInput, DayTradingBracketOrderState } from './DayTradingBracketOrder';

// --- Advanced Order Manager Types ---

export enum AdvancedOrderType {
  SCALPING_OCO = 'SCALPING_OCO',
  FAST_TRAILING_STOP = 'FAST_TRAILING_STOP',
  SESSION_CONDITIONAL = 'SESSION_CONDITIONAL',
  VOLATILITY_BASED = 'VOLATILITY_BASED',
  DAY_TRADING_BRACKET = 'DAY_TRADING_BRACKET',
}

export interface AdvancedOrderRequest {
  orderType: AdvancedOrderType;
  input: ScalpingOCOOrderInput | FastTrailingStopOrderInput | SessionConditionalOrderInput | VolatilityBasedOrderInput | DayTradingBracketOrderInput;
  priority?: number; // 1-10, 10 being highest priority
  metadata?: Record<string, any>;
}

export interface AdvancedOrderResponse {
  orderId: string;
  orderType: AdvancedOrderType;
  status: string;
  createdAt: Date;
  latencyMs: number;
  state: ScalpingOCOOrderState | FastTrailingStopOrderState | SessionConditionalOrderState | VolatilityBasedOrderState | DayTradingBracketOrderState;
}

export interface PerformanceMetrics {
  totalOrders: number;
  averageLatencyMs: number;
  maxLatencyMs: number;
  minLatencyMs: number;
  sub10msCount: number;
  sub10msPercentage: number;
  orderTypeBreakdown: Record<AdvancedOrderType, number>;
  lastUpdated: Date;
}

/**
 * Advanced Order Manager
 * Coordinates all advanced order types with unified interface and performance monitoring
 */
export class AdvancedOrderManager extends EventEmitter {
  private scalpingOCOManager: ScalpingOCOOrderManager;
  private trailingStopManager: FastTrailingStopOrderManager;
  private sessionConditionalManager: SessionConditionalOrderManager;
  private volatilityBasedManager: VolatilityBasedOrderManager;
  private bracketOrderManager: DayTradingBracketOrderManager;

  private logger: Logger;
  private performanceMetrics: PerformanceMetrics;
  private orderRegistry: Map<string, { type: AdvancedOrderType; managerId: string }> = new Map();

  constructor(logger: Logger) {
    super();
    this.logger = logger;

    // Initialize all order managers
    this.scalpingOCOManager = new ScalpingOCOOrderManager(logger);
    this.trailingStopManager = new FastTrailingStopOrderManager(logger);
    this.sessionConditionalManager = new SessionConditionalOrderManager(logger);
    this.volatilityBasedManager = new VolatilityBasedOrderManager(logger);
    this.bracketOrderManager = new DayTradingBracketOrderManager(logger);

    // Initialize performance metrics
    this.performanceMetrics = {
      totalOrders: 0,
      averageLatencyMs: 0,
      maxLatencyMs: 0,
      minLatencyMs: Infinity,
      sub10msCount: 0,
      sub10msPercentage: 0,
      orderTypeBreakdown: {
        [AdvancedOrderType.SCALPING_OCO]: 0,
        [AdvancedOrderType.FAST_TRAILING_STOP]: 0,
        [AdvancedOrderType.SESSION_CONDITIONAL]: 0,
        [AdvancedOrderType.VOLATILITY_BASED]: 0,
        [AdvancedOrderType.DAY_TRADING_BRACKET]: 0,
      },
      lastUpdated: new Date(),
    };

    this._setupEventListeners();
    this.logger.info('AdvancedOrderManager initialized with all order types');
  }

  /**
   * Create an advanced order with unified interface
   */
  public async createAdvancedOrder(request: AdvancedOrderRequest): Promise<AdvancedOrderResponse> {
    const startTime = performance.now();

    try {
      let state: any;
      let orderId: string;

      switch (request.orderType) {
        case AdvancedOrderType.SCALPING_OCO:
          state = await this.scalpingOCOManager.createAndActivateOrder(request.input as ScalpingOCOOrderInput);
          orderId = state.ocoGroupId;
          break;

        case AdvancedOrderType.FAST_TRAILING_STOP:
          state = await this.trailingStopManager.createTrailingStopOrder(request.input as FastTrailingStopOrderInput);
          orderId = state.trailingStopOrderId;
          break;

        case AdvancedOrderType.SESSION_CONDITIONAL:
          state = await this.sessionConditionalManager.createConditionalOrder(request.input as SessionConditionalOrderInput);
          orderId = state.sessionConditionalOrderId;
          break;

        case AdvancedOrderType.VOLATILITY_BASED:
          state = await this.volatilityBasedManager.createVolatilityOrder(request.input as VolatilityBasedOrderInput);
          orderId = state.volatilityOrderId;
          break;

        case AdvancedOrderType.DAY_TRADING_BRACKET:
          state = await this.bracketOrderManager.createBracketOrder(request.input as DayTradingBracketOrderInput);
          orderId = state.bracketOrderId;
          break;

        default:
          throw new Error(`Unsupported order type: ${request.orderType}`);
      }

      const latencyMs = performance.now() - startTime;

      // Register order for tracking
      this.orderRegistry.set(orderId, {
        type: request.orderType,
        managerId: orderId
      });

      // Update performance metrics
      this._updatePerformanceMetrics(request.orderType, latencyMs);

      const response: AdvancedOrderResponse = {
        orderId,
        orderType: request.orderType,
        status: state.status,
        createdAt: state.createdAt,
        latencyMs,
        state
      };

      this.logger.info(`Advanced order created: ${request.orderType} (${orderId}) in ${latencyMs.toFixed(2)}ms`);
      this.emit('advancedOrderCreated', response);

      return response;

    } catch (error: any) {
      const latencyMs = performance.now() - startTime;
      this.logger.error(`Failed to create advanced order: ${error.message}`, {
        orderType: request.orderType,
        latencyMs
      });
      throw error;
    }
  }

  /**
   * Cancel an advanced order by ID
   */
  public async cancelAdvancedOrder(orderId: string, reason: string = 'USER_REQUEST'): Promise<boolean> {
    const orderInfo = this.orderRegistry.get(orderId);
    if (!orderInfo) {
      this.logger.warn(`Order ${orderId} not found in registry`);
      return false;
    }

    try {
      switch (orderInfo.type) {
        case AdvancedOrderType.SCALPING_OCO:
          await this.scalpingOCOManager.cancelOCOOrder(orderId, reason);
          break;

        case AdvancedOrderType.FAST_TRAILING_STOP:
          await this.trailingStopManager.cancelTrailingStopOrder(orderId, reason);
          break;

        case AdvancedOrderType.SESSION_CONDITIONAL:
          await this.sessionConditionalManager.cancelConditionalOrder(orderId, reason);
          break;

        case AdvancedOrderType.VOLATILITY_BASED:
          await this.volatilityBasedManager.cancelVolatilityOrder(orderId, reason);
          break;

        case AdvancedOrderType.DAY_TRADING_BRACKET:
          await this.bracketOrderManager.cancelBracketOrder(orderId, reason);
          break;
      }

      this.logger.info(`Advanced order cancelled: ${orderInfo.type} (${orderId})`);
      this.emit('advancedOrderCancelled', { orderId, type: orderInfo.type, reason });
      return true;

    } catch (error: any) {
      this.logger.error(`Failed to cancel advanced order ${orderId}: ${error.message}`);
      return false;
    }
  }

  /**
   * Get order state by ID
   */
  public getOrderState(orderId: string): any | null {
    const orderInfo = this.orderRegistry.get(orderId);
    if (!orderInfo) return null;

    switch (orderInfo.type) {
      case AdvancedOrderType.SCALPING_OCO:
        return this.scalpingOCOManager.getOrderState(orderId);
      case AdvancedOrderType.FAST_TRAILING_STOP:
        return this.trailingStopManager.getOrderState(orderId);
      case AdvancedOrderType.SESSION_CONDITIONAL:
        return this.sessionConditionalManager.getOrderState(orderId);
      case AdvancedOrderType.VOLATILITY_BASED:
        return this.volatilityBasedManager.getOrderState(orderId);
      case AdvancedOrderType.DAY_TRADING_BRACKET:
        return this.bracketOrderManager.getBracketOrderState(orderId);
      default:
        return null;
    }
  }

  /**
   * Get performance metrics
   */
  public getPerformanceMetrics(): PerformanceMetrics {
    return { ...this.performanceMetrics };
  }

  /**
   * Get all active orders
   */
  public getAllActiveOrders(): Array<{ orderId: string; type: AdvancedOrderType; state: any }> {
    const activeOrders: Array<{ orderId: string; type: AdvancedOrderType; state: any }> = [];

    for (const [orderId, orderInfo] of this.orderRegistry) {
      const state = this.getOrderState(orderId);
      if (state && this._isActiveState(state, orderInfo.type)) {
        activeOrders.push({
          orderId,
          type: orderInfo.type,
          state
        });
      }
    }

    return activeOrders;
  }

  /**
   * Setup event listeners for all order managers
   */
  private _setupEventListeners(): void {
    // Scalping OCO events
    this.scalpingOCOManager.on('ocoFilled', (state) => {
      this.emit('orderFilled', { type: AdvancedOrderType.SCALPING_OCO, state });
    });

    // Trailing Stop events
    this.trailingStopManager.on('trailingStopFilled', (state) => {
      this.emit('orderFilled', { type: AdvancedOrderType.FAST_TRAILING_STOP, state });
    });

    // Session Conditional events
    this.sessionConditionalManager.on('sessionConditionalOrderFilled', (state) => {
      this.emit('orderFilled', { type: AdvancedOrderType.SESSION_CONDITIONAL, state });
    });

    // Volatility Based events
    this.volatilityBasedManager.on('volatilityOrderFilled', (state) => {
      this.emit('orderFilled', { type: AdvancedOrderType.VOLATILITY_BASED, state });
    });

    // Bracket Order events
    this.bracketOrderManager.on('bracketLegFilled', (data) => {
      this.emit('orderFilled', { type: AdvancedOrderType.DAY_TRADING_BRACKET, state: data.bracketOrder });
    });
  }

  /**
   * Update performance metrics
   */
  private _updatePerformanceMetrics(orderType: AdvancedOrderType, latencyMs: number): void {
    this.performanceMetrics.totalOrders++;
    this.performanceMetrics.orderTypeBreakdown[orderType]++;

    // Update latency metrics
    if (latencyMs < this.performanceMetrics.minLatencyMs) {
      this.performanceMetrics.minLatencyMs = latencyMs;
    }
    if (latencyMs > this.performanceMetrics.maxLatencyMs) {
      this.performanceMetrics.maxLatencyMs = latencyMs;
    }

    // Calculate average latency
    const totalLatency = this.performanceMetrics.averageLatencyMs * (this.performanceMetrics.totalOrders - 1) + latencyMs;
    this.performanceMetrics.averageLatencyMs = totalLatency / this.performanceMetrics.totalOrders;

    // Track sub-10ms performance
    if (latencyMs < 10) {
      this.performanceMetrics.sub10msCount++;
    }
    this.performanceMetrics.sub10msPercentage = (this.performanceMetrics.sub10msCount / this.performanceMetrics.totalOrders) * 100;

    this.performanceMetrics.lastUpdated = new Date();
  }

  /**
   * Check if order state is active
   */
  private _isActiveState(state: any, type: AdvancedOrderType): boolean {
    switch (type) {
      case AdvancedOrderType.SCALPING_OCO:
        return state.status === 'ACTIVE';
      case AdvancedOrderType.FAST_TRAILING_STOP:
        return state.status === 'ACTIVE';
      case AdvancedOrderType.SESSION_CONDITIONAL:
        return state.status === 'ORDER_PLACED' || state.status === 'ORDER_HELD';
      case AdvancedOrderType.VOLATILITY_BASED:
        return state.status === 'ORDER_PLACED';
      case AdvancedOrderType.DAY_TRADING_BRACKET:
        return state.status === 'ACTIVE' || state.status === 'ENTRY_PLACED';
      default:
        return false;
    }
  }

  /**
   * Cleanup and destroy all managers
   */
  public destroy(): void {
    this.sessionConditionalManager.destroy();
    this.volatilityBasedManager.destroy();
    this.orderRegistry.clear();
    this.removeAllListeners();
    this.logger.info('AdvancedOrderManager destroyed');
  }
}

export default AdvancedOrderManager;
