/**
 * Day Trading Bracket Order
 * Entry + Stop Loss + Take Profit order combination for intraday trading
 * 
 * This module provides bracket order functionality including:
 * - Parent entry order with child stop loss and take profit
 * - Automatic position management for day trading
 * - Risk-controlled entry with predefined exit strategy
 * - Session-based order management (enhanced)
 * - Dynamic adjustments based on intraday price action (future enhancement)
 *
 * Expected Benefits:
 * - Complete intraday trade automation
 * - Predefined risk/reward ratios
 * - Reduced emotional trading decisions
 * - Professional day trading order management
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { Logger } from 'winston'; // Assuming winston logger
import {
  OrderSide,
  OrderType,
  OrderStatus,
  TimeInForce,
  BaseOrderParams,
  Order,
  OrderFillEvent,
  // TradingSession, // If defined in ScalpingOCOOrder or a shared types file
} from './ScalpingOCOOrder'; // Reuse types

// --- Enhanced Bracket Order Specific Types ---

// Example TradingSession enum (if not imported)
export enum TradingSession { 
  ASIAN = 'ASIAN',
  LONDON = 'LONDON',
  NEW_YORK = 'NEW_YORK',
  OVERLAP_LDN_NY = 'OVERLAP_LDN_NY',
}

export interface BracketLeg extends Order {
  bracketOrderId: string;
  legType: 'ENTRY' | 'TAKE_PROFIT' | 'STOP_LOSS';
}

export interface DayTradingBracketOrderInput extends BaseOrderParams {
  entryOrderType: OrderType.MARKET | OrderType.LIMIT | OrderType.STOP;
  entryPrice?: number; // Required for LIMIT/STOP entry orders
  takeProfitDistance?: number; // In pips or price points from entry
  stopLossDistance?: number;   // In pips or price points from entry
  takeProfitPrice?: number; // Absolute price, alternative to distance
  stopLossPrice?: number;   // Absolute price, alternative to distance
  timeInForce?: TimeInForce;   // Default to DAY for day trading
  activateDuringSessions?: TradingSession[];
  cancelOutsideSessions?: boolean; // If true, cancel if market session closes
  // Future: Add trailing stop options, break-even triggers
}

export enum BracketStatus {
  PENDING_ENTRY = 'PENDING_ENTRY',
  ENTRY_PLACED = 'ENTRY_PLACED',
  ACTIVE = 'ACTIVE', // Entry filled, TP/SL active
  PARTIALLY_FILLED_ENTRY = 'PARTIALLY_FILLED_ENTRY',
  FILLED = 'FILLED', // TP or SL hit
  CANCELED = 'CANCELED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED',
}

export interface DayTradingBracketOrderState {
  bracketOrderId: string;
  clientOrderId?: string | undefined; // Allow undefined for exactOptionalPropertyTypes
  accountId: string;
  userId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  status: BracketStatus;
  entryOrder: BracketLeg | null;
  takeProfitOrder: BracketLeg | null;
  stopLossOrder: BracketLeg | null;
  createdAt: Date;
  updatedAt: Date;
  entryFilledAt?: Date;
  completedAt?: Date;
  triggeredLeg?: 'TAKE_PROFIT' | 'STOP_LOSS';
  actualEntryPrice?: number;
  actualTakeProfitPrice?: number; // Store the calculated TP price
  actualStopLossPrice?: number;   // Store the calculated SL price
  activationConditions?: {
    sessions?: TradingSession[];
  };
  cancelOutsideSessions?: boolean;
  executionLatencyMs?: number;

  // Store relevant parts of the original input for TP/SL calculation
  originalInputSide: OrderSide;
  originalInputEntryOrderType: OrderType.MARKET | OrderType.LIMIT | OrderType.STOP;
  originalInputEntryPrice?: number | undefined;
  originalInputTakeProfitPrice?: number | undefined;
  originalInputStopLossPrice?: number | undefined;
  originalInputTakeProfitDistance?: number | undefined;
  originalInputStopLossDistance?: number | undefined;
  originalInputTimeInForce?: TimeInForce | undefined;
}

/**
 * Manages Day Trading Bracket Orders.
 * This class handles the lifecycle of a bracket order: entry, take profit, and stop loss.
 */
export class DayTradingBracketOrderManager extends EventEmitter {
  private bracketOrders: Map<string, DayTradingBracketOrderState> = new Map();
  private logger: Logger;
  // private omsClient: OrderManagementSystem; // Hypothetical OMS client
  private sessionTimers: Map<string, NodeJS.Timeout> = new Map();

  constructor(logger: Logger /*, omsClient: OrderManagementSystem */) {
    super();
    this.logger = logger;
    // this.omsClient = omsClient;
    this.logger.info('DayTradingBracketOrderManager initialized.');
  }

  private _validateInput(input: DayTradingBracketOrderInput): void {
    if (!input.symbol || !input.side || input.quantity <= 0 || !input.accountId || !input.userId) {
      throw new Error('Missing required fields (symbol, side, quantity, accountId, userId).');
    }
    if ((input.entryOrderType === OrderType.LIMIT || input.entryOrderType === OrderType.STOP) && typeof input.entryPrice !== 'number') {
      throw new Error('Entry price is required and must be a number for LIMIT/STOP entry orders.');
    }
    if (!input.takeProfitDistance && typeof input.takeProfitPrice !== 'number') {
      throw new Error('Either takeProfitDistance (number) or takeProfitPrice (number) must be provided.');
    }
    if (!input.stopLossDistance && typeof input.stopLossPrice !== 'number') {
      throw new Error('Either stopLossDistance (number) or stopLossPrice (number) must be provided.');
    }
    if (input.takeProfitDistance && input.takeProfitDistance <= 0) {
      throw new Error('Take profit distance must be positive.');
    }
    if (input.stopLossDistance && input.stopLossDistance <= 0) {
      throw new Error('Stop loss distance must be positive.');
    }
    // Add more checks, e.g., SL/TP prices relative to entry for BUY/SELL
  }

  private _calculateExitPrices(state: DayTradingBracketOrderState, entryPrice: number): { tpPrice: number; slPrice: number } {
    let tpPrice = state.originalInputTakeProfitPrice;
    let slPrice = state.originalInputStopLossPrice;

    if (typeof state.originalInputTakeProfitDistance === 'number') {
      tpPrice = state.originalInputSide === OrderSide.BUY ? entryPrice + state.originalInputTakeProfitDistance : entryPrice - state.originalInputTakeProfitDistance;
    }
    if (typeof state.originalInputStopLossDistance === 'number') {
      slPrice = state.originalInputSide === OrderSide.BUY ? entryPrice - state.originalInputStopLossDistance : entryPrice + state.originalInputStopLossDistance;
    }

    if (typeof tpPrice !== 'number' || typeof slPrice !== 'number') {
        throw new Error('Could not definitively determine take profit or stop loss price from input.');
    }
    
    // Sanity check for prices
    if (state.originalInputSide === OrderSide.BUY && tpPrice <= slPrice) {
        throw new Error('For BUY orders, take profit price must be greater than stop loss price.');
    }
    if (state.originalInputSide === OrderSide.SELL && tpPrice >= slPrice) {
        throw new Error('For SELL orders, take profit price must be less than stop loss price.');
    }
    // Ensure prices are rounded to a reasonable precision for the symbol if necessary (not done here)
    return { tpPrice, slPrice };
  }

  public async createBracketOrder(input: DayTradingBracketOrderInput): Promise<DayTradingBracketOrderState> {
    const activationStartTime = performance.now();
    this._validateInput(input);

    const bracketOrderId = uuidv4();
    const now = new Date();

    const constructedActivationConditions: Partial<DayTradingBracketOrderState['activationConditions']> = {};
    if (input.activateDuringSessions && input.activateDuringSessions.length > 0) {
        constructedActivationConditions.sessions = input.activateDuringSessions;
    }

    const initialState: DayTradingBracketOrderState = {
      bracketOrderId,
      clientOrderId: input.clientOrderId,
      accountId: input.accountId,
      userId: input.userId,
      symbol: input.symbol,
      side: input.side,
      quantity: input.quantity,
      status: BracketStatus.PENDING_ENTRY,
      entryOrder: null,
      takeProfitOrder: null,
      stopLossOrder: null,
      createdAt: now,
      updatedAt: now,
      ...(Object.keys(constructedActivationConditions).length > 0 && { activationConditions: constructedActivationConditions }),
      cancelOutsideSessions: input.cancelOutsideSessions ?? false,
      // Store original input parts
      originalInputSide: input.side,
      originalInputEntryOrderType: input.entryOrderType,
      originalInputEntryPrice: input.entryPrice,
      originalInputTakeProfitPrice: input.takeProfitPrice,
      originalInputStopLossPrice: input.stopLossPrice,
      originalInputTakeProfitDistance: input.takeProfitDistance,
      originalInputStopLossDistance: input.stopLossDistance,
      originalInputTimeInForce: input.timeInForce,
    };

    this.bracketOrders.set(bracketOrderId, initialState);
    this.logger.info(`Bracket Order ${bracketOrderId} PENDING_ENTRY.`, initialState);
    this.emit('bracketOrderCreated', initialState);

    // TODO: Implement session checking logic if activateDuringSessions is set.
    // For now, proceed to place entry order.

    try {
      const entryOrderPayload: any = {
        symbol: input.symbol,
        side: input.side,
        quantity: input.quantity,
        accountId: input.accountId,
        userId: input.userId,
        type: input.entryOrderType,
        timeInForce: input.timeInForce || TimeInForce.DAY,
        metadata: { bracketOrderId, legType: 'ENTRY' as const },
      };

      if (input.clientOrderId) {
        entryOrderPayload.clientOrderId = `${input.clientOrderId}-ENTRY`;
      }
      if (input.entryOrderType === OrderType.LIMIT) {
        if (typeof input.entryPrice === 'number') entryOrderPayload.price = input.entryPrice;
        // Validation should catch if entryPrice is not a number here
      } else if (input.entryOrderType === OrderType.STOP) {
        if (typeof input.entryPrice === 'number') entryOrderPayload.stopPrice = input.entryPrice;
        // Validation should catch if entryPrice is not a number here
      }

      const entryOrderLeg = await this._placeOrderWithOMS(entryOrderPayload as Parameters<typeof this._placeOrderWithOMS>[0]);
      initialState.entryOrder = entryOrderLeg;
      initialState.status = BracketStatus.ENTRY_PLACED;
      initialState.updatedAt = new Date();
      initialState.executionLatencyMs = performance.now() - activationStartTime;

      this.logger.info(`Bracket ${bracketOrderId}: Entry order ${entryOrderLeg.id} placed. Status: ${initialState.status}. Latency: ${initialState.executionLatencyMs.toFixed(2)}ms`);
      this.emit('bracketEntryPlaced', initialState);

      if (initialState.cancelOutsideSessions && initialState.activationConditions?.sessions) {
        this._setupSessionTimer(bracketOrderId, initialState.activationConditions.sessions);
      }

    } catch (error: any) {
      this.logger.error(`Bracket ${bracketOrderId}: Failed to place entry order: ${error.message}`, { input });
      initialState.status = BracketStatus.REJECTED;
      initialState.updatedAt = new Date();
      this.bracketOrders.set(bracketOrderId, initialState);
      this.emit('bracketCreationFailed', initialState);
      throw error;
    }

    return initialState;
  }

  public async handleOrderUpdate(orderId: string, update: { status: OrderStatus; fill?: OrderFillEvent; filledQuantity?: number; averageFillPrice?: number }): Promise<void> {
    const bracketOrder = this._findBracketOrderByLegId(orderId);
    if (!bracketOrder) {
      this.logger.warn(`Received update for order ${orderId} not part of a known Bracket order.`);
      return;
    }

    const now = new Date();
    let legToUpdate: BracketLeg | null = null;

    if (bracketOrder.entryOrder?.id === orderId) legToUpdate = bracketOrder.entryOrder;
    else if (bracketOrder.takeProfitOrder?.id === orderId) legToUpdate = bracketOrder.takeProfitOrder;
    else if (bracketOrder.stopLossOrder?.id === orderId) legToUpdate = bracketOrder.stopLossOrder;

    if (!legToUpdate) return;

    legToUpdate.status = update.status;
    legToUpdate.updatedAt = now;
    // Ensure filledQuantity and averageFillPrice are numbers if provided in update
    if (typeof update.filledQuantity === 'number') legToUpdate.filledQuantity = update.filledQuantity;
    if (typeof update.averageFillPrice === 'number') legToUpdate.averageFillPrice = update.averageFillPrice;
    
    bracketOrder.updatedAt = now;

    if (legToUpdate.legType === 'ENTRY' && update.status === OrderStatus.FILLED) {
      const entryFillPrice = typeof update.averageFillPrice === 'number' ? update.averageFillPrice : legToUpdate.price;

      if (typeof entryFillPrice !== 'number') {
        this.logger.error(`Bracket ${bracketOrder.bracketOrderId}: Entry filled but no valid entry price (averageFillPrice or leg.price) found.`);
        bracketOrder.status = BracketStatus.REJECTED;
        this.emit('bracketCreationFailed', { bracketOrder, error: new Error('Missing entry fill price') });
        return;
      }
      bracketOrder.actualEntryPrice = entryFillPrice;
      bracketOrder.status = BracketStatus.ACTIVE;
      bracketOrder.entryFilledAt = now;
      this.logger.info(`Bracket ${bracketOrder.bracketOrderId}: Entry order ${orderId} FILLED at ${bracketOrder.actualEntryPrice}.`);
      this.emit('bracketEntryFilled', bracketOrder);

      const { tpPrice, slPrice } = this._calculateExitPrices(bracketOrder, bracketOrder.actualEntryPrice);
      bracketOrder.actualTakeProfitPrice = tpPrice;
      bracketOrder.actualStopLossPrice = slPrice;

      try {
        const tpLegPayload: any = {
          symbol: bracketOrder.symbol,
          side: bracketOrder.originalInputSide === OrderSide.BUY ? OrderSide.SELL : OrderSide.BUY,
          quantity: bracketOrder.quantity, 
          accountId: bracketOrder.accountId,
          userId: bracketOrder.userId,
          type: OrderType.LIMIT,
          price: tpPrice, // tpPrice is guaranteed to be a number by _calculateExitPrices
          timeInForce: bracketOrder.originalInputTimeInForce || TimeInForce.DAY,
          metadata: { bracketOrderId: bracketOrder.bracketOrderId, legType: 'TAKE_PROFIT' as const },
        };
        if (bracketOrder.clientOrderId) {
          tpLegPayload.clientOrderId = `${bracketOrder.clientOrderId}-TP`;
        }
        bracketOrder.takeProfitOrder = await this._placeOrderWithOMS(tpLegPayload as Parameters<typeof this._placeOrderWithOMS>[0]);

        const slLegPayload: any = {
          symbol: bracketOrder.symbol,
          side: bracketOrder.originalInputSide === OrderSide.BUY ? OrderSide.SELL : OrderSide.BUY,
          quantity: bracketOrder.quantity,
          accountId: bracketOrder.accountId,
          userId: bracketOrder.userId,
          type: OrderType.STOP,
          stopPrice: slPrice, // slPrice is guaranteed to be a number by _calculateExitPrices
          timeInForce: bracketOrder.originalInputTimeInForce || TimeInForce.DAY,
          metadata: { bracketOrderId: bracketOrder.bracketOrderId, legType: 'STOP_LOSS' as const },
        };
        if (bracketOrder.clientOrderId) {
          slLegPayload.clientOrderId = `${bracketOrder.clientOrderId}-SL`;
        }
        bracketOrder.stopLossOrder = await this._placeOrderWithOMS(slLegPayload as Parameters<typeof this._placeOrderWithOMS>[0]);

        this.logger.info(`Bracket ${bracketOrder.bracketOrderId}: TP (${tpPrice}) and SL (${slPrice}) orders placed.`);
        this.emit('bracketTpslPlaced', bracketOrder);
      } catch (error: any) {
        this.logger.error(`Bracket ${bracketOrder.bracketOrderId}: Failed to place TP/SL orders: ${error.message}`);
        bracketOrder.status = BracketStatus.REJECTED; // Or a state indicating partial failure
        // Attempt to cancel the entry order if possible, or flag for manual intervention
        if (bracketOrder.entryOrder && bracketOrder.entryOrder.status === OrderStatus.FILLED) {
            this.logger.warn(`Bracket ${bracketOrder.bracketOrderId}: Entry was filled, but TP/SL placement failed. Position is open without protection.`);
            // Potentially emit a high-priority alert here
        }
        this.emit('bracketTpslFailed', { bracketOrder, error });
      }
    } else if ((legToUpdate.legType === 'TAKE_PROFIT' || legToUpdate.legType === 'STOP_LOSS') && update.status === OrderStatus.FILLED) {
      bracketOrder.status = BracketStatus.FILLED;
      bracketOrder.triggeredLeg = legToUpdate.legType;
      bracketOrder.completedAt = now;
      this.logger.info(`Bracket ${bracketOrder.bracketOrderId}: ${legToUpdate.legType} order ${orderId} FILLED.`);
      this.emit('bracketLegFilled', { bracketOrder, legType: legToUpdate.legType, fill: update.fill });

      const otherLeg = legToUpdate.legType === 'TAKE_PROFIT' ? bracketOrder.stopLossOrder : bracketOrder.takeProfitOrder;
      if (otherLeg && (otherLeg.status === OrderStatus.ACTIVE || otherLeg.status === OrderStatus.PENDING)) {
        try {
          await this._cancelOrderLegWithOMS(otherLeg.id, bracketOrder.bracketOrderId, `bracket_leg_${update.status.toLowerCase()}`);
          this.logger.info(`Bracket ${bracketOrder.bracketOrderId}: Canceled other leg ${otherLeg.id}.`);
        } catch (error: any) {
          this.logger.error(`Bracket ${bracketOrder.bracketOrderId}: Failed to cancel other leg ${otherLeg.id}: ${error.message}`);
        }
      }
      this._clearSessionTimer(bracketOrder.bracketOrderId);
    } else if (update.status === OrderStatus.CANCELED || update.status === OrderStatus.REJECTED || update.status === OrderStatus.EXPIRED) {
      this.logger.info(`Bracket ${bracketOrder.bracketOrderId}: Leg ${legToUpdate.id} (${legToUpdate.legType}) is ${update.status}.`);
      // If entry order fails/cancels, cancel the whole bracket
      if (legToUpdate.legType === 'ENTRY' && bracketOrder.status !== BracketStatus.FILLED && bracketOrder.status !== BracketStatus.CANCELED) {
        await this.cancelBracketOrder(bracketOrder.bracketOrderId, `entry_leg_${update.status.toLowerCase()}`);
      }
      // If an active TP/SL leg is canceled externally, and the other is still active, we might need to cancel the other too.
      // This depends on desired behavior. For now, if one OCO leg is gone, the protection is partial.
      // The overall bracket status might need a review here.
      // If the bracket itself is still considered active, but one protection leg is gone, it's risky.
      // For simplicity, if a TP/SL leg is cancelled and the bracket is active, we might cancel the other leg too.
      else if ((legToUpdate.legType === 'TAKE_PROFIT' || legToUpdate.legType === 'STOP_LOSS') && bracketOrder.status === BracketStatus.ACTIVE) {
        const otherLeg = legToUpdate.legType === 'TAKE_PROFIT' ? bracketOrder.stopLossOrder : bracketOrder.takeProfitOrder;
        if (otherLeg && (otherLeg.status === OrderStatus.ACTIVE || otherLeg.status === OrderStatus.PENDING)) {
            this.logger.info(`Bracket ${bracketOrder.bracketOrderId}: One OCO leg (${legToUpdate.legType}) is ${update.status}. Cancelling the other leg ${otherLeg.id}.`);
            await this._cancelOrderLegWithOMS(otherLeg.id, bracketOrder.bracketOrderId, `other_leg_of_failed_${legToUpdate.legType}`);
        }
        // Consider if the whole bracket should be marked as CANCELED or a special state like 'MANUAL_INTERVENTION_REQUIRED'
        // For now, if both TP/SL are gone, and entry was filled, it's an unprotected position.
        // If entry was not filled, then cancelling the bracket is appropriate.
        if (bracketOrder.takeProfitOrder?.status !== OrderStatus.ACTIVE && bracketOrder.stopLossOrder?.status !== OrderStatus.ACTIVE) {
            if (bracketOrder.entryOrder?.status !== OrderStatus.FILLED) {
                 await this.cancelBracketOrder(bracketOrder.bracketOrderId, `all_legs_inactive_before_entry_fill`);
            } else {
                this.logger.warn(`Bracket ${bracketOrder.bracketOrderId}: Both TP/SL legs are inactive, but entry was filled. Position is unprotected.`);
                // Potentially change bracket status to something like 'UNPROTECTED'
            }
        }
      }
    }
    this.bracketOrders.set(bracketOrder.bracketOrderId, bracketOrder);
  }

  public async cancelBracketOrder(bracketOrderId: string, reason?: string): Promise<DayTradingBracketOrderState | null> {
    const bracketOrder = this.bracketOrders.get(bracketOrderId);
    if (!bracketOrder) {
      this.logger.warn(`Bracket Order ${bracketOrderId} not found for cancellation.`);
      return null;
    }

    if (bracketOrder.status === BracketStatus.FILLED || bracketOrder.status === BracketStatus.CANCELED || bracketOrder.status === BracketStatus.REJECTED) {
      this.logger.info(`Bracket Order ${bracketOrderId} is already in a terminal state: ${bracketOrder.status}. No action taken.`);
      return bracketOrder;
    }

    this.logger.info(`Cancelling Bracket Order ${bracketOrderId}. Reason: ${reason || 'N/A'}`);
    const cancelPromises: Promise<any>[] = [];
    if (bracketOrder.entryOrder && (bracketOrder.entryOrder.status === OrderStatus.ACTIVE || bracketOrder.entryOrder.status === OrderStatus.PENDING)) {
      cancelPromises.push(this._cancelOrderLegWithOMS(bracketOrder.entryOrder.id, bracketOrderId, reason));
    }
    if (bracketOrder.takeProfitOrder && (bracketOrder.takeProfitOrder.status === OrderStatus.ACTIVE || bracketOrder.takeProfitOrder.status === OrderStatus.PENDING)) {
      cancelPromises.push(this._cancelOrderLegWithOMS(bracketOrder.takeProfitOrder.id, bracketOrderId, reason));
    }
    if (bracketOrder.stopLossOrder && (bracketOrder.stopLossOrder.status === OrderStatus.ACTIVE || bracketOrder.stopLossOrder.status === OrderStatus.PENDING)) {
      cancelPromises.push(this._cancelOrderLegWithOMS(bracketOrder.stopLossOrder.id, bracketOrderId, reason));
    }

    try {
      await Promise.all(cancelPromises);
      bracketOrder.status = BracketStatus.CANCELED;
      bracketOrder.updatedAt = new Date();
      bracketOrder.completedAt = new Date(); // Mark completion time for cancellation too

      if(bracketOrder.entryOrder) bracketOrder.entryOrder.status = OrderStatus.CANCELED;
      if(bracketOrder.takeProfitOrder) bracketOrder.takeProfitOrder.status = OrderStatus.CANCELED;
      if(bracketOrder.stopLossOrder) bracketOrder.stopLossOrder.status = OrderStatus.CANCELED;

      this.logger.info(`Bracket Order ${bracketOrderId} canceled successfully.`);
      this.emit('bracketOrderCanceled', bracketOrder);
    } catch (error: any) {
      this.logger.error(`Bracket Order ${bracketOrderId}: Failed to cancel one or more legs: ${error.message}`);
      bracketOrder.status = BracketStatus.REJECTED; // Or a custom "PARTIALLY_CANCELED" status
      this.emit('bracketCancelFailed', { bracketOrder, error });
    }
    this._clearSessionTimer(bracketOrderId);
    this.bracketOrders.set(bracketOrderId, bracketOrder);
    return bracketOrder;
  }

  public getBracketOrderState(bracketOrderId: string): DayTradingBracketOrderState | undefined {
    return this.bracketOrders.get(bracketOrderId);
  }

  private _findBracketOrderByLegId(legOrderId: string): DayTradingBracketOrderState | undefined {
    for (const bracket of this.bracketOrders.values()) {
      if (bracket.entryOrder?.id === legOrderId || bracket.takeProfitOrder?.id === legOrderId || bracket.stopLossOrder?.id === legOrderId) {
        return bracket;
      }
    }
    return undefined;
  }

  // --- Simulated OMS Interaction (reuse or adapt from ScalpingOCOOrderManager or a shared service) ---
  private async _placeOrderWithOMS(params: BaseOrderParams & { type: OrderType; price?: number; stopPrice?: number; timeInForce: TimeInForce; metadata?: any }): Promise<BracketLeg> {
    await new Promise(resolve => setTimeout(resolve, Math.random() * 10 + 5)); // Simulate 5-15ms latency
    const orderId = uuidv4();
    const now = new Date();
    this.logger.debug(`OMS: Placing ${params.type} order for ${params.symbol}`, params);

    const placedOrder: BracketLeg = {
      id: orderId,
      symbol: params.symbol,
      side: params.side,
      quantity: params.quantity,
      accountId: params.accountId,
      userId: params.userId,
      type: params.type,
      status: OrderStatus.ACTIVE, // Assume active for simulation
      filledQuantity: 0,
      createdAt: now,
      updatedAt: now,
      timeInForce: params.timeInForce,
      metadata: params.metadata,
      bracketOrderId: params.metadata.bracketOrderId,
      legType: params.metadata.legType,
      ...(params.clientOrderId !== undefined && { clientOrderId: params.clientOrderId }),
      ...(params.price !== undefined && { price: params.price }),
      ...(params.stopPrice !== undefined && { stopPrice: params.stopPrice }),
    };
    this.emit('omsOrderPlaced', placedOrder);
    return placedOrder;
  }

  private async _cancelOrderLegWithOMS(orderId: string, bracketOrderId: string, reason?: string): Promise<{ success: boolean; orderId: string }> {
    await new Promise(resolve => setTimeout(resolve, Math.random() * 8 + 2)); // Simulate 2-10ms latency
    this.logger.debug(`OMS: Cancelling order ${orderId} (Bracket: ${bracketOrderId}). Reason: ${reason || 'N/A'}`);
    
    const bracketOrder = this.bracketOrders.get(bracketOrderId);
    if (bracketOrder) {
        let legToCancel: Order | null = null;
        if (bracketOrder.entryOrder?.id === orderId) legToCancel = bracketOrder.entryOrder;
        else if (bracketOrder.takeProfitOrder?.id === orderId) legToCancel = bracketOrder.takeProfitOrder;
        else if (bracketOrder.stopLossOrder?.id === orderId) legToCancel = bracketOrder.stopLossOrder;
        
        if (legToCancel && legToCancel.status !== OrderStatus.FILLED) {
            legToCancel.status = OrderStatus.CANCELED;
            legToCancel.updatedAt = new Date();
        }
    }
    this.emit('omsOrderCanceled', { orderId, bracketOrderId, reason });
    return { success: true, orderId };
  }

  // --- Session Management --- 
  private _isSessionActive(sessions?: TradingSession[]): boolean {
    if (!sessions || sessions.length === 0) return true; // No specific sessions means always active
    // This is a placeholder. Real implementation needs a robust way to check current trading session.
    // For example, by checking current UTC time against session definitions.
    const currentHour = new Date().getUTCHours();
    if (sessions.includes(TradingSession.LONDON) && currentHour >= 7 && currentHour < 16) return true; // Approx London
    if (sessions.includes(TradingSession.NEW_YORK) && currentHour >= 12 && currentHour < 21) return true; // Approx NY
    if (sessions.includes(TradingSession.ASIAN) && (currentHour >= 23 || currentHour < 8)) return true; // Approx Asian
    // Overlaps would need more precise logic
    this.logger.debug(`Session check: currentHourUTC=${currentHour}, requiredSessions=${sessions}. No active session matched.`);
    return false;
  }

  private _setupSessionTimer(bracketOrderId: string, sessions: TradingSession[]): void {
    // This is a simplified timer. A real system might use a cron job or a more robust scheduler
    // to check session boundaries and cancel orders if `cancelOutsideSessions` is true.
    // For this example, we'll just log a warning if it's not currently in session.
    if (!this._isSessionActive(sessions)) {
        this.logger.warn(`Bracket ${bracketOrderId}: Order created/activated outside of specified sessions [${sessions.join(', ')}]. It might not execute as expected or be canceled if cancelOutsideSessions is true.`);
        // If cancelOutsideSessions is true, and it's already outside session, cancel immediately.
        const order = this.bracketOrders.get(bracketOrderId);
        if(order && order.cancelOutsideSessions){
            this.logger.info(`Bracket ${bracketOrderId}: Cancelling immediately as it is outside active sessions and cancelOutsideSessions is true.`);
            this.cancelBracketOrder(bracketOrderId, "created_outside_active_session");
            return;
        }
    }
    
    // More robust timer logic would go here to check periodically or at session end times.
    // For now, this is a placeholder for the concept.
    const checkInterval = 60 * 60 * 1000; // Check every hour
    const timer = setInterval(() => {
        const order = this.bracketOrders.get(bracketOrderId);
        if (order && order.cancelOutsideSessions && 
            (order.status === BracketStatus.PENDING_ENTRY || order.status === BracketStatus.ENTRY_PLACED || order.status === BracketStatus.ACTIVE)) {
            if (!this._isSessionActive(order.activationConditions?.sessions)) {
                this.logger.info(`Bracket ${bracketOrderId}: Active session ended. Cancelling order due to cancelOutsideSessions policy.`);
                this.cancelBracketOrder(bracketOrderId, "session_ended");
            }
        }
    }, checkInterval);
    this.sessionTimers.set(bracketOrderId, timer);
  }

  private _clearSessionTimer(bracketOrderId: string): void {
    if (this.sessionTimers.has(bracketOrderId)) {
      clearInterval(this.sessionTimers.get(bracketOrderId)!);
      this.sessionTimers.delete(bracketOrderId);
      this.logger.debug(`Cleared session timer for bracket ${bracketOrderId}`);
    }
  }

  // Call this method when the application is shutting down to clean up timers
  public dispose(): void {
    this.logger.info('Disposing DayTradingBracketOrderManager, clearing all session timers.');
    for (const bracketId of this.sessionTimers.keys()) {
        this._clearSessionTimer(bracketId);
    }
    this.sessionTimers.clear(); // Ensure the map is empty
  }
}
