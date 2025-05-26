/**
 * Fast Trailing Stop Order
 * Dynamically adjusts the stop price based on market movements to protect profits while allowing room for favorable trends.
 *
 * This module provides:
 * - Trailing stop functionality with configurable trail type (percentage or points) and trail amount.
 * - Real-time market data integration for dynamic stop price updates.
 * - High-speed updates suitable for scalping and fast-moving markets.
 *
 * Expected Benefits:
 * - Automated profit protection.
 * - Maximized gains during strong trends.
 * - Reduced risk of premature stop-outs.
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
} from './ScalpingOCOOrder'; // Reuse common types

// --- Trailing Stop Specific Types ---

export enum TrailType {
  PERCENTAGE = 'PERCENTAGE',
  POINTS = 'POINTS', // Absolute price points/pips
}

export interface FastTrailingStopOrderInput extends BaseOrderParams {
  orderType: OrderType.MARKET | OrderType.LIMIT; // Initial entry order type
  entryPrice?: number; // Required for LIMIT entry
  trailType: TrailType;
  trailAmount: number; // Percentage (e.g., 0.5 for 0.5%) or points (e.g., 10 for 10 pips)
  initialStopPrice?: number; // Optional: if not provided, can be calculated from entry or first tick
  activateImmediately?: boolean; // If false, waits for a certain profit before activating trail
  activationThreshold?: number; // Profit in points/percentage before trail starts
  timeInForce?: TimeInForce;
}

export enum TrailingStopStatus {
  PENDING_ENTRY = 'PENDING_ENTRY',
  ENTRY_PLACED = 'ENTRY_PLACED',
  MONITORING_FOR_ACTIVATION = 'MONITORING_FOR_ACTIVATION', // Entry filled, waiting for activation threshold
  TRAILING_ACTIVE = 'TRAILING_ACTIVE', // Stop is actively trailing
  STOP_TRIGGERED = 'STOP_TRIGGERED', // Stop order sent to OMS
  FILLED = 'FILLED', // Position closed by stop
  CANCELED = 'CANCELED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED',
}

export interface FastTrailingStopOrderState {
  trailingStopOrderId: string;
  clientOrderId?: string | undefined;
  accountId: string;
  userId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  status: TrailingStopStatus;
  entryOrder: Order | null;
  stopOrder: Order | null; // The actual stop order placed with OMS
  createdAt: Date;
  updatedAt: Date;
  entryFilledAt?: Date | undefined;
  stopTriggeredAt?: Date | undefined;
  completedAt?: Date | undefined;
  
  // Trailing specific state
  trailType: TrailType;
  trailAmount: number;
  currentStopPrice: number | null; // The dynamically calculated stop price
  lastMarketPrice: number | null; // Last known market price used for trailing
  highestPriceSinceActive?: number | undefined; // For BUY orders, highest price reached
  lowestPriceSinceActive?: number | undefined;  // For SELL orders, lowest price reached
  activationThreshold?: number | undefined;
  isTrailActive: boolean; // Flag indicating if the trailing logic is active

  originalInput: FastTrailingStopOrderInput; // Store original input for reference
}

// --- Market Data Event (Simplified) ---
export interface MarketTickEvent {
  symbol: string;
  timestamp: Date;
  bid: number;
  ask: number;
  last?: number; // Last traded price, if available
}

/**
 * Manages Fast Trailing Stop Orders.
 * This class handles the lifecycle of a trailing stop order: entry, monitoring, and dynamic stop adjustment.
 */
export class FastTrailingStopOrderManager extends EventEmitter {
  private trailingOrders: Map<string, FastTrailingStopOrderState> = new Map();
  private logger: Logger;
  // private marketDataSubscriber: MarketDataSubscriber; // Hypothetical market data client
  // private omsClient: OrderManagementSystem; // Hypothetical OMS client

  constructor(logger: Logger /*, marketDataSubscriber: MarketDataSubscriber, omsClient: OrderManagementSystem */) {
    super();
    this.logger = logger;
    // this.marketDataSubscriber = marketDataSubscriber;
    // this.omsClient = omsClient;
    this.logger.info('FastTrailingStopOrderManager initialized.');
    // this._subscribeToMarketData(); // If applicable
  }

  private _validateInput(input: FastTrailingStopOrderInput): void {
    if (!input.symbol || !input.side || input.quantity <= 0 || !input.accountId || !input.userId) {
      throw new Error('Missing required fields (symbol, side, quantity, accountId, userId).');
    }
    if (input.orderType === OrderType.LIMIT && typeof input.entryPrice !== 'number') {
      throw new Error('Entry price is required for LIMIT entry orders.');
    }
    if (!input.trailType || (input.trailType !== TrailType.PERCENTAGE && input.trailType !== TrailType.POINTS)) {
        throw new Error('Invalid trailType. Must be PERCENTAGE or POINTS.');
    }
    if (typeof input.trailAmount !== 'number' || input.trailAmount <= 0) {
      throw new Error('Trail amount must be a positive number.');
    }
    if (input.activationThreshold && (typeof input.activationThreshold !== 'number' || input.activationThreshold <=0)) {
        throw new Error('Activation threshold must be a positive number if provided.');
    }
    if (input.initialStopPrice && typeof input.initialStopPrice !== 'number') {
        throw new Error('Initial stop price must be a number if provided.');
    }
  }

  public async createTrailingStopOrder(input: FastTrailingStopOrderInput): Promise<FastTrailingStopOrderState> {
    this._validateInput(input);
    const orderId = uuidv4();
    const now = new Date();

    const initialState: FastTrailingStopOrderState = {
      trailingStopOrderId: orderId,
      clientOrderId: input.clientOrderId,
      accountId: input.accountId,
      userId: input.userId,
      symbol: input.symbol,
      side: input.side,
      quantity: input.quantity,
      status: TrailingStopStatus.PENDING_ENTRY,
      entryOrder: null,
      stopOrder: null,
      createdAt: now,
      updatedAt: now,
      trailType: input.trailType,
      trailAmount: input.trailAmount,
      currentStopPrice: input.initialStopPrice ?? null,
      lastMarketPrice: null,
      activationThreshold: input.activationThreshold,
      isTrailActive: input.activateImmediately ?? !input.activationThreshold, // Activate if no threshold or explicitly told
      originalInput: { ...input }, // Shallow copy of input
    };

    this.trailingOrders.set(orderId, initialState);
    this.logger.info(`Trailing Stop Order ${orderId} PENDING_ENTRY.`, initialState);
    this.emit('trailingStopOrderCreated', initialState);

    // Place the initial entry order
    try {
      const entryOrderPayload: Omit<Order, 'id' | 'status' | 'createdAt' | 'updatedAt' | 'filledQuantity' | 'averageFillPrice'> & { type: OrderType.MARKET | OrderType.LIMIT, metadata: any } = {
        clientOrderId: input.clientOrderId ? `${input.clientOrderId}-ENTRY` : undefined,
        symbol: input.symbol,
        side: input.side,
        quantity: input.quantity,
        accountId: input.accountId,
        userId: input.userId,
        type: input.orderType,
        timeInForce: input.timeInForce || TimeInForce.GTC, // Good-Til-Canceled is common for entries with trailing stops
        metadata: { trailingStopOrderId: orderId, legType: 'ENTRY' },
        price: input.orderType === OrderType.LIMIT ? input.entryPrice : undefined,
      };
      
      const entryOrderLeg = await this._placeOrderWithOMS(entryOrderPayload);
      initialState.entryOrder = entryOrderLeg;
      initialState.status = TrailingStopStatus.ENTRY_PLACED;
      initialState.updatedAt = new Date();
      this.logger.info(`Trailing Stop ${orderId}: Entry order ${entryOrderLeg.id} placed. Status: ${initialState.status}.`);
      this.emit('trailingStopEntryPlaced', initialState);

      if (initialState.currentStopPrice && !initialState.isTrailActive) {
         this.logger.info(`Trailing Stop ${orderId}: Initial stop price ${initialState.currentStopPrice} set, awaiting activation or entry fill.`);
      }

    } catch (error: any) {
      this.logger.error(`Trailing Stop ${orderId}: Failed to place entry order: ${error.message}`, { input });
      initialState.status = TrailingStopStatus.REJECTED;
      initialState.updatedAt = new Date();
      this.trailingOrders.set(orderId, initialState);
      this.emit('trailingStopCreationFailed', initialState);
      throw error;
    }
    return initialState;
  }

  public async handleEntryOrderFill(trailingStopOrderId: string, fillEvent: OrderFillEvent): Promise<void> {
    const orderState = this.trailingOrders.get(trailingStopOrderId);
    if (!orderState || !orderState.entryOrder || orderState.entryOrder.id !== fillEvent.orderId) {
      this.logger.warn(`Trailing Stop: Received fill for unknown or mismatched entry order ${fillEvent.orderId}.`);
      return;
    }

    orderState.entryOrder.status = OrderStatus.FILLED;
    orderState.entryOrder.filledQuantity = (orderState.entryOrder.filledQuantity || 0) + fillEvent.quantity; // Changed from fillEvent.filledQuantity
    orderState.entryOrder.averageFillPrice = fillEvent.price; // Changed from fillEvent.averageFillPrice
    orderState.entryFilledAt = new Date();
    orderState.updatedAt = new Date();

    this.logger.info(`Trailing Stop ${trailingStopOrderId}: Entry order ${fillEvent.orderId} FILLED at ${fillEvent.price}.`); // Changed

    orderState.lastMarketPrice = fillEvent.price; // Changed
    if (orderState.side === OrderSide.BUY) {
        orderState.highestPriceSinceActive = fillEvent.price; // Changed
    } else {
        orderState.lowestPriceSinceActive = fillEvent.price; // Changed
    }
    
    if (orderState.currentStopPrice === null && fillEvent.price !== undefined) { // Changed
        if (orderState.side === OrderSide.BUY) {
            orderState.currentStopPrice = this._calculateStopPrice(fillEvent.price, orderState.trailAmount, orderState.trailType, OrderSide.BUY, true); // Changed
        } else { 
            orderState.currentStopPrice = this._calculateStopPrice(fillEvent.price, orderState.trailAmount, orderState.trailType, OrderSide.SELL, true); // Changed
        }
        this.logger.info(`Trailing Stop ${trailingStopOrderId}: Initial stop price calculated to ${orderState.currentStopPrice} based on entry.`);
    }

    if (orderState.isTrailActive) {
      orderState.status = TrailingStopStatus.TRAILING_ACTIVE;
      this.logger.info(`Trailing Stop ${trailingStopOrderId}: Trail is now ACTIVE. Current stop: ${orderState.currentStopPrice}.`);
      await this._updateOMSStopOrder(orderState); 
    } else {
      orderState.status = TrailingStopStatus.MONITORING_FOR_ACTIVATION;
      this.logger.info(`Trailing Stop ${trailingStopOrderId}: MONITORING FOR ACTIVATION. Current stop: ${orderState.currentStopPrice}. Threshold: ${orderState.activationThreshold}`);
    }
    this.emit('trailingStopEntryFilled', orderState);
  }

  public onMarketTick(tick: MarketTickEvent): void {
    this.trailingOrders.forEach(async orderState => { // Added async here
      if (orderState.symbol !== tick.symbol || 
         (orderState.status !== TrailingStopStatus.TRAILING_ACTIVE && orderState.status !== TrailingStopStatus.MONITORING_FOR_ACTIVATION) ||
         !orderState.entryOrder || orderState.entryOrder.status !== OrderStatus.FILLED) {
        return; 
      }

      const marketPrice = orderState.side === OrderSide.BUY ? tick.bid : tick.ask; 
      if (marketPrice === null || marketPrice === undefined) return;

      orderState.lastMarketPrice = marketPrice;

      if (orderState.status === TrailingStopStatus.MONITORING_FOR_ACTIVATION && orderState.activationThreshold && orderState.entryOrder.averageFillPrice) {
          let profit = 0;
          if (orderState.side === OrderSide.BUY) {
              profit = marketPrice - orderState.entryOrder.averageFillPrice;
          } else {
              profit = orderState.entryOrder.averageFillPrice - marketPrice;
          }
          if (profit >= orderState.activationThreshold) {
              orderState.isTrailActive = true;
              orderState.status = TrailingStopStatus.TRAILING_ACTIVE;
              this.logger.info(`Trailing Stop ${orderState.trailingStopOrderId}: Activation threshold met. Trail is now ACTIVE. Current stop: ${orderState.currentStopPrice}.`);
              this.emit('trailingStopActivated', orderState);
              if (orderState.side === OrderSide.BUY) orderState.highestPriceSinceActive = marketPrice;
              else orderState.lowestPriceSinceActive = marketPrice;
              await this._updateOMSStopOrder(orderState); // Added await
          } else {
              return; 
          }
      }
      
      if (!orderState.isTrailActive || orderState.status !== TrailingStopStatus.TRAILING_ACTIVE) return;

      let newStopPrice = orderState.currentStopPrice;
      let shouldUpdateStop = false;

      if (orderState.side === OrderSide.BUY) {
        if (orderState.highestPriceSinceActive === undefined || marketPrice > orderState.highestPriceSinceActive) {
          orderState.highestPriceSinceActive = marketPrice;
        }
        const calculatedStop = this._calculateStopPrice(orderState.highestPriceSinceActive!, orderState.trailAmount, orderState.trailType, OrderSide.BUY);
        if (orderState.currentStopPrice === null || calculatedStop > orderState.currentStopPrice) {
          newStopPrice = calculatedStop;
          shouldUpdateStop = true;
        }
      } else { 
        if (orderState.lowestPriceSinceActive === undefined || marketPrice < orderState.lowestPriceSinceActive) {
          orderState.lowestPriceSinceActive = marketPrice;
        }
        const calculatedStop = this._calculateStopPrice(orderState.lowestPriceSinceActive!, orderState.trailAmount, orderState.trailType, OrderSide.SELL);
        if (orderState.currentStopPrice === null || calculatedStop < orderState.currentStopPrice) {
          newStopPrice = calculatedStop;
          shouldUpdateStop = true;
        }
      }

      if (shouldUpdateStop && newStopPrice !== null) {
        orderState.currentStopPrice = newStopPrice;
        orderState.updatedAt = new Date();
        this.logger.debug(`Trailing Stop ${orderState.trailingStopOrderId}: Adjusted stop to ${newStopPrice} based on market price ${marketPrice}.`);
        this.emit('trailingStopAdjusted', orderState);
        await this._updateOMSStopOrder(orderState); // Added await
      }

      if (orderState.currentStopPrice !== null) { // Added parentheses
        const stopTriggered = (orderState.side === OrderSide.BUY && marketPrice <= orderState.currentStopPrice) ||
                              (orderState.side === OrderSide.SELL && marketPrice >= orderState.currentStopPrice);
        if (stopTriggered) {
          this.logger.info(`Trailing Stop ${orderState.trailingStopOrderId}: STOP TRIGGERED at market price ${marketPrice} (stop: ${orderState.currentStopPrice}).`);
          orderState.status = TrailingStopStatus.STOP_TRIGGERED;
          orderState.stopTriggeredAt = new Date();
          this.emit('trailingStopTriggered', orderState);
        }
      }
    });
  }

  private _calculateStopPrice(basePrice: number, trailAmount: number, trailType: TrailType, side: OrderSide, isInitialCalc: boolean = false): number {
    let stopPrice: number;
    if (trailType === TrailType.PERCENTAGE) {
      const percentage = trailAmount / 100;
      if (side === OrderSide.BUY) {
        stopPrice = basePrice * (1 - percentage);
      } else { 
        stopPrice = basePrice * (1 + percentage);
      }
    } else { 
      if (side === OrderSide.BUY) {
        stopPrice = basePrice - trailAmount;
      } else { 
        stopPrice = basePrice + trailAmount;
      }
    }
    return parseFloat(stopPrice.toFixed(5)); 
  }

  public async handleStopOrderFill(trailingStopOrderId: string, fillEvent: OrderFillEvent): Promise<void> {
    const orderState = this.trailingOrders.get(trailingStopOrderId);
    if (!orderState || !orderState.stopOrder || orderState.stopOrder.id !== fillEvent.orderId) {
      this.logger.warn(`Trailing Stop: Received fill for unknown or mismatched stop order ${fillEvent.orderId}.`);
      return;
    }
    if (orderState.status === TrailingStopStatus.FILLED) return; 

    orderState.stopOrder.status = OrderStatus.FILLED;
    orderState.stopOrder.filledQuantity = (orderState.stopOrder.filledQuantity || 0) + fillEvent.quantity; // Changed
    orderState.stopOrder.averageFillPrice = fillEvent.price; // Changed
    
    orderState.status = TrailingStopStatus.FILLED;
    orderState.completedAt = new Date();
    orderState.updatedAt = new Date();

    this.logger.info(`Trailing Stop ${trailingStopOrderId}: Stop order ${fillEvent.orderId} FILLED at ${fillEvent.price}. Position closed.`); // Changed
    this.emit('trailingStopFilled', orderState);
  }
  
  public async handleOrderUpdate(orderId: string, update: { status: OrderStatus; fill?: OrderFillEvent }): Promise<void> {
    let foundOrderState: FastTrailingStopOrderState | null = null;
    let legType: 'ENTRY' | 'STOP' | null = null;

    for (const state of this.trailingOrders.values()) {
        if (state.entryOrder?.id === orderId) {
            foundOrderState = state;
            legType = 'ENTRY';
            break;
        }
        if (state.stopOrder?.id === orderId) {
            foundOrderState = state;
            legType = 'STOP';
            break;
        }
    }

    if (!foundOrderState || !legType) {
        this.logger.warn(`TrailingStop: Received update for order ${orderId} not part of a known Trailing Stop order.`);
        return;
    }

    const now = new Date();
    foundOrderState.updatedAt = now;

    if (legType === 'ENTRY') {
        if (!foundOrderState.entryOrder) return;
        foundOrderState.entryOrder.status = update.status;
        foundOrderState.entryOrder.updatedAt = now;
        if (update.fill) {
            foundOrderState.entryOrder.filledQuantity = (foundOrderState.entryOrder.filledQuantity || 0) + update.fill.quantity; // Changed
            foundOrderState.entryOrder.averageFillPrice = update.fill.price; // Changed
        }

        if (update.status === OrderStatus.FILLED && update.fill) {
            await this.handleEntryOrderFill(foundOrderState.trailingStopOrderId, update.fill);
        } else if (update.status === OrderStatus.CANCELED || update.status === OrderStatus.REJECTED || update.status === OrderStatus.EXPIRED) {
            this.logger.info(`Trailing Stop ${foundOrderState.trailingStopOrderId}: Entry order ${orderId} is ${update.status}. Cancelling Trailing Stop.`);
            await this.cancelTrailingStopOrder(foundOrderState.trailingStopOrderId, `entry_leg_${update.status.toLowerCase()}`);
        }
    } else if (legType === 'STOP') {
        if (!foundOrderState.stopOrder) return;
        foundOrderState.stopOrder.status = update.status;
        foundOrderState.stopOrder.updatedAt = now;
        if (update.fill) {
            foundOrderState.stopOrder.filledQuantity = (foundOrderState.stopOrder.filledQuantity || 0) + update.fill.quantity; // Changed
            foundOrderState.stopOrder.averageFillPrice = update.fill.price; // Changed
        }

        if (update.status === OrderStatus.FILLED && update.fill) {
            await this.handleStopOrderFill(foundOrderState.trailingStopOrderId, update.fill);
        } else if (update.status === OrderStatus.CANCELED || update.status === OrderStatus.REJECTED || update.status === OrderStatus.EXPIRED) {
            this.logger.warn(`Trailing Stop ${foundOrderState.trailingStopOrderId}: Stop order ${orderId} is ${update.status}. Position may be unprotected.`);
            foundOrderState.status = TrailingStopStatus.CANCELED; 
            foundOrderState.completedAt = now;
            this.emit('trailingStopFailed', foundOrderState); 
        }
    }
  }

  public async cancelTrailingStopOrder(trailingStopOrderId: string, reason?: string): Promise<FastTrailingStopOrderState | null> {
    const orderState = this.trailingOrders.get(trailingStopOrderId);
    if (!orderState) {
      this.logger.warn(`Trailing Stop Order ${trailingStopOrderId} not found for cancellation.`);
      return null;
    }

    if (orderState.status === TrailingStopStatus.FILLED || orderState.status === TrailingStopStatus.CANCELED || orderState.status === TrailingStopStatus.REJECTED) {
      this.logger.info(`Trailing Stop Order ${trailingStopOrderId} is already in a terminal state: ${orderState.status}.`);
      return orderState;
    }

    this.logger.info(`Cancelling Trailing Stop Order ${trailingStopOrderId}. Reason: ${reason || 'N/A'}`);
    
    try {
      if (orderState.entryOrder && (orderState.entryOrder.status === OrderStatus.ACTIVE || orderState.entryOrder.status === OrderStatus.PENDING)) {
        await this._cancelOrderLegWithOMS(orderState.entryOrder.id, trailingStopOrderId, reason);
        if(orderState.entryOrder) orderState.entryOrder.status = OrderStatus.CANCELED;
      }
      if (orderState.stopOrder && (orderState.stopOrder.status === OrderStatus.ACTIVE || orderState.stopOrder.status === OrderStatus.PENDING)) {
        await this._cancelOrderLegWithOMS(orderState.stopOrder.id, trailingStopOrderId, reason);
         if(orderState.stopOrder) orderState.stopOrder.status = OrderStatus.CANCELED;
      }

      orderState.status = TrailingStopStatus.CANCELED;
      orderState.updatedAt = new Date();
      orderState.completedAt = new Date();
      this.logger.info(`Trailing Stop Order ${trailingStopOrderId} canceled successfully.`);
      this.emit('trailingStopOrderCanceled', orderState);
    } catch (error: any) {
      this.logger.error(`Trailing Stop Order ${trailingStopOrderId}: Failed to cancel one or more legs: ${error.message}`);
      orderState.status = TrailingStopStatus.REJECTED; 
      this.emit('trailingStopCancelFailed', { orderState, error });
    }
    this.trailingOrders.set(trailingStopOrderId, orderState);
    return orderState;
  }

  public getTrailingStopOrderState(trailingStopOrderId: string): FastTrailingStopOrderState | undefined {
    return this.trailingOrders.get(trailingStopOrderId);
  }

  private async _placeOrderWithOMS(params: Omit<Order, 'id' | 'status' | 'createdAt' | 'updatedAt' | 'filledQuantity' | 'averageFillPrice'> & { type: OrderType, metadata: any }): Promise<Order> {
    await new Promise(resolve => setTimeout(resolve, 5)); 
    const orderId = uuidv4();
    const now = new Date();
    this.logger.debug(`OMS: Placing ${params.type} order for ${params.symbol}`, params);
    
    const placedOrder: Order = {
      id: orderId,
      clientOrderId: params.clientOrderId,
      symbol: params.symbol,
      side: params.side,
      quantity: params.quantity,
      accountId: params.accountId,
      userId: params.userId,
      type: params.type,
      status: OrderStatus.ACTIVE, 
      filledQuantity: 0,
      averageFillPrice: undefined,
      createdAt: now,
      updatedAt: now,
      timeInForce: params.timeInForce!,
      metadata: params.metadata,
      price: params.price,
      stopPrice: params.stopPrice,
    };
    this.emit('omsOrderPlaced', placedOrder); 
    return placedOrder;
  }

  private async _updateOMSStopOrder(orderState: FastTrailingStopOrderState): Promise<void> {
    if (!orderState.currentStopPrice || !orderState.entryOrder || orderState.entryOrder.status !== OrderStatus.FILLED) {
      this.logger.debug(`Trailing Stop ${orderState.trailingStopOrderId}: Conditions not met to place/update stop order (no currentStopPrice or entry not filled).`);
      return;
    }

    const stopOrderPayload: Omit<Order, 'id' | 'status' | 'createdAt' | 'updatedAt' | 'filledQuantity' | 'averageFillPrice'> & { type: OrderType.STOP, metadata: any, stopPrice: number } = {
      clientOrderId: orderState.clientOrderId ? `${orderState.clientOrderId}-STOP` : undefined,
      symbol: orderState.symbol,
      side: orderState.side === OrderSide.BUY ? OrderSide.SELL : OrderSide.BUY, 
      quantity: orderState.quantity, 
      accountId: orderState.accountId,
      userId: orderState.userId,
      type: OrderType.STOP, 
      stopPrice: orderState.currentStopPrice,
      timeInForce: orderState.originalInput.timeInForce || TimeInForce.GTC,
      metadata: { trailingStopOrderId: orderState.trailingStopOrderId, legType: 'STOP' },
    };

    if (orderState.stopOrder && (orderState.stopOrder.status === OrderStatus.ACTIVE || orderState.stopOrder.status === OrderStatus.PENDING)) {
      try {
        await this._cancelOrderLegWithOMS(orderState.stopOrder.id, orderState.trailingStopOrderId, "trailing_stop_adjustment"); 
        const newStopOrder = await this._placeOrderWithOMS(stopOrderPayload); 
        orderState.stopOrder = newStopOrder;
        this.logger.info(`Trailing Stop ${orderState.trailingStopOrderId}: Modified existing stop order ${orderState.stopOrder.id} to new stopPrice ${orderState.currentStopPrice}.`);
        this.emit('omsOrderModified', orderState.stopOrder);
      } catch (error: any) {
        this.logger.error(`Trailing Stop ${orderState.trailingStopOrderId}: Failed to modify stop order ${orderState.stopOrder?.id}: ${error.message}`);
      }
    } else {
      try {
        const newStopOrder = await this._placeOrderWithOMS(stopOrderPayload);
        orderState.stopOrder = newStopOrder;
        this.logger.info(`Trailing Stop ${orderState.trailingStopOrderId}: Placed new stop order ${newStopOrder.id} at ${orderState.currentStopPrice}.`);
         this.emit('omsOrderPlaced', newStopOrder);
      } catch (error: any) {
        this.logger.error(`Trailing Stop ${orderState.trailingStopOrderId}: Failed to place new stop order: ${error.message}`);
      }
    }
  }

  private async _cancelOrderLegWithOMS(orderId: string, trailingStopOrderId: string, reason?: string): Promise<{ success: boolean; orderId: string }> {
    await new Promise(resolve => setTimeout(resolve, 5)); 
    this.logger.debug(`OMS: Cancelling order ${orderId} (TrailingStop: ${trailingStopOrderId}). Reason: ${reason || 'N/A'}`);
    this.emit('omsOrderCanceled', { orderId, trailingStopOrderId, reason });
    return { success: true, orderId };
  }
}
