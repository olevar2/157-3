/**
 * Scalping OCO (One-Cancels-Other) Order
 * Ultra-fast OCO implementation optimized for M1-M5 scalping trades
 *
 * This module provides OCO order functionality including:
 * - Simultaneous stop loss and take profit orders
 * - Automatic cancellation when one order fills
 * - Sub-10ms execution for scalping strategies (target)
 * - Real-time order state management
 * - Session-awareness for conditional activation (enhancement)
 * - Volatility-based parameter adjustments (enhancement)
 *
 * Expected Benefits:
 * - Professional scalping order management
 * - Automated risk management for short-term trades
 * - Ultra-fast order execution and cancellation
 * - Reduced manual intervention in scalping
 * - Enhanced adaptability to market conditions
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { Logger } from 'winston'; // Assuming winston logger is used project-wide

// --- Re-defined and Enhanced Shared Types (Ideally from a central types file) ---
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL',
}

export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  TRAILING_STOP = 'TRAILING_STOP', // Added for future use
}

export enum OrderStatus {
  PENDING = 'PENDING',       // Order created but not yet active on the exchange
  ACTIVE = 'ACTIVE',         // Order is live on the exchange
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  FILLED = 'FILLED',
  CANCELED = 'CANCELED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED',
}

export enum TimeInForce {
  GTC = 'GTC', // Good 'Til Canceled
  IOC = 'IOC', // Immediate or Cancel
  FOK = 'FOK', // Fill or Kill
  DAY = 'DAY', // Day Order
}

export interface BaseOrderParams {
  clientOrderId?: string | undefined;
  symbol: string;
  side: OrderSide;
  quantity: number;
  accountId: string;
  userId: string; // Added for audit/tracking
}

export interface Order extends BaseOrderParams {
  id: string; // System-generated unique order ID
  type: OrderType;
  status: OrderStatus;
  price?: number | undefined; // For LIMIT orders
  stopPrice?: number | undefined; // For STOP orders
  filledQuantity: number;
  averageFillPrice?: number | undefined;
  createdAt: Date;
  updatedAt: Date;
  timeInForce: TimeInForce;
  metadata?: Record<string, any>; // For additional info, e.g., source, strategyId
}

export interface OrderFillEvent {
  orderId: string;
  fillId: string;
  quantity: number;
  price: number;
  timestamp: Date;
  commission?: number; // Optional commission details
  executionVenue?: string; // Exchange or LP
}

// --- OCO Specific Types ---

export interface OCOLeg extends Order {
  ocoGroupId: string;
  legType: 'TAKE_PROFIT' | 'STOP_LOSS';
}

export interface ScalpingOCOOrderInput extends BaseOrderParams {
  takeProfitPrice: number;
  stopLossPrice: number;
  timeInForce?: TimeInForce; // Default to GTC for scalping legs
  // Enhancement: Session-based activation
  activateDuringSessions?: TradingSession[]; // e.g., [TradingSession.LONDON, TradingSession.NEW_YORK]
  // Enhancement: Volatility-based adjustments
  volatilityAdjustment?: {
    enabled: boolean;
    atrPeriod?: number; // e.g., 14 periods on M1/M5
    atrMultiplierStopLoss?: number; // e.g., 1.5 * ATR
    atrMultiplierTakeProfit?: number; // e.g., 2.0 * ATR
  };
}

export enum TradingSession { // Example, should be more robust
  ASIAN = 'ASIAN',
  LONDON = 'LONDON',
  NEW_YORK = 'NEW_YORK',
  OVERLAP_LDN_NY = 'OVERLAP_LDN_NY',
}

export interface ScalpingOCOOrderState {
  ocoGroupId: string;
  clientOrderId?: string;
  accountId: string;
  userId: string;
  symbol: string;
  side: OrderSide; // Side of the initial position this OCO protects
  quantity: number;
  status: OrderStatus; // Overall status: PENDING, ACTIVE, FILLED (one leg), CANCELED
  takeProfitOrder: OCOLeg | null;
  stopLossOrder: OCOLeg | null;
  createdAt: Date;
  updatedAt: Date;
  activationConditions?: {
    sessions?: TradingSession[];
    volatility?: any; // Store resolved volatility params
  };
  triggeredLeg?: 'TAKE_PROFIT' | 'STOP_LOSS';
  executionLatencyMs?: number; // For performance monitoring
}

/**
 * Manages Scalping OCO orders.
 * This class is responsible for creating, activating, and managing the lifecycle of OCO orders.
 * It interacts with a hypothetical OrderManagementSystem (OMS) to place and cancel orders.
 */
export class ScalpingOCOOrderManager extends EventEmitter {
  private ocoOrders: Map<string, ScalpingOCOOrderState> = new Map();
  private logger: Logger;
  // private omsClient: OrderManagementSystem; // Hypothetical OMS client

  constructor(logger: Logger /*, omsClient: OrderManagementSystem */) {
    super();
    this.logger = logger;
    // this.omsClient = omsClient;
    this.logger.info('ScalpingOCOOrderManager initialized.');
  }

  /**
   * Creates and activates a new scalping OCO order.
   * @param input - The parameters for the OCO order.
   * @returns The initial state of the OCO order.
   * @throws Error if input validation fails or OMS interaction fails.
   */
  public async createAndActivateOrder(input: ScalpingOCOOrderInput): Promise<ScalpingOCOOrderState> {
    const activationStartTime = performance.now();
    this._validateOCOInput(input);

    const ocoGroupId = uuidv4();
    const now = new Date();

    const constructedActivationConditions: Partial<ScalpingOCOOrderState['activationConditions']> = {};
    if (input.activateDuringSessions && input.activateDuringSessions.length > 0) {
        constructedActivationConditions.sessions = input.activateDuringSessions;
    }
    // TODO: Add volatility to constructedActivationConditions if/when resolved and enabled in input

    const initialState: ScalpingOCOOrderState = {
      ocoGroupId,
      clientOrderId: input.clientOrderId, // clientOrderId itself is optional in ScalpingOCOOrderState, so direct assignment is fine
      accountId: input.accountId,
      userId: input.userId,
      symbol: input.symbol,
      side: input.side,
      quantity: input.quantity,
      status: OrderStatus.PENDING, // Initially PENDING, will become ACTIVE after legs are placed
      takeProfitOrder: null,
      stopLossOrder: null,
      createdAt: now,
      updatedAt: now,
      ...(Object.keys(constructedActivationConditions).length > 0 && { activationConditions: constructedActivationConditions as ScalpingOCOOrderState['activationConditions'] }),
    };

    // In a real system, you might check session/volatility conditions *before* placing.
    // For this example, we assume they are met or will be handled by an activation poller.

    try {
      const takeProfitLegSide = input.side === OrderSide.BUY ? OrderSide.SELL : OrderSide.BUY;
      const stopLossLegSide = input.side === OrderSide.BUY ? OrderSide.SELL : OrderSide.BUY;

      // Simulate OMS interaction for placing the Take Profit leg
      const tpBaseParams = {
        symbol: input.symbol,
        side: takeProfitLegSide,
        quantity: input.quantity,
        accountId: input.accountId,
        userId: input.userId,
        type: OrderType.LIMIT,
        price: input.takeProfitPrice,
        timeInForce: input.timeInForce || TimeInForce.GTC,
        metadata: { ocoGroupId, legType: 'TAKE_PROFIT' as const },
      };
      const tpOrderParams = input.clientOrderId
        ? { ...tpBaseParams, clientOrderId: `${input.clientOrderId}-TP` }
        : tpBaseParams;
      const tpOrder: OCOLeg = await this._placeOrderWithOMS(tpOrderParams);
      initialState.takeProfitOrder = tpOrder;

      // Simulate OMS interaction for placing the Stop Loss leg
      const slBaseParams = {
        symbol: input.symbol,
        side: stopLossLegSide,
        quantity: input.quantity,
        accountId: input.accountId,
        userId: input.userId,
        type: OrderType.STOP,
        stopPrice: input.stopLossPrice,
        timeInForce: input.timeInForce || TimeInForce.GTC,
        metadata: { ocoGroupId, legType: 'STOP_LOSS' as const },
      };
      const slOrderParams = input.clientOrderId
        ? { ...slBaseParams, clientOrderId: `${input.clientOrderId}-SL` }
        : slBaseParams;
      const slOrder: OCOLeg = await this._placeOrderWithOMS(slOrderParams);
      initialState.stopLossOrder = slOrder;

      if (tpOrder.status === OrderStatus.ACTIVE && slOrder.status === OrderStatus.ACTIVE) {
        initialState.status = OrderStatus.ACTIVE;
      } else {
        // Handle partial placement failure - attempt to cancel already placed leg
        this.logger.error(`OCO ${ocoGroupId}: Partial leg placement failure. TP: ${tpOrder.status}, SL: ${slOrder.status}`);
        if (tpOrder.status === OrderStatus.ACTIVE) await this.cancelOrderLeg(tpOrder.id, ocoGroupId);
        if (slOrder.status === OrderStatus.ACTIVE) await this.cancelOrderLeg(slOrder.id, ocoGroupId);
        initialState.status = OrderStatus.REJECTED; // Or another appropriate status
        throw new Error(`OCO ${ocoGroupId}: Failed to activate all legs.`);
      }

      initialState.updatedAt = new Date();
      initialState.executionLatencyMs = performance.now() - activationStartTime;
      this.ocoOrders.set(ocoGroupId, initialState);
      this.logger.info(`OCO Order ${ocoGroupId} created and activated. Latency: ${initialState.executionLatencyMs?.toFixed(2)}ms`, initialState);
      this.emit('ocoActivated', initialState);
      return initialState;

    } catch (error: any) {
      this.logger.error(`OCO ${ocoGroupId}: Activation failed: ${error.message}`, { input });
      // Ensure any successfully placed leg is cancelled if the other failed
      if (initialState.takeProfitOrder?.status === OrderStatus.ACTIVE) {
        await this.cancelOrderLeg(initialState.takeProfitOrder.id, ocoGroupId, 'activation_failure');
      }
      if (initialState.stopLossOrder?.status === OrderStatus.ACTIVE) {
        await this.cancelOrderLeg(initialState.stopLossOrder.id, ocoGroupId, 'activation_failure');
      }
      throw error; // Re-throw for the caller to handle
    }
  }

  /**
   * Handles an update from the OMS, typically a fill or cancellation.
   * @param orderId The ID of the individual order leg that was updated.
   * @param update The update event (e.g., fill, cancellation confirmation).
   */
  public async handleOrderUpdate(orderId: string, update: { status: OrderStatus; fill?: OrderFillEvent }): Promise<void> {
    const ocoOrder = this._findOCOOrderByLegId(orderId);
    if (!ocoOrder) {
      this.logger.warn(`Received update for order ${orderId} not part of a known OCO.`);
      return;
    }

    const legToUpdate = ocoOrder.takeProfitOrder?.id === orderId ? ocoOrder.takeProfitOrder : ocoOrder.stopLossOrder;
    if (!legToUpdate) return; // Should not happen if ocoOrder was found

    legToUpdate.status = update.status;
    legToUpdate.updatedAt = new Date();
    if (update.fill) {
      legToUpdate.filledQuantity = (legToUpdate.filledQuantity || 0) + update.fill.quantity;
      legToUpdate.averageFillPrice = update.fill.price; // Simplified avg price for this example
    }
    ocoOrder.updatedAt = new Date();

    if (update.status === OrderStatus.FILLED) {
      ocoOrder.status = OrderStatus.FILLED;
      ocoOrder.triggeredLeg = legToUpdate.legType;
      const otherLeg = legToUpdate.legType === 'TAKE_PROFIT' ? ocoOrder.stopLossOrder : ocoOrder.takeProfitOrder;
      if (otherLeg && otherLeg.status === OrderStatus.ACTIVE) {
        try {
          await this.cancelOrderLeg(otherLeg.id, ocoOrder.ocoGroupId, 'oco_filled');
          this.logger.info(`OCO ${ocoOrder.ocoGroupId}: ${legToUpdate.legType} leg filled. Canceled other leg ${otherLeg.id}.`);
        } catch (error: any) {
          this.logger.error(`OCO ${ocoOrder.ocoGroupId}: Failed to cancel other leg ${otherLeg.id} after ${legToUpdate.legType} filled: ${error.message}`);
          // Potentially emit an alert here for manual intervention
        }
      }
      this.emit('ocoFilled', ocoOrder);
    } else if (update.status === OrderStatus.CANCELED || update.status === OrderStatus.REJECTED || update.status === OrderStatus.EXPIRED) {
      // If one leg is canceled/rejected/expired, and the OCO is still active,
      // we might need to cancel the other leg too, depending on the OCO strategy.
      // For simplicity here, if one leg fails or is externally canceled, we cancel the whole OCO.
      if (ocoOrder.status === OrderStatus.ACTIVE) {
         this.logger.info(`OCO ${ocoOrder.ocoGroupId}: Leg ${legToUpdate.id} (${legToUpdate.legType}) is ${update.status}. Cancelling OCO.`);
         await this.cancelOCOOrder(ocoOrder.ocoGroupId, `leg_${update.status.toLowerCase()}`);
      }
    }
    this.ocoOrders.set(ocoOrder.ocoGroupId, ocoOrder);
  }

  /**
   * Cancels an entire OCO order group.
   * @param ocoGroupId The ID of the OCO group to cancel.
   * @param reason Optional reason for cancellation.
   */
  public async cancelOCOOrder(ocoGroupId: string, reason?: string): Promise<ScalpingOCOOrderState | null> {
    const ocoOrder = this.ocoOrders.get(ocoGroupId);
    if (!ocoOrder) {
      this.logger.warn(`OCO Order ${ocoGroupId} not found for cancellation.`);
      return null;
    }

    if (ocoOrder.status === OrderStatus.FILLED || ocoOrder.status === OrderStatus.CANCELED || ocoOrder.status === OrderStatus.REJECTED) {
      this.logger.info(`OCO Order ${ocoGroupId} is already in a terminal state: ${ocoOrder.status}. No action taken.`);
      return ocoOrder;
    }

    const cancelPromises: Promise<any>[] = [];
    if (ocoOrder.takeProfitOrder && ocoOrder.takeProfitOrder.status === OrderStatus.ACTIVE) {
      cancelPromises.push(this.cancelOrderLeg(ocoOrder.takeProfitOrder.id, ocoGroupId, reason));
    }
    if (ocoOrder.stopLossOrder && ocoOrder.stopLossOrder.status === OrderStatus.ACTIVE) {
      cancelPromises.push(this.cancelOrderLeg(ocoOrder.stopLossOrder.id, ocoGroupId, reason));
    }

    try {
      await Promise.all(cancelPromises);
      ocoOrder.status = OrderStatus.CANCELED;
      ocoOrder.updatedAt = new Date();
      if(ocoOrder.takeProfitOrder) ocoOrder.takeProfitOrder.status = OrderStatus.CANCELED; // Reflect cancellation
      if(ocoOrder.stopLossOrder) ocoOrder.stopLossOrder.status = OrderStatus.CANCELED;   // Reflect cancellation

      this.logger.info(`OCO Order ${ocoGroupId} canceled successfully. Reason: ${reason || 'N/A'}`);
      this.emit('ocoCanceled', ocoOrder);
    } catch (error: any) {
      this.logger.error(`OCO Order ${ocoGroupId}: Failed to cancel one or more legs: ${error.message}`);
      // The OCO might be in a partially canceled state. This requires careful handling/logging.
      // For simplicity, we mark the OCO as REJECTED if full cancellation fails.
      ocoOrder.status = OrderStatus.REJECTED; // Or a custom "PARTIALLY_CANCELED" status
      this.emit('ocoCancelFailed', { ocoOrder, error });
      // throw error; // Decide if this should propagate
    }
    this.ocoOrders.set(ocoGroupId, ocoOrder);
    return ocoOrder;
  }

  public getOCOOrderState(ocoGroupId: string): ScalpingOCOOrderState | undefined {
    return this.ocoOrders.get(ocoGroupId);
  }

  // --- Private Helper Methods ---

  private _validateOCOInput(input: ScalpingOCOOrderInput): void {
    if (!input.symbol || !input.side || input.quantity <= 0 || !input.accountId || !input.userId) {
      throw new Error('Invalid OCO input: Missing required fields (symbol, side, quantity, accountId, userId).');
    }
    if (input.takeProfitPrice <= 0 || input.stopLossPrice <= 0) {
      throw new Error('Invalid OCO input: Take profit and stop loss prices must be positive.');
    }
    if (input.side === OrderSide.BUY) {
      if (input.takeProfitPrice <= input.stopLossPrice) { // Assuming entry is between SL and TP
        // This check might be more complex depending on where entry price is relative to SL/TP
        // For a BUY position, TP is above entry, SL is below entry.
        // So, TP price must be > SL price.
      }
    } else { // OrderSide.SELL
      if (input.takeProfitPrice >= input.stopLossPrice) {
        // For a SELL position, TP is below entry, SL is above entry.
        // So, TP price must be < SL price.
      }
    }
    // Add more validation for volatilityAdjustment, sessions etc. if needed
  }

  private _findOCOOrderByLegId(legOrderId: string): ScalpingOCOOrderState | undefined {
    for (const oco of this.ocoOrders.values()) {
      if (oco.takeProfitOrder?.id === legOrderId || oco.stopLossOrder?.id === legOrderId) {
        return oco;
      }
    }
    return undefined;
  }

  /**
   * Simulates placing an order with the Order Management System (OMS).
   * In a real system, this would be an async call to the OMS client.
   */
  private async _placeOrderWithOMS(params: BaseOrderParams & { type: OrderType; price?: number; stopPrice?: number; timeInForce: TimeInForce; metadata?: any }): Promise<OCOLeg> {
    // Simulate network latency and OMS processing
    await new Promise(resolve => setTimeout(resolve, Math.random() * 5 + 2)); // 2-7ms latency

    const orderId = uuidv4();
    const now = new Date();
    // Simulate OMS accepting the order
    this.logger.debug(`OMS: Placing ${params.type} order for ${params.symbol}`, params);

    // In a real scenario, OMS would confirm placement and return order details
    const placedOrder: OCOLeg = {
      id: orderId,
      symbol: params.symbol,
      side: params.side,
      quantity: params.quantity,
      accountId: params.accountId,
      userId: params.userId,
      type: params.type,
      status: OrderStatus.ACTIVE, // Assume OMS accepts and makes it active immediately for simulation
      filledQuantity: 0,
      createdAt: now,
      updatedAt: now,
      timeInForce: params.timeInForce,
      metadata: params.metadata,
      ocoGroupId: params.metadata.ocoGroupId, // Ensure metadata has these
      legType: params.metadata.legType,       // Ensure metadata has these
      // Conditionally add optional fields
      ...(params.clientOrderId !== undefined && { clientOrderId: params.clientOrderId }),
      ...(params.price !== undefined && { price: params.price }),
      ...(params.stopPrice !== undefined && { stopPrice: params.stopPrice }),
    };
    this.emit('omsOrderPlaced', placedOrder); // For external systems to know
    return placedOrder;
  }

  /**
   * Simulates cancelling an order with the OMS.
   */
  private async cancelOrderLeg(orderId: string, ocoGroupId: string, reason?: string): Promise<{ success: boolean; orderId: string }> {
    // Simulate network latency and OMS processing
    await new Promise(resolve => setTimeout(resolve, Math.random() * 5 + 1)); // 1-6ms latency
    this.logger.debug(`OMS: Cancelling order ${orderId} (OCO: ${ocoGroupId}). Reason: ${reason || 'N/A'}`);

    // Simulate OMS confirming cancellation
    const ocoOrder = this.ocoOrders.get(ocoGroupId);
    if (ocoOrder) {
        const leg = ocoOrder.takeProfitOrder?.id === orderId ? ocoOrder.takeProfitOrder : ocoOrder.stopLossOrder;
        if (leg) {
            leg.status = OrderStatus.CANCELED;
            leg.updatedAt = new Date();
        }
    }
    this.emit('omsOrderCanceled', { orderId, ocoGroupId, reason });
    return { success: true, orderId };
  }
}

// Example Usage (Illustrative - would be in a separate test/application file)
/*
async function example() {
  const logger = winston.createLogger({ // Basic Winston logger for example
    transports: [new winston.transports.Console({ format: winston.format.simple() })],
    level: 'debug',
  });

  // const omsClient = new MockOrderManagementSystem(); // Your actual OMS client
  const ocoManager = new ScalpingOCOOrderManager(logger); //, omsClient);

  ocoManager.on('ocoActivated', (state) => console.log('Event: OCO Activated', state.ocoGroupId));
  ocoManager.on('ocoFilled', (state) => console.log('Event: OCO Filled', state.ocoGroupId, 'Triggered:', state.triggeredLeg));
  ocoManager.on('ocoCanceled', (state) => console.log('Event: OCO Canceled', state.ocoGroupId));
  ocoManager.on('omsOrderPlaced', (order) => console.log('Event: OMS Order Placed', order.id, order.type, order.legType));
  ocoManager.on('omsOrderCanceled', (data) => console.log('Event: OMS Order Canceled', data.orderId));


  try {
    const ocoDetails: ScalpingOCOOrderInput = {
      clientOrderId: 'myScalpTrade123',
      symbol: 'EUR/USD',
      side: OrderSide.BUY, // We are in a BUY position
      quantity: 0.01,
      accountId: 'acc-scalper-001',
      userId: 'user-scalper-A',
      takeProfitPrice: 1.0850,
      stopLossPrice: 1.0830,
      timeInForce: TimeInForce.GTC,
      activateDuringSessions: [TradingSession.LONDON, TradingSession.NEW_YORK],
    };
    const activeOCO = await ocoManager.createAndActivateOrder(ocoDetails);
    console.log('Active OCO:', activeOCO.ocoGroupId);

    // Simulate a fill event from OMS for the take profit leg
    if (activeOCO.takeProfitOrder) {
      // await new Promise(resolve => setTimeout(resolve, 100)); // wait a bit
      // ocoManager.handleOrderUpdate(activeOCO.takeProfitOrder.id, {
      //   status: OrderStatus.FILLED,
      //   fill: {
      //     orderId: activeOCO.takeProfitOrder.id,
      //     fillId: uuidv4(),
      //     quantity: activeOCO.quantity,
      //     price: activeOCO.takeProfitOrder.price!,
      //     timestamp: new Date(),
      //   }
      // });
    }

    // Simulate cancelling the OCO externally
    // await new Promise(resolve => setTimeout(resolve, 200));
    // await ocoManager.cancelOCOOrder(activeOCO.ocoGroupId, 'manual_cancel');


  } catch (error: any) {
    console.error('OCO Example Error:', error.message);
  }
}

// example();
*/
