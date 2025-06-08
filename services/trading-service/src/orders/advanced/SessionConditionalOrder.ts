/**
 * Session Conditional Order
 * Activates or modifies order behavior based on predefined trading sessions (e.g., London, New York, Tokyo).
 *
 * This module provides:
 * - Definition of trading sessions and their typical active hours.
 * - Logic to check if an order should be active based on the current time and defined sessions.
 * - Ability to place, hold, or cancel orders depending on session state.
 *
 * Expected Benefits:
 * - Align trading strategies with specific market hours known for higher liquidity or volatility.
 * - Avoid order execution during unfavorable market conditions outside primary session times.
 * - Automated session-based order management.
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';

// Define common types since we're removing external dependencies
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT'
}

export enum OrderStatus {
  PENDING = 'PENDING',
  FILLED = 'FILLED',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  CANCELLED = 'CANCELLED',
  REJECTED = 'REJECTED'
}

export enum TimeInForce {
  GTC = 'GTC', // Good Till Cancelled
  IOC = 'IOC', // Immediate or Cancel
  FOK = 'FOK', // Fill or Kill
  DAY = 'DAY'  // Day order
}

export interface BaseOrderParams {
  symbol: string;
  side: OrderSide;
  type: OrderType;
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: TimeInForce;
}

export interface Order extends BaseOrderParams {
  id: string;
  status: OrderStatus;
  filledQuantity: number;
  averagePrice?: number;
  timestamp: Date;
  lastUpdated: Date;
}

export interface OrderFillEvent {
  orderId: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price: number;
  timestamp: Date;
}

// --- Session Definitions ---

export enum TradingSession {
  LONDON = 'LONDON',
  NEW_YORK = 'NEW_YORK',
  TOKYO = 'TOKYO',
  SYDNEY = 'SYDNEY',
  FRANKFURT = 'FRANKFURT',
  // Add more sessions as needed
}

// Define typical UTC active hours for sessions (can be made more configurable)
// These are simplified and don't account for DST or exact market opens/closes.
const SESSION_HOURS_UTC: Record<TradingSession, { start: number; end: number }> = {
  [TradingSession.LONDON]: { start: 7, end: 16 }, // 7:00 - 16:00 UTC
  [TradingSession.NEW_YORK]: { start: 12, end: 21 }, // 12:00 - 21:00 UTC
  [TradingSession.TOKYO]: { start: 0, end: 9 }, // 00:00 - 09:00 UTC
  [TradingSession.SYDNEY]: { start: 21, end: 6 }, // 21:00 - 06:00 UTC (spans across midnight)
  [TradingSession.FRANKFURT]: { start: 6, end: 15 }, // 06:00 - 15:00 UTC
};

// --- Session Conditional Order Specific Types ---

export interface SessionConditionalOrderInput extends BaseOrderParams {
  orderType: OrderType; // The type of the underlying order (MARKET, LIMIT, STOP)
  price?: number; // For LIMIT or STOP orders
  stopPrice?: number; // For STOP orders
  timeInForce?: TimeInForce;
  
  targetSessions: TradingSession[]; // Sessions during which this order should be active
  actionOutsideSession: 'HOLD' | 'CANCEL' | 'PLACE_ANYWAY'; // What to do if placed outside target sessions
  cancelAtSessionEnd?: boolean; // If true, cancel the order when all target sessions end for the day
  placeAtSessionStart?: boolean; // If true, and order is on HOLD, place it when a target session starts
}

export enum SessionConditionalStatus {
  PENDING_SESSION = 'PENDING_SESSION', // Waiting for a target session to start
  SESSION_ACTIVE_PENDING_ORDER = 'SESSION_ACTIVE_PENDING_ORDER', // Session active, underlying order not yet placed
  ORDER_PLACED = 'ORDER_PLACED',       // Underlying order sent to OMS
  ORDER_HELD = 'ORDER_HELD',           // Order created but held due to session rules
  FILLED = 'FILLED',
  CANCELED = 'CANCELED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED',
}

export interface SessionConditionalOrderState {
  sessionConditionalOrderId: string;
  clientOrderId?: string | undefined;
  accountId: string;
  userId: string;
  status: SessionConditionalStatus;
  underlyingOrder: Order | null;
  createdAt: Date;
  updatedAt: Date;
  
  targetSessions: TradingSession[];
  actionOutsideSession: 'HOLD' | 'CANCEL' | 'PLACE_ANYWAY';
  cancelAtSessionEnd: boolean;
  placeAtSessionStart: boolean;
  
  lastSessionCheck: Date | null;
  isActiveSession: boolean; // Is any of the target sessions currently active

  originalInput: SessionConditionalOrderInput;
}

/**
 * Manages Session Conditional Orders.
 * This class determines if an order should be placed, held, or canceled based on trading session times.
 */
export class SessionConditionalOrderManager extends EventEmitter {
  private conditionalOrders: Map<string, SessionConditionalOrderState> = new Map();
  private logger: Logger;
  private sessionCheckInterval: NodeJS.Timeout | null = null;

  constructor(logger: Logger, checkIntervalMs: number = 60000 /* 1 minute */) {
    super();
    console = logger;
    console.info('SessionConditionalOrderManager initialized.');
    this._startSessionMonitoring(checkIntervalMs);
  }

  private _isSessionActive(sessions: TradingSession[], nowUtc: Date = new Date()): boolean {
    const currentHour = nowUtc.getUTCHours();
    const currentMinute = nowUtc.getUTCMinutes(); // For more precise checks if needed

    for (const session of sessions) {
      const sessionTimes = SESSION_HOURS_UTC[session];
      if (!sessionTimes) continue;

      if (sessionTimes.start <= sessionTimes.end) { // Session does not span midnight
        if (currentHour >= sessionTimes.start && currentHour < sessionTimes.end) {
          return true;
        }
      } else { // Session spans midnight (e.g., Sydney)
        if (currentHour >= sessionTimes.start || currentHour < sessionTimes.end) {
          return true;
        }
      }
    }
    return false;
  }

  private _startSessionMonitoring(intervalMs: number): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
    }
    this.sessionCheckInterval = setInterval(() => {
      this.conditionalOrders.forEach(orderState => {
        this._checkAndUpdateOrderBasedOnSession(orderState.sessionConditionalOrderId);
      });
    }, intervalMs);
    console.info(`Session monitoring started with interval ${intervalMs}ms.`);
  }

  public stopSessionMonitoring(): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
      this.sessionCheckInterval = null;
      console.info('Session monitoring stopped.');
    }
  }
  
  private async _checkAndUpdateOrderBasedOnSession(orderId: string): Promise<void> {
    const orderState = this.conditionalOrders.get(orderId);
    if (!orderState || orderState.status === SessionConditionalStatus.FILLED || orderState.status === SessionConditionalStatus.CANCELED || orderState.status === SessionConditionalStatus.REJECTED) {
      return;
    }

    const wasActiveSession = orderState.isActiveSession;
    orderState.isActiveSession = this._isSessionActive(orderState.targetSessions);
    orderState.lastSessionCheck = new Date();

    if (orderState.isActiveSession && !wasActiveSession) { // Session just started
      console.info(`SessionConditionalOrder ${orderId}: Target session started.`);
      this.emit('sessionStarted', orderState);
      if (orderState.status === SessionConditionalStatus.ORDER_HELD && orderState.placeAtSessionStart) {
        console.info(`SessionConditionalOrder ${orderId}: Attempting to place held order as session started.`);
        await this._placeUnderlyingOrder(orderId);
      }
    } else if (!orderState.isActiveSession && wasActiveSession) { // Session just ended
      console.info(`SessionConditionalOrder ${orderId}: Target session ended.`);
      this.emit('sessionEnded', orderState);
      if (orderState.underlyingOrder && orderState.underlyingOrder.status === OrderStatus.ACTIVE && orderState.cancelAtSessionEnd) {
        console.info(`SessionConditionalOrder ${orderId}: Attempting to cancel order as session ended.`);
        await this.cancelConditionalOrder(orderId, 'SESSION_ENDED');
      }
    }
    
    // Initial placement logic if status is PENDING_SESSION
    if (orderState.status === SessionConditionalStatus.PENDING_SESSION && orderState.isActiveSession) {
        console.info(`SessionConditionalOrder ${orderId}: Target session is active. Proceeding to place order.`);
        await this._placeUnderlyingOrder(orderId);
    }
  }


  public async createConditionalOrder(input: SessionConditionalOrderInput): Promise<SessionConditionalOrderState> {
    if (!input.targetSessions || input.targetSessions.length === 0) {
      throw new Error('Target trading sessions must be specified.');
    }
    const orderId = uuidv4();
    const now = new Date();
    const isActiveNow = this._isSessionActive(input.targetSessions, now);

    const initialState: SessionConditionalOrderState = {
      sessionConditionalOrderId: orderId,
      clientOrderId: input.clientOrderId,
      accountId: input.accountId,
      userId: input.userId,
      status: SessionConditionalStatus.PENDING_SESSION, // Initial status
      underlyingOrder: null,
      createdAt: now,
      updatedAt: now,
      targetSessions: [...input.targetSessions],
      actionOutsideSession: input.actionOutsideSession,
      cancelAtSessionEnd: input.cancelAtSessionEnd ?? false,
      placeAtSessionStart: input.placeAtSessionStart ?? false,
      lastSessionCheck: now,
      isActiveSession: isActiveNow,
      originalInput: { ...input },
    };

    this.conditionalOrders.set(orderId, initialState);
    console.info(`Session Conditional Order ${orderId} created. Target sessions: ${input.targetSessions.join(', ')}. Active now: ${isActiveNow}.`);
    this.emit('sessionConditionalOrderCreated', initialState);

    if (isActiveNow) {
      initialState.status = SessionConditionalStatus.SESSION_ACTIVE_PENDING_ORDER;
      await this._placeUnderlyingOrder(orderId);
    } else {
      switch (input.actionOutsideSession) {
        case 'HOLD':
          initialState.status = SessionConditionalStatus.ORDER_HELD;
          console.info(`SessionConditionalOrder ${orderId}: Order HELD as it's outside target sessions.`);
          this.emit('sessionConditionalOrderHeld', initialState);
          break;
        case 'CANCEL':
          initialState.status = SessionConditionalStatus.CANCELED;
          initialState.updatedAt = new Date();
          console.info(`SessionConditionalOrder ${orderId}: Order CANCELED as it's outside target sessions and action is CANCEL.`);
          this.emit('sessionConditionalOrderCanceled', initialState);
          // No OMS interaction needed as it was never placed
          break;
        case 'PLACE_ANYWAY':
          initialState.status = SessionConditionalStatus.SESSION_ACTIVE_PENDING_ORDER; // Treat as if session is active for placement
          console.info(`SessionConditionalOrder ${orderId}: Placing order despite being outside target sessions (PLACE_ANYWAY).`);
          await this._placeUnderlyingOrder(orderId);
          break;
        default:
          initialState.status = SessionConditionalStatus.ORDER_HELD; // Default to HOLD
          console.warn(`SessionConditionalOrder ${orderId}: Unknown actionOutsideSession '${input.actionOutsideSession}'. Defaulting to HOLD.`);
          this.emit('sessionConditionalOrderHeld', initialState);
          break;
      }
    }
    
    this.conditionalOrders.set(orderId, initialState); // Ensure state is updated after async ops
    return initialState;
  }

  private async _placeUnderlyingOrder(conditionalOrderId: string): Promise<void> {
    const orderState = this.conditionalOrders.get(conditionalOrderId);
    if (!orderState || orderState.underlyingOrder || orderState.status === SessionConditionalStatus.FILLED || orderState.status === SessionConditionalStatus.CANCELED) {
      console.warn(`SessionConditionalOrder ${conditionalOrderId}: Cannot place underlying order. State: ${orderState?.status}, Has Underlying: ${!!orderState?.underlyingOrder}`);
      return;
    }

    const input = orderState.originalInput;
    try {
      // Simulate OMS placement
      const underlyingOrderPayload: Omit<Order, 'id' | 'status' | 'createdAt' | 'updatedAt' | 'filledQuantity' | 'averageFillPrice'> & { metadata: any } = {
        clientOrderId: input.clientOrderId ? `${input.clientOrderId}-SESS` : undefined,
        symbol: input.symbol,
        side: input.side,
        quantity: input.quantity,
        accountId: input.accountId,
        userId: input.userId,
        type: input.orderType,
        price: input.price,
        stopPrice: input.stopPrice,
        timeInForce: input.timeInForce || TimeInForce.GTC,
        metadata: { conditionalOrderId: conditionalOrderId, strategy: 'sessionBased' },
      };
      
      const placedOrder = await this._sendToOMS(underlyingOrderPayload); // Simulated
      
      orderState.underlyingOrder = placedOrder;
      orderState.status = SessionConditionalStatus.ORDER_PLACED;
      orderState.updatedAt = new Date();
      console.info(`SessionConditionalOrder ${conditionalOrderId}: Underlying order ${placedOrder.id} placed with OMS.`);
      this.emit('sessionConditionalUnderlyingOrderPlaced', orderState);

    } catch (error: any) {
      console.error(`SessionConditionalOrder ${conditionalOrderId}: Failed to place underlying order: ${error.message}`, { input });
      orderState.status = SessionConditionalStatus.REJECTED; // Or some other failure status
      orderState.updatedAt = new Date();
      this.emit('sessionConditionalOrderPlacementFailed', { orderState, error });
      // Potentially re-throw or handle retry logic
    }
  }

  public async handleUnderlyingOrderUpdate(conditionalOrderId: string, update: Partial<Order>): Promise<void> {
    const orderState = this.conditionalOrders.get(conditionalOrderId);
    if (!orderState || !orderState.underlyingOrder) {
      console.warn(`SessionConditionalOrder: Received update for non-existent or unlinked underlying order for ${conditionalOrderId}.`);
      return;
    }

    // Update the underlying order state
    orderState.underlyingOrder = { ...orderState.underlyingOrder, ...update, updatedAt: new Date() };
    orderState.updatedAt = new Date();

    console.info(`SessionConditionalOrder ${conditionalOrderId}: Underlying order ${orderState.underlyingOrder.id} updated. Status: ${update.status}`);
    this.emit('sessionConditionalOrderUpdated', orderState);

    if (update.status === OrderStatus.FILLED) {
      orderState.status = SessionConditionalStatus.FILLED;
      this.emit('sessionConditionalOrderFilled', orderState);
    } else if (update.status === OrderStatus.CANCELED) {
      // If canceled externally, reflect this. If canceled by this manager, it's already CANCELED.
      if (orderState.status !== SessionConditionalStatus.CANCELED) {
          orderState.status = SessionConditionalStatus.CANCELED;
          console.info(`SessionConditionalOrder ${conditionalOrderId}: Underlying order was canceled externally.`);
          this.emit('sessionConditionalOrderCanceled', { orderState, reason: 'EXTERNAL_CANCEL' });
      }
    } else if (update.status === OrderStatus.REJECTED) {
      orderState.status = SessionConditionalStatus.REJECTED;
      this.emit('sessionConditionalOrderRejected', orderState);
    }
  }
  
  public async handleUnderlyingOrderFill(conditionalOrderId: string, fillEvent: OrderFillEvent): Promise<void> {
    const orderState = this.conditionalOrders.get(conditionalOrderId);
    if (!orderState || !orderState.underlyingOrder || orderState.underlyingOrder.id !== fillEvent.orderId) {
        console.warn(`SessionConditionalOrder: Received fill for unknown or mismatched underlying order ${fillEvent.orderId} for conditional ${conditionalOrderId}.`);
        return;
    }

    orderState.underlyingOrder.status = OrderStatus.FILLED;
    orderState.underlyingOrder.filledQuantity = (orderState.underlyingOrder.filledQuantity || 0) + fillEvent.quantity;
    orderState.underlyingOrder.averageFillPrice = fillEvent.price; // Assuming fill event provides the average price for this fill
    orderState.underlyingOrder.updatedAt = new Date();
    
    orderState.status = SessionConditionalStatus.FILLED;
    orderState.updatedAt = new Date();
    
    console.info(`SessionConditionalOrder ${conditionalOrderId}: Underlying order ${fillEvent.orderId} FILLED.`);
    this.emit('sessionConditionalOrderFilled', orderState);
  }


  public async cancelConditionalOrder(conditionalOrderId: string, reason: string = 'USER_REQUEST'): Promise<void> {
    const orderState = this.conditionalOrders.get(conditionalOrderId);
    if (!orderState) {
      console.warn(`SessionConditionalOrder: Attempted to cancel non-existent order ${conditionalOrderId}.`);
      return;
    }

    if (orderState.status === SessionConditionalStatus.CANCELED || orderState.status === SessionConditionalStatus.FILLED) {
      console.info(`SessionConditionalOrder ${conditionalOrderId}: Already ${orderState.status}, no action needed for cancellation.`);
      return;
    }

    const oldStatus = orderState.status;
    orderState.status = SessionConditionalStatus.CANCELED;
    orderState.updatedAt = new Date();

    if (orderState.underlyingOrder && orderState.underlyingOrder.status !== OrderStatus.CANCELED && orderState.underlyingOrder.status !== OrderStatus.FILLED) {
      try {
        await this._cancelOMSOrder(orderState.underlyingOrder.id, reason); // Simulated
        if(orderState.underlyingOrder) { // TS null check
            orderState.underlyingOrder.status = OrderStatus.CANCELED;
            orderState.underlyingOrder.updatedAt = new Date();
        }
        console.info(`SessionConditionalOrder ${conditionalOrderId}: Underlying order ${orderState.underlyingOrder?.id} cancellation requested from OMS.`);
      } catch (error: any) {
        console.error(`SessionConditionalOrder ${conditionalOrderId}: Failed to cancel underlying order ${orderState.underlyingOrder?.id}: ${error.message}`);
        // Revert status or handle error appropriately, e.g., mark as PENDING_CANCEL_FAILED
        orderState.status = oldStatus; // Example: revert status
        throw error; // Re-throw to signal failure
      }
    } else {
         console.info(`SessionConditionalOrder ${conditionalOrderId}: No active underlying order to cancel, or order already terminal. Marking as CANCELED.`);
    }
    
    this.emit('sessionConditionalOrderCanceled', { orderState, reason });
  }

  public getOrderState(conditionalOrderId: string): SessionConditionalOrderState | undefined {
    return this.conditionalOrders.get(conditionalOrderId);
  }

  // --- Simulated OMS Interactions ---
  private async _sendToOMS(
    payload: Omit<Order, 'id' | 'status' | 'createdAt' | 'updatedAt' | 'filledQuantity' | 'averageFillPrice'> & { metadata: any }
  ): Promise<Order> {
    const now = new Date();
    // Simulate network delay and OMS processing
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100)); 
    
    const placedOrder: Order = {
      ...payload,
      id: uuidv4(),
      status: OrderStatus.ACTIVE, // Assume it becomes active immediately for simulation
      filledQuantity: 0,
      createdAt: now,
      updatedAt: now,
    };
    console.debug(`Simulated OMS: Order ${placedOrder.id} placed.`, placedOrder);
    return placedOrder;
  }

  private async _cancelOMSOrder(orderId: string, reason: string): Promise<void> {
    // Simulate network delay and OMS processing
    await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 50));
    console.debug(`Simulated OMS: Order ${orderId} cancellation processed. Reason: ${reason}.`);
    // In a real system, OMS would send an update that then calls handleUnderlyingOrderUpdate
  }
  
  public destroy(): void {
    this.stopSessionMonitoring();
    this.conditionalOrders.clear();
    this.removeAllListeners();
    console.info('SessionConditionalOrderManager destroyed.');
  }
}

// Example Usage (for testing or integration)
/*
async function testSessionOrders() {
  const logger = console; // Replace with actual Winston logger

  const sessionManager = new SessionConditionalOrderManager(logger as any, 5000); // Check every 5s

  const orderInputNY: SessionConditionalOrderInput = {
    clientOrderId: 'test-ny-001',
    symbol: 'EUR/USD',
    side: OrderSide.BUY,
    quantity: 10000,
    accountId: 'acc-123',
    userId: 'user-456',
    orderType: OrderType.LIMIT,
    price: 1.0850,
    targetSessions: [TradingSession.NEW_YORK],
    actionOutsideSession: 'HOLD',
    cancelAtSessionEnd: true,
    placeAtSessionStart: true,
  };
  
  const orderInputLondonCancel: SessionConditionalOrderInput = {
    clientOrderId: 'test-lon-002',
    symbol: 'GBP/JPY',
    side: OrderSide.SELL,
    quantity: 5000,
    accountId: 'acc-123',
    userId: 'user-456',
    orderType: OrderType.MARKET,
    targetSessions: [TradingSession.LONDON],
    actionOutsideSession: 'CANCEL',
  };

  try {
    const nyOrder = await sessionManager.createConditionalOrder(orderInputNY);
    logger.info('NY Order Created:', nyOrder);

    const lonOrder = await sessionManager.createConditionalOrder(orderInputLondonCancel);
    logger.info('London Order Created (or canceled):', lonOrder);

    // Simulate time passing or manually trigger session checks for testing
    // sessionManager._checkAndUpdateOrderBasedOnSession(nyOrder.sessionConditionalOrderId);

    // Simulate an external fill for the NY order if it gets placed
    // setTimeout(async () => {
    //   const currentState = sessionManager.getOrderState(nyOrder.sessionConditionalOrderId);
    //   if (currentState && currentState.underlyingOrder && currentState.status === SessionConditionalStatus.ORDER_PLACED) {
    //     logger.info('Simulating fill for NY order...');
    //     const fill: OrderFillEvent = {
    //       orderId: currentState.underlyingOrder.id,
    //       fillId: uuidv4(),
    //       quantity: currentState.underlyingOrder.quantity,
    //       price: currentState.underlyingOrder.price!,
    //       timestamp: new Date(),
    //     };
    //     await sessionManager.handleUnderlyingOrderFill(nyOrder.sessionConditionalOrderId, fill);
    //   }
    // }, 20000); // After 20 seconds


  } catch (error) {
    logger.error('Error in test:', error);
  }

  // setTimeout(() => sessionManager.destroy(), 60000); // Clean up after 1 minute
}

// testSessionOrders();
*/
