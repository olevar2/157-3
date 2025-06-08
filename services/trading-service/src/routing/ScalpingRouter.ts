/**
 * Scalping Router
 * Optimized for ultra-low latency and best execution for scalping strategies.
 * Considers venue latency, fill probability, and fees.
 *
 * This module provides:
 * - Logic to select the optimal execution venue for scalping orders.
 * - Real-time monitoring of venue performance (simulated).
 * - Prioritization of direct market access (DMA) if available.
 *
 * Expected Benefits:
 * - Reduced slippage for scalping orders.
 * - Improved fill rates at desired prices.
 * - Lower overall transaction costs for high-frequency trading.
 */

// --- Common Types ---

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

export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  quantity: number;
  price?: number;
}

// --- Venue and Routing Specific Types ---

export interface ExecutionVenue {
  id: string;
  name: string;
  supportedOrderTypes: OrderType[];
  // Simulated performance metrics
  averageLatencyMs: number; // Average round-trip time to this venue
  fillProbability: number; // Likelihood of filling a market/limit order (0.0 to 1.0)
  makerFee: number; // Fee for adding liquidity (can be negative - rebate)
  takerFee: number; // Fee for removing liquidity
  minOrderSize: number;
  maxOrderSize: number;
  isDMA: boolean; // Direct Market Access
  healthStatus: 'ONLINE' | 'OFFLINE' | 'DEGRADED';
  currentLoad?: number; // 0.0 to 1.0, how busy the venue is
}

export interface RouteDecision {
  venue: ExecutionVenue;
  reason: string; // Why this venue was chosen
  estimatedCost: number;
  estimatedLatencyMs: number;
}

export interface RoutingRequest {
  order: Pick<Order, 'symbol' | 'side' | 'quantity' | 'type' | 'price'>;
  // Preferences for routing
  priority: 'SPEED' | 'COST' | 'FILL_PROBABILITY';
  maxAllowedLatencyMs?: number;
  maxAllowedCost?: number; // Per unit or total
}

/**
 * ScalpingRouter: Selects the best venue for high-frequency scalping orders.
 */
export class ScalpingRouter {
  private venues: ExecutionVenue[] = [];

  constructor(initialVenues: ExecutionVenue[] = []) {
    this.venues = initialVenues;
    console.log(`ScalpingRouter initialized with ${initialVenues.length} venues.`);
    // In a real system, this would subscribe to venue status updates.
  }
  public addVenue(venue: ExecutionVenue): void {
    if (!this.venues.find(v => v.id === venue.id)) {
      this.venues.push(venue);
      console.log(`Venue ${venue.name} added to ScalpingRouter.`);
    } else {
      console.warn(`Venue ${venue.name} already exists.`);
    }
  }

  public updateVenueStatus(venueId: string, updates: Partial<ExecutionVenue>): void {
    const venue = this.venues.find(v => v.id === venueId);
    if (venue) {
      Object.assign(venue, updates);
      console.log(`Venue ${venue.name} status updated. Latency: ${venue.averageLatencyMs}ms, Health: ${venue.healthStatus}`);    } else {
      console.warn(`Attempted to update non-existent venue ${venueId}.`);
    }
  }

  public getAvailableVenues(): Readonly<ExecutionVenue[]> {
    return this.venues.filter(v => v.healthStatus === 'ONLINE');
  }

  private _scoreVenue(venue: ExecutionVenue, request: RoutingRequest): number {
    let score = 1000;
    const { order, priority, maxAllowedCost, maxAllowedLatencyMs } = request;

    // Penalize for not supporting order type
    if (!venue.supportedOrderTypes.includes(order.type)) {
      return -Infinity; // Cannot route
    }

    // Penalize for order size constraints
    if (order.quantity < venue.minOrderSize || order.quantity > venue.maxOrderSize) {
        return -Infinity; // Cannot route
    }

    // Latency Score (lower is better)
    score -= venue.averageLatencyMs * (priority === 'SPEED' ? 2 : 1);
    if (maxAllowedLatencyMs && venue.averageLatencyMs > maxAllowedLatencyMs) {
        score -= 500; // Heavy penalty if exceeds max allowed
    }

    // Cost Score (lower is better)
    // Assuming taker fee for market orders or aggressive limit orders
    const fee = (order.type === OrderType.LIMIT && order.price /* and is passive */) ? venue.makerFee : venue.takerFee;
    const estimatedCost = fee * order.quantity; // Simplified cost
    score -= estimatedCost * (priority === 'COST' ? 200 : 100); // Cost is significant
     if (maxAllowedCost && estimatedCost > maxAllowedCost) {
        score -= 500; // Heavy penalty
    }

    // Fill Probability Score (higher is better)
    score += venue.fillProbability * (priority === 'FILL_PROBABILITY' ? 200 : 100);

    // Bonus for DMA
    if (venue.isDMA) {
      score += 50;
    }

    // Penalize for degraded health or high load
    if (venue.healthStatus === 'DEGRADED') score -= 200;
    if (venue.currentLoad && venue.currentLoad > 0.8) score -= (venue.currentLoad - 0.8) * 500;

    return score;
  }

  public async findBestRoute(request: RoutingRequest): Promise<RouteDecision | null> {
    const availableVenues = this.getAvailableVenues();    if (availableVenues.length === 0) {
      console.warn('No available venues for routing.');
      return null;
    }

    let bestVenue: ExecutionVenue | null = null;
    let highestScore = -Infinity;
    let reason = 'No suitable venue found.';

    for (const venue of availableVenues) {
      const score = this._scoreVenue(venue, request);
      console.log(`Venue ${venue.name} scored ${score} for order ${request.order.symbol} ${request.order.type}`);
      if (score > highestScore) {
        highestScore = score;
        bestVenue = venue;
      }
    }

    if (bestVenue) {
      const fee = (request.order.type === OrderType.LIMIT && request.order.price) ? bestVenue.makerFee : bestVenue.takerFee;
      const estimatedCost = fee * request.order.quantity;
      reason = `Highest score (${highestScore.toFixed(2)}) based on priority: ${request.priority}, Latency: ${bestVenue.averageLatencyMs}ms, FillProb: ${bestVenue.fillProbability}, Cost: ${estimatedCost.toFixed(4)}`;
      
      const decision: RouteDecision = {
        venue: bestVenue,
        reason: reason,
        estimatedCost: estimatedCost,
        estimatedLatencyMs: bestVenue.averageLatencyMs,
      };
      console.log(`Best route for ${request.order.symbol} ${request.order.type} Qty ${request.order.quantity}: Venue ${bestVenue.name}. Reason: ${reason}`);
      this.emit('routeSelected', decision, request.order);
      return decision;
    }

    console.warn(`Could not find a suitable route for order ${request.order.symbol} ${request.order.type}.`);
    this.emit('noRouteFound', request.order);
    return null;
  }
  
  // EventEmitter methods if ScalpingRouter needs to emit events
  private _emitter = new EventEmitter();
  public on(event: string | symbol, listener: (...args: any[]) => void): this {
    this._emitter.on(event, listener);
    return this;
  }
  public emit(event: string | symbol, ...args: any[]): boolean {
    return this._emitter.emit(event, ...args);
  }
  public removeAllListeners(event?: string | symbol): this {
    this._emitter.removeAllListeners(event);
    return this;
  }
}

// --- Example Usage (for testing or integration) ---
/*
function testScalpingRouter() {
  const logger = console; // Replace with actual Winston logger

  const venues: ExecutionVenue[] = [
    {
      id: 'venue-a', name: 'VenueA-FAST-DMA', supportedOrderTypes: [OrderType.MARKET, OrderType.LIMIT],
      averageLatencyMs: 5, fillProbability: 0.95, makerFee: -0.0001, takerFee: 0.0002,
      minOrderSize: 100, maxOrderSize: 100000, isDMA: true, healthStatus: 'ONLINE', currentLoad: 0.2
    },
    {
      id: 'venue-b', name: 'VenueB-CHEAP', supportedOrderTypes: [OrderType.MARKET, OrderType.LIMIT],
      averageLatencyMs: 20, fillProbability: 0.99, makerFee: 0.0000, takerFee: 0.0001,
      minOrderSize: 1, maxOrderSize: 500000, isDMA: false, healthStatus: 'ONLINE', currentLoad: 0.5
    },
    {
      id: 'venue-c', name: 'VenueC-SLOW-RELIABLE', supportedOrderTypes: [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP],
      averageLatencyMs: 50, fillProbability: 0.999, makerFee: -0.0002, takerFee: 0.0003,
      minOrderSize: 1000, maxOrderSize: 1000000, isDMA: false, healthStatus: 'ONLINE', currentLoad: 0.1
    },
     {
      id: 'venue-d', name: 'VenueD-DEGRADED', supportedOrderTypes: [OrderType.MARKET, OrderType.LIMIT],
      averageLatencyMs: 15, fillProbability: 0.80, makerFee: 0.0000, takerFee: 0.00015,
      minOrderSize: 100, maxOrderSize: 100000, isDMA: false, healthStatus: 'DEGRADED', currentLoad: 0.9
    }
  ];

  const router = new ScalpingRouter(venues);

  router.on('routeSelected', (decision, order) => {
    console.log(`EVENT: Route selected for ${order.symbol}: ${decision.venue.name}`);
  });
  router.on('noRouteFound', (order) => {
    console.warn(`EVENT: No route found for ${order.symbol}`);
  });

  const routingRequestSpeed: RoutingRequest = {
    order: { symbol: 'EUR/USD', side: OrderSide.BUY, quantity: 10000, type: OrderType.MARKET },
    priority: 'SPEED',
  };

  const routingRequestCost: RoutingRequest = {
    order: { symbol: 'GBP/JPY', side: OrderSide.SELL, quantity: 50000, type: OrderType.LIMIT, price: 190.50 },
    priority: 'COST',
    maxAllowedLatencyMs: 30,
  };
  
  const routingRequestFill: RoutingRequest = {
    order: { symbol: 'USD/CAD', side: OrderSide.BUY, quantity: 25000, type: OrderType.MARKET },
    priority: 'FILL_PROBABILITY',
  };

  (async () => {
    console.log('\n--- Routing for SPEED ---');
    const routeSpeed = await router.findBestRoute(routingRequestSpeed);
    // if (routeSpeed) console.log('Chosen for SPEED:', routeSpeed.venue.name, routeSpeed.reason);

    console.log('\n--- Routing for COST ---');
    const routeCost = await router.findBestRoute(routingRequestCost);
    // if (routeCost) console.log('Chosen for COST:', routeCost.venue.name, routeCost.reason);
    
    console.log('\n--- Routing for FILL PROBABILITY ---');
    const routeFill = await router.findBestRoute(routingRequestFill);
    // if (routeFill) console.log('Chosen for FILL:', routeFill.venue.name, routeFill.reason);

    console.log('\n--- Simulating VenueA degraded ---');
    router.updateVenueStatus('venue-a', { healthStatus: 'DEGRADED', averageLatencyMs: 70 });
    const routeSpeedAfterDegrade = await router.findBestRoute(routingRequestSpeed);
    // if (routeSpeedAfterDegrade) console.log('Chosen for SPEED (VenueA degraded):', routeSpeedAfterDegrade.venue.name, routeSpeedAfterDegrade.reason);

  })();
}

// testScalpingRouter();
*/
