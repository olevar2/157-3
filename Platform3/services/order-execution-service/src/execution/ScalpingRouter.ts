/**
 * Scalping Router
 * Ultra-fast order routing optimized for M1-M5 scalping strategies
 * 
 * This module provides intelligent order routing including:
 * - Sub-10ms routing decisions for scalping trades
 * - Liquidity provider selection based on spread and latency
 * - Real-time venue performance monitoring
 * - Smart order fragmentation for large scalping positions
 * - Session-aware routing optimization
 * 
 * Expected Benefits:
 * - Optimal execution prices for scalping trades
 * - Minimized slippage through intelligent routing
 * - Reduced latency through venue optimization
 * - Enhanced fill rates for time-sensitive scalping
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { Logger } from 'winston';

// --- Types ---
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL',
}

export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
}

export enum VenueType {
  ECN = 'ECN',
  MARKET_MAKER = 'MARKET_MAKER',
  STP = 'STP',
  PRIME_BROKER = 'PRIME_BROKER',
}

export interface VenueInfo {
  id: string;
  name: string;
  type: VenueType;
  averageLatencyMs: number;
  averageSpread: number;
  fillRate: number; // 0-1
  isActive: boolean;
  maxOrderSize: number;
  minOrderSize: number;
  supportedSymbols: string[];
  sessionPreference: TradingSession[];
}

export enum TradingSession {
  ASIAN = 'ASIAN',
  LONDON = 'LONDON',
  NEW_YORK = 'NEW_YORK',
  OVERLAP_LDN_NY = 'OVERLAP_LDN_NY',
}

export interface OrderRequest {
  id: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  type: OrderType;
  price?: number;
  stopPrice?: number;
  accountId: string;
  userId: string;
  urgency: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'; // For scalping prioritization
  maxSlippage?: number; // In pips
  timeoutMs?: number; // Max time to wait for fill
}

export interface RoutingDecision {
  requestId: string;
  selectedVenue: VenueInfo;
  routingReason: string;
  expectedLatencyMs: number;
  expectedSpread: number;
  fragmentedOrders?: OrderFragment[];
  routingScore: number; // 0-100, higher is better
  decisionTimeMs: number;
}

export interface OrderFragment {
  id: string;
  parentRequestId: string;
  venue: VenueInfo;
  quantity: number;
  priority: number; // 1-5, 1 is highest
}

export interface VenuePerformanceMetrics {
  venueId: string;
  timestamp: Date;
  averageLatencyMs: number;
  averageSpread: number;
  fillRate: number;
  rejectionRate: number;
  slippageAverage: number;
  volumeHandled: number;
}

/**
 * Ultra-fast scalping order router
 * Optimizes order routing for sub-10ms execution
 */
export class ScalpingRouter extends EventEmitter {
  private venues: Map<string, VenueInfo> = new Map();
  private performanceHistory: Map<string, VenuePerformanceMetrics[]> = new Map();
  private routingDecisions: Map<string, RoutingDecision> = new Map();
  private logger: Logger;
  private currentSession: TradingSession;

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this.currentSession = this._getCurrentSession();
    this._initializeDefaultVenues();
    this.logger.info('ScalpingRouter initialized for ultra-fast order routing');
  }

  /**
   * Routes an order to the optimal venue for scalping execution
   */
  public async routeOrder(request: OrderRequest): Promise<RoutingDecision> {
    const startTime = performance.now();
    
    try {
      // Fast venue filtering for scalping requirements
      const eligibleVenues = this._getEligibleVenues(request);
      
      if (eligibleVenues.length === 0) {
        throw new Error(`No eligible venues found for ${request.symbol}`);
      }

      // Score venues based on scalping criteria
      const venueScores = this._scoreVenues(eligibleVenues, request);
      
      // Select best venue
      const selectedVenue = venueScores[0].venue;
      const routingScore = venueScores[0].score;

      // Check if order needs fragmentation
      const fragments = this._shouldFragment(request, selectedVenue) 
        ? this._createOrderFragments(request, eligibleVenues)
        : undefined;

      const decision: RoutingDecision = {
        requestId: request.id,
        selectedVenue,
        routingReason: this._generateRoutingReason(venueScores[0]),
        expectedLatencyMs: selectedVenue.averageLatencyMs,
        expectedSpread: selectedVenue.averageSpread,
        fragmentedOrders: fragments,
        routingScore,
        decisionTimeMs: performance.now() - startTime,
      };

      this.routingDecisions.set(request.id, decision);
      
      this.logger.info(`Order ${request.id} routed to ${selectedVenue.name} in ${decision.decisionTimeMs.toFixed(2)}ms`, {
        symbol: request.symbol,
        quantity: request.quantity,
        urgency: request.urgency,
        score: routingScore,
      });

      this.emit('orderRouted', decision);
      return decision;

    } catch (error: any) {
      this.logger.error(`Routing failed for order ${request.id}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Updates venue performance metrics in real-time
   */
  public updateVenuePerformance(metrics: VenuePerformanceMetrics): void {
    const venue = this.venues.get(metrics.venueId);
    if (!venue) {
      this.logger.warn(`Unknown venue ${metrics.venueId} in performance update`);
      return;
    }

    // Update venue info with latest metrics
    venue.averageLatencyMs = metrics.averageLatencyMs;
    venue.averageSpread = metrics.averageSpread;
    venue.fillRate = metrics.fillRate;

    // Store historical data
    if (!this.performanceHistory.has(metrics.venueId)) {
      this.performanceHistory.set(metrics.venueId, []);
    }
    
    const history = this.performanceHistory.get(metrics.venueId)!;
    history.push(metrics);
    
    // Keep only last 100 metrics for performance
    if (history.length > 100) {
      history.shift();
    }

    this.emit('venuePerformanceUpdated', metrics);
  }

  /**
   * Gets current routing statistics
   */
  public getRoutingStats(): any {
    const stats = {
      totalRoutingDecisions: this.routingDecisions.size,
      averageDecisionTimeMs: 0,
      venueUtilization: new Map<string, number>(),
      currentSession: this.currentSession,
    };

    // Calculate average decision time
    const decisions = Array.from(this.routingDecisions.values());
    if (decisions.length > 0) {
      stats.averageDecisionTimeMs = decisions.reduce((sum, d) => sum + d.decisionTimeMs, 0) / decisions.length;
    }

    // Calculate venue utilization
    for (const decision of decisions) {
      const venueId = decision.selectedVenue.id;
      stats.venueUtilization.set(venueId, (stats.venueUtilization.get(venueId) || 0) + 1);
    }

    return stats;
  }

  // --- Private Methods ---

  private _initializeDefaultVenues(): void {
    const defaultVenues: VenueInfo[] = [
      {
        id: 'ecn-prime-1',
        name: 'ECN Prime Liquidity',
        type: VenueType.ECN,
        averageLatencyMs: 2.5,
        averageSpread: 0.1,
        fillRate: 0.98,
        isActive: true,
        maxOrderSize: 10000000,
        minOrderSize: 1000,
        supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
        sessionPreference: [TradingSession.LONDON, TradingSession.NEW_YORK],
      },
      {
        id: 'mm-fast-1',
        name: 'Market Maker Fast',
        type: VenueType.MARKET_MAKER,
        averageLatencyMs: 1.8,
        averageSpread: 0.2,
        fillRate: 0.95,
        isActive: true,
        maxOrderSize: 5000000,
        minOrderSize: 1000,
        supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY'],
        sessionPreference: [TradingSession.ASIAN, TradingSession.LONDON],
      },
      {
        id: 'stp-ultra-1',
        name: 'STP Ultra Low Latency',
        type: VenueType.STP,
        averageLatencyMs: 1.2,
        averageSpread: 0.15,
        fillRate: 0.92,
        isActive: true,
        maxOrderSize: 2000000,
        minOrderSize: 1000,
        supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        sessionPreference: [TradingSession.NEW_YORK, TradingSession.OVERLAP_LDN_NY],
      },
    ];

    for (const venue of defaultVenues) {
      this.venues.set(venue.id, venue);
    }
  }

  private _getEligibleVenues(request: OrderRequest): VenueInfo[] {
    return Array.from(this.venues.values()).filter(venue => 
      venue.isActive &&
      venue.supportedSymbols.includes(request.symbol) &&
      request.quantity >= venue.minOrderSize &&
      request.quantity <= venue.maxOrderSize
    );
  }

  private _scoreVenues(venues: VenueInfo[], request: OrderRequest): Array<{venue: VenueInfo, score: number}> {
    const scored = venues.map(venue => ({
      venue,
      score: this._calculateVenueScore(venue, request),
    }));

    return scored.sort((a, b) => b.score - a.score);
  }

  private _calculateVenueScore(venue: VenueInfo, request: OrderRequest): number {
    let score = 0;

    // Latency score (40% weight) - critical for scalping
    const latencyScore = Math.max(0, 100 - (venue.averageLatencyMs * 10));
    score += latencyScore * 0.4;

    // Spread score (30% weight) - important for scalping profitability
    const spreadScore = Math.max(0, 100 - (venue.averageSpread * 100));
    score += spreadScore * 0.3;

    // Fill rate score (20% weight)
    score += venue.fillRate * 100 * 0.2;

    // Session preference bonus (10% weight)
    if (venue.sessionPreference.includes(this.currentSession)) {
      score += 10;
    }

    // Urgency bonus for ultra-low latency venues
    if (request.urgency === 'CRITICAL' && venue.averageLatencyMs < 2.0) {
      score += 15;
    }

    return Math.min(100, score);
  }

  private _shouldFragment(request: OrderRequest, venue: VenueInfo): boolean {
    // Fragment large orders or when multiple venues might be beneficial
    return request.quantity > venue.maxOrderSize * 0.5 || request.urgency === 'CRITICAL';
  }

  private _createOrderFragments(request: OrderRequest, venues: VenueInfo[]): OrderFragment[] {
    const fragments: OrderFragment[] = [];
    const topVenues = venues.slice(0, Math.min(3, venues.length)); // Use top 3 venues
    const fragmentSize = Math.floor(request.quantity / topVenues.length);
    let remainingQuantity = request.quantity;

    for (let i = 0; i < topVenues.length && remainingQuantity > 0; i++) {
      const quantity = i === topVenues.length - 1 ? remainingQuantity : fragmentSize;
      
      fragments.push({
        id: uuidv4(),
        parentRequestId: request.id,
        venue: topVenues[i],
        quantity,
        priority: i + 1,
      });

      remainingQuantity -= quantity;
    }

    return fragments;
  }

  private _generateRoutingReason(scoredVenue: {venue: VenueInfo, score: number}): string {
    const venue = scoredVenue.venue;
    return `Selected ${venue.name} (score: ${scoredVenue.score.toFixed(1)}) - ` +
           `Latency: ${venue.averageLatencyMs}ms, Spread: ${venue.averageSpread}, ` +
           `Fill Rate: ${(venue.fillRate * 100).toFixed(1)}%`;
  }

  private _getCurrentSession(): TradingSession {
    const now = new Date();
    const utcHour = now.getUTCHours();

    // Simplified session detection based on UTC hours
    if (utcHour >= 0 && utcHour < 7) return TradingSession.ASIAN;
    if (utcHour >= 7 && utcHour < 12) return TradingSession.LONDON;
    if (utcHour >= 12 && utcHour < 17) return TradingSession.OVERLAP_LDN_NY;
    if (utcHour >= 17 && utcHour < 22) return TradingSession.NEW_YORK;
    return TradingSession.ASIAN;
  }
}
