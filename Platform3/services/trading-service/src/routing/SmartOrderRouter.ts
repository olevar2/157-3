/**
 * Smart Order Router
 * Intelligent order routing for optimal execution and minimal slippage
 * 
 * This module provides smart order routing including:
 * - Multi-venue price discovery and comparison
 * - Liquidity aggregation across brokers/exchanges
 * - Slippage minimization algorithms
 * - Execution cost optimization
 * - Real-time venue selection
 * 
 * Expected Benefits:
 * - Minimal slippage on rapid entries/exits
 * - Optimal price discovery for short-term trades
 * - Reduced execution costs
 * - Enhanced fill rates and execution quality
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { Logger } from 'winston';

export interface VenueQuote {
  venueId: string;
  venueName: string;
  symbol: string;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  spread: number;
  timestamp: Date;
  latency: number; // milliseconds
  reliability: number; // 0-1 score
  executionCost: number; // estimated cost per unit
}

export interface RoutingRequest {
  id: string;
  userId: string;
  accountId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  orderType: 'market' | 'limit' | 'stop';
  limitPrice?: number;
  stopPrice?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  maxSlippage: number; // percentage
  urgency: 'immediate' | 'normal' | 'patient';
  metadata?: Record<string, any>;
}

export interface RoutingDecision {
  requestId: string;
  selectedVenue: string;
  executionStrategy: 'single' | 'split' | 'iceberg';
  orderFragments: OrderFragment[];
  estimatedSlippage: number;
  estimatedCost: number;
  estimatedFillTime: number; // milliseconds
  confidence: number; // 0-1
  reasoning: string[];
}

export interface OrderFragment {
  venueId: string;
  quantity: number;
  price?: number;
  priority: number;
  estimatedFillTime: number;
}

export interface ExecutionResult {
  requestId: string;
  fragmentId: string;
  venueId: string;
  filledQuantity: number;
  fillPrice: number;
  commission: number;
  slippage: number;
  executionTime: number;
  timestamp: Date;
}

export class SmartOrderRouter extends EventEmitter {
  private logger: Logger;
  private venues: Map<string, VenueQuote> = new Map();
  private routingHistory: Map<string, RoutingDecision> = new Map();
  private performanceMetrics: Map<string, any> = new Map();

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this.logger.info('Smart Order Router initialized');
  }

  /**
   * Route an order to the optimal venue(s)
   */
  async routeOrder(request: RoutingRequest): Promise<RoutingDecision> {
    const startTime = performance.now();

    try {
      // Get current market data from all venues
      const quotes = await this.gatherVenueQuotes(request.symbol);

      // Analyze routing options
      const routingOptions = await this.analyzeRoutingOptions(request, quotes);

      // Select optimal routing strategy
      const decision = await this.selectOptimalRouting(request, routingOptions);

      // Store decision for performance tracking
      this.routingHistory.set(request.id, decision);

      const routingTime = performance.now() - startTime;
      this.logger.info(`Order routed in ${routingTime.toFixed(2)}ms: ${request.id} -> ${decision.selectedVenue}`);

      this.emit('orderRouted', { request, decision, routingTime });

      return decision;

    } catch (error) {
      this.logger.error(`Error routing order ${request.id}:`, error);
      this.emit('routingError', { request, error });
      throw error;
    }
  }

  /**
   * Update venue quote
   */
  updateVenueQuote(quote: VenueQuote): void {
    this.venues.set(quote.venueId, quote);
    this.emit('quoteUpdated', quote);
  }

  /**
   * Record execution result for performance tracking
   */
  recordExecutionResult(result: ExecutionResult): void {
    const decision = this.routingHistory.get(result.requestId);
    if (decision) {
      this.updatePerformanceMetrics(decision, result);
    }

    this.emit('executionRecorded', result);
  }

  /**
   * Get venue performance metrics
   */
  getVenuePerformance(venueId: string): any {
    return this.performanceMetrics.get(venueId) || {
      totalOrders: 0,
      avgSlippage: 0,
      avgExecutionTime: 0,
      fillRate: 0,
      reliability: 0
    };
  }

  /**
   * Gather quotes from all available venues
   */
  private async gatherVenueQuotes(symbol: string): Promise<VenueQuote[]> {
    const quotes: VenueQuote[] = [];
    const currentTime = new Date();

    // Filter recent quotes (within last 1 second)
    for (const [venueId, quote] of this.venues) {
      if (quote.symbol === symbol && 
          (currentTime.getTime() - quote.timestamp.getTime()) < 1000) {
        quotes.push(quote);
      }
    }

    // Sort by best execution potential
    quotes.sort((a, b) => {
      const scoreA = this.calculateVenueScore(a);
      const scoreB = this.calculateVenueScore(b);
      return scoreB - scoreA;
    });

    return quotes;
  }

  /**
   * Calculate venue score for ranking
   */
  private calculateVenueScore(quote: VenueQuote): number {
    const spreadScore = 1 / (1 + quote.spread);
    const latencyScore = 1 / (1 + quote.latency / 100);
    const reliabilityScore = quote.reliability;
    const costScore = 1 / (1 + quote.executionCost);

    return (spreadScore * 0.3) + (latencyScore * 0.3) + (reliabilityScore * 0.2) + (costScore * 0.2);
  }

  /**
   * Analyze routing options
   */
  private async analyzeRoutingOptions(
    request: RoutingRequest, 
    quotes: VenueQuote[]
  ): Promise<any[]> {
    const options = [];

    for (const quote of quotes) {
      // Single venue execution
      const singleVenueOption = await this.analyzeSingleVenue(request, quote);
      if (singleVenueOption) {
        options.push(singleVenueOption);
      }

      // Split execution (for large orders)
      if (request.quantity > quote.bidSize || request.quantity > quote.askSize) {
        const splitOption = await this.analyzeSplitExecution(request, quotes);
        if (splitOption) {
          options.push(splitOption);
        }
      }
    }

    return options;
  }

  /**
   * Analyze single venue execution
   */
  private async analyzeSingleVenue(request: RoutingRequest, quote: VenueQuote): Promise<any> {
    const price = request.side === 'buy' ? quote.ask : quote.bid;
    const availableSize = request.side === 'buy' ? quote.askSize : quote.bidSize;

    if (request.quantity > availableSize) {
      return null; // Not enough liquidity
    }

    // Calculate estimated slippage
    const estimatedSlippage = this.calculateEstimatedSlippage(request, quote);
    
    if (estimatedSlippage > request.maxSlippage) {
      return null; // Exceeds max slippage
    }

    return {
      type: 'single',
      venueId: quote.venueId,
      price,
      estimatedSlippage,
      estimatedCost: quote.executionCost * request.quantity,
      estimatedFillTime: quote.latency + 50, // Base execution time
      confidence: quote.reliability
    };
  }

  /**
   * Analyze split execution across multiple venues
   */
  private async analyzeSplitExecution(request: RoutingRequest, quotes: VenueQuote[]): Promise<any> {
    const fragments: OrderFragment[] = [];
    let remainingQuantity = request.quantity;
    let totalCost = 0;
    let maxFillTime = 0;

    for (const quote of quotes) {
      if (remainingQuantity <= 0) break;

      const availableSize = request.side === 'buy' ? quote.askSize : quote.bidSize;
      const fragmentSize = Math.min(remainingQuantity, availableSize);

      if (fragmentSize > 0) {
        fragments.push({
          venueId: quote.venueId,
          quantity: fragmentSize,
          price: request.side === 'buy' ? quote.ask : quote.bid,
          priority: fragments.length + 1,
          estimatedFillTime: quote.latency + 50
        });

        remainingQuantity -= fragmentSize;
        totalCost += quote.executionCost * fragmentSize;
        maxFillTime = Math.max(maxFillTime, quote.latency + 50);
      }
    }

    if (remainingQuantity > 0) {
      return null; // Cannot fill entire order
    }

    return {
      type: 'split',
      fragments,
      estimatedSlippage: this.calculateSplitSlippage(request, fragments),
      estimatedCost: totalCost,
      estimatedFillTime: maxFillTime,
      confidence: 0.8 // Lower confidence for split orders
    };
  }

  /**
   * Select optimal routing strategy
   */
  private async selectOptimalRouting(
    request: RoutingRequest, 
    options: any[]
  ): Promise<RoutingDecision> {
    if (options.length === 0) {
      throw new Error('No viable routing options found');
    }

    // Score options based on request urgency and preferences
    const scoredOptions = options.map(option => ({
      ...option,
      score: this.calculateOptionScore(option, request)
    }));

    // Select best option
    const bestOption = scoredOptions.reduce((best, current) => 
      current.score > best.score ? current : best
    );

    // Create routing decision
    const decision: RoutingDecision = {
      requestId: request.id,
      selectedVenue: bestOption.venueId || 'multiple',
      executionStrategy: bestOption.type,
      orderFragments: bestOption.fragments || [{
        venueId: bestOption.venueId,
        quantity: request.quantity,
        price: bestOption.price,
        priority: 1,
        estimatedFillTime: bestOption.estimatedFillTime
      }],
      estimatedSlippage: bestOption.estimatedSlippage,
      estimatedCost: bestOption.estimatedCost,
      estimatedFillTime: bestOption.estimatedFillTime,
      confidence: bestOption.confidence,
      reasoning: this.generateReasoning(bestOption, request)
    };

    return decision;
  }

  /**
   * Calculate option score based on request preferences
   */
  private calculateOptionScore(option: any, request: RoutingRequest): number {
    let score = 0;

    // Slippage weight (higher for urgent orders)
    const slippageWeight = request.urgency === 'immediate' ? 0.4 : 0.3;
    score += (1 - option.estimatedSlippage / request.maxSlippage) * slippageWeight;

    // Cost weight
    const costWeight = 0.2;
    score += (1 / (1 + option.estimatedCost / 1000)) * costWeight;

    // Speed weight (higher for urgent orders)
    const speedWeight = request.urgency === 'immediate' ? 0.3 : 0.2;
    score += (1 / (1 + option.estimatedFillTime / 1000)) * speedWeight;

    // Confidence weight
    const confidenceWeight = 0.3;
    score += option.confidence * confidenceWeight;

    return score;
  }

  /**
   * Calculate estimated slippage
   */
  private calculateEstimatedSlippage(request: RoutingRequest, quote: VenueQuote): number {
    // Simplified slippage calculation
    const marketImpact = request.quantity / (quote.bidSize + quote.askSize);
    const baseSlippage = quote.spread / 2;
    
    return (baseSlippage + (marketImpact * quote.spread)) / 
           (request.side === 'buy' ? quote.ask : quote.bid) * 100;
  }

  /**
   * Calculate slippage for split execution
   */
  private calculateSplitSlippage(request: RoutingRequest, fragments: OrderFragment[]): number {
    // Weighted average slippage across fragments
    let totalSlippage = 0;
    let totalQuantity = 0;

    for (const fragment of fragments) {
      const fragmentSlippage = 0.01; // Simplified calculation
      totalSlippage += fragmentSlippage * fragment.quantity;
      totalQuantity += fragment.quantity;
    }

    return totalQuantity > 0 ? totalSlippage / totalQuantity : 0;
  }

  /**
   * Generate reasoning for routing decision
   */
  private generateReasoning(option: any, request: RoutingRequest): string[] {
    const reasons = [];

    if (option.type === 'single') {
      reasons.push(`Single venue execution for optimal speed`);
      reasons.push(`Low slippage: ${option.estimatedSlippage.toFixed(3)}%`);
    } else {
      reasons.push(`Split execution across ${option.fragments.length} venues`);
      reasons.push(`Aggregated liquidity for large order`);
    }

    if (request.urgency === 'immediate') {
      reasons.push(`Prioritized speed for immediate execution`);
    }

    return reasons;
  }

  /**
   * Update performance metrics
   */
  private updatePerformanceMetrics(decision: RoutingDecision, result: ExecutionResult): void {
    const venueId = result.venueId;
    const metrics = this.performanceMetrics.get(venueId) || {
      totalOrders: 0,
      totalSlippage: 0,
      totalExecutionTime: 0,
      successfulFills: 0
    };

    metrics.totalOrders++;
    metrics.totalSlippage += result.slippage;
    metrics.totalExecutionTime += result.executionTime;
    metrics.successfulFills++;

    metrics.avgSlippage = metrics.totalSlippage / metrics.totalOrders;
    metrics.avgExecutionTime = metrics.totalExecutionTime / metrics.totalOrders;
    metrics.fillRate = metrics.successfulFills / metrics.totalOrders;

    this.performanceMetrics.set(venueId, metrics);
  }
}
