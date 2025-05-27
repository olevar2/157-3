/**
 * Liquidity Aggregator
 * Multi-venue liquidity aggregation for optimal order execution
 * 
 * This module provides intelligent liquidity aggregation including:
 * - Real-time liquidity pool monitoring across multiple venues
 * - Smart order distribution for best execution
 * - Depth-of-market analysis and optimization
 * - Cross-venue arbitrage detection
 * - Latency-aware venue selection
 * 
 * Expected Benefits:
 * - Access to deeper liquidity pools
 * - Better execution prices through venue competition
 * - Reduced market impact through order distribution
 * - Enhanced fill rates for large orders
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { Logger } from 'winston';

// --- Types ---
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL',
}

export interface LiquidityProvider {
  id: string;
  name: string;
  type: 'ECN' | 'BANK' | 'MARKET_MAKER' | 'RETAIL_AGGREGATOR';
  isActive: boolean;
  latencyMs: number;
  reliability: number; // 0-1
  supportedSymbols: string[];
  minOrderSize: number;
  maxOrderSize: number;
  commission: number; // Per million
}

export interface MarketDepth {
  symbol: string;
  providerId: string;
  bids: PriceLevel[];
  asks: PriceLevel[];
  timestamp: Date;
  totalBidVolume: number;
  totalAskVolume: number;
}

export interface PriceLevel {
  price: number;
  size: number;
  count: number; // Number of orders at this level
}

export interface AggregatedDepth {
  symbol: string;
  bids: AggregatedPriceLevel[];
  asks: AggregatedPriceLevel[];
  bestBid: number;
  bestAsk: number;
  spread: number;
  totalBidVolume: number;
  totalAskVolume: number;
  timestamp: Date;
  providerCount: number;
}

export interface AggregatedPriceLevel extends PriceLevel {
  providers: string[]; // Which providers contribute to this level
  weightedPrice: number; // Volume-weighted price
}

export interface LiquidityAllocation {
  orderId: string;
  symbol: string;
  side: OrderSide;
  totalQuantity: number;
  allocations: VenueAllocation[];
  expectedFillPrice: number;
  expectedFillTime: number;
  totalCost: number;
  priceImprovement: number;
}

export interface VenueAllocation {
  providerId: string;
  quantity: number;
  price: number;
  priority: number; // 1-5, 1 is highest
  estimatedLatency: number;
  commission: number;
}

/**
 * Multi-venue liquidity aggregation engine
 */
export class LiquidityAggregator extends EventEmitter {
  private providers: Map<string, LiquidityProvider> = new Map();
  private marketDepths: Map<string, Map<string, MarketDepth>> = new Map(); // symbol -> providerId -> depth
  private aggregatedDepths: Map<string, AggregatedDepth> = new Map();
  private logger: Logger;
  private updateInterval: NodeJS.Timeout | null = null;

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this._initializeProviders();
    this._startDepthAggregation();
    this.logger.info('LiquidityAggregator initialized for multi-venue execution');
  }

  /**
   * Adds a liquidity provider to the aggregation pool
   */
  public addProvider(provider: LiquidityProvider): void {
    this.providers.set(provider.id, provider);
    this.logger.info(`Added liquidity provider: ${provider.name} (${provider.type})`);
    this.emit('providerAdded', provider);
  }

  /**
   * Updates market depth from a specific provider
   */
  public updateMarketDepth(depth: MarketDepth): void {
    if (!this.marketDepths.has(depth.symbol)) {
      this.marketDepths.set(depth.symbol, new Map());
    }

    this.marketDepths.get(depth.symbol)!.set(depth.providerId, depth);
    this._aggregateDepth(depth.symbol);
    this.emit('depthUpdated', depth);
  }

  /**
   * Allocates an order across multiple liquidity providers for optimal execution
   */
  public allocateLiquidity(
    symbol: string,
    side: OrderSide,
    quantity: number,
    maxPrice?: number
  ): LiquidityAllocation {
    const aggregatedDepth = this.aggregatedDepths.get(symbol);
    if (!aggregatedDepth) {
      throw new Error(`No aggregated depth available for ${symbol}`);
    }

    const levels = side === OrderSide.BUY ? aggregatedDepth.asks : aggregatedDepth.bids;
    const allocations: VenueAllocation[] = [];
    let remainingQuantity = quantity;
    let totalCost = 0;
    let weightedPriceSum = 0;

    // Sort levels by price (best first)
    const sortedLevels = [...levels].sort((a, b) => 
      side === OrderSide.BUY ? a.price - b.price : b.price - a.price
    );

    for (const level of sortedLevels) {
      if (remainingQuantity <= 0) break;
      if (maxPrice && ((side === OrderSide.BUY && level.price > maxPrice) || 
                       (side === OrderSide.SELL && level.price < maxPrice))) {
        break;
      }

      const availableQuantity = Math.min(level.size, remainingQuantity);
      
      // Distribute quantity among providers at this level
      const providerAllocations = this._distributeAmongProviders(
        level.providers,
        availableQuantity,
        level.price,
        symbol
      );

      allocations.push(...providerAllocations);
      
      const levelCost = availableQuantity * level.price;
      totalCost += levelCost;
      weightedPriceSum += levelCost;
      remainingQuantity -= availableQuantity;
    }

    if (remainingQuantity > 0) {
      this.logger.warn(`Could not allocate full quantity for ${symbol}. Remaining: ${remainingQuantity}`);
    }

    const expectedFillPrice = quantity > 0 ? weightedPriceSum / (quantity - remainingQuantity) : 0;
    const bestPrice = side === OrderSide.BUY ? aggregatedDepth.bestAsk : aggregatedDepth.bestBid;
    const priceImprovement = Math.abs(expectedFillPrice - bestPrice);

    const allocation: LiquidityAllocation = {
      orderId: uuidv4(),
      symbol,
      side,
      totalQuantity: quantity,
      allocations,
      expectedFillPrice,
      expectedFillTime: this._calculateExpectedFillTime(allocations),
      totalCost,
      priceImprovement,
    };

    this.logger.info(`Allocated liquidity for ${symbol}`, {
      quantity,
      allocations: allocations.length,
      expectedPrice: expectedFillPrice,
      priceImprovement,
    });

    this.emit('liquidityAllocated', allocation);
    return allocation;
  }

  /**
   * Gets current aggregated market depth for a symbol
   */
  public getAggregatedDepth(symbol: string): AggregatedDepth | undefined {
    return this.aggregatedDepths.get(symbol);
  }

  /**
   * Gets liquidity statistics across all providers
   */
  public getLiquidityStats(): any {
    const stats = {
      totalProviders: this.providers.size,
      activeProviders: 0,
      symbolsCovered: new Set<string>(),
      averageLatency: 0,
      totalVolume: 0,
    };

    let latencySum = 0;
    for (const provider of this.providers.values()) {
      if (provider.isActive) {
        stats.activeProviders++;
        latencySum += provider.latencyMs;
        provider.supportedSymbols.forEach(symbol => stats.symbolsCovered.add(symbol));
      }
    }

    stats.averageLatency = stats.activeProviders > 0 ? latencySum / stats.activeProviders : 0;

    // Calculate total volume from aggregated depths
    for (const depth of this.aggregatedDepths.values()) {
      stats.totalVolume += depth.totalBidVolume + depth.totalAskVolume;
    }

    return {
      ...stats,
      symbolsCovered: stats.symbolsCovered.size,
    };
  }

  // --- Private Methods ---

  private _initializeProviders(): void {
    const defaultProviders: LiquidityProvider[] = [
      {
        id: 'ecn-prime',
        name: 'ECN Prime',
        type: 'ECN',
        isActive: true,
        latencyMs: 2.1,
        reliability: 0.99,
        supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
        minOrderSize: 10000,
        maxOrderSize: 100000000,
        commission: 0.1,
      },
      {
        id: 'bank-tier1',
        name: 'Tier 1 Bank',
        type: 'BANK',
        isActive: true,
        latencyMs: 3.5,
        reliability: 0.98,
        supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        minOrderSize: 50000,
        maxOrderSize: 500000000,
        commission: 0.05,
      },
      {
        id: 'mm-fast',
        name: 'Fast Market Maker',
        type: 'MARKET_MAKER',
        isActive: true,
        latencyMs: 1.8,
        reliability: 0.95,
        supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY'],
        minOrderSize: 1000,
        maxOrderSize: 10000000,
        commission: 0.2,
      },
    ];

    for (const provider of defaultProviders) {
      this.providers.set(provider.id, provider);
    }
  }

  private _startDepthAggregation(): void {
    this.updateInterval = setInterval(() => {
      for (const symbol of this.marketDepths.keys()) {
        this._aggregateDepth(symbol);
      }
    }, 100); // Update every 100ms
  }

  private _aggregateDepth(symbol: string): void {
    const providerDepths = this.marketDepths.get(symbol);
    if (!providerDepths || providerDepths.size === 0) return;

    const allBids: Map<number, AggregatedPriceLevel> = new Map();
    const allAsks: Map<number, AggregatedPriceLevel> = new Map();

    // Aggregate all price levels from all providers
    for (const [providerId, depth] of providerDepths) {
      const provider = this.providers.get(providerId);
      if (!provider || !provider.isActive) continue;

      // Process bids
      for (const bid of depth.bids) {
        const existing = allBids.get(bid.price);
        if (existing) {
          existing.size += bid.size;
          existing.count += bid.count;
          existing.providers.push(providerId);
          existing.weightedPrice = (existing.weightedPrice * existing.size + bid.price * bid.size) / 
                                  (existing.size + bid.size);
        } else {
          allBids.set(bid.price, {
            price: bid.price,
            size: bid.size,
            count: bid.count,
            providers: [providerId],
            weightedPrice: bid.price,
          });
        }
      }

      // Process asks
      for (const ask of depth.asks) {
        const existing = allAsks.get(ask.price);
        if (existing) {
          existing.size += ask.size;
          existing.count += ask.count;
          existing.providers.push(providerId);
          existing.weightedPrice = (existing.weightedPrice * existing.size + ask.price * ask.size) / 
                                  (existing.size + ask.size);
        } else {
          allAsks.set(ask.price, {
            price: ask.price,
            size: ask.size,
            count: ask.count,
            providers: [providerId],
            weightedPrice: ask.price,
          });
        }
      }
    }

    // Sort and create aggregated depth
    const sortedBids = Array.from(allBids.values()).sort((a, b) => b.price - a.price);
    const sortedAsks = Array.from(allAsks.values()).sort((a, b) => a.price - b.price);

    const bestBid = sortedBids.length > 0 ? sortedBids[0].price : 0;
    const bestAsk = sortedAsks.length > 0 ? sortedAsks[0].price : 0;

    const aggregated: AggregatedDepth = {
      symbol,
      bids: sortedBids,
      asks: sortedAsks,
      bestBid,
      bestAsk,
      spread: bestAsk - bestBid,
      totalBidVolume: sortedBids.reduce((sum, bid) => sum + bid.size, 0),
      totalAskVolume: sortedAsks.reduce((sum, ask) => sum + ask.size, 0),
      timestamp: new Date(),
      providerCount: providerDepths.size,
    };

    this.aggregatedDepths.set(symbol, aggregated);
    this.emit('depthAggregated', aggregated);
  }

  private _distributeAmongProviders(
    providerIds: string[],
    quantity: number,
    price: number,
    symbol: string
  ): VenueAllocation[] {
    const allocations: VenueAllocation[] = [];
    const activeProviders = providerIds
      .map(id => this.providers.get(id))
      .filter(p => p && p.isActive && p.supportedSymbols.includes(symbol))
      .sort((a, b) => (a!.latencyMs * a!.commission) - (b!.latencyMs * b!.commission)); // Sort by cost

    if (activeProviders.length === 0) return allocations;

    const quantityPerProvider = Math.floor(quantity / activeProviders.length);
    let remainingQuantity = quantity;

    for (let i = 0; i < activeProviders.length && remainingQuantity > 0; i++) {
      const provider = activeProviders[i]!;
      const allocationQuantity = i === activeProviders.length - 1 ? 
        remainingQuantity : Math.min(quantityPerProvider, remainingQuantity);

      allocations.push({
        providerId: provider.id,
        quantity: allocationQuantity,
        price,
        priority: i + 1,
        estimatedLatency: provider.latencyMs,
        commission: (allocationQuantity * price * provider.commission) / 1000000,
      });

      remainingQuantity -= allocationQuantity;
    }

    return allocations;
  }

  private _calculateExpectedFillTime(allocations: VenueAllocation[]): number {
    if (allocations.length === 0) return 0;
    
    // Return the maximum latency among all allocations (parallel execution)
    return Math.max(...allocations.map(a => a.estimatedLatency));
  }

  public destroy(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}
