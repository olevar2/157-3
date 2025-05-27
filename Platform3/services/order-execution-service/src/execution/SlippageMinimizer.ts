/**
 * Slippage Minimizer
 * Advanced slippage reduction algorithms for optimal trade execution
 * 
 * This module provides intelligent slippage minimization including:
 * - Real-time market impact analysis
 * - Adaptive order sizing based on liquidity
 * - Time-weighted average price (TWAP) execution
 * - Volume-weighted average price (VWAP) strategies
 * - Smart order timing based on market microstructure
 * 
 * Expected Benefits:
 * - Reduced execution costs through slippage minimization
 * - Improved fill prices for large orders
 * - Market impact awareness and mitigation
 * - Enhanced profitability for scalping and day trading
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';

// --- Types ---
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL',
}

export interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  bidSize: number;
  askSize: number;
  lastPrice: number;
  volume: number;
  timestamp: Date;
}

export interface OrderExecution {
  id: string;
  symbol: string;
  side: OrderSide;
  requestedQuantity: number;
  requestedPrice?: number;
  executedQuantity: number;
  executedPrice: number;
  slippage: number; // In pips
  slippagePercent: number;
  marketImpact: number;
  timestamp: Date;
}

export interface SlippageAnalysis {
  symbol: string;
  averageSlippage: number;
  maxSlippage: number;
  minSlippage: number;
  slippageStdDev: number;
  marketImpactFactor: number;
  liquidityScore: number; // 0-100
  recommendedMaxOrderSize: number;
  optimalExecutionTime: Date;
}

export interface ExecutionStrategy {
  name: string;
  description: string;
  maxOrderSize: number;
  timeSlices: number;
  delayBetweenSlicesMs: number;
  adaptToVolume: boolean;
  respectSpread: boolean;
}

/**
 * Advanced slippage minimization engine
 */
export class SlippageMinimizer extends EventEmitter {
  private marketData: Map<string, MarketData> = new Map();
  private executionHistory: Map<string, OrderExecution[]> = new Map();
  private slippageAnalysis: Map<string, SlippageAnalysis> = new Map();
  private logger: Logger;

  // Predefined execution strategies
  private strategies: Map<string, ExecutionStrategy> = new Map();

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this._initializeStrategies();
    this.logger.info('SlippageMinimizer initialized for optimal execution');
  }

  /**
   * Analyzes potential slippage for an order
   */
  public analyzeSlippage(symbol: string, side: OrderSide, quantity: number): SlippageAnalysis {
    const marketData = this.marketData.get(symbol);
    if (!marketData) {
      throw new Error(`No market data available for ${symbol}`);
    }

    const history = this.executionHistory.get(symbol) || [];
    const recentExecutions = history.slice(-50); // Last 50 executions

    // Calculate historical slippage statistics
    const slippages = recentExecutions.map(exec => exec.slippage);
    const averageSlippage = slippages.length > 0 ? slippages.reduce((a, b) => a + b, 0) / slippages.length : 0;
    const maxSlippage = slippages.length > 0 ? Math.max(...slippages) : 0;
    const minSlippage = slippages.length > 0 ? Math.min(...slippages) : 0;
    const slippageStdDev = this._calculateStandardDeviation(slippages);

    // Calculate market impact factor
    const marketImpactFactor = this._calculateMarketImpact(marketData, quantity, side);

    // Calculate liquidity score
    const liquidityScore = this._calculateLiquidityScore(marketData);

    // Recommend optimal order size
    const recommendedMaxOrderSize = this._calculateOptimalOrderSize(marketData, marketImpactFactor);

    // Determine optimal execution time
    const optimalExecutionTime = this._calculateOptimalExecutionTime(marketData);

    const analysis: SlippageAnalysis = {
      symbol,
      averageSlippage,
      maxSlippage,
      minSlippage,
      slippageStdDev,
      marketImpactFactor,
      liquidityScore,
      recommendedMaxOrderSize,
      optimalExecutionTime,
    };

    this.slippageAnalysis.set(symbol, analysis);
    this.emit('slippageAnalyzed', analysis);

    return analysis;
  }

  /**
   * Recommends optimal execution strategy
   */
  public recommendStrategy(symbol: string, quantity: number, urgency: 'LOW' | 'MEDIUM' | 'HIGH'): ExecutionStrategy {
    const analysis = this.slippageAnalysis.get(symbol);
    if (!analysis) {
      throw new Error(`No slippage analysis available for ${symbol}. Run analyzeSlippage first.`);
    }

    let strategyName: string;

    if (urgency === 'HIGH' || quantity <= analysis.recommendedMaxOrderSize) {
      strategyName = 'IMMEDIATE';
    } else if (analysis.liquidityScore > 70) {
      strategyName = 'TWAP';
    } else {
      strategyName = 'VWAP';
    }

    const strategy = this.strategies.get(strategyName);
    if (!strategy) {
      throw new Error(`Strategy ${strategyName} not found`);
    }

    this.logger.info(`Recommended ${strategyName} strategy for ${symbol}`, {
      quantity,
      urgency,
      liquidityScore: analysis.liquidityScore,
      marketImpact: analysis.marketImpactFactor,
    });

    return strategy;
  }

  /**
   * Updates market data for slippage calculations
   */
  public updateMarketData(data: MarketData): void {
    this.marketData.set(data.symbol, data);
    this.emit('marketDataUpdated', data);
  }

  /**
   * Records execution for slippage analysis
   */
  public recordExecution(execution: OrderExecution): void {
    if (!this.executionHistory.has(execution.symbol)) {
      this.executionHistory.set(execution.symbol, []);
    }

    const history = this.executionHistory.get(execution.symbol)!;
    history.push(execution);

    // Keep only last 200 executions per symbol
    if (history.length > 200) {
      history.shift();
    }

    this.logger.debug(`Recorded execution for ${execution.symbol}`, {
      slippage: execution.slippage,
      marketImpact: execution.marketImpact,
      quantity: execution.executedQuantity,
    });

    this.emit('executionRecorded', execution);
  }

  /**
   * Gets slippage statistics for a symbol
   */
  public getSlippageStats(symbol: string): any {
    const history = this.executionHistory.get(symbol) || [];
    const analysis = this.slippageAnalysis.get(symbol);

    return {
      symbol,
      totalExecutions: history.length,
      analysis,
      recentExecutions: history.slice(-10),
    };
  }

  // --- Private Methods ---

  private _initializeStrategies(): void {
    this.strategies.set('IMMEDIATE', {
      name: 'IMMEDIATE',
      description: 'Execute immediately for urgent orders',
      maxOrderSize: 1000000,
      timeSlices: 1,
      delayBetweenSlicesMs: 0,
      adaptToVolume: false,
      respectSpread: true,
    });

    this.strategies.set('TWAP', {
      name: 'TWAP',
      description: 'Time-weighted average price execution',
      maxOrderSize: 5000000,
      timeSlices: 10,
      delayBetweenSlicesMs: 30000, // 30 seconds
      adaptToVolume: false,
      respectSpread: true,
    });

    this.strategies.set('VWAP', {
      name: 'VWAP',
      description: 'Volume-weighted average price execution',
      maxOrderSize: 10000000,
      timeSlices: 20,
      delayBetweenSlicesMs: 60000, // 1 minute
      adaptToVolume: true,
      respectSpread: true,
    });

    this.strategies.set('ICEBERG', {
      name: 'ICEBERG',
      description: 'Hide large orders with small visible quantities',
      maxOrderSize: 50000000,
      timeSlices: 50,
      delayBetweenSlicesMs: 5000, // 5 seconds
      adaptToVolume: true,
      respectSpread: false,
    });
  }

  private _calculateMarketImpact(marketData: MarketData, quantity: number, side: OrderSide): number {
    const relevantSize = side === OrderSide.BUY ? marketData.askSize : marketData.bidSize;
    const impactRatio = quantity / relevantSize;
    
    // Simple market impact model - can be enhanced with more sophisticated algorithms
    let impact = 0;
    if (impactRatio > 1) {
      impact = Math.log(impactRatio) * marketData.spread * 0.5;
    } else {
      impact = impactRatio * marketData.spread * 0.1;
    }

    return Math.max(0, impact);
  }

  private _calculateLiquidityScore(marketData: MarketData): number {
    const totalSize = marketData.bidSize + marketData.askSize;
    const spreadTightness = 1 / (marketData.spread + 0.0001); // Avoid division by zero
    const volumeScore = Math.min(100, marketData.volume / 1000000 * 50); // Normalize volume

    // Combine factors for liquidity score
    const sizeScore = Math.min(100, totalSize / 10000 * 50);
    const spreadScore = Math.min(100, spreadTightness * 10);

    return (sizeScore * 0.4 + spreadScore * 0.4 + volumeScore * 0.2);
  }

  private _calculateOptimalOrderSize(marketData: MarketData, marketImpact: number): number {
    const baseSize = Math.min(marketData.bidSize, marketData.askSize) * 0.1;
    const impactAdjustment = Math.max(0.1, 1 - marketImpact);
    
    return Math.floor(baseSize * impactAdjustment);
  }

  private _calculateOptimalExecutionTime(marketData: MarketData): Date {
    // Simple heuristic - execute when spread is tight and volume is high
    const now = new Date();
    const delayMs = marketData.spread > 0.0002 ? 30000 : 5000; // Wait if spread is wide
    
    return new Date(now.getTime() + delayMs);
  }

  private _calculateStandardDeviation(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    
    return Math.sqrt(avgSquaredDiff);
  }
}
