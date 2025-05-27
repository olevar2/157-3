/**
 * Volatility Adjusted Risk Manager
 * Dynamic risk management based on real-time market volatility
 * 
 * This module provides volatility-aware risk management including:
 * - Real-time volatility calculation and monitoring
 * - Dynamic position sizing based on volatility regimes
 * - Adaptive stop-loss and take-profit levels
 * - Volatility-based exposure limits
 * - Risk scaling for different market conditions
 * 
 * Expected Benefits:
 * - Reduced risk during high volatility periods
 * - Optimized position sizes for market conditions
 * - Dynamic risk parameters that adapt to market changes
 * - Enhanced capital preservation during volatile markets
 */

import { EventEmitter } from 'events';
import { Logger } from 'winston';

// --- Types ---
export enum VolatilityRegime {
  VERY_LOW = 'VERY_LOW',
  LOW = 'LOW',
  NORMAL = 'NORMAL',
  HIGH = 'HIGH',
  EXTREME = 'EXTREME',
}

export interface VolatilityMetrics {
  symbol: string;
  currentVolatility: number;
  averageVolatility: number;
  volatilityPercentile: number;
  regime: VolatilityRegime;
  atr: number; // Average True Range
  garchVolatility: number;
  realizedVolatility: number;
  impliedVolatility?: number;
  timestamp: Date;
}

export interface VolatilityAdjustment {
  symbol: string;
  originalRisk: RiskParameters;
  adjustedRisk: RiskParameters;
  adjustmentFactor: number;
  regime: VolatilityRegime;
  reason: string;
  timestamp: Date;
}

export interface RiskParameters {
  maxPositionSize: number;
  stopLossDistance: number;
  takeProfitDistance: number;
  maxExposure: number;
  riskPerTrade: number;
  leverageMultiplier: number;
}

export interface VolatilityConfig {
  lookbackPeriods: number;
  atrPeriods: number;
  garchAlpha: number;
  garchBeta: number;
  regimeThresholds: {
    veryLow: number;
    low: number;
    normal: number;
    high: number;
    extreme: number;
  };
  adjustmentFactors: Record<VolatilityRegime, number>;
}

export interface PriceData {
  symbol: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Volatility-based risk adjustment engine
 */
export class VolatilityAdjustedRisk extends EventEmitter {
  private config: VolatilityConfig;
  private priceHistory: Map<string, PriceData[]> = new Map();
  private volatilityMetrics: Map<string, VolatilityMetrics> = new Map();
  private riskAdjustments: Map<string, VolatilityAdjustment> = new Map();
  private logger: Logger;

  constructor(config: VolatilityConfig, logger: Logger) {
    super();
    this.config = config;
    this.logger = logger;
    this.logger.info('VolatilityAdjustedRisk initialized for dynamic risk management');
  }

  /**
   * Updates price data and recalculates volatility metrics
   */
  public updatePriceData(priceData: PriceData): void {
    if (!this.priceHistory.has(priceData.symbol)) {
      this.priceHistory.set(priceData.symbol, []);
    }

    const history = this.priceHistory.get(priceData.symbol)!;
    history.push(priceData);

    // Keep only required lookback periods
    if (history.length > this.config.lookbackPeriods) {
      history.shift();
    }

    // Recalculate volatility metrics
    if (history.length >= Math.min(20, this.config.lookbackPeriods)) {
      const metrics = this.calculateVolatilityMetrics(priceData.symbol, history);
      this.volatilityMetrics.set(priceData.symbol, metrics);
      this.emit('volatilityUpdated', metrics);
    }
  }

  /**
   * Adjusts risk parameters based on current volatility regime
   */
  public adjustRiskForVolatility(symbol: string, baseRisk: RiskParameters): VolatilityAdjustment {
    const metrics = this.volatilityMetrics.get(symbol);
    if (!metrics) {
      throw new Error(`No volatility metrics available for ${symbol}`);
    }

    const adjustmentFactor = this.config.adjustmentFactors[metrics.regime];
    const adjustedRisk = this.applyVolatilityAdjustment(baseRisk, adjustmentFactor, metrics);

    const adjustment: VolatilityAdjustment = {
      symbol,
      originalRisk: { ...baseRisk },
      adjustedRisk,
      adjustmentFactor,
      regime: metrics.regime,
      reason: this.generateAdjustmentReason(metrics),
      timestamp: new Date(),
    };

    this.riskAdjustments.set(symbol, adjustment);
    this.emit('riskAdjusted', adjustment);

    this.logger.info(`Risk adjusted for ${symbol}`, {
      regime: metrics.regime,
      adjustmentFactor,
      volatility: metrics.currentVolatility,
    });

    return adjustment;
  }

  /**
   * Gets current volatility metrics for a symbol
   */
  public getVolatilityMetrics(symbol: string): VolatilityMetrics | undefined {
    return this.volatilityMetrics.get(symbol);
  }

  /**
   * Gets current risk adjustment for a symbol
   */
  public getRiskAdjustment(symbol: string): VolatilityAdjustment | undefined {
    return this.riskAdjustments.get(symbol);
  }

  /**
   * Calculates optimal position size based on volatility
   */
  public calculateVolatilityAdjustedPositionSize(
    symbol: string,
    accountBalance: number,
    riskPercentage: number
  ): number {
    const metrics = this.volatilityMetrics.get(symbol);
    if (!metrics) {
      throw new Error(`No volatility metrics available for ${symbol}`);
    }

    const baseRiskAmount = accountBalance * (riskPercentage / 100);
    const volatilityAdjustment = this.config.adjustmentFactors[metrics.regime];
    const adjustedRiskAmount = baseRiskAmount * volatilityAdjustment;

    // Calculate position size based on ATR for stop loss
    const stopLossDistance = metrics.atr * 2; // 2x ATR stop loss
    const positionSize = adjustedRiskAmount / stopLossDistance;

    return Math.max(0, positionSize);
  }

  // --- Private Methods ---

  private calculateVolatilityMetrics(symbol: string, history: PriceData[]): VolatilityMetrics {
    const returns = this.calculateReturns(history);
    const currentVolatility = this.calculateRealizedVolatility(returns);
    const atr = this.calculateATR(history);
    const garchVolatility = this.calculateGARCHVolatility(returns);

    // Calculate historical volatility percentiles
    const historicalVolatilities = this.calculateRollingVolatilities(returns);
    const volatilityPercentile = this.calculatePercentile(historicalVolatilities, currentVolatility);

    const regime = this.determineVolatilityRegime(currentVolatility);
    const averageVolatility = historicalVolatilities.reduce((a, b) => a + b, 0) / historicalVolatilities.length;

    return {
      symbol,
      currentVolatility,
      averageVolatility,
      volatilityPercentile,
      regime,
      atr,
      garchVolatility,
      realizedVolatility: currentVolatility,
      timestamp: new Date(),
    };
  }

  private calculateReturns(history: PriceData[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < history.length; i++) {
      const return_ = Math.log(history[i].close / history[i - 1].close);
      returns.push(return_);
    }
    return returns;
  }

  private calculateRealizedVolatility(returns: number[]): number {
    if (returns.length === 0) return 0;

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    // Annualized volatility (assuming daily data)
    return Math.sqrt(variance * 252);
  }

  private calculateATR(history: PriceData[]): number {
    if (history.length < 2) return 0;

    const trueRanges: number[] = [];
    for (let i = 1; i < history.length; i++) {
      const current = history[i];
      const previous = history[i - 1];
      
      const tr1 = current.high - current.low;
      const tr2 = Math.abs(current.high - previous.close);
      const tr3 = Math.abs(current.low - previous.close);
      
      trueRanges.push(Math.max(tr1, tr2, tr3));
    }

    // Simple moving average of true ranges
    const atrPeriods = Math.min(this.config.atrPeriods, trueRanges.length);
    const recentTRs = trueRanges.slice(-atrPeriods);
    
    return recentTRs.reduce((a, b) => a + b, 0) / recentTRs.length;
  }

  private calculateGARCHVolatility(returns: number[]): number {
    if (returns.length < 10) return this.calculateRealizedVolatility(returns);

    let variance = 0.0001; // Initial variance
    const alpha = this.config.garchAlpha;
    const beta = this.config.garchBeta;

    for (const return_ of returns) {
      variance = (1 - alpha - beta) * 0.0001 + alpha * Math.pow(return_, 2) + beta * variance;
    }

    return Math.sqrt(variance * 252); // Annualized
  }

  private calculateRollingVolatilities(returns: number[]): number[] {
    const volatilities: number[] = [];
    const windowSize = 20; // 20-day rolling window

    for (let i = windowSize; i <= returns.length; i++) {
      const window = returns.slice(i - windowSize, i);
      const vol = this.calculateRealizedVolatility(window);
      volatilities.push(vol);
    }

    return volatilities;
  }

  private calculatePercentile(values: number[], target: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    let count = 0;
    
    for (const value of sorted) {
      if (value <= target) count++;
    }

    return (count / sorted.length) * 100;
  }

  private determineVolatilityRegime(volatility: number): VolatilityRegime {
    const thresholds = this.config.regimeThresholds;

    if (volatility <= thresholds.veryLow) return VolatilityRegime.VERY_LOW;
    if (volatility <= thresholds.low) return VolatilityRegime.LOW;
    if (volatility <= thresholds.normal) return VolatilityRegime.NORMAL;
    if (volatility <= thresholds.high) return VolatilityRegime.HIGH;
    return VolatilityRegime.EXTREME;
  }

  private applyVolatilityAdjustment(
    baseRisk: RiskParameters,
    adjustmentFactor: number,
    metrics: VolatilityMetrics
  ): RiskParameters {
    return {
      maxPositionSize: baseRisk.maxPositionSize * adjustmentFactor,
      stopLossDistance: Math.max(baseRisk.stopLossDistance, metrics.atr * 1.5),
      takeProfitDistance: baseRisk.takeProfitDistance * (1 / adjustmentFactor),
      maxExposure: baseRisk.maxExposure * adjustmentFactor,
      riskPerTrade: baseRisk.riskPerTrade * adjustmentFactor,
      leverageMultiplier: baseRisk.leverageMultiplier * adjustmentFactor,
    };
  }

  private generateAdjustmentReason(metrics: VolatilityMetrics): string {
    const regime = metrics.regime;
    const percentile = metrics.volatilityPercentile.toFixed(1);

    switch (regime) {
      case VolatilityRegime.VERY_LOW:
        return `Very low volatility (${percentile}th percentile) - increased position sizing allowed`;
      case VolatilityRegime.LOW:
        return `Low volatility (${percentile}th percentile) - slightly increased position sizing`;
      case VolatilityRegime.NORMAL:
        return `Normal volatility (${percentile}th percentile) - standard risk parameters`;
      case VolatilityRegime.HIGH:
        return `High volatility (${percentile}th percentile) - reduced position sizing for protection`;
      case VolatilityRegime.EXTREME:
        return `Extreme volatility (${percentile}th percentile) - significant risk reduction required`;
      default:
        return `Volatility regime: ${regime}`;
    }
  }
}
