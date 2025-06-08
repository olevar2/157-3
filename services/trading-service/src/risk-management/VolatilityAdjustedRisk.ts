import axios from 'axios';
import { EventEmitter } from 'events';

/**
 * Volatility Adjusted Risk Management
 * Dynamically adjusts position sizes and risk parameters based on market volatility
 */

export interface VolatilityMetrics {
  symbol: string;
  atr: number;                    // Average True Range
  volatilityPercent: number;      // Volatility as percentage
  volatilityRank: number;         // Percentile rank (0-100)
  riskAdjustment: number;         // Risk adjustment factor (0.5 - 2.0)
  timestamp: Date;
}

export interface VolatilityThresholds {
  low: number;      // Below 25th percentile
  medium: number;   // 25th to 75th percentile
  high: number;     // Above 75th percentile
  extreme: number;  // Above 95th percentile
}

export interface RiskAdjustmentConfig {
  baseRiskPercent: number;           // Base risk per trade (e.g., 1%)
  volatilityLookback: number;        // Days to look back for volatility calculation
  minRiskMultiplier: number;         // Minimum risk multiplier (e.g., 0.5)
  maxRiskMultiplier: number;         // Maximum risk multiplier (e.g., 2.0)
  thresholds: VolatilityThresholds;  // Volatility thresholds
  adjustmentFactors: {               // Risk adjustment factors by volatility regime
    low: number;      // Increase risk in low volatility
    medium: number;   // Normal risk in medium volatility
    high: number;     // Reduce risk in high volatility
    extreme: number;  // Severely reduce risk in extreme volatility
  };
}

export interface AdjustedPositionSize {
  symbol: string;
  originalSize: number;
  adjustedSize: number;
  adjustmentFactor: number;
  volatilityRegime: 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
  stopLoss: number;
  reasoning: string;
}

export class VolatilityAdjustedRisk extends EventEmitter {
  private config: RiskAdjustmentConfig;
  private volatilityHistory: Map<string, VolatilityMetrics[]> = new Map();
  private pythonEngineUrl: string;

  constructor(config?: Partial<RiskAdjustmentConfig>) {
    super();
    this.pythonEngineUrl = process.env.PYTHON_ENGINE_URL || 'http://localhost:8000';
    
    // Default configuration
    this.config = {
      baseRiskPercent: 0.01,        // 1% base risk per trade
      volatilityLookback: 20,       // 20 days lookback
      minRiskMultiplier: 0.3,       // Minimum 0.3x risk
      maxRiskMultiplier: 2.0,       // Maximum 2.0x risk
      thresholds: {
        low: 25,       // 25th percentile
        medium: 75,    // 75th percentile
        high: 90,      // 90th percentile
        extreme: 95    // 95th percentile
      },
      adjustmentFactors: {
        low: 1.5,      // Increase risk by 50% in low volatility
        medium: 1.0,   // Normal risk in medium volatility
        high: 0.7,     // Reduce risk by 30% in high volatility
        extreme: 0.4   // Reduce risk by 60% in extreme volatility
      },
      ...config
    };

    console.log('VolatilityAdjustedRisk initialized with config:', this.config);
  }

  /**
   * Calculate volatility metrics for a symbol
   */
  public async calculateVolatilityMetrics(symbol: string, priceData: number[]): Promise<VolatilityMetrics> {
    if (priceData.length < 2) {
      throw new Error('Insufficient price data for volatility calculation');
    }

    // Calculate simple volatility (standard deviation of returns)
    const returns = [];
    for (let i = 1; i < priceData.length; i++) {
      returns.push((priceData[i] - priceData[i - 1]) / priceData[i - 1]);
    }

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const volatilityPercent = Math.sqrt(variance) * 100;

    // Calculate ATR approximation (simplified)
    const atr = this.calculateATR(priceData);

    // Get historical volatility for percentile ranking
    const history = this.volatilityHistory.get(symbol) || [];
    const volatilityRank = this.calculateVolatilityRank(volatilityPercent, history);

    // Calculate risk adjustment factor
    const riskAdjustment = this.calculateRiskAdjustmentFactor(volatilityRank);

    const metrics: VolatilityMetrics = {
      symbol,
      atr,
      volatilityPercent,
      volatilityRank,
      riskAdjustment,
      timestamp: new Date()
    };

    // Store in history
    this.updateVolatilityHistory(symbol, metrics);

    console.log(`Volatility metrics for ${symbol}: ${volatilityPercent.toFixed(2)}% (${volatilityRank.toFixed(0)}th percentile), Adjustment: ${riskAdjustment.toFixed(2)}x`);

    return metrics;
  }

  /**
   * Calculate simplified ATR
   */
  private calculateATR(priceData: number[], period: number = 14): number {
    if (priceData.length < period + 1) return 0;

    const trueRanges = [];
    for (let i = 1; i < priceData.length; i++) {
      // Simplified TR calculation (just using close prices)
      const tr = Math.abs(priceData[i] - priceData[i - 1]);
      trueRanges.push(tr);
    }

    // Calculate average of last 'period' true ranges
    const recentTRs = trueRanges.slice(-period);
    return recentTRs.reduce((sum, tr) => sum + tr, 0) / recentTRs.length;
  }

  /**
   * Calculate volatility percentile rank
   */
  private calculateVolatilityRank(currentVolatility: number, history: VolatilityMetrics[]): number {
    if (history.length < 10) return 50; // Default to median if insufficient history

    const volatilities = history.map(h => h.volatilityPercent);
    const belowCurrent = volatilities.filter(v => v < currentVolatility).length;
    return (belowCurrent / volatilities.length) * 100;
  }

  /**
   * Calculate risk adjustment factor based on volatility rank
   */
  private calculateRiskAdjustmentFactor(volatilityRank: number): number {
    const { thresholds, adjustmentFactors } = this.config;

    let factor: number;
    if (volatilityRank <= thresholds.low) {
      factor = adjustmentFactors.low;
    } else if (volatilityRank <= thresholds.medium) {
      factor = adjustmentFactors.medium;
    } else if (volatilityRank <= thresholds.high) {
      factor = adjustmentFactors.high;
    } else {
      factor = adjustmentFactors.extreme;
    }

    // Clamp to min/max multipliers
    return Math.max(this.config.minRiskMultiplier, Math.min(this.config.maxRiskMultiplier, factor));
  }

  /**
   * Update volatility history for a symbol
   */
  private updateVolatilityHistory(symbol: string, metrics: VolatilityMetrics): void {
    if (!this.volatilityHistory.has(symbol)) {
      this.volatilityHistory.set(symbol, []);
    }

    const history = this.volatilityHistory.get(symbol)!;
    history.push(metrics);

    // Keep only recent history (based on lookback period)
    const maxHistory = this.config.volatilityLookback * 2; // Keep extra for better percentile calculation
    if (history.length > maxHistory) {
      history.splice(0, history.length - maxHistory);
    }
  }

  /**
   * Calculate adjusted position size based on volatility
   */
  public async calculateAdjustedPosition(
    symbol: string,
    accountBalance: number,
    entryPrice: number,
    stopLossPrice: number,
    priceData: number[]
  ): Promise<AdjustedPositionSize> {
    // Get volatility metrics
    const volatilityMetrics = await this.calculateVolatilityMetrics(symbol, priceData);

    // Calculate base position size (1% risk)
    const riskAmount = accountBalance * this.config.baseRiskPercent;
    const stopLossDistance = Math.abs(entryPrice - stopLossPrice);
    const baseSize = riskAmount / stopLossDistance;

    // Apply volatility adjustment
    const adjustedSize = baseSize * volatilityMetrics.riskAdjustment;

    // Determine volatility regime
    let regime: 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
    if (volatilityMetrics.volatilityRank <= this.config.thresholds.low) {
      regime = 'LOW';
    } else if (volatilityMetrics.volatilityRank <= this.config.thresholds.medium) {
      regime = 'MEDIUM';
    } else if (volatilityMetrics.volatilityRank <= this.config.thresholds.high) {
      regime = 'HIGH';
    } else {
      regime = 'EXTREME';
    }

    const result: AdjustedPositionSize = {
      symbol,
      originalSize: baseSize,
      adjustedSize,
      adjustmentFactor: volatilityMetrics.riskAdjustment,
      volatilityRegime: regime,
      stopLoss: stopLossPrice,
      reasoning: `${regime} volatility (${volatilityMetrics.volatilityRank.toFixed(0)}th percentile) - ${volatilityMetrics.riskAdjustment > 1 ? 'increased' : 'reduced'} position size`
    };

    console.log(`Position size adjusted for ${symbol}: ${baseSize.toFixed(2)} → ${adjustedSize.toFixed(2)} (${(volatilityMetrics.riskAdjustment * 100).toFixed(0)}%) - ${regime} volatility`);

    // Emit adjustment event
    this.emit('positionAdjusted', result);

    // Notify Python engines
    await this.notifyPythonEngines('position_adjusted', result);

    return result;
  }

  /**
   * Get volatility regime for a symbol
   */
  public async getVolatilityRegime(symbol: string, priceData: number[]): Promise<{
    regime: 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
    metrics: VolatilityMetrics;
    recommendation: string;
  }> {
    const metrics = await this.calculateVolatilityMetrics(symbol, priceData);
    
    let regime: 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
    let recommendation: string;

    if (metrics.volatilityRank <= this.config.thresholds.low) {
      regime = 'LOW';
      recommendation = 'Consider increasing position sizes. Market is calm and predictable.';
    } else if (metrics.volatilityRank <= this.config.thresholds.medium) {
      regime = 'MEDIUM';
      recommendation = 'Use normal position sizing. Market volatility is within normal range.';
    } else if (metrics.volatilityRank <= this.config.thresholds.high) {
      regime = 'HIGH';
      recommendation = 'Reduce position sizes. Market is more volatile than usual.';
    } else {
      regime = 'EXTREME';
      recommendation = 'Significantly reduce position sizes or avoid trading. Market is extremely volatile.';
    }

    return { regime, metrics, recommendation };
  }

  /**
   * Update configuration
   */
  public updateConfig(newConfig: Partial<RiskAdjustmentConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log('Volatility risk config updated:', newConfig);
  }

  /**
   * Get volatility history for a symbol
   */
  public getVolatilityHistory(symbol: string): VolatilityMetrics[] {
    return this.volatilityHistory.get(symbol) || [];
  }

  /**
   * Clear volatility history (useful for backtesting)
   */
  public clearHistory(symbol?: string): void {
    if (symbol) {
      this.volatilityHistory.delete(symbol);
      console.log(`Cleared volatility history for ${symbol}`);
    } else {
      this.volatilityHistory.clear();
      console.log('Cleared all volatility history');
    }
  }

  /**
   * Notify Python engines about volatility events
   */
  private async notifyPythonEngines(eventType: string, data: any): Promise<void> {
    try {
      await axios.post(`${this.pythonEngineUrl}/api/risk/volatility-event`, {
        eventType,
        data,
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      console.error(`Failed to notify Python engines about ${eventType}:`, error.message);
    }
  }

  /**
   * Get current configuration
   */
  public getConfig(): RiskAdjustmentConfig {
    return { ...this.config };
  }
}
