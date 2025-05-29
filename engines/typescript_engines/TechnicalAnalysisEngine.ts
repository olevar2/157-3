// Technical Analysis Engine - RSI, MACD, Bollinger Bands, Moving Averages
// Provides comprehensive technical indicator calculations and trend analysis

import { Logger } from 'winston';
import * as TI from 'technicalindicators';
import { mean, standardDeviation } from 'simple-statistics';

export interface MarketData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TechnicalAnalysisResult {
  symbol: string;
  timestamp: number;
  indicators: {
    rsi: RSIAnalysis;
    macd: MACDAnalysis;
    bollingerBands: BollingerBandsAnalysis;
    movingAverages: MovingAveragesAnalysis;
    stochastic: StochasticAnalysis;
    atr: ATRAnalysis;
  };
  signals: TradingSignal[];
  trend: TrendAnalysis;
  support: number[];
  resistance: number[];
  sentiment: SentimentAnalysis;
}

export interface RSIAnalysis {
  current: number;
  signal: 'oversold' | 'overbought' | 'neutral';
  strength: number; // 0-1
  divergence?: 'bullish' | 'bearish' | null;
}

export interface MACDAnalysis {
  macd: number;
  signal: number;
  histogram: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  crossover?: 'bullish' | 'bearish' | null;
}

export interface BollingerBandsAnalysis {
  upper: number;
  middle: number;
  lower: number;
  position: 'above_upper' | 'below_lower' | 'middle' | 'upper_half' | 'lower_half';
  squeeze: boolean;
  expansion: boolean;
}

export interface MovingAveragesAnalysis {
  sma20: number;
  sma50: number;
  sma200: number;
  ema12: number;
  ema26: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  crossovers: string[];
}

export interface StochasticAnalysis {
  k: number;
  d: number;
  signal: 'oversold' | 'overbought' | 'neutral';
  crossover?: 'bullish' | 'bearish' | null;
}

export interface ATRAnalysis {
  current: number;
  average: number;
  volatility: 'high' | 'medium' | 'low';
}

export interface TradingSignal {
  type: 'buy' | 'sell' | 'hold';
  strength: number; // 0-1
  source: string;
  description: string;
  confidence: number; // 0-1
}

export interface TrendAnalysis {
  direction: 'uptrend' | 'downtrend' | 'sideways';
  strength: number; // 0-1
  duration: number; // periods
  reliability: number; // 0-1
}

export interface SentimentAnalysis {
  score: number; // -1 to 1 (bearish to bullish)
  label: 'very_bearish' | 'bearish' | 'neutral' | 'bullish' | 'very_bullish';
  confidence: number; // 0-1
  factors: string[];
}

export class TechnicalAnalysisEngine {
  private logger: Logger;
  private ready: boolean = false;

  constructor(logger: Logger) {
    this.logger = logger;
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing Technical Analysis Engine...');
    
    // Validate technical indicators library
    try {
      const testData = [1, 2, 3, 4, 5];
      TI.SMA.calculate({ period: 3, values: testData });
      this.ready = true;
      this.logger.info('✅ Technical Analysis Engine initialized');
    } catch (error) {
      this.logger.error('❌ Technical Analysis Engine initialization failed:', error);
      throw error;
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  async analyze(symbol: string, marketData: MarketData[]): Promise<TechnicalAnalysisResult> {
    if (!this.ready) {
      throw new Error('Technical Analysis Engine not initialized');
    }

    if (marketData.length < 50) {
      throw new Error('Insufficient data for technical analysis (minimum 50 periods required)');
    }

    this.logger.debug(`Performing technical analysis for ${symbol} with ${marketData.length} data points`);

    const closes = marketData.map(d => d.close);
    const highs = marketData.map(d => d.high);
    const lows = marketData.map(d => d.low);
    const volumes = marketData.map(d => d.volume || 0);

    // Calculate all indicators
    const rsi = this.calculateRSI(closes);
    const macd = this.calculateMACD(closes);
    const bollingerBands = this.calculateBollingerBands(closes);
    const movingAverages = this.calculateMovingAverages(closes);
    const stochastic = this.calculateStochastic(highs, lows, closes);
    const atr = this.calculateATR(marketData);

    // Analyze support and resistance
    const supportResistance = this.findSupportResistance(highs, lows);

    // Generate trading signals
    const signals = this.generateTradingSignals(rsi, macd, bollingerBands, movingAverages, stochastic);

    // Analyze trend
    const trend = this.analyzeTrend(closes, movingAverages);

    // Calculate sentiment
    const sentiment = this.calculateSentiment(rsi, macd, trend, signals);

    return {
      symbol,
      timestamp: Date.now(),
      indicators: {
        rsi,
        macd,
        bollingerBands,
        movingAverages,
        stochastic,
        atr
      },
      signals,
      trend,
      support: supportResistance.support,
      resistance: supportResistance.resistance,
      sentiment
    };
  }

  private calculateRSI(closes: number[]): RSIAnalysis {
    const rsiValues = TI.RSI.calculate({ period: 14, values: closes });
    const current = rsiValues[rsiValues.length - 1];

    let signal: 'oversold' | 'overbought' | 'neutral' = 'neutral';
    if (current < 30) signal = 'oversold';
    else if (current > 70) signal = 'overbought';

    const strength = current > 50 ? (current - 50) / 50 : (50 - current) / 50;

    // Check for divergence (simplified)
    const divergence = this.checkRSIDivergence(rsiValues, closes);

    return {
      current,
      signal,
      strength,
      divergence
    };
  }

  private calculateMACD(closes: number[]): MACDAnalysis {
    const macdData = TI.MACD.calculate({
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      values: closes
    });

    const latest = macdData[macdData.length - 1];
    const macd = latest.MACD;
    const signal = latest.signal;
    const histogram = latest.histogram;

    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (macd > signal && histogram > 0) trend = 'bullish';
    else if (macd < signal && histogram < 0) trend = 'bearish';

    // Check for crossover
    const crossover = this.checkMACDCrossover(macdData);

    return {
      macd,
      signal,
      histogram,
      trend,
      crossover
    };
  }

  private calculateBollingerBands(closes: number[]): BollingerBandsAnalysis {
    const bbData = TI.BollingerBands.calculate({
      period: 20,
      stdDev: 2,
      values: closes
    });

    const latest = bbData[bbData.length - 1];
    const currentPrice = closes[closes.length - 1];

    let position: BollingerBandsAnalysis['position'] = 'middle';
    if (currentPrice > latest.upper) position = 'above_upper';
    else if (currentPrice < latest.lower) position = 'below_lower';
    else if (currentPrice > latest.middle) position = 'upper_half';
    else position = 'lower_half';

    // Check for squeeze and expansion
    const bandWidth = (latest.upper - latest.lower) / latest.middle;
    const avgBandWidth = mean(bbData.slice(-20).map(bb => (bb.upper - bb.lower) / bb.middle));
    
    const squeeze = bandWidth < avgBandWidth * 0.8;
    const expansion = bandWidth > avgBandWidth * 1.2;

    return {
      upper: latest.upper,
      middle: latest.middle,
      lower: latest.lower,
      position,
      squeeze,
      expansion
    };
  }

  private calculateMovingAverages(closes: number[]): MovingAveragesAnalysis {
    const sma20 = TI.SMA.calculate({ period: 20, values: closes });
    const sma50 = TI.SMA.calculate({ period: 50, values: closes });
    const sma200 = TI.SMA.calculate({ period: 200, values: closes });
    const ema12 = TI.EMA.calculate({ period: 12, values: closes });
    const ema26 = TI.EMA.calculate({ period: 26, values: closes });

    const currentSMA20 = sma20[sma20.length - 1];
    const currentSMA50 = sma50[sma50.length - 1];
    const currentSMA200 = sma200[sma200.length - 1];
    const currentEMA12 = ema12[ema12.length - 1];
    const currentEMA26 = ema26[ema26.length - 1];

    // Determine trend
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (currentSMA20 > currentSMA50 && currentSMA50 > currentSMA200) trend = 'bullish';
    else if (currentSMA20 < currentSMA50 && currentSMA50 < currentSMA200) trend = 'bearish';

    // Check for crossovers
    const crossovers = this.checkMACrossovers(sma20, sma50, ema12, ema26);

    return {
      sma20: currentSMA20,
      sma50: currentSMA50,
      sma200: currentSMA200,
      ema12: currentEMA12,
      ema26: currentEMA26,
      trend,
      crossovers
    };
  }

  private calculateStochastic(highs: number[], lows: number[], closes: number[]): StochasticAnalysis {
    const stochData = TI.Stochastic.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: 14,
      signalPeriod: 3
    });

    const latest = stochData[stochData.length - 1];
    const k = latest.k;
    const d = latest.d;

    let signal: 'oversold' | 'overbought' | 'neutral' = 'neutral';
    if (k < 20 && d < 20) signal = 'oversold';
    else if (k > 80 && d > 80) signal = 'overbought';

    // Check for crossover
    const crossover = this.checkStochasticCrossover(stochData);

    return {
      k,
      d,
      signal,
      crossover
    };
  }

  private calculateATR(marketData: MarketData[]): ATRAnalysis {
    const atrData = TI.ATR.calculate({
      high: marketData.map(d => d.high),
      low: marketData.map(d => d.low),
      close: marketData.map(d => d.close),
      period: 14
    });

    const current = atrData[atrData.length - 1];
    const average = mean(atrData.slice(-20));

    let volatility: 'high' | 'medium' | 'low' = 'medium';
    if (current > average * 1.5) volatility = 'high';
    else if (current < average * 0.5) volatility = 'low';

    return {
      current,
      average,
      volatility
    };
  }

  // Helper methods for analysis
  private checkRSIDivergence(rsiValues: number[], closes: number[]): 'bullish' | 'bearish' | null {
    // Simplified divergence detection
    if (rsiValues.length < 20) return null;

    const recentRSI = rsiValues.slice(-10);
    const recentPrices = closes.slice(-10);

    const rsiTrend = recentRSI[recentRSI.length - 1] - recentRSI[0];
    const priceTrend = recentPrices[recentPrices.length - 1] - recentPrices[0];

    if (rsiTrend > 0 && priceTrend < 0) return 'bullish';
    if (rsiTrend < 0 && priceTrend > 0) return 'bearish';

    return null;
  }

  private checkMACDCrossover(macdData: any[]): 'bullish' | 'bearish' | null {
    if (macdData.length < 2) return null;

    const current = macdData[macdData.length - 1];
    const previous = macdData[macdData.length - 2];

    if (previous.MACD <= previous.signal && current.MACD > current.signal) return 'bullish';
    if (previous.MACD >= previous.signal && current.MACD < current.signal) return 'bearish';

    return null;
  }

  private checkMACrossovers(sma20: number[], sma50: number[], ema12: number[], ema26: number[]): string[] {
    const crossovers: string[] = [];

    // Golden Cross / Death Cross
    if (sma20.length >= 2 && sma50.length >= 2) {
      const currentSMA20 = sma20[sma20.length - 1];
      const previousSMA20 = sma20[sma20.length - 2];
      const currentSMA50 = sma50[sma50.length - 1];
      const previousSMA50 = sma50[sma50.length - 2];

      if (previousSMA20 <= previousSMA50 && currentSMA20 > currentSMA50) {
        crossovers.push('Golden Cross (SMA20 > SMA50)');
      }
      if (previousSMA20 >= previousSMA50 && currentSMA20 < currentSMA50) {
        crossovers.push('Death Cross (SMA20 < SMA50)');
      }
    }

    return crossovers;
  }

  private checkStochasticCrossover(stochData: any[]): 'bullish' | 'bearish' | null {
    if (stochData.length < 2) return null;

    const current = stochData[stochData.length - 1];
    const previous = stochData[stochData.length - 2];

    if (previous.k <= previous.d && current.k > current.d) return 'bullish';
    if (previous.k >= previous.d && current.k < current.d) return 'bearish';

    return null;
  }

  private findSupportResistance(highs: number[], lows: number[]): { support: number[], resistance: number[] } {
    // Simplified support/resistance detection using pivot points
    const support: number[] = [];
    const resistance: number[] = [];

    const lookback = 10;
    
    for (let i = lookback; i < lows.length - lookback; i++) {
      const isSupport = lows.slice(i - lookback, i).every(low => low >= lows[i]) &&
                       lows.slice(i + 1, i + lookback + 1).every(low => low >= lows[i]);
      
      if (isSupport) {
        support.push(lows[i]);
      }
    }

    for (let i = lookback; i < highs.length - lookback; i++) {
      const isResistance = highs.slice(i - lookback, i).every(high => high <= highs[i]) &&
                          highs.slice(i + 1, i + lookback + 1).every(high => high <= highs[i]);
      
      if (isResistance) {
        resistance.push(highs[i]);
      }
    }

    return {
      support: support.slice(-3), // Last 3 support levels
      resistance: resistance.slice(-3) // Last 3 resistance levels
    };
  }

  private generateTradingSignals(
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    bb: BollingerBandsAnalysis,
    ma: MovingAveragesAnalysis,
    stoch: StochasticAnalysis
  ): TradingSignal[] {
    const signals: TradingSignal[] = [];

    // RSI signals
    if (rsi.signal === 'oversold') {
      signals.push({
        type: 'buy',
        strength: rsi.strength,
        source: 'RSI',
        description: 'RSI indicates oversold condition',
        confidence: 0.7
      });
    } else if (rsi.signal === 'overbought') {
      signals.push({
        type: 'sell',
        strength: rsi.strength,
        source: 'RSI',
        description: 'RSI indicates overbought condition',
        confidence: 0.7
      });
    }

    // MACD signals
    if (macd.crossover === 'bullish') {
      signals.push({
        type: 'buy',
        strength: 0.8,
        source: 'MACD',
        description: 'MACD bullish crossover detected',
        confidence: 0.8
      });
    } else if (macd.crossover === 'bearish') {
      signals.push({
        type: 'sell',
        strength: 0.8,
        source: 'MACD',
        description: 'MACD bearish crossover detected',
        confidence: 0.8
      });
    }

    // Bollinger Bands signals
    if (bb.position === 'below_lower') {
      signals.push({
        type: 'buy',
        strength: 0.6,
        source: 'Bollinger Bands',
        description: 'Price below lower Bollinger Band',
        confidence: 0.6
      });
    } else if (bb.position === 'above_upper') {
      signals.push({
        type: 'sell',
        strength: 0.6,
        source: 'Bollinger Bands',
        description: 'Price above upper Bollinger Band',
        confidence: 0.6
      });
    }

    // Moving Average signals
    if (ma.trend === 'bullish') {
      signals.push({
        type: 'buy',
        strength: 0.7,
        source: 'Moving Averages',
        description: 'Bullish moving average alignment',
        confidence: 0.7
      });
    } else if (ma.trend === 'bearish') {
      signals.push({
        type: 'sell',
        strength: 0.7,
        source: 'Moving Averages',
        description: 'Bearish moving average alignment',
        confidence: 0.7
      });
    }

    return signals;
  }

  private analyzeTrend(closes: number[], ma: MovingAveragesAnalysis): TrendAnalysis {
    const recentCloses = closes.slice(-20);
    const priceChange = recentCloses[recentCloses.length - 1] - recentCloses[0];
    
    let direction: 'uptrend' | 'downtrend' | 'sideways' = 'sideways';
    if (priceChange > 0 && ma.trend === 'bullish') direction = 'uptrend';
    else if (priceChange < 0 && ma.trend === 'bearish') direction = 'downtrend';

    const strength = Math.abs(priceChange) / recentCloses[0];
    const reliability = ma.trend !== 'neutral' ? 0.8 : 0.4;

    return {
      direction,
      strength: Math.min(strength * 10, 1), // Normalize to 0-1
      duration: 20, // Using 20 periods for analysis
      reliability
    };
  }

  private calculateSentiment(
    rsi: RSIAnalysis,
    macd: MACDAnalysis,
    trend: TrendAnalysis,
    signals: TradingSignal[]
  ): SentimentAnalysis {
    let score = 0;
    const factors: string[] = [];

    // RSI contribution
    if (rsi.signal === 'oversold') {
      score += 0.3;
      factors.push('RSI oversold');
    } else if (rsi.signal === 'overbought') {
      score -= 0.3;
      factors.push('RSI overbought');
    }

    // MACD contribution
    if (macd.trend === 'bullish') {
      score += 0.3;
      factors.push('MACD bullish');
    } else if (macd.trend === 'bearish') {
      score -= 0.3;
      factors.push('MACD bearish');
    }

    // Trend contribution
    if (trend.direction === 'uptrend') {
      score += 0.4 * trend.strength;
      factors.push('Uptrend detected');
    } else if (trend.direction === 'downtrend') {
      score -= 0.4 * trend.strength;
      factors.push('Downtrend detected');
    }

    // Normalize score to -1 to 1
    score = Math.max(-1, Math.min(1, score));

    let label: SentimentAnalysis['label'] = 'neutral';
    if (score > 0.6) label = 'very_bullish';
    else if (score > 0.2) label = 'bullish';
    else if (score < -0.6) label = 'very_bearish';
    else if (score < -0.2) label = 'bearish';

    const confidence = Math.abs(score);

    return {
      score,
      label,
      confidence,
      factors
    };
  }

  async analyzeSentiment(symbol: string, marketData: MarketData[]): Promise<SentimentAnalysis> {
    const analysis = await this.analyze(symbol, marketData);
    return analysis.sentiment;
  }
}
