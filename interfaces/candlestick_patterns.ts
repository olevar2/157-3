/**
 * TypeScript interfaces for Japanese Candlestick Patterns
 * Platform3 Trading System
 */

export enum PatternType {
  BULLISH = "bullish",
  BEARISH = "bearish",
  NEUTRAL = "neutral",
  REVERSAL = "reversal",
  CONTINUATION = "continuation"
}

export enum CandlestickPatternName {
  // Single candle patterns
  STANDARD_DOJI = "Standard Doji",
  DRAGONFLY_DOJI = "Dragonfly Doji",
  GRAVESTONE_DOJI = "Gravestone Doji",
  LONG_LEGGED_DOJI = "Long-legged Doji",
  HAMMER = "Hammer",
  HANGING_MAN = "Hanging Man",
  INVERTED_HAMMER = "Inverted Hammer",
  SHOOTING_STAR = "Shooting Star",
  BULLISH_MARUBOZU = "Bullish Marubozu",
  BEARISH_MARUBOZU = "Bearish Marubozu",
  SPINNING_TOP = "Spinning Top",
  HIGH_WAVE_CANDLE = "High Wave Candle",
  
  // Two candle patterns
  BULLISH_ENGULFING = "Bullish Engulfing",
  BEARISH_ENGULFING = "Bearish Engulfing",
  PIERCING_LINE = "Piercing Line",
  DARK_CLOUD_COVER = "Dark Cloud Cover",
  BULLISH_HARAMI = "Bullish Harami",
  BEARISH_HARAMI = "Bearish Harami",
  TWEEZER_BOTTOM = "Tweezer Bottom",
  TWEEZER_TOP = "Tweezer Top",
  
  // Three candle patterns
  MORNING_STAR = "Morning Star",
  EVENING_STAR = "Evening Star",
  THREE_WHITE_SOLDIERS = "Three White Soldiers",
  THREE_BLACK_CROWS = "Three Black Crows",
  THREE_INSIDE_UP = "Three Inside Up",
  THREE_INSIDE_DOWN = "Three Inside Down",
  ABANDONED_BABY = "Abandoned Baby"
}

export interface CandleData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: Date;
}

export interface PatternResult {
  patternName: CandlestickPatternName;
  patternType: PatternType;
  strength: number;  // 0-100
  confidence: number;  // 0-1
  position: number;
  candlesInvolved: CandleData[];
  description: string;
  metadata?: Record<string, any>;
}

export interface PatternDetectionResult {
  patterns: PatternResult[];
  patternCount: number;
  analysis: PatternAnalysis;
  strongestPattern: PatternResult | null;
  trendContext: TrendContext;
}

export interface PatternAnalysis {
  typeDistribution: Record<PatternType, number>;
  averageStrengths: Record<PatternType, number>;
  recentPatterns: string[];
  dominantSentiment: PatternType;
}

export interface TrendContext {
  trend: "uptrend" | "downtrend" | "sideways" | "unknown";
  strength: number;
  slope: number;
}

export interface CandlestickPatternConfig {
  dojiThreshold?: number;
  shadowRatioThreshold?: number;
  trendPeriod?: number;
  volumeConfirmation?: boolean;
}

// Individual pattern results
export interface DojiPattern {
  type: "standard" | "dragonfly" | "gravestone" | "long_legged";
  bodyRatio: number;
  upperShadowRatio: number;
  lowerShadowRatio: number;
}

export interface HammerPattern {
  type: "hammer" | "hanging_man" | "inverted_hammer" | "shooting_star";
  bodyRatio: number;
  shadowRatio: number;
  trendAlignment: boolean;
}

export interface MarubozuPattern {
  type: "bullish_marubozu" | "bearish_marubozu";
  bodyRatio: number;
  upperShadowRatio: number;
  lowerShadowRatio: number;
  volumeConfirmed: boolean;
}

// Service interface
export interface ICandlestickPatternService {
  detectPatterns(data: CandleData[]): Promise<PatternDetectionResult>;
  detectSinglePattern(candle: CandleData, context: TrendContext): PatternResult[];
  getPatternStrength(pattern: PatternResult): number;
  validatePattern(pattern: PatternResult, priceData: CandleData[]): boolean;
}
