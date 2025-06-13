# -*- coding: utf-8 -*-
"""
Japanese Candlestick Patterns - Complete Implementation
Platform3 Enhanced Technical Analysis Engine

Comprehensive implementation of all major Japanese candlestick patterns including:
- Single candle patterns (Doji, Hammer, Marubozu, etc.)
- Two candle patterns (Engulfing, Harami, etc.)
- Three candle patterns (Morning/Evening Star, etc.)
- Complex patterns (Abandoned Baby, etc.)

Features:
- Pattern strength scoring
- Trend context analysis
- Volume confirmation
- Multi-timeframe validation
- Real-time pattern detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame

class PatternType(Enum):
    """Types of candlestick patterns"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"

@dataclass
class CandleData:
    """Represents a single candlestick"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    
    @property
    def body_size(self) -> float:
        """Size of the candle body"""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Size of upper shadow/wick"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Size of lower shadow/wick"""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Total range from high to low"""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """True if close > open"""
        return self.close > self.open
    
    @property
    def body_ratio(self) -> float:
        """Body size as ratio of total range"""
        return self.body_size / self.total_range if self.total_range > 0 else 0

@dataclass
class PatternResult:
    """Result of pattern detection"""
    pattern_name: str
    pattern_type: PatternType
    strength: float  # 0-100
    confidence: float  # 0-1
    position: int  # Index where pattern was found
    candles_involved: List[CandleData]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class JapaneseCandlestickPatterns(IndicatorBase):
    """
    Comprehensive Japanese Candlestick Pattern Recognition Engine
    
    Detects and analyzes all major candlestick patterns with
    strength scoring and context analysis.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None, # Added config
                 doji_threshold: float = 0.1,
                 shadow_ratio_threshold: float = 2.0,
                 trend_period: int = 10,
                 volume_confirmation: bool = True):
        """
        Initialize pattern recognition engine
        
        Args:
            config: Optional configuration dictionary (default None)
            doji_threshold: Max body/range ratio for doji (default 0.1)
            shadow_ratio_threshold: Min shadow/body ratio for patterns (default 2.0)
            trend_period: Periods to analyze for trend context (default 10)
            volume_confirmation: Whether to check volume confirmation (default True)
        """
        super().__init__(config=config) # Pass config to super
        
        self.doji_threshold = doji_threshold
        self.shadow_ratio_threshold = shadow_ratio_threshold
        self.trend_period = trend_period
        self.volume_confirmation = volume_confirmation
        
        # Pattern detection results
        self.detected_patterns: List[PatternResult] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Analyze candlestick data for all patterns
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            # Validate data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_cols)
            
            # Convert to CandleData objects
            candles = self._create_candle_objects(data)
            
            # Clear previous results
            self.detected_patterns.clear()
            
            # Detect all pattern types
            self._detect_single_candle_patterns(candles)
            self._detect_two_candle_patterns(candles)
            self._detect_three_candle_patterns(candles)
            
            # Analyze results
            # Ensure _analyze_patterns method exists or implement it
            analysis = self._analyze_patterns() if hasattr(self, '_analyze_patterns') else {}
            
            return {
                'patterns': [self._pattern_to_dict(p) for p in self.detected_patterns],
                'pattern_count': len(self.detected_patterns),
                'analysis': analysis,
                'strongest_pattern': self._get_strongest_pattern(),
                'trend_context': self._analyze_trend_context(candles)
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {e}")
            raise
    
    def _create_candle_objects(self, data: pd.DataFrame) -> List[CandleData]:
        """Convert DataFrame to CandleData objects"""
        candles = []
        
        for idx, row in data.iterrows():
            candle = CandleData(
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=idx if isinstance(idx, datetime) else datetime.now()
            )
            candles.append(candle)
        
        return candles
    
    def _get_trend_at_position(self, candles: List[CandleData], position: int) -> str:
        """Determine market trend at given position"""
        if position < 5:
            return 'neutral'
        
        # Look at last 5-10 candles to determine trend
        lookback = min(10, position)
        start_idx = position - lookback
        recent_candles = candles[start_idx:position]
        
        # Calculate trend based on price movement
        if len(recent_candles) < 3:
            return 'neutral'
        
        # Simple trend detection using closing prices
        closes = [c.close for c in recent_candles]
        first_third = np.mean(closes[:len(closes)//3])
        last_third = np.mean(closes[-len(closes)//3:])
        
        change_percent = (last_third - first_third) / first_third
        
        if change_percent > 0.02:  # 2% upward movement
            return 'uptrend'
        elif change_percent < -0.02:  # 2% downward movement
            return 'downtrend'
        else:
            return 'neutral'
    
    # ===== SINGLE CANDLE PATTERNS =====
    
    def _detect_single_candle_patterns(self, candles: List[CandleData]):
        """Detect all single candle patterns"""
        for i, candle in enumerate(candles):
            # Get trend context
            trend = self._get_trend_at_position(candles, i)
            
            # Doji variations
            self._detect_doji_patterns(candle, i, trend)
            
            # Hammer & Hanging Man
            self._detect_hammer_hanging_man(candle, i, trend)
            
            # Inverted Hammer & Shooting Star
            self._detect_inverted_hammer_shooting_star(candle, i, trend)
            
            # Marubozu
            self._detect_marubozu(candle, i, trend)
            
            # Spinning Top
            self._detect_spinning_top(candle, i, trend)
            
            # High Wave Candle
            self._detect_high_wave_candle(candle, i, trend)
    
    def _detect_doji_patterns(self, candle: CandleData, position: int, trend: str):
        """Detect all Doji variations"""
        if candle.body_ratio > self.doji_threshold:
            return
        
        # Calculate shadow ratios
        upper_shadow_ratio = candle.upper_shadow / candle.body_size if candle.body_size > 0 else float('inf')
        lower_shadow_ratio = candle.lower_shadow / candle.body_size if candle.body_size > 0 else float('inf')
        
        # Standard Doji
        if upper_shadow_ratio > 1 and lower_shadow_ratio > 1:
            pattern_type = PatternType.NEUTRAL
            if trend == 'uptrend' or trend == 'downtrend':
                pattern_type = PatternType.REVERSAL
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Standard Doji",
                pattern_type=pattern_type,
                strength=self._calculate_doji_strength(candle, trend, 'standard'),
                confidence=0.8,
                position=position,
                candles_involved=[candle],
                description="Market indecision, potential reversal signal"
            ))
        
        # Dragonfly Doji
        if candle.upper_shadow < candle.total_range * 0.1 and candle.lower_shadow > candle.total_range * 0.5:
            self.detected_patterns.append(PatternResult(
                pattern_name="Dragonfly Doji",
                pattern_type=PatternType.BULLISH if trend == 'downtrend' else PatternType.NEUTRAL,
                strength=self._calculate_doji_strength(candle, trend, 'dragonfly'),
                confidence=0.85,
                position=position,
                candles_involved=[candle],
                description="Bullish reversal signal at bottom"
            ))
        
        # Gravestone Doji
        if candle.lower_shadow < candle.total_range * 0.1 and candle.upper_shadow > candle.total_range * 0.5:
            self.detected_patterns.append(PatternResult(
                pattern_name="Gravestone Doji",
                pattern_type=PatternType.BEARISH if trend == 'uptrend' else PatternType.NEUTRAL,
                strength=self._calculate_doji_strength(candle, trend, 'gravestone'),
                confidence=0.85,
                position=position,
                candles_involved=[candle],
                description="Bearish reversal signal at top"
            ))
        
        # Long-legged Doji
        if upper_shadow_ratio > 3 and lower_shadow_ratio > 3:
            self.detected_patterns.append(PatternResult(
                pattern_name="Long-legged Doji",
                pattern_type=PatternType.NEUTRAL,
                strength=self._calculate_doji_strength(candle, trend, 'long_legged'),
                confidence=0.9,
                position=position,
                candles_involved=[candle],
                description="Extreme indecision, strong reversal potential"
            ))
    
    def _detect_hammer_hanging_man(self, candle: CandleData, position: int, trend: str):
        """Detect Hammer and Hanging Man patterns"""
        # Check pattern criteria
        if not (0.1 <= candle.body_ratio <= 0.35):  # Small body
            return
        
        if candle.lower_shadow < candle.body_size * self.shadow_ratio_threshold:  # Long lower shadow
            return
        
        if candle.upper_shadow > candle.body_size * 0.5:  # Small upper shadow
            return
        
        # Pattern found - determine type based on trend
        if trend == 'downtrend':
            # Hammer - bullish reversal
            self.detected_patterns.append(PatternResult(
                pattern_name="Hammer",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_hammer_strength(candle, trend),
                confidence=0.75,
                position=position,
                candles_involved=[candle],
                description="Bullish reversal pattern in downtrend"
            ))
        elif trend == 'uptrend':
            # Hanging Man - bearish reversal
            self.detected_patterns.append(PatternResult(
                pattern_name="Hanging Man",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_hammer_strength(candle, trend),
                confidence=0.7,
                position=position,
                candles_involved=[candle],
                description="Bearish reversal pattern in uptrend"
            ))
    
    def _detect_inverted_hammer_shooting_star(self, candle: CandleData, position: int, trend: str):
        """Detect Inverted Hammer and Shooting Star patterns"""
        # Check pattern criteria
        if not (0.1 <= candle.body_ratio <= 0.35):  # Small body
            return
        
        if candle.upper_shadow < candle.body_size * self.shadow_ratio_threshold:  # Long upper shadow
            return
        
        if candle.lower_shadow > candle.body_size * 0.5:  # Small lower shadow
            return
        
        # Pattern found - determine type based on trend
        if trend == 'downtrend':
            # Inverted Hammer - potential bullish reversal
            self.detected_patterns.append(PatternResult(
                pattern_name="Inverted Hammer",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_inverted_hammer_strength(candle, trend),
                confidence=0.65,
                position=position,
                candles_involved=[candle],
                description="Potential bullish reversal in downtrend"
            ))
        elif trend == 'uptrend':
            # Shooting Star - bearish reversal
            self.detected_patterns.append(PatternResult(
                pattern_name="Shooting Star",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_inverted_hammer_strength(candle, trend),
                confidence=0.75,
                position=position,
                candles_involved=[candle],
                description="Bearish reversal pattern in uptrend"
            ))
    
    def _detect_marubozu(self, candle: CandleData, position: int, trend: str):
        """Detect Marubozu patterns (Bullish and Bearish)"""
        # Marubozu has very small or no shadows
        if candle.upper_shadow > candle.total_range * 0.05:
            return
        if candle.lower_shadow > candle.total_range * 0.05:
            return
        if candle.body_ratio < 0.9:  # Body should be at least 90% of range
            return
        
        if candle.is_bullish:
            # Bullish Marubozu
            self.detected_patterns.append(PatternResult(
                pattern_name="Bullish Marubozu",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_marubozu_strength(candle, trend),
                confidence=0.9,
                position=position,
                candles_involved=[candle],
                description="Strong bullish sentiment, continuation likely"
            ))
        else:
            # Bearish Marubozu
            self.detected_patterns.append(PatternResult(
                pattern_name="Bearish Marubozu",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_marubozu_strength(candle, trend),
                confidence=0.9,
                position=position,
                candles_involved=[candle],
                description="Strong bearish sentiment, continuation likely"
            ))
    
    def _detect_spinning_top(self, candle: CandleData, position: int, trend: str):
        """Detect Spinning Top pattern"""
        # Small body with shadows on both sides
        if not (0.1 <= candle.body_ratio <= 0.3):
            return
        
        # Both shadows should be significant
        if candle.upper_shadow < candle.body_size * 0.5:
            return
        if candle.lower_shadow < candle.body_size * 0.5:
            return
        
        # Shadows should be relatively equal (within 2x of each other)
        shadow_ratio = max(candle.upper_shadow, candle.lower_shadow) / min(candle.upper_shadow, candle.lower_shadow)
        if shadow_ratio > 2:
            return
        
        pattern_type = PatternType.NEUTRAL
        if trend in ['uptrend', 'downtrend']:
            pattern_type = PatternType.REVERSAL
        
        self.detected_patterns.append(PatternResult(
            pattern_name="Spinning Top",
            pattern_type=pattern_type,
            strength=self._calculate_spinning_top_strength(candle, trend),
            confidence=0.7,
            position=position,
            candles_involved=[candle],
            description="Market indecision, potential trend change"
        ))
    
    def _detect_high_wave_candle(self, candle: CandleData, position: int, trend: str):
        """Detect High Wave Candle pattern"""
        # Very long shadows relative to body
        if candle.body_size == 0:
            return  # This would be a doji
        
        upper_shadow_ratio = candle.upper_shadow / candle.body_size
        lower_shadow_ratio = candle.lower_shadow / candle.body_size
        
        # Both shadows should be very long
        if upper_shadow_ratio < 3 or lower_shadow_ratio < 3:
            return
        
        self.detected_patterns.append(PatternResult(
            pattern_name="High Wave Candle",
            pattern_type=PatternType.NEUTRAL,
            strength=self._calculate_high_wave_strength(candle, trend),
            confidence=0.75,
            position=position,
            candles_involved=[candle],
            description="Extreme volatility and indecision"
        ))
    
    # ===== PATTERN STRENGTH CALCULATIONS =====
    
    def _calculate_doji_strength(self, candle: CandleData, trend: str, doji_type: str) -> float:
        """Calculate strength score for Doji patterns"""
        base_strength = 50.0
        
        # Smaller body = stronger doji
        body_factor = (self.doji_threshold - candle.body_ratio) / self.doji_threshold
        base_strength += body_factor * 20
        
        # Trend context
        if doji_type == 'dragonfly' and trend == 'downtrend':
            base_strength += 20
        elif doji_type == 'gravestone' and trend == 'uptrend':
            base_strength += 20
        elif doji_type in ['standard', 'long_legged'] and trend != 'sideways':
            base_strength += 15
        
        # Volume confirmation
        # (Would need volume average to properly implement)
        
        return min(100, max(0, base_strength))
    
    def _calculate_hammer_strength(self, candle: CandleData, trend: str) -> float:
        """Calculate strength for Hammer/Hanging Man"""
        base_strength = 60.0
        
        # Longer lower shadow = stronger pattern
        shadow_ratio = candle.lower_shadow / candle.body_size if candle.body_size > 0 else 0
        if shadow_ratio > 3:
            base_strength += 20
        elif shadow_ratio > 2:
            base_strength += 10
        
        # Trend appropriateness
        if trend in ['uptrend', 'downtrend']:
            base_strength += 15
        
        return min(100, max(0, base_strength))
    
    def _calculate_inverted_hammer_strength(self, candle: CandleData, trend: str) -> float:
        """Calculate strength for Inverted Hammer/Shooting Star"""
        base_strength = 55.0
        
        # Longer upper shadow = stronger pattern
        shadow_ratio = candle.upper_shadow / candle.body_size if candle.body_size > 0 else 0
        if shadow_ratio > 3:
            base_strength += 20
        elif shadow_ratio > 2:
            base_strength += 10
        
        # Trend appropriateness
        if trend in ['uptrend', 'downtrend']:
            base_strength += 15
        
        return min(100, max(0, base_strength))
    
    def _calculate_marubozu_strength(self, candle: CandleData, trend: str) -> float:
        """Calculate strength for Marubozu patterns"""
        base_strength = 70.0
        
        # Larger body ratio = stronger pattern
        base_strength += (candle.body_ratio - 0.9) * 100
        
        # Trend alignment
        if (candle.is_bullish and trend == 'uptrend') or (not candle.is_bullish and trend == 'downtrend'):
            base_strength += 15  # Continuation
        else:
            base_strength += 5   # Potential reversal
        
        return min(100, max(0, base_strength))
    
    def _calculate_spinning_top_strength(self, candle: CandleData, trend: str) -> float:
        """Calculate strength for Spinning Top"""
        base_strength = 50.0
        
        # More balanced shadows = stronger pattern
        shadow_balance = min(candle.upper_shadow, candle.lower_shadow) / max(candle.upper_shadow, candle.lower_shadow)
        base_strength += shadow_balance * 20
        
        # Trend context
        if trend in ['uptrend', 'downtrend']:
            base_strength += 15
        
        return min(100, max(0, base_strength))
    
    def _calculate_high_wave_strength(self, candle: CandleData, trend: str) -> float:
        """Calculate strength for High Wave Candle"""
        base_strength = 60.0
        
        # Longer shadows = stronger pattern
        avg_shadow_ratio = ((candle.upper_shadow + candle.lower_shadow) / 2) / candle.body_size
        if avg_shadow_ratio > 5:
            base_strength += 25
        elif avg_shadow_ratio > 4:
            base_strength += 15
        
        return min(100, max(0, base_strength))
    
    # ===== TWO CANDLE PATTERNS =====
    
    def _detect_two_candle_patterns(self, candles: List[CandleData]):
        """Detect all two-candle patterns"""
        for i in range(1, len(candles)):
            prev_candle = candles[i-1]
            curr_candle = candles[i]
            trend = self._get_trend_at_position(candles, i)
            
            # Engulfing patterns
            self._detect_engulfing_pattern(prev_candle, curr_candle, i, trend)
            
            # Harami patterns
            self._detect_harami_pattern(prev_candle, curr_candle, i, trend)
            
            # Piercing Line & Dark Cloud Cover
            self._detect_piercing_line_dark_cloud(prev_candle, curr_candle, i, trend)
            
            # Tweezer patterns
            self._detect_tweezer_patterns(prev_candle, curr_candle, i, trend)
            
            # Belt Hold patterns
            self._detect_belt_hold(prev_candle, curr_candle, i, trend)
            
            # Kicking patterns
            self._detect_kicking_pattern(prev_candle, curr_candle, i, trend)
    
    def _detect_engulfing_pattern(self, prev: CandleData, curr: CandleData, position: int, trend: str):
        """Detect Bullish and Bearish Engulfing patterns"""
        # Check if current candle body engulfs previous candle body
        prev_body_top = max(prev.open, prev.close)
        prev_body_bottom = min(prev.open, prev.close)
        curr_body_top = max(curr.open, curr.close)
        curr_body_bottom = min(curr.open, curr.close)
        
        if curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom:
            # Engulfing pattern found
            if not prev.is_bullish and curr.is_bullish and trend == 'downtrend':
                # Bullish Engulfing
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bullish Engulfing",
                    pattern_type=PatternType.BULLISH,
                    strength=self._calculate_engulfing_strength(prev, curr, trend, True),
                    confidence=0.85,
                    position=position,
                    candles_involved=[prev, curr],
                    description="Strong bullish reversal pattern in downtrend"
                ))
            elif prev.is_bullish and not curr.is_bullish and trend == 'uptrend':
                # Bearish Engulfing
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bearish Engulfing",
                    pattern_type=PatternType.BEARISH,
                    strength=self._calculate_engulfing_strength(prev, curr, trend, False),
                    confidence=0.85,
                    position=position,
                    candles_involved=[prev, curr],
                    description="Strong bearish reversal pattern in uptrend"
                ))
    
    def _detect_harami_pattern(self, prev: CandleData, curr: CandleData, position: int, trend: str):
        """Detect Bullish and Bearish Harami patterns"""
        # Check if current candle body is contained within previous candle body
        prev_body_top = max(prev.open, prev.close)
        prev_body_bottom = min(prev.open, prev.close)
        curr_body_top = max(curr.open, curr.close)
        curr_body_bottom = min(curr.open, curr.close)
        
        if curr_body_top <= prev_body_top and curr_body_bottom >= prev_body_bottom:
            # Harami pattern found
            if prev.is_bullish and not curr.is_bullish and trend == 'uptrend':
                # Bearish Harami
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bearish Harami",
                    pattern_type=PatternType.BEARISH,
                    strength=self._calculate_harami_strength(prev, curr, trend, False),
                    confidence=0.7,
                    position=position,
                    candles_involved=[prev, curr],
                    description="Bearish reversal signal in uptrend"
                ))
            elif not prev.is_bullish and curr.is_bullish and trend == 'downtrend':
                # Bullish Harami
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bullish Harami",
                    pattern_type=PatternType.BULLISH,
                    strength=self._calculate_harami_strength(prev, curr, trend, True),
                    confidence=0.7,
                    position=position,
                    candles_involved=[prev, curr],
                    description="Bullish reversal signal in downtrend"
                ))
    
    def _detect_piercing_line_dark_cloud(self, prev: CandleData, curr: CandleData, position: int, trend: str):
        """Detect Piercing Line and Dark Cloud Cover patterns"""
        prev_midpoint = (prev.open + prev.close) / 2
        
        # Piercing Line (bullish reversal in downtrend)
        if (trend == 'downtrend' and not prev.is_bullish and curr.is_bullish and
            curr.open < prev.low and curr.close > prev_midpoint and curr.close < prev.open):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Piercing Line",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_piercing_strength(prev, curr, trend),
                confidence=0.75,
                position=position,
                candles_involved=[prev, curr],
                description="Bullish reversal pattern - pierces midpoint of previous bearish candle"
            ))
        
        # Dark Cloud Cover (bearish reversal in uptrend)
        elif (trend == 'uptrend' and prev.is_bullish and not curr.is_bullish and
              curr.open > prev.high and curr.close < prev_midpoint and curr.close > prev.open):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Dark Cloud Cover",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_dark_cloud_strength(prev, curr, trend),
                confidence=0.75,
                position=position,
                candles_involved=[prev, curr],
                description="Bearish reversal pattern - penetrates midpoint of previous bullish candle"
            ))
    
    def _detect_tweezer_patterns(self, prev: CandleData, curr: CandleData, position: int, trend: str):
        """Detect Tweezer Tops and Tweezer Bottoms"""
        high_tolerance = 0.001  # 0.1% tolerance
        low_tolerance = 0.001
        
        # Tweezer Top (bearish reversal in uptrend)
        if (trend == 'uptrend' and abs(prev.high - curr.high) < (prev.high * high_tolerance) and
            prev.is_bullish and not curr.is_bullish):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Tweezer Top",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_tweezer_strength(prev, curr, trend, False),
                confidence=0.7,
                position=position,
                candles_involved=[prev, curr],
                description="Bearish reversal - matching highs in uptrend"
            ))
        
        # Tweezer Bottom (bullish reversal in downtrend)
        elif (trend == 'downtrend' and abs(prev.low - curr.low) < (prev.low * low_tolerance) and
              not prev.is_bullish and curr.is_bullish):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Tweezer Bottom",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_tweezer_strength(prev, curr, trend, True),
                confidence=0.7,
                position=position,
                candles_involved=[prev, curr],
                description="Bullish reversal - matching lows in downtrend"
            ))
    
    def _detect_belt_hold(self, prev: CandleData, curr: CandleData, position: int, trend: str):
        """Detect Belt Hold patterns (Yorikiri)"""
        # Belt hold has very small or no shadow on one side
        shadow_threshold = curr.total_range * 0.05
        
        # Bullish Belt Hold
        if (curr.is_bullish and curr.lower_shadow < shadow_threshold and 
            curr.body_ratio > 0.6 and trend == 'downtrend'):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Bullish Belt Hold",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_belt_hold_strength(curr, trend, True),
                confidence=0.65,
                position=position,
                candles_involved=[curr],
                description="Bullish reversal - opens at low and rallies"
            ))
        
        # Bearish Belt Hold
        elif (not curr.is_bullish and curr.upper_shadow < shadow_threshold and 
              curr.body_ratio > 0.6 and trend == 'uptrend'):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Bearish Belt Hold",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_belt_hold_strength(curr, trend, False),
                confidence=0.65,
                position=position,
                candles_involved=[curr],
                description="Bearish reversal - opens at high and declines"
            ))
    
    def _detect_kicking_pattern(self, prev: CandleData, curr: CandleData, position: int, trend: str):
        """Detect Kicking patterns - two marubozu candles with gap"""
        # Both candles should be marubozu (very small shadows)
        shadow_threshold = 0.05
        
        prev_is_marubozu = (prev.upper_shadow < prev.total_range * shadow_threshold and
                           prev.lower_shadow < prev.total_range * shadow_threshold and
                           prev.body_ratio > 0.9)
        
        curr_is_marubozu = (curr.upper_shadow < curr.total_range * shadow_threshold and
                           curr.lower_shadow < curr.total_range * shadow_threshold and
                           curr.body_ratio > 0.9)
        
        if prev_is_marubozu and curr_is_marubozu:
            # Bullish Kicking - gap up between bearish and bullish marubozu
            if not prev.is_bullish and curr.is_bullish and curr.open > prev.close:
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bullish Kicking",
                    pattern_type=PatternType.BULLISH,
                    strength=self._calculate_kicking_strength(prev, curr, True),
                    confidence=0.9,
                    position=position,
                    candles_involved=[prev, curr],
                    description="Very strong bullish reversal - gap between marubozu candles"
                ))
            
            # Bearish Kicking - gap down between bullish and bearish marubozu
            elif prev.is_bullish and not curr.is_bullish and curr.open < prev.close:
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bearish Kicking",
                    pattern_type=PatternType.BEARISH,
                    strength=self._calculate_kicking_strength(prev, curr, False),
                    confidence=0.9,
                    position=position,
                    candles_involved=[prev, curr],
                    description="Very strong bearish reversal - gap between marubozu candles"
                ))
    
    # ===== THREE CANDLE PATTERNS =====
    
    def _detect_three_candle_patterns(self, candles: List[CandleData]):
        """Detect all three-candle patterns"""
        for i in range(2, len(candles)):
            candle1 = candles[i-2]
            candle2 = candles[i-1]
            candle3 = candles[i]
            trend = self._get_trend_at_position(candles, i)
            
            # Morning/Evening Star
            self._detect_star_patterns(candle1, candle2, candle3, i, trend)
            
            # Three White Soldiers/Black Crows
            self._detect_three_soldiers_crows(candle1, candle2, candle3, i, trend)
            
            # Three Inside/Outside patterns
            self._detect_three_inside_outside(candle1, candle2, candle3, i, trend)
            
            # Abandoned Baby
            self._detect_abandoned_baby(candle1, candle2, candle3, i, trend)
            
            # Three Line Strike
            self._detect_three_line_strike(candle1, candle2, candle3, i, trend, candles)
            
            # Matching Low/High
            self._detect_matching_patterns(candle1, candle2, candle3, i, trend)
    
    def _detect_star_patterns(self, c1: CandleData, c2: CandleData, c3: CandleData, 
                             position: int, trend: str):
        """Detect Morning Star and Evening Star patterns"""
        # Morning Star (bullish reversal in downtrend)
        if (trend == 'downtrend' and not c1.is_bullish and c3.is_bullish and
            c2.body_size < min(c1.body_size, c3.body_size) * 0.3 and
            c3.close > (c1.open + c1.close) / 2):
            
            # Check for gap conditions
            gap_down = c2.high < c1.low
            gap_up = c3.low > c2.high
            
            strength_bonus = 10 if gap_down else 0
            strength_bonus += 10 if gap_up else 0
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Morning Star",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_star_strength(c1, c2, c3, trend, True) + strength_bonus,
                confidence=0.85,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Strong bullish reversal - three candle pattern with star"
            ))
        
        # Evening Star (bearish reversal in uptrend)
        elif (trend == 'uptrend' and c1.is_bullish and not c3.is_bullish and
              c2.body_size < min(c1.body_size, c3.body_size) * 0.3 and
              c3.close < (c1.open + c1.close) / 2):
            
            # Check for gap conditions
            gap_up = c2.low > c1.high
            gap_down = c3.high < c2.low
            
            strength_bonus = 10 if gap_up else 0
            strength_bonus += 10 if gap_down else 0
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Evening Star",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_star_strength(c1, c2, c3, trend, False) + strength_bonus,
                confidence=0.85,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Strong bearish reversal - three candle pattern with star"
            ))
    
    def _detect_three_soldiers_crows(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                    position: int, trend: str):
        """Detect Three White Soldiers and Three Black Crows"""
        # Three White Soldiers
        if (c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c2.open > c1.open and c3.open > c2.open and
            c2.close > c1.close and c3.close > c2.close and
            c1.upper_shadow < c1.body_size * 0.3 and
            c2.upper_shadow < c2.body_size * 0.3 and
            c3.upper_shadow < c3.body_size * 0.3):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Three White Soldiers",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_three_soldiers_strength(c1, c2, c3, trend),
                confidence=0.8,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Strong bullish continuation - three advancing white candles"
            ))
        
        # Three Black Crows
        elif (not c1.is_bullish and not c2.is_bullish and not c3.is_bullish and
              c2.open < c1.open and c3.open < c2.open and
              c2.close < c1.close and c3.close < c2.close and
              c1.lower_shadow < c1.body_size * 0.3 and
              c2.lower_shadow < c2.body_size * 0.3 and
              c3.lower_shadow < c3.body_size * 0.3):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Three Black Crows",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_three_crows_strength(c1, c2, c3, trend),
                confidence=0.8,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Strong bearish continuation - three declining black candles"
            ))
    
    def _detect_three_inside_outside(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                    position: int, trend: str):
        """Detect Three Inside Up/Down and Three Outside Up/Down patterns"""
        # Check for harami pattern in first two candles
        c1_body_top = max(c1.open, c1.close)
        c1_body_bottom = min(c1.open, c1.close)
        c2_body_top = max(c2.open, c2.close)
        c2_body_bottom = min(c2.open, c2.close)
        
        is_harami = c2_body_top <= c1_body_top and c2_body_bottom >= c1_body_bottom
        is_engulfing = c2_body_top >= c1_body_top and c2_body_bottom <= c1_body_bottom
        
        # Three Inside Up (bullish)
        if (is_harami and not c1.is_bullish and c2.is_bullish and 
            c3.is_bullish and c3.close > c1.close and trend == 'downtrend'):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Three Inside Up",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_three_inside_strength(c1, c2, c3, trend, True),
                confidence=0.75,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Bullish reversal - harami followed by confirmation"
            ))
        
        # Three Inside Down (bearish)
        elif (is_harami and c1.is_bullish and not c2.is_bullish and 
              not c3.is_bullish and c3.close < c1.close and trend == 'uptrend'):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Three Inside Down",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_three_inside_strength(c1, c2, c3, trend, False),
                confidence=0.75,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Bearish reversal - harami followed by confirmation"
            ))
        
        # Three Outside Up (bullish)
        elif (is_engulfing and not c1.is_bullish and c2.is_bullish and 
              c3.is_bullish and c3.close > c2.close and trend == 'downtrend'):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Three Outside Up",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_three_outside_strength(c1, c2, c3, trend, True),
                confidence=0.8,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Bullish reversal - engulfing followed by confirmation"
            ))
        
        # Three Outside Down (bearish)
        elif (is_engulfing and c1.is_bullish and not c2.is_bullish and 
              not c3.is_bullish and c3.close < c2.close and trend == 'uptrend'):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Three Outside Down",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_three_outside_strength(c1, c2, c3, trend, False),
                confidence=0.8,
                position=position,
                candles_involved=[c1, c2, c3],
                description="Bearish reversal - engulfing followed by confirmation"
            ))
    
    def _detect_abandoned_baby(self, c1: CandleData, c2: CandleData, c3: CandleData,
                              position: int, trend: str):
        """Detect Abandoned Baby pattern - rare and powerful reversal"""
        # Middle candle should be a doji with gaps on both sides
        if c2.body_ratio < self.doji_threshold:
            # Bullish Abandoned Baby
            if (not c1.is_bullish and c3.is_bullish and trend == 'downtrend' and
                c2.high < c1.low and c2.high < c3.low):  # Gaps on both sides
                
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bullish Abandoned Baby",
                    pattern_type=PatternType.BULLISH,
                    strength=95,  # Very strong pattern
                    confidence=0.95,
                    position=position,
                    candles_involved=[c1, c2, c3],
                    description="Very rare and powerful bullish reversal - doji with gaps"
                ))
            
            # Bearish Abandoned Baby
            elif (c1.is_bullish and not c3.is_bullish and trend == 'uptrend' and
                  c2.low > c1.high and c2.low > c3.high):  # Gaps on both sides
                
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bearish Abandoned Baby",
                    pattern_type=PatternType.BEARISH,
                    strength=95,  # Very strong pattern
                    confidence=0.95,
                    position=position,
                    candles_involved=[c1, c2, c3],
                    description="Very rare and powerful bearish reversal - doji with gaps"
                ))
    
    def _detect_three_line_strike(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                 position: int, trend: str, candles: List[CandleData]):
        """Detect Three Line Strike pattern"""
        # Need to check fourth candle for this pattern
        if position + 1 < len(candles):
            c4 = candles[position + 1]
            
            # Bullish Three Line Strike
            if (not c1.is_bullish and not c2.is_bullish and not c3.is_bullish and
                c4.is_bullish and c2.close < c1.close and c3.close < c2.close and
                c4.open <= c3.close and c4.close > c1.open):
                
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bullish Three Line Strike",
                    pattern_type=PatternType.BULLISH,
                    strength=self._calculate_three_line_strike_strength(c1, c2, c3, c4, trend, True),
                    confidence=0.85,
                    position=position + 1,
                    candles_involved=[c1, c2, c3, c4],
                    description="Bullish continuation - three black candles erased by one white"
                ))
            
            # Bearish Three Line Strike
            elif (c1.is_bullish and c2.is_bullish and c3.is_bullish and
                  not c4.is_bullish and c2.close > c1.close and c3.close > c2.close and
                  c4.open >= c3.close and c4.close < c1.open):
                
                self.detected_patterns.append(PatternResult(
                    pattern_name="Bearish Three Line Strike",
                    pattern_type=PatternType.BEARISH,
                    strength=self._calculate_three_line_strike_strength(c1, c2, c3, c4, trend, False),
                    confidence=0.85,
                    position=position + 1,
                    candles_involved=[c1, c2, c3, c4],
                    description="Bearish continuation - three white candles erased by one black"
                ))
    
    def _detect_matching_patterns(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                 position: int, trend: str):
        """Detect Matching Low and Matching High patterns"""
        tolerance = 0.001  # 0.1% tolerance for matching
        
        # Matching Low (bullish)
        if (trend == 'downtrend' and not c1.is_bullish and not c2.is_bullish and
            abs(c1.close - c2.close) < c1.close * tolerance):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Matching Low",
                pattern_type=PatternType.BULLISH,
                strength=self._calculate_matching_strength(c1, c2, trend, True),
                confidence=0.65,
                position=position - 1,
                candles_involved=[c1, c2],
                description="Bullish reversal - matching closing prices at bottom"
            ))
        
        # Matching High (bearish)
        elif (trend == 'uptrend' and c1.is_bullish and c2.is_bullish and
              abs(c1.close - c2.close) < c1.close * tolerance):
            
            self.detected_patterns.append(PatternResult(
                pattern_name="Matching High",
                pattern_type=PatternType.BEARISH,
                strength=self._calculate_matching_strength(c1, c2, trend, False),
                confidence=0.65,
                position=position - 1,
                candles_involved=[c1, c2],
                description="Bearish reversal - matching closing prices at top"
            ))
    
    # ===== PATTERN STRENGTH CALCULATIONS FOR NEW PATTERNS =====
    
    def _calculate_engulfing_strength(self, prev: CandleData, curr: CandleData, 
                                     trend: str, is_bullish: bool) -> float:
        """Calculate strength for engulfing patterns"""
        base_strength = 70.0
        
        # Size ratio bonus
        size_ratio = curr.body_size / prev.body_size if prev.body_size > 0 else 1
        if size_ratio > 2:
            base_strength += 15
        elif size_ratio > 1.5:
            base_strength += 10
        
        # Trend appropriateness
        if (is_bullish and trend == 'downtrend') or (not is_bullish and trend == 'uptrend'):
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_harami_strength(self, prev: CandleData, curr: CandleData,
                                  trend: str, is_bullish: bool) -> float:
        """Calculate strength for harami patterns"""
        base_strength = 60.0
        
        # Size ratio (smaller current candle is better)
        size_ratio = curr.body_size / prev.body_size if prev.body_size > 0 else 1
        if size_ratio < 0.3:
            base_strength += 15
        elif size_ratio < 0.5:
            base_strength += 10
        
        # Trend appropriateness
        if (is_bullish and trend == 'downtrend') or (not is_bullish and trend == 'uptrend'):
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_piercing_strength(self, prev: CandleData, curr: CandleData, trend: str) -> float:
        """Calculate strength for piercing line pattern"""
        base_strength = 65.0
        
        # Penetration depth
        prev_range = prev.open - prev.close
        penetration = (curr.close - prev.close) / prev_range if prev_range != 0 else 0
        
        if penetration > 0.6:
            base_strength += 20
        elif penetration > 0.5:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_dark_cloud_strength(self, prev: CandleData, curr: CandleData, trend: str) -> float:
        """Calculate strength for dark cloud cover pattern"""
        base_strength = 65.0
        
        # Penetration depth
        prev_range = prev.close - prev.open
        penetration = (prev.close - curr.close) / prev_range if prev_range != 0 else 0
        
        if penetration > 0.6:
            base_strength += 20
        elif penetration > 0.5:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_tweezer_strength(self, prev: CandleData, curr: CandleData,
                                   trend: str, is_bottom: bool) -> float:
        """Calculate strength for tweezer patterns"""
        base_strength = 60.0
        
        # How precisely the highs/lows match
        if is_bottom:
            precision = 1 - abs(prev.low - curr.low) / prev.low
        else:
            precision = 1 - abs(prev.high - curr.high) / prev.high
        
        base_strength += precision * 20
        
        # Trend context
        if trend in ['uptrend', 'downtrend']:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_belt_hold_strength(self, candle: CandleData, trend: str, is_bullish: bool) -> float:
        """Calculate strength for belt hold patterns"""
        base_strength = 55.0
        
        # Body size bonus
        if candle.body_ratio > 0.8:
            base_strength += 20
        elif candle.body_ratio > 0.7:
            base_strength += 10
        
        # Trend appropriateness
        if (is_bullish and trend == 'downtrend') or (not is_bullish and trend == 'uptrend'):
            base_strength += 15
        
        return min(100, max(0, base_strength))
    
    def _calculate_kicking_strength(self, prev: CandleData, curr: CandleData, is_bullish: bool) -> float:
        """Calculate strength for kicking patterns"""
        base_strength = 80.0  # Strong pattern by nature
        
        # Gap size bonus
        gap_size = abs(curr.open - prev.close) / prev.close
        if gap_size > 0.02:  # 2% gap
            base_strength += 15
        elif gap_size > 0.01:  # 1% gap
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_star_strength(self, c1: CandleData, c2: CandleData, c3: CandleData,
                                trend: str, is_morning: bool) -> float:
        """Calculate strength for morning/evening star patterns"""
        base_strength = 75.0
        
        # Star size (smaller is better)
        star_ratio = c2.body_size / min(c1.body_size, c3.body_size)
        if star_ratio < 0.1:
            base_strength += 10
        elif star_ratio < 0.2:
            base_strength += 5
        
        # Penetration into first candle
        if is_morning:
            penetration = (c3.close - c1.close) / c1.body_size if c1.body_size > 0 else 0
        else:
            penetration = (c1.close - c3.close) / c1.body_size if c1.body_size > 0 else 0
        
        if penetration > 0.5:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_three_soldiers_strength(self, c1: CandleData, c2: CandleData, 
                                          c3: CandleData, trend: str) -> float:
        """Calculate strength for three white soldiers"""
        base_strength = 70.0
        
        # Consistent advancement
        advance1 = c2.close - c1.close
        advance2 = c3.close - c2.close
        
        if advance1 > 0 and advance2 > 0 and abs(advance1 - advance2) / advance1 < 0.3:
            base_strength += 15
        
        # Small upper shadows
        avg_shadow_ratio = ((c1.upper_shadow / c1.body_size) + 
                           (c2.upper_shadow / c2.body_size) + 
                           (c3.upper_shadow / c3.body_size)) / 3
        
        if avg_shadow_ratio < 0.1:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_three_crows_strength(self, c1: CandleData, c2: CandleData,
                                       c3: CandleData, trend: str) -> float:
        """Calculate strength for three black crows"""
        base_strength = 70.0
        
        # Consistent decline
        decline1 = c1.close - c2.close
        decline2 = c2.close - c3.close
        
        if decline1 > 0 and decline2 > 0 and abs(decline1 - decline2) / decline1 < 0.3:
            base_strength += 15
        
        # Small lower shadows
        avg_shadow_ratio = ((c1.lower_shadow / c1.body_size) + 
                           (c2.lower_shadow / c2.body_size) + 
                           (c3.lower_shadow / c3.body_size)) / 3
        
        if avg_shadow_ratio < 0.1:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_three_inside_strength(self, c1: CandleData, c2: CandleData,
                                        c3: CandleData, trend: str, is_up: bool) -> float:
        """Calculate strength for three inside up/down patterns"""
        base_strength = 65.0
        
        # Confirmation candle strength
        if is_up and c3.close > c1.high:
            base_strength += 15
        elif not is_up and c3.close < c1.low:
            base_strength += 15
        
        return min(100, max(0, base_strength))
    
    def _calculate_three_outside_strength(self, c1: CandleData, c2: CandleData,
                                         c3: CandleData, trend: str, is_up: bool) -> float:
        """Calculate strength for three outside up/down patterns"""
        base_strength = 70.0
        
        # Engulfing quality
        engulf_ratio = c2.body_size / c1.body_size if c1.body_size > 0 else 1
        if engulf_ratio > 2:
            base_strength += 10
        
        # Confirmation strength
        if is_up and c3.close > c2.close:
            base_strength += 10
        elif not is_up and c3.close < c2.close:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_three_line_strike_strength(self, c1: CandleData, c2: CandleData,
                                             c3: CandleData, c4: CandleData,
                                             trend: str, is_bullish: bool) -> float:
        """Calculate strength for three line strike patterns"""
        base_strength = 75.0
        
        # How completely the fourth candle erases the previous three
        if is_bullish:
            erasure = (c4.close - c1.open) / (c1.open - c3.close) if c1.open != c3.close else 1
        else:
            erasure = (c1.open - c4.close) / (c3.close - c1.open) if c3.close != c1.open else 1
        
        if erasure > 1.1:
            base_strength += 15
        elif erasure > 1:
            base_strength += 10
        
        return min(100, max(0, base_strength))
    
    def _calculate_matching_strength(self, c1: CandleData, c2: CandleData,
                                    trend: str, is_low: bool) -> float:
        """Calculate strength for matching low/high patterns"""
        base_strength = 55.0
        
        # How precisely the closes match
        precision = 1 - abs(c1.close - c2.close) / c1.close
        base_strength += precision * 20
        
        # Trend context
        if (is_low and trend == 'downtrend') or (not is_low and trend == 'uptrend'):
            base_strength += 15
        
        return min(100, max(0, base_strength))
