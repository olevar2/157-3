# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Belt Hold Pattern Scanner - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

Belt Hold (Yorikiri) is a single-candle reversal pattern that appears at trend
extremes. It's characterized by a long body with little to no shadow on one side,
indicating strong directional momentum that may signal trend reversal.

Pattern Characteristics:
- Single candle pattern
- Bullish Belt Hold: Opens at/near low, closes near high (no lower shadow)
- Bearish Belt Hold: Opens at/near high, closes near low (no upper shadow)
- Long body relative to recent candles
- Appears at trend extremes

Key Features:
- Strong reversal signal potential
- Momentum shift indication
- Volume confirmation analysis
- Pattern strength measurement
- Support/resistance level validation

Trading Applications:
- Trend reversal identification
- Entry/exit timing optimization
- Momentum change detection
- Risk management enhancement
- Market sentiment analysis

Mathematical Foundation:
- Body Ratio > 60% of total range
- Shadow Ratio < 5% of total range (relevant side)
- Volume > Average Volume (preferred)
- Trend context validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame

class BeltHoldType(Enum):
    """Types of Belt Hold patterns"""
    BULLISH_BELT_HOLD = "bullish_belt_hold"     # Opens low, closes high
    BEARISH_BELT_HOLD = "bearish_belt_hold"     # Opens high, closes low
    STRONG_BULLISH = "strong_bullish_belt"      # Extra strong bullish
    STRONG_BEARISH = "strong_bearish_belt"      # Extra strong bearish

@dataclass
class CandleData:
    """Represents a single candlestick"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: any
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def body_ratio(self) -> float:
        return self.body_size / self.total_range if self.total_range > 0 else 0
    
    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

@dataclass
class BeltHoldResult:
    """Result of Belt Hold pattern detection"""
    pattern_type: BeltHoldType
    strength: float              # 0-100
    confidence: float            # 0-1
    position: int               # Index where pattern was found
    candle: CandleData          # The belt hold candle
    body_ratio: float           # Body size as percentage of total range
    shadow_ratio: float         # Relevant shadow as percentage
    volume_confirmation: bool   # Volume support
    reversal_level: float       # Key reversal level
    target_levels: List[float]  # Potential price targets
    stop_loss: float           # Suggested stop loss
    metadata: Dict[str, Any] = field(default_factory=dict)

class BeltHoldPattern(IndicatorBase):
    """
    Belt Hold Pattern Recognition and Analysis Engine
    
    Specialized scanner for detecting and analyzing Belt Hold patterns
    with comprehensive strength assessment and trading signal generation.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.6,
                 max_shadow_ratio: float = 0.05,
                 volume_threshold: float = 1.2,
                 trend_lookback: int = 10,
                 min_size_multiplier: float = 1.2):
        """
        Initialize Belt Hold pattern scanner
        
        Args:
            min_body_ratio: Minimum body size as percentage of total range (0.6 = 60%)
            max_shadow_ratio: Maximum shadow size as percentage (0.05 = 5%)
            volume_threshold: Volume multiplier vs average (1.2 = 20% above average)
            trend_lookback: Number of periods to analyze for trend context
            min_size_multiplier: Minimum candle size vs average (1.2 = 20% larger)
        """
        super().__init__()
        self.min_body_ratio = min_body_ratio
        self.max_shadow_ratio = max_shadow_ratio
        self.volume_threshold = volume_threshold
        self.trend_lookback = trend_lookback
        self.min_size_multiplier = min_size_multiplier
        
        self.detected_patterns: List[BeltHoldResult] = []
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate Belt Hold patterns in the given data
        
        Args:
            data: DataFrame with OHLCV data
              Returns:
            IndicatorResult containing detected patterns
        """
        try:
            if len(data) < 5:  # Need some history for context
                return IndicatorResult(
                    timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                    indicator_name="BeltHoldPattern",
                    indicator_type=IndicatorType.PATTERN,
                    timeframe=TimeFrame.D1,
                    value=[],
                    raw_data={"error": "Insufficient data for pattern detection"}
                )
            
            # Reset previous results
            self.detected_patterns = []
            
            # Convert to candle objects
            candles = self._create_candle_objects(data)
            
            # Detect patterns
            self._detect_belt_hold_patterns(candles)
            
            # Create signals
            signals = self._create_trading_signals()            
            return IndicatorResult(
                timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                indicator_name="BeltHoldPattern",
                indicator_type=IndicatorType.PATTERN,
                timeframe=TimeFrame.D1,
                value=self.detected_patterns,
                raw_data={
                    "total_patterns": len(self.detected_patterns),
                    "pattern_types": self._get_pattern_type_counts(),
                    "strongest_pattern": self._get_strongest_pattern()                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in Belt Hold calculation: {e}")
            return IndicatorResult(
                timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                indicator_name="BeltHoldPattern",
                indicator_type=IndicatorType.PATTERN,
                timeframe=TimeFrame.D1,
                value=[],
                raw_data={"error": str(e)}
            )
    
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
                timestamp=idx
            )
            candles.append(candle)
        
        return candles
    
    def _detect_belt_hold_patterns(self, candles: List[CandleData]):
        """Detect Belt Hold patterns in the candle data"""
        for i in range(self.trend_lookback, len(candles)):
            candle = candles[i]
            
            # Check for Bullish Belt Hold
            if self._is_bullish_belt_hold(candle):
                self._create_belt_hold_result(candle, candles, i, is_bullish=True)
            
            # Check for Bearish Belt Hold
            elif self._is_bearish_belt_hold(candle):
                self._create_belt_hold_result(candle, candles, i, is_bullish=False)
    
    def _is_bullish_belt_hold(self, candle: CandleData) -> bool:
        """Check if candle forms a Bullish Belt Hold pattern"""
        # Must be bullish
        if not candle.is_bullish:
            return False
        
        # Body ratio check
        if candle.body_ratio < self.min_body_ratio:
            return False
        
        # Lower shadow should be minimal (opens at/near low)
        if candle.total_range > 0:
            lower_shadow_ratio = candle.lower_shadow / candle.total_range
            if lower_shadow_ratio > self.max_shadow_ratio:
                return False
        
        return True
    
    def _is_bearish_belt_hold(self, candle: CandleData) -> bool:
        """Check if candle forms a Bearish Belt Hold pattern"""
        # Must be bearish
        if candle.is_bullish:
            return False
        
        # Body ratio check
        if candle.body_ratio < self.min_body_ratio:
            return False
        
        # Upper shadow should be minimal (opens at/near high)
        if candle.total_range > 0:
            upper_shadow_ratio = candle.upper_shadow / candle.total_range
            if upper_shadow_ratio > self.max_shadow_ratio:
                return False
        
        return True
    
    def _create_belt_hold_result(self, candle: CandleData, candles: List[CandleData], 
                               position: int, is_bullish: bool):
        """Create a Belt Hold pattern result"""
        # Check if candle is significant size
        avg_range = self._calculate_average_range(candles, position)
        if avg_range > 0 and candle.total_range < (avg_range * self.min_size_multiplier):
            return  # Candle too small
        
        # Determine pattern type and strength
        strength = self._calculate_pattern_strength(candle, candles, position, is_bullish)
        
        # Only proceed if strength is adequate
        if strength < 50:
            return
        
        # Determine specific pattern type
        if is_bullish:
            if strength >= 80:
                pattern_type = BeltHoldType.STRONG_BULLISH
            else:
                pattern_type = BeltHoldType.BULLISH_BELT_HOLD
        else:
            if strength >= 80:
                pattern_type = BeltHoldType.STRONG_BEARISH
            else:
                pattern_type = BeltHoldType.BEARISH_BELT_HOLD
        
        # Calculate pattern metrics
        confidence = self._calculate_confidence(candle, is_bullish, strength)
        shadow_ratio = (candle.lower_shadow if is_bullish else candle.upper_shadow) / candle.total_range
        
        # Check volume confirmation
        volume_confirmation = self._check_volume_confirmation(candle, candles, position)
        
        # Calculate trading levels
        reversal_level = candle.close
        targets = self._calculate_target_levels(candle, candles, position, is_bullish)
        stop_loss = self._calculate_stop_loss(candle, is_bullish)
        
        # Create pattern result
        pattern = BeltHoldResult(
            pattern_type=pattern_type,
            strength=strength,
            confidence=confidence,
            position=position,
            candle=candle,
            body_ratio=candle.body_ratio,
            shadow_ratio=shadow_ratio,
            volume_confirmation=volume_confirmation,
            reversal_level=reversal_level,
            target_levels=targets,
            stop_loss=stop_loss,
            metadata={
                "trend_context": self._get_trend_context(candles, position),
                "size_vs_average": candle.total_range / avg_range if avg_range > 0 else 0,
                "atr": self._calculate_atr(candles, position)
            }
        )
        
        self.detected_patterns.append(pattern)
    
    def _calculate_average_range(self, candles: List[CandleData], position: int, period: int = 10) -> float:
        """Calculate average candle range over specified period"""
        start_idx = max(0, position - period)
        ranges = [c.total_range for c in candles[start_idx:position]]
        return np.mean(ranges) if ranges else 0.0
    
    def _calculate_pattern_strength(self, candle: CandleData, candles: List[CandleData], 
                                  position: int, is_bullish: bool) -> float:
        """Calculate the strength of the Belt Hold pattern (0-100)"""
        strength = 50.0  # Base strength
        
        # Body ratio bonus
        body_bonus = (candle.body_ratio - self.min_body_ratio) * 100
        strength += min(20, body_bonus)
        
        # Shadow ratio bonus (smaller shadow = stronger)
        relevant_shadow = candle.lower_shadow if is_bullish else candle.upper_shadow
        shadow_ratio = relevant_shadow / candle.total_range if candle.total_range > 0 else 0
        shadow_bonus = (self.max_shadow_ratio - shadow_ratio) * 200
        strength += min(15, max(0, shadow_bonus))
        
        # Size bonus (larger candle = stronger)
        avg_range = self._calculate_average_range(candles, position)
        if avg_range > 0:
            size_ratio = candle.total_range / avg_range
            if size_ratio > 2.0:
                strength += 15
            elif size_ratio > 1.5:
                strength += 10
            elif size_ratio > 1.2:
                strength += 5
        
        # Trend context bonus
        trend = self._get_trend_context(candles, position)
        if is_bullish and trend in ["downtrend", "strong_downtrend"]:
            strength += 15
        elif not is_bullish and trend in ["uptrend", "strong_uptrend"]:
            strength += 15
        
        # Volume confirmation bonus
        if self._check_volume_confirmation(candle, candles, position):
            strength += 10
        
        # Gap bonus (if there's a gap)
        if position > 0:
            prev_candle = candles[position - 1]
            if is_bullish and candle.open > prev_candle.high:
                strength += 10  # Gap up
            elif not is_bullish and candle.open < prev_candle.low:
                strength += 10  # Gap down
        
        return min(100, max(0, strength))
    
    def _calculate_confidence(self, candle: CandleData, is_bullish: bool, strength: float) -> float:
        """Calculate confidence level (0-1)"""
        base_confidence = 0.6
        
        # Strength influence
        strength_bonus = (strength - 50) / 100  # Normalize to 0-0.5
        base_confidence += strength_bonus * 0.3
        
        # Body ratio influence
        if candle.body_ratio > 0.8:
            base_confidence += 0.1
        elif candle.body_ratio > 0.7:
            base_confidence += 0.05
        
        # Shadow minimality influence
        relevant_shadow = candle.lower_shadow if is_bullish else candle.upper_shadow
        shadow_ratio = relevant_shadow / candle.total_range if candle.total_range > 0 else 0
        if shadow_ratio < 0.02:  # Very small shadow
            base_confidence += 0.1
        elif shadow_ratio < 0.03:
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _get_trend_context(self, candles: List[CandleData], position: int) -> str:
        """Determine trend context at the pattern position"""
        if position < self.trend_lookback:
            return "insufficient_data"
        
        lookback_candles = candles[position - self.trend_lookback:position]
        closes = [c.close for c in lookback_candles]
        
        # Simple trend analysis
        start_price = closes[0]
        end_price = closes[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        
        if change_pct >= 15:
            return "strong_uptrend"
        elif change_pct >= 5:
            return "uptrend"
        elif change_pct <= -15:
            return "strong_downtrend"
        elif change_pct <= -5:
            return "downtrend"
        else:
            return "sideways"
    
    def _check_volume_confirmation(self, candle: CandleData, 
                                 candles: List[CandleData], position: int) -> bool:
        """Check if volume confirms the pattern"""
        if position < 10:
            return False
        
        recent_volumes = [c.volume for c in candles[position-10:position]]
        avg_volume = np.mean(recent_volumes)
        
        return candle.volume >= (avg_volume * self.volume_threshold)
    
    def _calculate_target_levels(self, candle: CandleData, candles: List[CandleData], 
                               position: int, is_bullish: bool) -> List[float]:
        """Calculate potential price targets"""
        atr = self._calculate_atr(candles, position)
        
        if is_bullish:
            # Bullish targets
            target1 = candle.close + (atr * 1.0)
            target2 = candle.close + (atr * 2.0)
            target3 = candle.close + (atr * 3.0)
        else:
            # Bearish targets
            target1 = candle.close - (atr * 1.0)
            target2 = candle.close - (atr * 2.0)
            target3 = candle.close - (atr * 3.0)
        
        return [target1, target2, target3]
    
    def _calculate_stop_loss(self, candle: CandleData, is_bullish: bool) -> float:
        """Calculate suggested stop loss level"""
        if is_bullish:
            # Stop below the low (but the low should be very close to open)
            return candle.low - (candle.body_size * 0.1)
        else:
            # Stop above the high (but the high should be very close to open)
            return candle.high + (candle.body_size * 0.1)
    
    def _calculate_atr(self, candles: List[CandleData], position: int, period: int = 14) -> float:
        """Calculate Average True Range"""
        if position < period:
            return 0.0
        
        tr_values = []
        for i in range(position - period, position):
            if i > 0:
                high_low = candles[i].high - candles[i].low
                high_close = abs(candles[i].high - candles[i-1].close)
                low_close = abs(candles[i].low - candles[i-1].close)
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
        
        return np.mean(tr_values) if tr_values else 0.0
    
    def _create_trading_signals(self) -> List[Dict[str, Any]]:
        """Create trading signals from detected patterns"""
        signals = []
        
        for pattern in self.detected_patterns:
            if pattern.strength >= 65:  # Quality threshold
                signal_type = "BUY" if "bullish" in pattern.pattern_type.value else "SELL"
                
                signal = {
                    "type": signal_type,
                    "pattern": f"{pattern.pattern_type.value.replace('_', ' ').title()}",
                    "strength": pattern.strength,
                    "confidence": pattern.confidence,
                    "entry_price": pattern.candle.close,
                    "stop_loss": pattern.stop_loss,
                    "targets": pattern.target_levels,
                    "position": pattern.position,
                    "volume_confirmed": pattern.volume_confirmation,
                    "body_ratio": pattern.body_ratio,
                    "shadow_ratio": pattern.shadow_ratio
                }
                signals.append(signal)
        
        return signals
    
    def _get_pattern_type_counts(self) -> Dict[str, int]:
        """Get count of each pattern type"""
        counts = {ptype.value: 0 for ptype in BeltHoldType}
        for pattern in self.detected_patterns:
            counts[pattern.pattern_type.value] += 1
        return counts
    
    def _get_strongest_pattern(self) -> Optional[Dict[str, Any]]:
        """Get the strongest detected pattern"""
        if not self.detected_patterns:
            return None
        
        strongest = max(self.detected_patterns, key=lambda p: p.strength)
        return {
            "strength": strongest.strength,
            "confidence": strongest.confidence,
            "position": strongest.position,
            "pattern_type": strongest.pattern_type.value,
            "body_ratio": strongest.body_ratio,
            "shadow_ratio": strongest.shadow_ratio
        }
    
    def get_pattern_description(self) -> str:
        """Get description of the pattern"""
        return (
            "Belt Hold (Yorikiri): A single-candle reversal pattern with a long body "
            "and minimal shadow on one side. Bullish Belt Hold opens near low and closes "
            "near high; Bearish Belt Hold opens near high and closes near low. "
            "Indicates strong momentum that may signal trend reversal."
        )
