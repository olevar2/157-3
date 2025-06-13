# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Tweezer Patterns Scanner - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

Tweezer patterns are two-candle reversal formations where the candles have
nearly identical highs (Tweezer Top) or lows (Tweezer Bottom). These patterns
suggest potential trend reversals when they occur at significant price levels.

Pattern Characteristics:
- Two-candle pattern
- Tweezer Top: Similar highs in uptrend (bearish reversal)
- Tweezer Bottom: Similar lows in downtrend (bullish reversal)
- High/low levels should be within small tolerance
- Different colored candles preferred

Key Features:
- Support/resistance level identification
- Trend reversal signal generation
- Pattern strength measurement
- Volume confirmation analysis
- Entry/exit timing optimization

Trading Applications:
- Reversal identification at key levels
- Support/resistance validation
- Entry/exit signal generation
- Risk management enhancement
- Market turning point detection

Mathematical Foundation:
- Tolerance: |High1 - High2| / High1 < threshold (typically 0.1-0.5%)
- Volume confirmation: Volume2 > Average_Volume
- Trend context validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from engines.indicator_base import (
    IndicatorBase,
    IndicatorResult,
    IndicatorType,
    TimeFrame,
)


class TweezerType(Enum):
    """Types of Tweezer patterns"""

    TWEEZER_TOP = "tweezer_top"  # Bearish reversal at highs
    TWEEZER_BOTTOM = "tweezer_bottom"  # Bullish reversal at lows
    DOJI_TWEEZER_TOP = "doji_tweezer_top"  # Tweezer top with doji
    DOJI_TWEEZER_BOTTOM = "doji_tweezer_bottom"  # Tweezer bottom with doji


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
    def is_doji(self) -> bool:
        total_range = self.high - self.low
        return total_range > 0 and self.body_size / total_range < 0.1

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class TweezerResult:
    """Result of Tweezer pattern detection"""

    pattern_type: TweezerType
    strength: float  # 0-100
    confidence: float  # 0-1
    position: int  # Index where pattern was found
    candle1: CandleData  # First candle
    candle2: CandleData  # Second candle
    level_tolerance: float  # Tolerance in price matching
    volume_confirmation: bool  # Volume support
    key_level: float  # Support/resistance level
    target_levels: List[float]  # Potential price targets
    stop_loss: float  # Suggested stop loss
    metadata: Dict[str, Any] = field(default_factory=dict)


class TweezerPatterns(IndicatorBase):
    """
    Tweezer Patterns Recognition and Analysis Engine

    Specialized scanner for detecting and analyzing Tweezer Top and Bottom patterns
    with comprehensive strength assessment and trading signal generation.
    """

    def __init__(
        self,
        tolerance_percentage: float = 0.002,  # 0.2% tolerance
        volume_threshold: float = 1.1,
        trend_lookback: int = 10,
        min_candle_size: float = 0.1,
    ):
        """
        Initialize Tweezer pattern scanner

        Args:
            tolerance_percentage: Maximum price difference as percentage (0.002 = 0.2%)
            volume_threshold: Volume multiplier vs average (1.1 = 10% above average)
            trend_lookback: Number of periods to analyze for trend context
            min_candle_size: Minimum candle range as percentage of average
        """
        super().__init__()
        self.tolerance_percentage = tolerance_percentage
        self.volume_threshold = volume_threshold
        self.trend_lookback = trend_lookback
        self.min_candle_size = min_candle_size

        self.detected_patterns: List[TweezerResult] = []
        self.logger = logging.getLogger(__name__)

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate Tweezer patterns in the given data

        Args:            data: DataFrame with OHLCV data

        Returns:
            IndicatorResult containing detected patterns
        """
        try:
            if len(data) < 2:
                return IndicatorResult(
                    timestamp=datetime.now(),
                    indicator_name="Tweezer Patterns",
                    indicator_type=IndicatorType.PATTERN,
                    timeframe=TimeFrame.D1,
                    value=[],
                    signal=[],
                    raw_data={"error": "Insufficient data for pattern detection"},
                )

            # Reset previous results
            self.detected_patterns = []

            # Convert to candle objects
            candles = self._create_candle_objects(data)

            # Detect patterns
            self._detect_tweezer_patterns(candles)
            # Create signals
            signals = self._create_trading_signals()

            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name="Tweezer Patterns",
                indicator_type=IndicatorType.PATTERN,
                timeframe=TimeFrame.D1,
                value=self.detected_patterns,
                signal=signals,
                raw_data={
                    "total_patterns": len(self.detected_patterns),
                    "pattern_types": self._get_pattern_type_counts(),
                    "strongest_pattern": self._get_strongest_pattern(),
                },
            )

        except Exception as e:
            self.logger.error(f"Error in Tweezer pattern calculation: {e}")
            return IndicatorResult(
                timestamp=datetime.now(),
                indicator_name="Tweezer Patterns",
                indicator_type=IndicatorType.PATTERN,
                timeframe=TimeFrame.D1,
                value=[],
                signal=[],
                raw_data={"error": str(e)},
            )

    def _create_candle_objects(self, data: pd.DataFrame) -> List[CandleData]:
        """Convert DataFrame to CandleData objects"""
        candles = []

        for idx, row in data.iterrows():
            candle = CandleData(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timestamp=idx,
            )
            candles.append(candle)

        return candles

    def _detect_tweezer_patterns(self, candles: List[CandleData]):
        """Detect Tweezer patterns in the candle data"""
        for i in range(1, len(candles)):
            candle1 = candles[i - 1]  # First candle
            candle2 = candles[i]  # Second candle

            # Check for Tweezer Top
            if self._is_tweezer_top(candle1, candle2):
                self._create_tweezer_result(candle1, candle2, candles, i, is_top=True)

            # Check for Tweezer Bottom
            elif self._is_tweezer_bottom(candle1, candle2):
                self._create_tweezer_result(candle1, candle2, candles, i, is_top=False)

    def _is_tweezer_top(self, candle1: CandleData, candle2: CandleData) -> bool:
        """Check if two candles form a Tweezer Top pattern"""
        # Check if highs are similar within tolerance
        high_diff = abs(candle1.high - candle2.high)
        tolerance = candle1.high * self.tolerance_percentage

        if high_diff > tolerance:
            return False

        # Prefer different colored candles (not required but stronger)
        # At least one candle should have meaningful size
        avg_range = (candle1.high - candle1.low + candle2.high - candle2.low) / 2
        if avg_range == 0:
            return False

        return True

    def _is_tweezer_bottom(self, candle1: CandleData, candle2: CandleData) -> bool:
        """Check if two candles form a Tweezer Bottom pattern"""
        # Check if lows are similar within tolerance
        low_diff = abs(candle1.low - candle2.low)
        tolerance = candle1.low * self.tolerance_percentage

        if low_diff > tolerance:
            return False

        # At least one candle should have meaningful size
        avg_range = (candle1.high - candle1.low + candle2.high - candle2.low) / 2
        if avg_range == 0:
            return False

        return True

    def _create_tweezer_result(
        self,
        candle1: CandleData,
        candle2: CandleData,
        candles: List[CandleData],
        position: int,
        is_top: bool,
    ):
        """Create a Tweezer pattern result"""
        # Determine pattern type
        if is_top:
            if candle1.is_doji or candle2.is_doji:
                pattern_type = TweezerType.DOJI_TWEEZER_TOP
            else:
                pattern_type = TweezerType.TWEEZER_TOP
            key_level = max(candle1.high, candle2.high)
        else:
            if candle1.is_doji or candle2.is_doji:
                pattern_type = TweezerType.DOJI_TWEEZER_BOTTOM
            else:
                pattern_type = TweezerType.TWEEZER_BOTTOM
            key_level = min(candle1.low, candle2.low)

        # Calculate pattern metrics
        strength = self._calculate_pattern_strength(
            candle1, candle2, candles, position, is_top
        )
        confidence = self._calculate_confidence(candle1, candle2, is_top)
        level_tolerance = self._calculate_level_tolerance(candle1, candle2, is_top)

        # Check volume confirmation
        volume_confirmation = self._check_volume_confirmation(
            candle2, candles, position
        )

        # Calculate trading levels
        targets = self._calculate_target_levels(candle1, candle2, is_top)
        stop_loss = self._calculate_stop_loss(candle1, candle2, is_top)

        # Create pattern result
        pattern = TweezerResult(
            pattern_type=pattern_type,
            strength=strength,
            confidence=confidence,
            position=position,
            candle1=candle1,
            candle2=candle2,
            level_tolerance=level_tolerance,
            volume_confirmation=volume_confirmation,
            key_level=key_level,
            target_levels=targets,
            stop_loss=stop_loss,
            metadata={
                "trend_context": self._get_trend_context(candles, position),
                "is_doji_pattern": candle1.is_doji or candle2.is_doji,
                "color_difference": candle1.is_bullish != candle2.is_bullish,
                "atr": self._calculate_atr(candles, position),
            },
        )

        self.detected_patterns.append(pattern)

    def _calculate_level_tolerance(
        self, candle1: CandleData, candle2: CandleData, is_top: bool
    ) -> float:
        """Calculate the actual tolerance used for level matching"""
        if is_top:
            return abs(candle1.high - candle2.high) / max(candle1.high, candle2.high)
        else:
            return abs(candle1.low - candle2.low) / max(candle1.low, candle2.low)

    def _calculate_pattern_strength(
        self,
        candle1: CandleData,
        candle2: CandleData,
        candles: List[CandleData],
        position: int,
        is_top: bool,
    ) -> float:
        """Calculate the strength of the Tweezer pattern (0-100)"""
        strength = 50.0  # Base strength

        # Tolerance bonus (tighter tolerance = stronger)
        tolerance = self._calculate_level_tolerance(candle1, candle2, is_top)
        if tolerance < 0.001:  # Very tight tolerance
            strength += 20
        elif tolerance < 0.002:
            strength += 15
        elif tolerance < 0.005:
            strength += 10

        # Color difference bonus
        if candle1.is_bullish != candle2.is_bullish:
            strength += 15

        # Doji presence bonus
        if candle1.is_doji or candle2.is_doji:
            strength += 10

        # Shadow length bonus (rejection of levels)
        if is_top:
            # Long upper shadows suggest strong rejection
            avg_body = (candle1.body_size + candle2.body_size) / 2
            if avg_body > 0:
                shadow1_ratio = candle1.upper_shadow / avg_body
                shadow2_ratio = candle2.upper_shadow / avg_body
                if shadow1_ratio > 1 or shadow2_ratio > 1:
                    strength += 10
        else:
            # Long lower shadows suggest strong rejection
            avg_body = (candle1.body_size + candle2.body_size) / 2
            if avg_body > 0:
                shadow1_ratio = candle1.lower_shadow / avg_body
                shadow2_ratio = candle2.lower_shadow / avg_body
                if shadow1_ratio > 1 or shadow2_ratio > 1:
                    strength += 10

        # Trend context bonus
        trend = self._get_trend_context(candles, position)
        if is_top and trend in ["uptrend", "strong_uptrend"]:
            strength += 15
        elif not is_top and trend in ["downtrend", "strong_downtrend"]:
            strength += 15

        # Volume confirmation bonus
        if self._check_volume_confirmation(candle2, candles, position):
            strength += 10

        return min(100, max(0, strength))

    def _calculate_confidence(
        self, candle1: CandleData, candle2: CandleData, is_top: bool
    ) -> float:
        """Calculate confidence level (0-1)"""
        base_confidence = 0.6

        # Tolerance influence
        tolerance = self._calculate_level_tolerance(candle1, candle2, is_top)
        if tolerance < 0.001:
            base_confidence += 0.2
        elif tolerance < 0.002:
            base_confidence += 0.15
        elif tolerance < 0.005:
            base_confidence += 0.1

        # Color difference influence
        if candle1.is_bullish != candle2.is_bullish:
            base_confidence += 0.1

        # Doji influence
        if candle1.is_doji or candle2.is_doji:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _get_trend_context(self, candles: List[CandleData], position: int) -> str:
        """Determine trend context at the pattern position"""
        if position < self.trend_lookback:
            return "insufficient_data"

        lookback_candles = candles[position - self.trend_lookback : position]
        closes = [c.close for c in lookback_candles]

        # Simple trend analysis
        start_price = closes[0]
        end_price = closes[-1]
        change_pct = ((end_price - start_price) / start_price) * 100

        if change_pct >= 10:
            return "strong_uptrend"
        elif change_pct >= 3:
            return "uptrend"
        elif change_pct <= -10:
            return "strong_downtrend"
        elif change_pct <= -3:
            return "downtrend"
        else:
            return "sideways"

    def _check_volume_confirmation(
        self, candle2: CandleData, candles: List[CandleData], position: int
    ) -> bool:
        """Check if volume confirms the pattern"""
        if position < 10:
            return False

        recent_volumes = [c.volume for c in candles[position - 10 : position]]
        avg_volume = np.mean(recent_volumes)

        return candle2.volume >= (avg_volume * self.volume_threshold)

    def _calculate_target_levels(
        self, candle1: CandleData, candle2: CandleData, is_top: bool
    ) -> List[float]:
        """Calculate potential price targets"""
        if is_top:
            # For bearish reversal
            pattern_high = max(candle1.high, candle2.high)
            pattern_low = min(candle1.low, candle2.low)
            pattern_height = pattern_high - pattern_low

            target1 = pattern_low - (pattern_height * 0.5)  # 50% projection
            target2 = pattern_low - pattern_height  # 100% projection
            target3 = pattern_low - (pattern_height * 1.618)  # Fibonacci extension
        else:
            # For bullish reversal
            pattern_high = max(candle1.high, candle2.high)
            pattern_low = min(candle1.low, candle2.low)
            pattern_height = pattern_high - pattern_low

            target1 = pattern_high + (pattern_height * 0.5)  # 50% projection
            target2 = pattern_high + pattern_height  # 100% projection
            target3 = pattern_high + (pattern_height * 1.618)  # Fibonacci extension

        return [target1, target2, target3]

    def _calculate_stop_loss(
        self, candle1: CandleData, candle2: CandleData, is_top: bool
    ) -> float:
        """Calculate suggested stop loss level"""
        if is_top:
            # Stop above the tweezer top
            return max(candle1.high, candle2.high) + (
                abs(candle2.close - candle2.open) * 0.2
            )
        else:
            # Stop below the tweezer bottom
            return min(candle1.low, candle2.low) - (
                abs(candle2.close - candle2.open) * 0.2
            )

    def _calculate_atr(
        self, candles: List[CandleData], position: int, period: int = 14
    ) -> float:
        """Calculate Average True Range"""
        if position < period:
            return 0.0

        tr_values = []
        for i in range(position - period, position):
            if i > 0:
                high_low = candles[i].high - candles[i].low
                high_close = abs(candles[i].high - candles[i - 1].close)
                low_close = abs(candles[i].low - candles[i - 1].close)
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)

        return np.mean(tr_values) if tr_values else 0.0

    def _create_trading_signals(self) -> List[Dict[str, Any]]:
        """Create trading signals from detected patterns"""
        signals = []

        for pattern in self.detected_patterns:
            if pattern.strength >= 65:  # Quality threshold
                signal_type = "SELL" if "top" in pattern.pattern_type.value else "BUY"

                signal = {
                    "type": signal_type,
                    "pattern": f"Tweezer {pattern.pattern_type.value.replace('_', ' ').title()}",
                    "strength": pattern.strength,
                    "confidence": pattern.confidence,
                    "entry_price": pattern.candle2.close,
                    "stop_loss": pattern.stop_loss,
                    "targets": pattern.target_levels,
                    "position": pattern.position,
                    "volume_confirmed": pattern.volume_confirmation,
                    "key_level": pattern.key_level,
                    "level_tolerance": pattern.level_tolerance,
                }
                signals.append(signal)

        return signals

    def _get_pattern_type_counts(self) -> Dict[str, int]:
        """Get count of each pattern type"""
        counts = {ptype.value: 0 for ptype in TweezerType}
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
            "key_level": strongest.key_level,
            "level_tolerance": strongest.level_tolerance,
        }

    def get_pattern_description(self) -> str:
        """Get description of the pattern"""
        return (
            "Tweezer Patterns: Two-candle reversal patterns where candles have nearly "
            "identical highs (Tweezer Top - bearish) or lows (Tweezer Bottom - bullish). "
            "These patterns suggest potential trend reversals at key support/resistance levels."
        )
