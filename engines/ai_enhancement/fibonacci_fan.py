"""
Platform3 Fibonacci Fan Indicator
==================================

Individual implementation of Fibonacci Fan analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Fibonacci Fan Constants
FIBONACCI_FAN_RATIOS = [0.382, 0.500, 0.618]


@dataclass
class SwingPoint:
    """Represents a swing high or low point"""

    index: int
    price: float
    swing_type: str  # 'high' or 'low'
    strength: float = 1.0
    timestamp: Optional[datetime] = None


@dataclass
class FibonacciLevel:
    """Represents a Fibonacci level"""

    ratio: float
    price: float
    level_type: str
    distance_from_current: float = 0.0
    support_resistance: str = "neutral"


@dataclass
class FanLine:
    """Represents a Fibonacci Fan line"""

    ratio: float
    angle: float
    slope: float
    current_price: float
    support_resistance: str
    line_strength: float


@dataclass
class FibonacciFanResult:
    """Result structure for Fibonacci Fan analysis"""

    base_swing: SwingPoint
    trend_swing: SwingPoint
    fan_lines: List[FanLine]
    active_fan_line: Optional[FanLine]
    fan_direction: str  # 'bullish', 'bearish', 'neutral'
    angle_strength: float
    support_resistance_quality: float


class FibonacciFan:
    """
    Fibonacci Fan Indicator

    Creates fan lines from a swing point using Fibonacci ratios
    to identify dynamic support and resistance levels.
    """

    def __init__(self, swing_window: int = 10, sensitivity: float = 0.02, **kwargs):
        """
        Initialize Fibonacci Fan indicator

        Args:
            swing_window: Number of periods to look for swing points
            sensitivity: Sensitivity for swing point detection
        """
        self.swing_window = swing_window
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciFanResult]:
        """
        Calculate Fibonacci Fan for given data.

        Args:
            data: Price data (DataFrame with OHLC, dict, or array)

        Returns:
            FibonacciFanResult with fan lines and signals
        """
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                closes = data["close"].values
                highs = (
                    data.get("high", closes).values
                    if "high" in data.columns
                    else closes
                )
                lows = (
                    data.get("low", closes).values if "low" in data.columns else closes
                )
            elif isinstance(data, dict):
                closes = np.array(data.get("close", []))
                highs = np.array(data.get("high", closes))
                lows = np.array(data.get("low", closes))
            elif isinstance(data, np.ndarray):
                closes = data.flatten()
                highs = lows = closes
            else:
                return None

            if len(closes) < self.swing_window:
                return None

            # Find base and trend swing points
            base_swing, trend_swing = self._find_fan_base_points(closes, highs, lows)

            # Calculate fan lines
            fan_lines = self._calculate_fan_lines(base_swing, trend_swing, len(closes))

            # Find active fan line
            active_fan_line = self._find_active_fan_line(fan_lines, closes[-1])

            # Determine fan direction
            fan_direction = self._determine_fan_direction(base_swing, trend_swing)

            # Calculate angle strength
            angle_strength = self._calculate_angle_strength(base_swing, trend_swing)

            # Calculate support/resistance quality
            sr_quality = self._calculate_sr_quality(fan_lines, closes)

            return FibonacciFanResult(
                base_swing=base_swing,
                trend_swing=trend_swing,
                fan_lines=fan_lines,
                active_fan_line=active_fan_line,
                fan_direction=fan_direction,
                angle_strength=angle_strength,
                support_resistance_quality=sr_quality,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Fan: {e}")
            return None

    def _find_fan_base_points(
        self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> tuple:
        """Find base and trend swing points for fan construction"""
        data_len = len(closes)
        window = min(self.swing_window, data_len // 2)

        # Find significant swing points in recent data
        # Base point: significant low/high in first half of window
        base_start = max(0, data_len - 2 * window)
        base_end = max(base_start + 1, data_len - window)

        # Trend point: opposite swing in second half
        trend_start = base_end
        trend_end = data_len

        # Find lowest low in base period
        base_low_idx = np.argmin(lows[base_start:base_end]) + base_start
        base_high_idx = np.argmax(highs[base_start:base_end]) + base_start

        # Find highest high in trend period
        trend_high_idx = np.argmax(highs[trend_start:trend_end]) + trend_start
        trend_low_idx = np.argmin(lows[trend_start:trend_end]) + trend_start

        # Determine which combination makes most sense
        if highs[trend_high_idx] > highs[base_high_idx]:
            # Uptrend: base low to trend high
            base_swing = SwingPoint(base_low_idx, lows[base_low_idx], "low")
            trend_swing = SwingPoint(trend_high_idx, highs[trend_high_idx], "high")
        else:
            # Downtrend: base high to trend low
            base_swing = SwingPoint(base_high_idx, highs[base_high_idx], "high")
            trend_swing = SwingPoint(trend_low_idx, lows[trend_low_idx], "low")

        return base_swing, trend_swing

    def _calculate_fan_lines(
        self, base_swing: SwingPoint, trend_swing: SwingPoint, data_length: int
    ) -> List[FanLine]:
        """Calculate Fibonacci fan lines from base to trend points"""
        fan_lines = []

        # Calculate base trend line
        time_diff = trend_swing.index - base_swing.index
        price_diff = trend_swing.price - base_swing.price

        if time_diff == 0:
            return fan_lines

        base_slope = price_diff / time_diff
        base_angle = np.arctan(base_slope) * 180 / np.pi

        # Create fan lines using Fibonacci ratios
        for ratio in FIBONACCI_FAN_RATIOS:
            fan_slope = base_slope * ratio
            fan_angle = np.arctan(fan_slope) * 180 / np.pi

            # Calculate current price on this fan line
            current_time_diff = (data_length - 1) - base_swing.index
            current_fan_price = base_swing.price + (fan_slope * current_time_diff)

            # Determine support/resistance based on trend direction
            if trend_swing.price > base_swing.price:  # Uptrend
                sr_type = "support"
                line_strength = ratio  # Higher ratio = stronger support
            else:  # Downtrend
                sr_type = "resistance"
                line_strength = ratio

            fan_lines.append(
                FanLine(
                    ratio=ratio,
                    angle=fan_angle,
                    slope=fan_slope,
                    current_price=current_fan_price,
                    support_resistance=sr_type,
                    line_strength=line_strength,
                )
            )

        return fan_lines

    def _find_active_fan_line(
        self, fan_lines: List[FanLine], current_price: float
    ) -> Optional[FanLine]:
        """Find the fan line closest to current price"""
        if not fan_lines:
            return None

        return min(fan_lines, key=lambda line: abs(line.current_price - current_price))

    def _determine_fan_direction(
        self, base_swing: SwingPoint, trend_swing: SwingPoint
    ) -> str:
        """Determine overall fan direction"""
        if trend_swing.price > base_swing.price:
            return "bullish"
        elif trend_swing.price < base_swing.price:
            return "bearish"
        else:
            return "neutral"

    def _calculate_angle_strength(
        self, base_swing: SwingPoint, trend_swing: SwingPoint
    ) -> float:
        """Calculate the strength of the fan angle"""
        time_diff = abs(trend_swing.index - base_swing.index)
        price_diff = abs(trend_swing.price - base_swing.price)

        if time_diff == 0:
            return 0.0

        # Calculate slope steepness
        slope = price_diff / time_diff

        # Normalize slope to 0-1 range (arbitrary scaling)
        # Steeper slopes get higher strength scores
        max_reasonable_slope = price_diff * 0.1  # 10% per period is quite steep
        strength = min(1.0, slope / max_reasonable_slope)

        return strength

    def _calculate_sr_quality(
        self, fan_lines: List[FanLine], closes: np.ndarray
    ) -> float:
        """Calculate support/resistance quality of fan lines"""
        if not fan_lines or len(closes) < 10:
            return 0.0

        total_quality = 0.0
        valid_lines = 0

        for line in fan_lines:
            # Check how well this line has acted as support/resistance
            # Simple approximation: check recent price action relative to line

            line_quality = 0.0
            recent_prices = closes[-10:]  # Last 10 periods

            # Count how many times price respected this level
            touches = 0
            for price in recent_prices:
                distance = abs(price - line.current_price)
                relative_distance = (
                    distance / line.current_price if line.current_price > 0 else 1.0
                )

                if relative_distance < 0.02:  # Within 2%
                    touches += 1

            # Quality based on touches (more touches = better quality)
            line_quality = min(1.0, touches / len(recent_prices))

            total_quality += line_quality * line.line_strength
            valid_lines += 1

        return total_quality / valid_lines if valid_lines > 0 else 0.0
