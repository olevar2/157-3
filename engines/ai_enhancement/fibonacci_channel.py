"""
Platform3 Fibonacci Channel Indicator
======================================

Individual implementation of Fibonacci Channel analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Fibonacci Channel Constants
FIBONACCI_CHANNEL_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000]


@dataclass
class SwingPoint:
    """Represents a swing high or low point"""

    index: int
    price: float
    swing_type: str  # 'high' or 'low'
    strength: float = 1.0
    timestamp: Optional[datetime] = None


@dataclass
class ChannelLine:
    """Represents a Fibonacci Channel line"""

    ratio: float
    current_price: float
    slope: float
    y_intercept: float
    line_type: str  # 'support', 'resistance', 'center'
    line_strength: float


@dataclass
class FibonacciChannelResult:
    """Result structure for Fibonacci Channel analysis"""

    base_line_start: SwingPoint
    base_line_end: SwingPoint
    parallel_point: SwingPoint
    channel_lines: List[ChannelLine]
    active_channel_line: Optional[ChannelLine]
    channel_direction: str  # 'bullish', 'bearish', 'sideways'
    channel_width: float
    current_position: float  # 0-1 position within channel
    breakout_probability: float
    channel_strength: float


class FibonacciChannel:
    """
    Fibonacci Channel Indicator

    Creates parallel channel lines using Fibonacci ratios
    to identify support/resistance zones and breakout potential.
    """

    def __init__(self, swing_window: int = 10, sensitivity: float = 0.02, **kwargs):
        """
        Initialize Fibonacci Channel indicator

        Args:
            swing_window: Number of periods to look for swing points
            sensitivity: Sensitivity for swing point detection
        """
        self.swing_window = swing_window
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciChannelResult]:
        """
        Calculate Fibonacci Channel for given data.

        Args:
            data: Price data (DataFrame with OHLC, dict, or array)

        Returns:
            FibonacciChannelResult with channel lines and breakout analysis
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

            # Find three points to define the channel
            base_start, base_end, parallel_point = self._find_channel_points(
                closes, highs, lows
            )

            # Calculate channel lines
            channel_lines = self._calculate_channel_lines(
                base_start, base_end, parallel_point, len(closes)
            )

            # Find active channel line
            current_price = closes[-1]
            active_line = self._find_active_channel_line(channel_lines, current_price)

            # Determine channel direction
            channel_direction = self._determine_channel_direction(base_start, base_end)

            # Calculate channel metrics
            channel_width = self._calculate_channel_width(channel_lines)
            current_position = self._calculate_current_position(
                channel_lines, current_price
            )
            breakout_probability = self._calculate_breakout_probability(
                channel_lines, current_price, closes
            )
            channel_strength = self._calculate_channel_strength(
                channel_lines, closes, highs, lows
            )

            return FibonacciChannelResult(
                base_line_start=base_start,
                base_line_end=base_end,
                parallel_point=parallel_point,
                channel_lines=channel_lines,
                active_channel_line=active_line,
                channel_direction=channel_direction,
                channel_width=channel_width,
                current_position=current_position,
                breakout_probability=breakout_probability,
                channel_strength=channel_strength,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Channel: {e}")
            return None

    def _find_channel_points(
        self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> tuple:
        """Find three points to define the Fibonacci channel"""
        data_len = len(closes)

        # Base line: Two points defining the main trend line
        # Start point: significant swing in first third
        start_section = data_len // 3
        start_high_idx = np.argmax(highs[:start_section])
        start_low_idx = np.argmin(lows[:start_section])

        # End point: significant swing in middle third
        middle_start = start_section
        middle_end = 2 * data_len // 3
        end_high_idx = np.argmax(highs[middle_start:middle_end]) + middle_start
        end_low_idx = np.argmin(lows[middle_start:middle_end]) + middle_start

        # Parallel point: opposite swing to define channel width
        parallel_start = middle_end
        parallel_high_idx = np.argmax(highs[parallel_start:]) + parallel_start
        parallel_low_idx = np.argmin(lows[parallel_start:]) + parallel_start

        # Determine best combination for channel
        # Prefer uptrend channels (low to high base line)
        if highs[end_high_idx] > highs[start_high_idx]:
            # Uptrend channel
            base_start = SwingPoint(start_low_idx, lows[start_low_idx], "low")
            base_end = SwingPoint(end_high_idx, highs[end_high_idx], "high")
            parallel_point = SwingPoint(
                parallel_high_idx, highs[parallel_high_idx], "high"
            )
        else:
            # Downtrend channel
            base_start = SwingPoint(start_high_idx, highs[start_high_idx], "high")
            base_end = SwingPoint(end_low_idx, lows[end_low_idx], "low")
            parallel_point = SwingPoint(parallel_low_idx, lows[parallel_low_idx], "low")

        return base_start, base_end, parallel_point

    def _calculate_channel_lines(
        self,
        base_start: SwingPoint,
        base_end: SwingPoint,
        parallel_point: SwingPoint,
        data_length: int,
    ) -> List[ChannelLine]:
        """Calculate Fibonacci channel lines"""
        channel_lines = []

        # Calculate base line slope
        time_diff = base_end.index - base_start.index
        price_diff = base_end.price - base_start.price

        if time_diff == 0:
            return channel_lines

        base_slope = price_diff / time_diff
        base_y_intercept = base_start.price - (base_slope * base_start.index)

        # Calculate parallel line distance
        parallel_price_at_base_time = (
            base_slope * parallel_point.index + base_y_intercept
        )
        channel_width = abs(parallel_point.price - parallel_price_at_base_time)

        # Create channel lines using Fibonacci ratios
        current_time = data_length - 1

        for ratio in FIBONACCI_CHANNEL_RATIOS:
            # Calculate offset from base line
            if parallel_point.price > parallel_price_at_base_time:
                # Parallel point is above base line
                offset = channel_width * ratio
                line_type = "resistance" if ratio > 0.5 else "support"
            else:
                # Parallel point is below base line
                offset = -channel_width * ratio
                line_type = "support" if ratio > 0.5 else "resistance"

            # Special case for center line
            if abs(ratio - 0.5) < 0.01:
                line_type = "center"

            # Calculate current price on this channel line
            y_intercept = base_y_intercept + offset
            current_price = base_slope * current_time + y_intercept

            # Calculate line strength (golden ratio and center lines are strongest)
            if abs(ratio - 0.618) < 0.01 or abs(ratio - 0.5) < 0.01:
                line_strength = 1.0
            elif abs(ratio - 0.382) < 0.01 or abs(ratio - 0.786) < 0.01:
                line_strength = 0.8
            else:
                line_strength = 0.6

            channel_lines.append(
                ChannelLine(
                    ratio=ratio,
                    current_price=current_price,
                    slope=base_slope,
                    y_intercept=y_intercept,
                    line_type=line_type,
                    line_strength=line_strength,
                )
            )

        return channel_lines

    def _find_active_channel_line(
        self, channel_lines: List[ChannelLine], current_price: float
    ) -> Optional[ChannelLine]:
        """Find the channel line closest to current price"""
        if not channel_lines:
            return None

        return min(
            channel_lines, key=lambda line: abs(line.current_price - current_price)
        )

    def _determine_channel_direction(
        self, base_start: SwingPoint, base_end: SwingPoint
    ) -> str:
        """Determine overall channel direction"""
        if base_end.price > base_start.price:
            return "bullish"
        elif base_end.price < base_start.price:
            return "bearish"
        else:
            return "sideways"

    def _calculate_channel_width(self, channel_lines: List[ChannelLine]) -> float:
        """Calculate relative channel width"""
        if len(channel_lines) < 2:
            return 0.0

        prices = [line.current_price for line in channel_lines]
        max_price = max(prices)
        min_price = min(prices)
        avg_price = sum(prices) / len(prices)

        # Relative width as percentage of average price
        if avg_price > 0:
            return (max_price - min_price) / avg_price
        else:
            return 0.0

    def _calculate_current_position(
        self, channel_lines: List[ChannelLine], current_price: float
    ) -> float:
        """Calculate current position within the channel (0-1)"""
        if not channel_lines:
            return 0.5

        prices = [line.current_price for line in channel_lines]
        max_price = max(prices)
        min_price = min(prices)

        if max_price == min_price:
            return 0.5

        # Position relative to channel bounds
        position = (current_price - min_price) / (max_price - min_price)
        return max(0.0, min(1.0, position))

    def _calculate_breakout_probability(
        self, channel_lines: List[ChannelLine], current_price: float, closes: np.ndarray
    ) -> float:
        """Calculate probability of channel breakout"""
        if not channel_lines or len(closes) < 5:
            return 0.0

        # Find channel boundaries
        prices = [line.current_price for line in channel_lines]
        upper_bound = max(prices)
        lower_bound = min(prices)

        # Check recent price action relative to boundaries
        recent_closes = closes[-5:]  # Last 5 periods
        boundary_tests = 0

        for price in recent_closes:
            if price >= upper_bound * 0.98 or price <= lower_bound * 1.02:
                boundary_tests += 1

        # More boundary tests = higher breakout probability
        breakout_probability = boundary_tests / len(recent_closes)

        # Adjust based on current position
        current_position = self._calculate_current_position(
            channel_lines, current_price
        )
        if current_position > 0.9 or current_position < 0.1:
            breakout_probability *= 1.5

        return min(1.0, breakout_probability)

    def _calculate_channel_strength(
        self,
        channel_lines: List[ChannelLine],
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> float:
        """Calculate overall strength of the channel formation"""
        if not channel_lines or len(closes) < 10:
            return 0.0

        total_strength = 0.0
        valid_lines = 0

        # Test how well each line has acted as support/resistance
        for line in channel_lines:
            line_strength = 0.0
            touches = 0

            # Check recent price action against this line
            recent_data = min(10, len(closes))
            for i in range(-recent_data, 0):
                price_high = highs[i]
                price_low = lows[i]

                # Check if price tested this line
                line_price = line.slope * (len(closes) + i - 1) + line.y_intercept

                # Test for touches (within 1%)
                if (
                    abs(price_high - line_price) / line_price < 0.01
                    or abs(price_low - line_price) / line_price < 0.01
                ):
                    touches += 1

            # More touches = stronger line
            if recent_data > 0:
                line_strength = touches / recent_data
                total_strength += line_strength * line.line_strength
                valid_lines += 1

        return total_strength / valid_lines if valid_lines > 0 else 0.0
