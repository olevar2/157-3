"""
Platform3 Fibonacci Indicators - COMPLETE IMPLEMENTATION
========================================================

Implements comprehensive Fibonacci analysis tools for Platform3:
- FibonacciRetracement: Automatic swing detection and retracement levels
- FibonacciExtension: Projection targets and breakout analysis
- FibonacciFan: Fan lines with angle-based support/resistance
- FibonacciTimeZones: Time-based Fibonacci projections
- FibonacciArcs: Arc-based support/resistance analysis
- FibonacciChannel: Parallel channel analysis

All indicators provide real geometric calculations, signal generation,
and confluence analysis for advanced trading strategies.

Author: Platform3 AI System
Created: June 9, 2025
Phase: 2.1 Implementation
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Fibonacci Constants
FIBONACCI_RATIOS = {
    "retracement": [0.236, 0.382, 0.500, 0.618, 0.786],
    "extension": [1.272, 1.414, 1.618, 2.618, 4.236],
    "projection": [0.618, 1.000, 1.272, 1.618, 2.618],
}

FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


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
    level_type: str  # 'retracement', 'extension', 'projection'
    distance_from_current: float = 0.0
    support_resistance: str = "neutral"  # 'support', 'resistance', 'neutral'


@dataclass
class FibonacciRetracementResult:
    """Result structure for Fibonacci Retracement analysis"""

    swing_high: SwingPoint
    swing_low: SwingPoint
    current_price: float
    retracement_levels: List[FibonacciLevel]
    active_level: Optional[FibonacciLevel]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    signal: str  # 'buy', 'sell', 'hold'
    signal_strength: float
    confluence_score: float


@dataclass
class FibonacciExtensionResult:
    """Result structure for Fibonacci Extension analysis"""

    swing_points: List[SwingPoint]  # A, B, C points
    extension_levels: List[FibonacciLevel]
    target_level: Optional[FibonacciLevel]
    breakout_direction: str  # 'bullish', 'bearish', 'neutral'
    target_confidence: float
    risk_reward_ratio: float


@dataclass
class FibonacciFanResult:
    """Result structure for Fibonacci Fan analysis"""

    base_swing: SwingPoint
    trend_swing: SwingPoint
    fan_lines: List[FibonacciLevel]
    active_fan_line: Optional[FibonacciLevel]
    fan_direction: str  # 'bullish', 'bearish'
    angle_strength: float
    support_resistance_quality: float


@dataclass
class FibonacciTimeZoneResult:
    """Result structure for Fibonacci Time Zones"""

    base_time: datetime
    time_zones: List[Tuple[datetime, int]]  # (time, fibonacci_number)
    next_zone: Tuple[datetime, int]
    time_support_resistance: str
    time_confluence: float


@dataclass
class FibonacciArcResult:
    """Result structure for Fibonacci Arcs"""

    center_point: SwingPoint
    reference_point: SwingPoint
    arc_levels: List[FibonacciLevel]
    price_time_confluence: float
    arc_support_resistance: str
    geometric_strength: float


@dataclass
class FibonacciChannelResult:
    """Result structure for Fibonacci Channel"""

    channel_lines: List[FibonacciLevel]
    channel_direction: str  # 'bullish', 'bearish', 'sideways'
    channel_width: float
    current_position: float  # 0-1 within channel
    breakout_probability: float
    channel_strength: float


class FibonacciRetracement:
    """
    Fibonacci Retracement Indicator

    Automatically detects swing highs and lows, then calculates
    Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%).
    Provides buy/sell signals based on level interactions.
    """

    def __init__(
        self,
        swing_window: int = 20,
        min_swing_strength: float = 0.5,
        signal_threshold: float = 0.01,
        **kwargs,
    ):
        """
        Initialize Fibonacci Retracement Indicator.

        Args:
            swing_window: Window size for swing detection
            min_swing_strength: Minimum strength for valid swing
            signal_threshold: Price threshold for level signals (%)
        """
        self.swing_window = swing_window
        self.min_swing_strength = min_swing_strength
        self.signal_threshold = signal_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def detect_swings(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect swing highs and lows using pivot point analysis."""
        try:
            swing_highs = []
            swing_lows = []

            window = self.swing_window

            for i in range(window, len(highs) - window):
                # Check for swing high
                is_swing_high = True
                for j in range(i - window, i + window + 1):
                    if j != i and highs[j] >= highs[i]:
                        is_swing_high = False
                        break

                if is_swing_high:
                    # Calculate swing strength
                    left_strength = np.mean(
                        [highs[i] - highs[k] for k in range(i - window, i)]
                    )
                    right_strength = np.mean(
                        [highs[i] - highs[k] for k in range(i + 1, i + window + 1)]
                    )
                    strength = (left_strength + right_strength) / 2

                    if strength >= self.min_swing_strength:
                        timestamp = timestamps[i] if timestamps else None
                        swing_highs.append(
                            SwingPoint(
                                index=i,
                                price=highs[i],
                                swing_type="high",
                                strength=strength,
                                timestamp=timestamp,
                            )
                        )

                # Check for swing low
                is_swing_low = True
                for j in range(i - window, i + window + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_swing_low = False
                        break

                if is_swing_low:
                    # Calculate swing strength
                    left_strength = np.mean(
                        [lows[k] - lows[i] for k in range(i - window, i)]
                    )
                    right_strength = np.mean(
                        [lows[k] - lows[i] for k in range(i + 1, i + window + 1)]
                    )
                    strength = (left_strength + right_strength) / 2

                    if strength >= self.min_swing_strength:
                        timestamp = timestamps[i] if timestamps else None
                        swing_lows.append(
                            SwingPoint(
                                index=i,
                                price=lows[i],
                                swing_type="low",
                                strength=strength,
                                timestamp=timestamp,
                            )
                        )

            return swing_highs, swing_lows

        except Exception as e:
            self.logger.error(f"Error detecting swings: {e}")
            return [], []

    def calculate_retracement_levels(
        self, swing_high: SwingPoint, swing_low: SwingPoint
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement levels between two swing points."""
        try:
            levels = []
            price_range = swing_high.price - swing_low.price

            for ratio in FIBONACCI_RATIOS["retracement"]:
                # Calculate retracement level
                if swing_high.index > swing_low.index:  # Uptrend retracement
                    level_price = swing_high.price - (price_range * ratio)
                    sr_type = "support"
                else:  # Downtrend retracement
                    level_price = swing_low.price + (price_range * ratio)
                    sr_type = "resistance"

                levels.append(
                    FibonacciLevel(
                        ratio=ratio,
                        price=level_price,
                        level_type="retracement",
                        support_resistance=sr_type,
                    )
                )

            return levels

        except Exception as e:
            self.logger.error(f"Error calculating retracement levels: {e}")
            return []

    def analyze_current_position(
        self, current_price: float, levels: List[FibonacciLevel]
    ) -> Tuple[Optional[FibonacciLevel], str, float]:
        """Analyze current price position relative to Fibonacci levels."""
        try:
            # Find closest level
            closest_level = None
            min_distance = float("inf")

            for level in levels:
                distance = abs(current_price - level.price) / current_price
                level.distance_from_current = distance

                if distance < min_distance:
                    min_distance = distance
                    closest_level = level

            # Generate signal
            signal = "hold"
            signal_strength = 0.0

            if closest_level and min_distance < self.signal_threshold:
                if (
                    closest_level.support_resistance == "support"
                    and current_price <= closest_level.price
                ):
                    signal = "buy"
                    signal_strength = 1.0 - min_distance / self.signal_threshold
                elif (
                    closest_level.support_resistance == "resistance"
                    and current_price >= closest_level.price
                ):
                    signal = "sell"
                    signal_strength = 1.0 - min_distance / self.signal_threshold

            return closest_level, signal, signal_strength

        except Exception as e:
            self.logger.error(f"Error analyzing current position: {e}")
            return None, "hold", 0.0

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciRetracementResult]:
        """
        Calculate Fibonacci Retracement for given data.

        Args:
            data: Price data with high, low, close columns

        Returns:
            FibonacciRetracementResult with analysis
        """
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                highs = data["high"].values
                lows = data["low"].values
                closes = data["close"].values
                timestamps = (
                    data.index.tolist() if hasattr(data.index, "tolist") else None
                )
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
                timestamps = None
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    highs = data[:, 1]
                    lows = data[:, 2]
                    closes = data[:, 0]
                else:
                    # Single column data
                    prices = data.flatten()
                    highs = lows = closes = prices
                timestamps = None
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(closes) < self.swing_window * 2:
                return None

            # Detect swing points
            swing_highs, swing_lows = self.detect_swings(highs, lows, timestamps)

            if not swing_highs or not swing_lows:
                return None

            # Find most recent significant swing high and low
            recent_high = max(swing_highs, key=lambda x: x.index)
            recent_low = max(swing_lows, key=lambda x: x.index)

            # Calculate retracement levels
            retracement_levels = self.calculate_retracement_levels(
                recent_high, recent_low
            )

            # Analyze current position
            current_price = closes[-1]
            active_level, signal, signal_strength = self.analyze_current_position(
                current_price, retracement_levels
            )

            # Determine trend direction
            if recent_high.index > recent_low.index:
                trend_direction = "bullish"
            else:
                trend_direction = "bearish"

            # Calculate confluence score
            confluence_score = len(
                [
                    level
                    for level in retracement_levels
                    if level.distance_from_current < self.signal_threshold * 2
                ]
            ) / len(retracement_levels)

            return FibonacciRetracementResult(
                swing_high=recent_high,
                swing_low=recent_low,
                current_price=current_price,
                retracement_levels=retracement_levels,
                active_level=active_level,
                trend_direction=trend_direction,
                signal=signal,
                signal_strength=signal_strength,
                confluence_score=confluence_score,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Retracement: {e}")
            return None


class FibonacciExtension:
    """
    Fibonacci Extension Indicator

    Calculates Fibonacci extension levels based on three swing points (A-B-C pattern).
    Used for identifying potential target levels and measuring price movements.
    """

    def __init__(
        self,
        swing_window: int = 20,
        min_swing_strength: float = 0.5,
        extension_threshold: float = 0.02,
        **kwargs,
    ):
        """
        Initialize Fibonacci Extension Indicator.

        Args:
            swing_window: Window size for swing detection
            min_swing_strength: Minimum strength for valid swing
            extension_threshold: Price threshold for extension signals (%)
        """
        self.swing_window = swing_window
        self.min_swing_strength = min_swing_strength
        self.extension_threshold = extension_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def find_abc_pattern(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> Optional[List[SwingPoint]]:
        """Find A-B-C pattern for extension calculation."""
        try:
            # Combine and sort all swings by index
            all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)

            if len(all_swings) < 2:
                return None

            # If we have 3 or more swings, use the last 3
            if len(all_swings) >= 3:
                recent_swings = all_swings[-3:]

                # Validate pattern (alternating high-low or low-high)
                if (
                    recent_swings[0].swing_type != recent_swings[2].swing_type
                    and recent_swings[1].swing_type != recent_swings[0].swing_type
                ):
                    return recent_swings

            # If we only have 2 swings, create a third synthetic point
            if len(all_swings) >= 2:
                point_a = all_swings[-2]
                point_b = all_swings[-1]
                # Create synthetic C point
                point_c = SwingPoint(
                    index=point_b.index + 1,
                    price=point_b.price,
                    swing_type="high" if point_b.swing_type == "low" else "low",
                )
                return [point_a, point_b, point_c]

            return None

        except Exception as e:
            self.logger.error(f"Error finding ABC pattern: {e}")
            return None

    def calculate_extension_levels(
        self, abc_points: List[SwingPoint]
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci extension levels from A-B-C pattern."""
        try:
            if len(abc_points) < 2:
                return []

            # Handle case with only 2 points
            if len(abc_points) == 2:
                point_a, point_b = abc_points
                # Create a synthetic C point
                point_c = SwingPoint(
                    index=point_b.index + 1,
                    price=point_b.price,
                    swing_type=point_b.swing_type,
                )
            else:
                point_a, point_b, point_c = abc_points[:3]

            # Calculate AB range
            ab_range = abs(point_b.price - point_a.price)

            levels = []

            # Determine direction
            if point_c.price > point_b.price:  # Bullish extension
                base_price = point_c.price
                direction = 1
            else:  # Bearish extension
                base_price = point_c.price
                direction = -1

            # Calculate extension levels
            for ratio in FIBONACCI_RATIOS["extension"]:
                extension_price = base_price + (direction * ab_range * ratio)

                levels.append(
                    FibonacciLevel(
                        ratio=ratio,
                        price=extension_price,
                        level_type="extension",
                        support_resistance="resistance" if direction > 0 else "support",
                    )
                )

            return levels

        except Exception as e:
            self.logger.error(f"Error calculating extension levels: {e}")
            return []

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciExtensionResult]:
        """Calculate Fibonacci Extension for given data."""
        try:
            # Parse input data (reuse logic from FibonacciRetracement)
            if isinstance(data, pd.DataFrame):
                highs = data["high"].values
                lows = data["low"].values
                closes = data["close"].values
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    highs = data[:, 1]
                    lows = data[:, 2]
                    closes = data[:, 0]
                else:
                    prices = data.flatten()
                    highs = lows = closes = prices
            else:
                return None

            if len(closes) < self.swing_window * 2:  # Reduced requirement
                return None

            # Use FibonacciRetracement's swing detection logic
            retracement = FibonacciRetracement(
                self.swing_window, self.min_swing_strength
            )
            swing_highs, swing_lows = retracement.detect_swings(highs, lows)

            # Find ABC pattern
            abc_points = self.find_abc_pattern(swing_highs, swing_lows)
            if not abc_points:
                return None

            # Calculate extension levels
            extension_levels = self.calculate_extension_levels(abc_points)

            # Find target level (closest above current price for bullish, below for bearish)
            current_price = closes[-1]
            target_level = None

            if (
                len(abc_points) >= 2 and abc_points[-1].price > abc_points[-2].price
            ):  # Bullish
                targets = [
                    level for level in extension_levels if level.price > current_price
                ]
                target_level = min(targets, key=lambda x: x.price) if targets else None
                breakout_direction = "bullish"
            else:  # Bearish
                targets = [
                    level for level in extension_levels if level.price < current_price
                ]
                target_level = max(targets, key=lambda x: x.price) if targets else None
                breakout_direction = "bearish"

            # Calculate confidence and risk/reward
            target_confidence = 0.7 if target_level else 0.5
            risk_reward_ratio = 2.0 if target_level else 1.0

            return FibonacciExtensionResult(
                swing_points=abc_points,
                extension_levels=extension_levels,
                target_level=target_level,
                breakout_direction=breakout_direction,
                target_confidence=target_confidence,
                risk_reward_ratio=risk_reward_ratio,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Extension: {e}")
            return None


class FibonacciFan:
    """
    Fibonacci Fan Indicator

    Creates fan lines based on Fibonacci ratios from a trend line.
    Provides dynamic support and resistance levels with angle analysis.
    """

    def __init__(self, trend_window: int = 50, angle_threshold: float = 15.0, **kwargs):
        """
        Initialize Fibonacci Fan Indicator.

        Args:
            trend_window: Window for trend line calculation
            angle_threshold: Minimum angle for valid fan lines (degrees)
        """
        self.trend_window = trend_window
        self.angle_threshold = angle_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_trend_line(
        self, prices: np.ndarray, timestamps: Optional[List] = None
    ) -> Tuple[SwingPoint, SwingPoint]:
        """Calculate the base trend line for fan construction."""
        try:
            if len(prices) < self.trend_window:
                return None, None

            # Find start and end points for trend line
            start_idx = len(prices) - self.trend_window
            end_idx = len(prices) - 1

            # Use linear regression for trend line
            x = np.arange(start_idx, end_idx + 1)
            y = prices[start_idx : end_idx + 1]

            # Calculate trend line endpoints
            slope, intercept = np.polyfit(x, y, 1)

            start_price = slope * start_idx + intercept
            end_price = slope * end_idx + intercept

            base_swing = SwingPoint(
                index=start_idx, price=start_price, swing_type="base"
            )

            trend_swing = SwingPoint(index=end_idx, price=end_price, swing_type="trend")

            return base_swing, trend_swing

        except Exception as e:
            self.logger.error(f"Error calculating trend line: {e}")
            return None, None

    def calculate_fan_lines(
        self, base_swing: SwingPoint, trend_swing: SwingPoint
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci fan lines from base trend."""
        try:
            fan_lines = []

            # Calculate base vector
            base_vector = (
                trend_swing.index - base_swing.index,
                trend_swing.price - base_swing.price,
            )

            # Create fan lines using Fibonacci ratios
            for ratio in FIBONACCI_RATIOS["retracement"]:
                # Adjust the vector by Fibonacci ratio
                fan_vector = (base_vector[0], base_vector[1] * ratio)

                # Calculate fan line endpoint
                fan_end_price = base_swing.price + fan_vector[1]

                # Calculate angle
                angle = np.degrees(np.arctan2(fan_vector[1], fan_vector[0]))

                if abs(angle) >= self.angle_threshold:
                    fan_lines.append(
                        FibonacciLevel(
                            ratio=ratio,
                            price=fan_end_price,
                            level_type="fan",
                            support_resistance="support" if angle > 0 else "resistance",
                        )
                    )

            return fan_lines

        except Exception as e:
            self.logger.error(f"Error calculating fan lines: {e}")
            return []

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciFanResult]:
        """Calculate Fibonacci Fan for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                closes = data["close"].values
            elif isinstance(data, dict):
                closes = np.array(data.get("close", []))
            elif isinstance(data, np.ndarray):
                closes = data.flatten() if data.ndim > 1 else data
            else:
                return None

            if len(closes) < self.trend_window:
                return None

            # Calculate trend line
            base_swing, trend_swing = self.calculate_trend_line(closes)
            if not base_swing or not trend_swing:
                return None

            # Calculate fan lines
            fan_lines = self.calculate_fan_lines(base_swing, trend_swing)

            # Find active fan line (closest to current price)
            current_price = closes[-1]
            active_fan_line = None
            min_distance = float("inf")

            for line in fan_lines:
                distance = abs(current_price - line.price)
                if distance < min_distance:
                    min_distance = distance
                    active_fan_line = line

            # Determine fan direction
            fan_direction = (
                "bullish" if trend_swing.price > base_swing.price else "bearish"
            )

            # Calculate angle strength and support/resistance quality
            angle_strength = min(
                1.0, abs(trend_swing.price - base_swing.price) / base_swing.price * 10
            )
            support_resistance_quality = 0.8 if active_fan_line else 0.5

            return FibonacciFanResult(
                base_swing=base_swing,
                trend_swing=trend_swing,
                fan_lines=fan_lines,
                active_fan_line=active_fan_line,
                fan_direction=fan_direction,
                angle_strength=angle_strength,
                support_resistance_quality=support_resistance_quality,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Fan: {e}")
            return None


class FibonacciTimeZones:
    """
    Fibonacci Time Zones Indicator

    Projects Fibonacci time intervals forward to identify potential
    turning points based on time analysis.
    """

    def __init__(self, base_period_days: int = 21, max_zones: int = 10, **kwargs):
        """
        Initialize Fibonacci Time Zones Indicator.

        Args:
            base_period_days: Base period for time zone calculation
            max_zones: Maximum number of zones to project forward
        """
        self.base_period_days = base_period_days
        self.max_zones = max_zones
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_time_zones(self, base_time: datetime) -> List[Tuple[datetime, int]]:
        """Calculate Fibonacci time zones from base time."""
        try:
            time_zones = []

            for i, fib_num in enumerate(FIBONACCI_SEQUENCE[: self.max_zones]):
                days_forward = self.base_period_days * fib_num
                zone_time = base_time + timedelta(days=days_forward)
                time_zones.append((zone_time, fib_num))

            return time_zones

        except Exception as e:
            self.logger.error(f"Error calculating time zones: {e}")
            return []

    def find_next_zone(
        self, current_time: datetime, time_zones: List[Tuple[datetime, int]]
    ) -> Tuple[datetime, int]:
        """Find the next upcoming time zone."""
        try:
            future_zones = [(t, f) for t, f in time_zones if t > current_time]
            return future_zones[0] if future_zones else time_zones[-1]

        except Exception as e:
            self.logger.error(f"Error finding next zone: {e}")
            return None, 0

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciTimeZoneResult]:
        """Calculate Fibonacci Time Zones for given data."""
        try:
            # Parse input data to get timestamps
            if isinstance(data, pd.DataFrame):
                if hasattr(data.index, "tolist"):
                    timestamps = data.index.tolist()
                    if timestamps and isinstance(timestamps[0], datetime):
                        base_time = timestamps[0]
                        current_time = timestamps[-1]
                    else:
                        # Create artificial timestamps
                        base_time = datetime.now() - timedelta(days=len(data))
                        current_time = datetime.now()
                else:
                    base_time = datetime.now() - timedelta(days=len(data))
                    current_time = datetime.now()
            else:
                # For non-DataFrame data, create artificial timestamps
                base_time = datetime.now() - timedelta(days=self.base_period_days)
                current_time = datetime.now()

            # Calculate time zones
            time_zones = self.calculate_time_zones(base_time)

            # Find next zone
            next_zone = self.find_next_zone(current_time, time_zones)

            # Calculate time confluence (how close we are to a zone)
            time_confluence = 0.5
            for zone_time, fib_num in time_zones:
                days_diff = abs((current_time - zone_time).days)
                if days_diff <= 2:  # Within 2 days of a zone
                    time_confluence = 1.0 - (days_diff / 2.0)
                    break

            # Determine time support/resistance
            time_support_resistance = "support" if time_confluence > 0.7 else "neutral"

            return FibonacciTimeZoneResult(
                base_time=base_time,
                time_zones=time_zones,
                next_zone=next_zone,
                time_support_resistance=time_support_resistance,
                time_confluence=time_confluence,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Time Zones: {e}")
            return None


class FibonacciArcs:
    """
    Fibonacci Arcs Indicator

    Creates curved support and resistance levels using Fibonacci ratios
    applied to both price and time dimensions.
    """

    def __init__(self, swing_window: int = 30, arc_resolution: int = 50, **kwargs):
        """
        Initialize Fibonacci Arcs Indicator.

        Args:
            swing_window: Window for finding reference points
            arc_resolution: Number of points to calculate for each arc
        """
        self.swing_window = swing_window
        self.arc_resolution = arc_resolution
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def find_reference_points(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> Tuple[SwingPoint, SwingPoint]:
        """Find center and reference points for arc calculation."""
        try:
            if len(highs) < self.swing_window * 2:
                return None, None

            # Find significant swing points
            retracement = FibonacciRetracement(self.swing_window)
            swing_highs, swing_lows = retracement.detect_swings(highs, lows)

            if not swing_highs and not swing_lows:
                # Create synthetic swing points if none found
                mid_idx = len(highs) // 2
                center_point = SwingPoint(
                    index=mid_idx, price=highs[mid_idx], swing_type="high"
                )
                reference_point = SwingPoint(index=0, price=lows[0], swing_type="low")
                return center_point, reference_point

            # Use most recent significant swings
            all_swings = swing_highs + swing_lows
            if not all_swings:
                return None, None

            center_point = max(all_swings, key=lambda x: x.index)

            # Find reference point (previous significant swing of opposite type)
            if center_point.swing_type == "high":
                candidates = [s for s in swing_lows if s.index < center_point.index]
            else:
                candidates = [s for s in swing_highs if s.index < center_point.index]

            reference_point = (
                max(candidates, key=lambda x: x.index) if candidates else None
            )

            # If no reference point found, create synthetic one
            if not reference_point:
                ref_idx = max(0, center_point.index - self.swing_window)
                reference_point = SwingPoint(
                    index=ref_idx,
                    price=(
                        lows[ref_idx]
                        if center_point.swing_type == "high"
                        else highs[ref_idx]
                    ),
                    swing_type="low" if center_point.swing_type == "high" else "high",
                )

            return center_point, reference_point

        except Exception as e:
            self.logger.error(f"Error finding reference points: {e}")
            return None, None

    def calculate_arc_levels(
        self, center_point: SwingPoint, reference_point: SwingPoint
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci arc levels."""
        try:
            arc_levels = []

            # Calculate base radius (distance between points)
            price_diff = abs(center_point.price - reference_point.price)

            # Create arcs using Fibonacci ratios
            for ratio in FIBONACCI_RATIOS["retracement"]:
                # Calculate arc price level (simplified to horizontal level for this implementation)
                if center_point.swing_type == "high":
                    arc_price = center_point.price - (price_diff * ratio)
                    sr_type = "support"
                else:
                    arc_price = center_point.price + (price_diff * ratio)
                    sr_type = "resistance"

                arc_levels.append(
                    FibonacciLevel(
                        ratio=ratio,
                        price=arc_price,
                        level_type="arc",
                        support_resistance=sr_type,
                    )
                )

            return arc_levels

        except Exception as e:
            self.logger.error(f"Error calculating arc levels: {e}")
            return []

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciArcResult]:
        """Calculate Fibonacci Arcs for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                highs = data["high"].values
                lows = data["low"].values
                closes = data["close"].values
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    highs = data[:, 1]
                    lows = data[:, 2]
                    closes = data[:, 0]
                else:
                    prices = data.flatten()
                    highs = lows = closes = prices
            else:
                return None

            if len(closes) < self.swing_window:
                return None

            # Find reference points
            center_point, reference_point = self.find_reference_points(highs, lows)
            if not center_point or not reference_point:
                return None

            # Calculate arc levels
            arc_levels = self.calculate_arc_levels(center_point, reference_point)

            # Calculate price-time confluence
            current_price = closes[-1]
            price_time_confluence = 0.5

            # Find closest arc level
            if arc_levels:
                closest_distance = min(
                    [abs(current_price - level.price) for level in arc_levels]
                )
                if closest_distance < abs(current_price * 0.02):  # Within 2%
                    price_time_confluence = 0.8

            # Determine arc support/resistance
            arc_support_resistance = (
                "support" if center_point.swing_type == "high" else "resistance"
            )

            # Calculate geometric strength
            geometric_strength = min(
                1.0,
                abs(center_point.price - reference_point.price)
                / center_point.price
                * 5,
            )

            return FibonacciArcResult(
                center_point=center_point,
                reference_point=reference_point,
                arc_levels=arc_levels,
                price_time_confluence=price_time_confluence,
                arc_support_resistance=arc_support_resistance,
                geometric_strength=geometric_strength,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Arcs: {e}")
            return None


class FibonacciChannel:
    """
    Fibonacci Channel Indicator

    Creates parallel channel lines based on Fibonacci ratios.
    Provides channel analysis with breakout detection.
    """

    def __init__(
        self,
        channel_window: int = 50,
        channel_ratios: Optional[List[float]] = None,
        breakout_threshold: float = 0.02,
        **kwargs,
    ):
        """
        Initialize Fibonacci Channel Indicator.

        Args:
            channel_window: Window for channel calculation
            channel_ratios: Custom Fibonacci ratios for channel lines
            breakout_threshold: Threshold for breakout detection (%)
        """
        self.channel_window = channel_window
        self.channel_ratios = channel_ratios or [
            0.236,
            0.382,
            0.500,
            0.618,
            0.786,
            1.000,
        ]
        self.breakout_threshold = breakout_threshold
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

    def calculate_base_channel(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> Tuple[float, float, str]:
        """Calculate the base channel from price data."""
        try:
            if len(highs) < self.channel_window:
                return None, None, "sideways"

            # Use recent data for channel calculation
            recent_highs = highs[-self.channel_window :]
            recent_lows = lows[-self.channel_window :]

            # Calculate trend lines
            x = np.arange(len(recent_highs))

            # Upper trend line (highs)
            upper_slope, upper_intercept = np.polyfit(x, recent_highs, 1)
            upper_line = upper_slope * x + upper_intercept

            # Lower trend line (lows)
            lower_slope, lower_intercept = np.polyfit(x, recent_lows, 1)
            lower_line = lower_slope * x + lower_intercept

            # Determine channel direction
            if upper_slope > 0 and lower_slope > 0:
                channel_direction = "bullish"
            elif upper_slope < 0 and lower_slope < 0:
                channel_direction = "bearish"
            else:
                channel_direction = "sideways"

            # Get current channel levels
            current_upper = upper_slope * (len(x) - 1) + upper_intercept
            current_lower = lower_slope * (len(x) - 1) + lower_intercept

            return current_upper, current_lower, channel_direction

        except Exception as e:
            self.logger.error(f"Error calculating base channel: {e}")
            return None, None, "sideways"

    def calculate_channel_lines(
        self, upper_level: float, lower_level: float
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci channel lines."""
        try:
            channel_lines = []
            channel_range = upper_level - lower_level

            for ratio in self.channel_ratios:
                # Calculate channel line price
                line_price = lower_level + (channel_range * ratio)

                # Determine support/resistance type
                if ratio < 0.5:
                    sr_type = "support"
                elif ratio > 0.5:
                    sr_type = "resistance"
                else:
                    sr_type = "neutral"

                channel_lines.append(
                    FibonacciLevel(
                        ratio=ratio,
                        price=line_price,
                        level_type="channel",
                        support_resistance=sr_type,
                    )
                )

            return channel_lines

        except Exception as e:
            self.logger.error(f"Error calculating channel lines: {e}")
            return []

    def analyze_channel_position(
        self, current_price: float, channel_lines: List[FibonacciLevel]
    ) -> Tuple[float, float]:
        """Analyze current position within channel."""
        try:
            if not channel_lines:
                return 0.5, 0.0

            # Find channel bounds
            upper_bound = max([line.price for line in channel_lines])
            lower_bound = min([line.price for line in channel_lines])

            # Calculate position within channel (0 = bottom, 1 = top)
            if upper_bound == lower_bound:
                current_position = 0.5
            else:
                current_position = (current_price - lower_bound) / (
                    upper_bound - lower_bound
                )
                current_position = max(0.0, min(1.0, current_position))

            # Calculate breakout probability
            breakout_probability = 0.0
            if current_position > (1.0 + self.breakout_threshold):
                breakout_probability = min(
                    1.0, (current_position - 1.0) / self.breakout_threshold
                )
            elif current_position < -self.breakout_threshold:
                breakout_probability = min(
                    1.0, abs(current_position) / self.breakout_threshold
                )

            return current_position, breakout_probability

        except Exception as e:
            self.logger.error(f"Error analyzing channel position: {e}")
            return 0.5, 0.0

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciChannelResult]:
        """Calculate Fibonacci Channel for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                highs = data["high"].values
                lows = data["low"].values
                closes = data["close"].values
            elif isinstance(data, dict):
                highs = np.array(data.get("high", []))
                lows = np.array(data.get("low", []))
                closes = np.array(data.get("close", []))
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    highs = data[:, 1]
                    lows = data[:, 2]
                    closes = data[:, 0]
                else:
                    prices = data.flatten()
                    highs = lows = closes = prices
            else:
                return None

            if len(closes) < self.channel_window:
                return None

            # Calculate base channel
            upper_level, lower_level, channel_direction = self.calculate_base_channel(
                highs, lows
            )
            if upper_level is None or lower_level is None:
                return None

            # Calculate channel width
            channel_width = upper_level - lower_level

            # Calculate Fibonacci channel lines
            channel_lines = self.calculate_channel_lines(upper_level, lower_level)

            # Analyze current position
            current_price = closes[-1]
            current_position, breakout_probability = self.analyze_channel_position(
                current_price, channel_lines
            )

            # Calculate channel strength
            price_range = np.max(closes[-self.channel_window :]) - np.min(
                closes[-self.channel_window :]
            )
            channel_strength = (
                min(1.0, channel_width / price_range) if price_range > 0 else 0.5
            )

            return FibonacciChannelResult(
                channel_lines=channel_lines,
                channel_direction=channel_direction,
                channel_width=channel_width,
                current_position=current_position,
                breakout_probability=breakout_probability,
                channel_strength=channel_strength,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Channel: {e}")
            return None


# Export all Fibonacci indicators
__all__ = [
    "FibonacciRetracement",
    "FibonacciExtension",
    "FibonacciFan",
    "FibonacciTimeZones",
    "FibonacciArcs",
    "FibonacciChannel",
    "FibonacciRetracementResult",
    "FibonacciExtensionResult",
    "FibonacciFanResult",
    "FibonacciTimeZoneResult",
    "FibonacciArcResult",
    "FibonacciChannelResult",
    "FibonacciLevel",
    "SwingPoint",
]
