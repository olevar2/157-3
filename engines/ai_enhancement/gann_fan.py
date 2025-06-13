"""
Platform3 Gann Fan Indicator
============================

Real implementation of Gann Fan indicator for advanced geometric market analysis.
Based on W.D. Gann's mathematical principles for angular support and resistance lines.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Gann Constants
GANN_ANGLES = {
    "1x8": 7.125,
    "1x4": 14.036,
    "1x3": 18.435,
    "1x2": 26.565,
    "1x1": 45.0,
    "2x1": 63.435,
    "3x1": 71.565,
    "4x1": 75.964,
    "8x1": 82.875,
}


@dataclass
class GannLevel:
    """Represents a Gann analysis level"""

    price: float
    angle: float
    level_type: str  # 'support', 'resistance', 'angle_line'
    time_projection: Optional[int] = None
    strength: float = 1.0
    square_position: Optional[Tuple[int, int]] = None


@dataclass
class GannFanResult:
    """Results from Gann Fan analysis"""

    pivot_point: Tuple[int, float]  # (index, price)
    fan_lines: Dict[str, List[Tuple[int, float]]]  # angle_name -> [(index, price), ...]
    support_levels: List[GannLevel]
    resistance_levels: List[GannLevel]
    active_angles: List[str]
    trend_strength: float
    confidence: float


class GannFan:
    """
    Gann Fan Indicator - Creates angular support and resistance lines

    This indicator implements W.D. Gann's famous fan lines methodology for identifying
    support and resistance levels based on geometric price-time relationships.

    Key Features:
    - All 9 primary Gann angles (1x8, 1x4, 1x3, 1x2, 1x1, 2x1, 3x1, 4x1, 8x1)
    - Dynamic pivot point detection
    - Angular strength assessment
    - Support/resistance level identification
    - Trend direction analysis
    """

    def __init__(
        self,
        min_pivot_strength: float = 0.02,
        angle_tolerance: float = 0.1,
        lookback_period: int = 50,
        price_scale: float = 1.0,
    ):
        """
        Initialize Gann Fan indicator

        Args:
            min_pivot_strength: Minimum strength for pivot point detection
            angle_tolerance: Tolerance for angle line validation
            lookback_period: Number of periods to look back for pivot detection
            price_scale: Scale factor for price-time conversion
        """
        self.min_pivot_strength = min_pivot_strength
        self.angle_tolerance = angle_tolerance
        self.lookback_period = lookback_period
        self.price_scale = price_scale

        self.logger = logging.getLogger(__name__)

    def calculate(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> GannFanResult:
        """
        Calculate Gann Fan lines and levels

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            GannFanResult with fan lines and analysis
        """
        try:
            # Find significant pivot point
            pivot_point = self._find_pivot_point(high, low, close)

            if pivot_point is None:
                return self._empty_result()

            # Calculate fan lines from pivot point
            fan_lines = self._calculate_fan_lines(pivot_point, len(close))

            # Identify support and resistance levels
            support_levels, resistance_levels = self._identify_levels(
                fan_lines, high, low, close, pivot_point
            )

            # Assess trend strength and active angles
            active_angles = self._identify_active_angles(fan_lines, high, low, close)
            trend_strength = self._calculate_trend_strength(
                active_angles, close, pivot_point
            )

            # Calculate confidence based on angle interactions
            confidence = self._calculate_confidence(
                support_levels, resistance_levels, active_angles
            )

            return GannFanResult(
                pivot_point=pivot_point,
                fan_lines=fan_lines,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                active_angles=active_angles,
                trend_strength=trend_strength,
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Error in Gann Fan calculation: {e}")
            return self._empty_result()

    def _find_pivot_point(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Optional[Tuple[int, float]]:
        """Find the most significant pivot point for fan line origin"""
        try:
            # Calculate pivot strength using price swings
            pivot_highs = []
            pivot_lows = []

            for i in range(self.lookback_period, len(close) - self.lookback_period):
                # Check for swing high
                if (
                    high.iloc[i] > high.iloc[i - self.lookback_period : i].max()
                    and high.iloc[i]
                    > high.iloc[i + 1 : i + self.lookback_period + 1].max()
                ):

                    strength = (
                        high.iloc[i]
                        - low.iloc[
                            i - self.lookback_period : i + self.lookback_period + 1
                        ].min()
                    ) / close.iloc[i]
                    if strength >= self.min_pivot_strength:
                        pivot_highs.append((i, high.iloc[i], strength))

                # Check for swing low
                if (
                    low.iloc[i] < low.iloc[i - self.lookback_period : i].min()
                    and low.iloc[i]
                    < low.iloc[i + 1 : i + self.lookback_period + 1].min()
                ):

                    strength = (
                        high.iloc[
                            i - self.lookback_period : i + self.lookback_period + 1
                        ].max()
                        - low.iloc[i]
                    ) / close.iloc[i]
                    if strength >= self.min_pivot_strength:
                        pivot_lows.append((i, low.iloc[i], strength))

            # Select the most recent significant pivot
            all_pivots = pivot_highs + pivot_lows
            if not all_pivots:
                # Fallback to recent significant high/low
                recent_period = min(50, len(close) // 4)
                recent_high_idx = high.iloc[-recent_period:].idxmax()
                recent_low_idx = low.iloc[-recent_period:].idxmin()

                if high.iloc[recent_high_idx] > low.iloc[recent_low_idx]:
                    return (recent_high_idx, high.iloc[recent_high_idx])
                else:
                    return (recent_low_idx, low.iloc[recent_low_idx])

            # Return most recent pivot with highest strength
            all_pivots.sort(key=lambda x: (x[0], x[2]), reverse=True)
            return (all_pivots[0][0], all_pivots[0][1])

        except Exception as e:
            self.logger.error(f"Error finding pivot point: {e}")
            return None

    def _calculate_fan_lines(
        self, pivot_point: Tuple[int, float], total_length: int
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Calculate all Gann fan lines from pivot point"""
        pivot_idx, pivot_price = pivot_point
        fan_lines = {}

        for angle_name, angle_degrees in GANN_ANGLES.items():
            fan_lines[angle_name] = []

            # Convert angle to slope
            angle_radians = math.radians(angle_degrees)
            slope = math.tan(angle_radians) * self.price_scale

            # Calculate line points forward and backward from pivot
            for i in range(max(0, pivot_idx - 100), min(total_length, pivot_idx + 100)):
                time_diff = i - pivot_idx
                price = pivot_price + (slope * time_diff)
                fan_lines[angle_name].append((i, price))

        return fan_lines

    def _identify_levels(
        self,
        fan_lines: Dict[str, List[Tuple[int, float]]],
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        pivot_point: Tuple[int, float],
    ) -> Tuple[List[GannLevel], List[GannLevel]]:
        """Identify support and resistance levels from fan line interactions"""
        support_levels = []
        resistance_levels = []

        pivot_idx, pivot_price = pivot_point
        current_price = close.iloc[-1]

        for angle_name, line_points in fan_lines.items():
            if not line_points:
                continue

            # Find current price level on this angle line
            current_line_price = None
            for idx, price in line_points:
                if idx >= len(close) - 1:
                    current_line_price = price
                    break

            if current_line_price is None:
                continue

            # Determine if this is support or resistance
            angle_degrees = GANN_ANGLES[angle_name]
            strength = self._calculate_level_strength(
                angle_name, line_points, high, low, close
            )

            level = GannLevel(
                price=current_line_price,
                angle=angle_degrees,
                level_type=(
                    "support" if current_line_price < current_price else "resistance"
                ),
                strength=strength,
            )

            if current_line_price < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)

        # Sort by proximity to current price
        support_levels.sort(key=lambda x: abs(x.price - current_price))
        resistance_levels.sort(key=lambda x: abs(x.price - current_price))

        return support_levels[:5], resistance_levels[:5]  # Return top 5 of each

    def _calculate_level_strength(
        self,
        angle_name: str,
        line_points: List[Tuple[int, float]],
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> float:
        """Calculate the strength of a Gann angle level"""
        # Base strength on angle importance
        angle_strengths = {
            "1x1": 1.0,  # Most important
            "1x2": 0.8,  # Strong
            "2x1": 0.8,  # Strong
            "1x3": 0.6,  # Medium
            "3x1": 0.6,  # Medium
            "1x4": 0.4,  # Weak
            "4x1": 0.4,  # Weak
            "1x8": 0.2,  # Very weak
            "8x1": 0.2,  # Very weak
        }

        base_strength = angle_strengths.get(angle_name, 0.3)

        # Enhance strength based on price interactions
        interactions = 0
        total_strength = base_strength

        for idx, line_price in line_points:
            if 0 <= idx < len(close):
                # Check if price touched this level
                if low.iloc[idx] <= line_price <= high.iloc[idx]:
                    interactions += 1
                    total_strength += 0.1

        return min(total_strength, 1.0)

    def _identify_active_angles(
        self,
        fan_lines: Dict[str, List[Tuple[int, float]]],
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> List[str]:
        """Identify which Gann angles are currently active (being respected)"""
        active_angles = []
        current_price = close.iloc[-1]

        for angle_name, line_points in fan_lines.items():
            if not line_points:
                continue

            # Check recent price action against this angle
            recent_touches = 0
            for i in range(max(0, len(close) - 20), len(close)):
                for idx, line_price in line_points:
                    if (
                        idx == i
                        and abs(line_price - close.iloc[i]) / close.iloc[i]
                        < self.angle_tolerance
                    ):
                        recent_touches += 1
                        break

            if recent_touches >= 2:  # At least 2 touches in recent period
                active_angles.append(angle_name)

        return active_angles

    def _calculate_trend_strength(
        self, active_angles: List[str], close: pd.Series, pivot_point: Tuple[int, float]
    ) -> float:
        """Calculate overall trend strength based on active angles"""
        if not active_angles:
            return 0.0

        pivot_idx, pivot_price = pivot_point
        current_price = close.iloc[-1]

        # Base strength on price movement from pivot
        price_change = (current_price - pivot_price) / pivot_price
        base_strength = min(abs(price_change), 1.0)

        # Enhance based on number and quality of active angles
        angle_quality = sum(GANN_ANGLES[angle] / 45.0 for angle in active_angles) / len(
            active_angles
        )

        return min(base_strength * (1 + angle_quality), 1.0)

    def _calculate_confidence(
        self,
        support_levels: List[GannLevel],
        resistance_levels: List[GannLevel],
        active_angles: List[str],
    ) -> float:
        """Calculate confidence in the Gann Fan analysis"""
        confidence = 0.0

        # Base confidence on number of levels
        confidence += min(len(support_levels) * 0.1, 0.3)
        confidence += min(len(resistance_levels) * 0.1, 0.3)

        # Enhance based on active angles
        if "1x1" in active_angles:
            confidence += 0.2
        confidence += min(len(active_angles) * 0.05, 0.2)

        # Enhance based on level strengths
        if support_levels:
            avg_support_strength = sum(
                level.strength for level in support_levels
            ) / len(support_levels)
            confidence += avg_support_strength * 0.2

        if resistance_levels:
            avg_resistance_strength = sum(
                level.strength for level in resistance_levels
            ) / len(resistance_levels)
            confidence += avg_resistance_strength * 0.2

        return min(confidence, 1.0)

    def _empty_result(self) -> GannFanResult:
        """Return empty result when calculation fails"""
        return GannFanResult(
            pivot_point=(0, 0.0),
            fan_lines={},
            support_levels=[],
            resistance_levels=[],
            active_angles=[],
            trend_strength=0.0,
            confidence=0.0,
        )

    def get_signals(self, result: GannFanResult, close: pd.Series) -> Dict[str, float]:
        """Generate trading signals from Gann Fan analysis"""
        signals = {
            "trend_strength": result.trend_strength,
            "confidence": result.confidence,
            "support_signal": 0.0,
            "resistance_signal": 0.0,
            "breakout_signal": 0.0,
        }

        if not result.support_levels and not result.resistance_levels:
            return signals

        current_price = close.iloc[-1]

        # Support signal
        if result.support_levels:
            nearest_support = result.support_levels[0]
            distance_to_support = (
                current_price - nearest_support.price
            ) / current_price
            if 0 < distance_to_support < 0.02:  # Within 2% of support
                signals["support_signal"] = nearest_support.strength

        # Resistance signal
        if result.resistance_levels:
            nearest_resistance = result.resistance_levels[0]
            distance_to_resistance = (
                nearest_resistance.price - current_price
            ) / current_price
            if 0 < distance_to_resistance < 0.02:  # Within 2% of resistance
                signals["resistance_signal"] = nearest_resistance.strength

        # Breakout signal
        if "1x1" in result.active_angles and result.trend_strength > 0.6:
            signals["breakout_signal"] = result.trend_strength * result.confidence

        return signals
