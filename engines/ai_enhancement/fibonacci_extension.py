"""
Platform3 Fibonacci Extension Indicator
========================================

Individual implementation of Fibonacci Extension analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Fibonacci Extension Constants
FIBONACCI_EXTENSION_RATIOS = [1.272, 1.414, 1.618, 2.618, 4.236]


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
class FibonacciExtensionResult:
    """Result structure for Fibonacci Extension analysis"""

    swing_points: List[SwingPoint]  # A, B, C pattern
    extension_levels: List[FibonacciLevel]
    target_level: Optional[FibonacciLevel]
    breakout_direction: str  # 'bullish', 'bearish', 'neutral'
    target_confidence: float
    risk_reward_ratio: float


class FibonacciExtension:
    """
    Fibonacci Extension Indicator

    Calculates extension levels based on A-B-C swing pattern using
    Fibonacci ratios: 127.2%, 141.4%, 161.8%, 261.8%, 423.6%
    """

    def __init__(
        self, swing_window: int = 10, min_swing_strength: float = 0.02, **kwargs
    ):
        """
        Initialize Fibonacci Extension indicator

        Args:
            swing_window: Number of periods to look for swing points
            min_swing_strength: Minimum price movement to qualify as swing
        """
        self.swing_window = swing_window
        self.min_swing_strength = min_swing_strength
        self.sensitivity = kwargs.get("sensitivity", 0.02)
        self.logger = logging.getLogger(__name__)

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciExtensionResult]:
        """
        Calculate Fibonacci Extension for given data.

        Args:
            data: Price data (DataFrame with OHLC, dict, or array)

        Returns:
            FibonacciExtensionResult with extension levels and targets
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

            if len(closes) < 20:  # Need enough data for A-B-C pattern
                return None

            # Find A-B-C swing pattern
            swing_points = self._find_abc_pattern(closes, highs, lows)

            if len(swing_points) < 3:
                return None

            # Calculate extension levels
            extension_levels = self._calculate_extension_levels(swing_points)

            # Determine target level and breakout direction
            target_level, breakout_direction = self._determine_target_and_direction(
                swing_points, extension_levels
            )

            # Calculate target confidence
            target_confidence = self._calculate_target_confidence(
                swing_points, extension_levels
            )

            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                swing_points, target_level
            )

            return FibonacciExtensionResult(
                swing_points=swing_points,
                extension_levels=extension_levels,
                target_level=target_level,
                breakout_direction=breakout_direction,
                target_confidence=target_confidence,
                risk_reward_ratio=risk_reward_ratio,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Extension: {e}")
            return None

    def _find_abc_pattern(
        self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> List[SwingPoint]:
        """Find A-B-C swing pattern for extension calculation"""
        data_len = len(closes)

        # Simple A-B-C pattern identification
        # A: Starting point (1/3 through data)
        # B: Peak/Trough (2/3 through data)
        # C: Current correction (end of data)

        a_idx = max(0, data_len // 3)
        b_idx = max(a_idx + 1, 2 * data_len // 3)
        c_idx = data_len - 1

        # Determine if pattern is bullish or bearish
        if highs[b_idx] > highs[a_idx]:  # Bullish pattern
            a_point = SwingPoint(a_idx, lows[a_idx], "low")
            b_point = SwingPoint(b_idx, highs[b_idx], "high")
            c_point = SwingPoint(c_idx, lows[c_idx], "low")
        else:  # Bearish pattern
            a_point = SwingPoint(a_idx, highs[a_idx], "high")
            b_point = SwingPoint(b_idx, lows[b_idx], "low")
            c_point = SwingPoint(c_idx, highs[c_idx], "high")

        return [a_point, b_point, c_point]

    def _calculate_extension_levels(
        self, swing_points: List[SwingPoint]
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci extension levels from A-B-C pattern"""
        if len(swing_points) < 3:
            return []

        a_point, b_point, c_point = swing_points[:3]

        # Calculate A-B wave range
        ab_range = abs(b_point.price - a_point.price)

        extension_levels = []

        # Determine extension direction
        if a_point.swing_type == "low" and b_point.swing_type == "high":
            # Bullish extension from C point
            for ratio in FIBONACCI_EXTENSION_RATIOS:
                ext_price = c_point.price + (ab_range * ratio)
                extension_levels.append(
                    FibonacciLevel(
                        ratio=ratio,
                        price=ext_price,
                        level_type="extension",
                        support_resistance="resistance",
                    )
                )
        else:
            # Bearish extension from C point
            for ratio in FIBONACCI_EXTENSION_RATIOS:
                ext_price = c_point.price - (ab_range * ratio)
                extension_levels.append(
                    FibonacciLevel(
                        ratio=ratio,
                        price=ext_price,
                        level_type="extension",
                        support_resistance="support",
                    )
                )

        return extension_levels

    def _determine_target_and_direction(
        self, swing_points: List[SwingPoint], extension_levels: List[FibonacciLevel]
    ) -> tuple:
        """Determine primary target level and breakout direction"""
        if not extension_levels or len(swing_points) < 3:
            return None, "neutral"

        a_point, b_point, c_point = swing_points[:3]

        # Determine direction based on pattern
        if a_point.swing_type == "low" and b_point.swing_type == "high":
            breakout_direction = "bullish"
            # Use 161.8% extension as primary target
            target_level = next(
                (
                    level
                    for level in extension_levels
                    if abs(level.ratio - 1.618) < 0.01
                ),
                extension_levels[0] if extension_levels else None,
            )
        else:
            breakout_direction = "bearish"
            target_level = next(
                (
                    level
                    for level in extension_levels
                    if abs(level.ratio - 1.618) < 0.01
                ),
                extension_levels[0] if extension_levels else None,
            )

        return target_level, breakout_direction

    def _calculate_target_confidence(
        self, swing_points: List[SwingPoint], extension_levels: List[FibonacciLevel]
    ) -> float:
        """Calculate confidence level for extension target"""
        if len(swing_points) < 3 or not extension_levels:
            return 0.0

        a_point, b_point, c_point = swing_points[:3]

        # Calculate pattern strength factors
        ab_range = abs(b_point.price - a_point.price)
        bc_range = abs(c_point.price - b_point.price)

        # Confidence based on retracement depth
        retracement_ratio = bc_range / ab_range if ab_range > 0 else 0

        # Ideal retracement is 38.2% to 61.8%
        if 0.382 <= retracement_ratio <= 0.618:
            confidence = 0.8
        elif 0.236 <= retracement_ratio <= 0.786:
            confidence = 0.6
        else:
            confidence = 0.4

        return min(1.0, confidence)

    def _calculate_risk_reward_ratio(
        self, swing_points: List[SwingPoint], target_level: Optional[FibonacciLevel]
    ) -> float:
        """Calculate risk-reward ratio for the extension trade"""
        if len(swing_points) < 3 or not target_level:
            return 1.0

        c_point = swing_points[2]

        # Calculate potential reward
        reward = abs(target_level.price - c_point.price)

        # Calculate risk (stop loss typically at C point invalidation)
        # Use 10% of AB range as risk estimate
        ab_range = abs(swing_points[1].price - swing_points[0].price)
        risk = ab_range * 0.1

        if risk > 0:
            return reward / risk
        else:
            return 2.0  # Default ratio
