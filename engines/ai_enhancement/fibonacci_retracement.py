"""
Platform3 Fibonacci Retracement Indicator
==========================================

Individual implementation of Fibonacci Retracement analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Fibonacci Constants
FIBONACCI_RETRACEMENT_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]


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


class FibonacciRetracement:
    """
    Fibonacci Retracement Indicator

    Calculates retracement levels based on swing highs and lows using
    Fibonacci ratios: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    """

    def __init__(self, swing_window: int = 10, **kwargs):
        """
        Initialize Fibonacci Retracement indicator

        Args:
            swing_window: Number of periods to look for swing points
        """
        self.swing_window = swing_window
        self.logger = logging.getLogger(__name__)

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciRetracementResult]:
        """
        Calculate Fibonacci Retracement for given data.

        Args:
            data: Price data (DataFrame with OHLC, dict, or array)

        Returns:
            FibonacciRetracementResult with retracement levels and signals
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

            # Find swing high and low points
            swing_high, swing_low = self._find_swing_points(highs, lows)

            # Calculate retracement levels
            retracement_levels = self._calculate_retracement_levels(
                swing_high, swing_low
            )

            current_price = closes[-1]

            # Find active level (closest to current price)
            active_level = self._find_active_level(retracement_levels, current_price)

            # Determine trend direction
            trend_direction = (
                "bullish" if swing_high.index > swing_low.index else "bearish"
            )

            # Generate trading signal
            signal, signal_strength = self._generate_signal(
                current_price, retracement_levels, trend_direction
            )

            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(
                current_price, retracement_levels
            )

            return FibonacciRetracementResult(
                swing_high=swing_high,
                swing_low=swing_low,
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

    def _find_swing_points(self, highs: np.ndarray, lows: np.ndarray) -> tuple:
        """Find swing high and low points within the window"""
        window = min(self.swing_window, len(highs))

        # Find highest high and lowest low in recent window
        recent_high_idx = np.argmax(highs[-window:]) + len(highs) - window
        recent_low_idx = np.argmin(lows[-window:]) + len(lows) - window

        swing_high = SwingPoint(recent_high_idx, highs[recent_high_idx], "high")
        swing_low = SwingPoint(recent_low_idx, lows[recent_low_idx], "low")

        return swing_high, swing_low

    def _calculate_retracement_levels(
        self, swing_high: SwingPoint, swing_low: SwingPoint
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement levels"""
        price_range = swing_high.price - swing_low.price
        levels = []

        for ratio in FIBONACCI_RETRACEMENT_RATIOS:
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

    def _find_active_level(
        self, levels: List[FibonacciLevel], current_price: float
    ) -> Optional[FibonacciLevel]:
        """Find the closest Fibonacci level to current price"""
        if not levels:
            return None
        return min(levels, key=lambda x: abs(x.price - current_price))

    def _generate_signal(
        self, current_price: float, levels: List[FibonacciLevel], trend: str
    ) -> tuple:
        """Generate trading signal based on price position relative to Fibonacci levels"""
        if not levels:
            return "hold", 0.0

        # Find distance to nearest level
        distances = [abs(level.price - current_price) for level in levels]
        min_distance = min(distances)

        # Calculate signal strength based on proximity to level
        signal_strength = max(0.1, 1.0 - (min_distance / (max(distances) + 1e-8)))

        # Simple signal logic
        active_level = self._find_active_level(levels, current_price)
        if active_level:
            if trend == "bullish" and active_level.support_resistance == "support":
                return "buy", signal_strength
            elif trend == "bearish" and active_level.support_resistance == "resistance":
                return "sell", signal_strength

        return "hold", signal_strength * 0.5

    def _calculate_confluence_score(
        self, current_price: float, levels: List[FibonacciLevel]
    ) -> float:
        """Calculate confluence score based on multiple level proximity"""
        if not levels:
            return 0.0

        # Count levels within 2% of current price
        nearby_levels = 0
        price_tolerance = current_price * 0.02

        for level in levels:
            if abs(level.price - current_price) <= price_tolerance:
                nearby_levels += 1

        # Normalize confluence score
        max_levels = len(levels)
        return min(1.0, nearby_levels / max_levels * 2.0)
