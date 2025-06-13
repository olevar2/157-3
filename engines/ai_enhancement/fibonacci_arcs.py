"""
Platform3 Fibonacci Arcs Indicator
===================================

Individual implementation of Fibonacci Arcs analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Fibonacci Ratios for Arcs
FIBONACCI_ARC_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]


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
class ArcLevel:
    """Represents a Fibonacci Arc level"""

    ratio: float
    radius: float
    center_price: float
    current_arc_price: float
    support_resistance: str
    geometric_strength: float


@dataclass
class FibonacciArcResult:
    """Result structure for Fibonacci Arc analysis"""

    center_point: SwingPoint
    reference_point: SwingPoint
    arc_levels: List[ArcLevel]
    active_arc: Optional[ArcLevel]
    price_time_confluence: float
    arc_support_resistance: str
    geometric_strength: float


class FibonacciArcs:
    """
    Fibonacci Arcs Indicator

    Creates arc levels using Fibonacci ratios to identify
    price-time confluence zones for support and resistance.
    """

    def __init__(
        self, swing_window: int = 10, min_swing_strength: float = 0.02, **kwargs
    ):
        """
        Initialize Fibonacci Arcs indicator

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
    ) -> Optional[FibonacciArcResult]:
        """
        Calculate Fibonacci Arcs for given data.

        Args:
            data: Price data (DataFrame with OHLC, dict, or array)

        Returns:
            FibonacciArcResult with arc levels and confluence analysis
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

            # Find center and reference points for arc construction
            center_point, reference_point = self._find_arc_base_points(
                closes, highs, lows
            )

            # Calculate arc levels
            arc_levels = self._calculate_arc_levels(
                center_point, reference_point, len(closes)
            )

            # Find active arc (closest to current price)
            current_price = closes[-1]
            active_arc = self._find_active_arc(arc_levels, current_price)

            # Calculate price-time confluence
            price_time_confluence = self._calculate_price_time_confluence(
                arc_levels, current_price, len(closes)
            )

            # Determine overall arc support/resistance
            arc_sr = self._determine_arc_support_resistance(arc_levels, current_price)

            # Calculate geometric strength
            geometric_strength = self._calculate_geometric_strength(
                center_point, reference_point
            )

            return FibonacciArcResult(
                center_point=center_point,
                reference_point=reference_point,
                arc_levels=arc_levels,
                active_arc=active_arc,
                price_time_confluence=price_time_confluence,
                arc_support_resistance=arc_sr,
                geometric_strength=geometric_strength,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Arcs: {e}")
            return None

    def _find_arc_base_points(
        self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray
    ) -> tuple:
        """Find center and reference points for arc construction"""
        data_len = len(closes)

        # Center point: significant swing in middle section
        center_start = max(0, data_len // 3)
        center_end = min(data_len, 2 * data_len // 3)

        # Find most significant swing in center section
        center_high_idx = np.argmax(highs[center_start:center_end]) + center_start
        center_low_idx = np.argmin(lows[center_start:center_end]) + center_start

        # Choose the swing point with larger magnitude move
        high_magnitude = highs[center_high_idx] - np.mean(
            closes[center_start:center_end]
        )
        low_magnitude = np.mean(closes[center_start:center_end]) - lows[center_low_idx]

        if high_magnitude > low_magnitude:
            center_point = SwingPoint(center_high_idx, highs[center_high_idx], "high")
        else:
            center_point = SwingPoint(center_low_idx, lows[center_low_idx], "low")

        # Reference point: opposite swing earlier in data
        ref_start = max(0, center_start - self.swing_window)
        ref_end = center_start

        if center_point.swing_type == "high":
            # If center is high, find reference low
            ref_idx = np.argmin(lows[ref_start:ref_end]) + ref_start
            reference_point = SwingPoint(ref_idx, lows[ref_idx], "low")
        else:
            # If center is low, find reference high
            ref_idx = np.argmax(highs[ref_start:ref_end]) + ref_start
            reference_point = SwingPoint(ref_idx, highs[ref_idx], "high")

        return center_point, reference_point

    def _calculate_arc_levels(
        self, center_point: SwingPoint, reference_point: SwingPoint, data_length: int
    ) -> List[ArcLevel]:
        """Calculate Fibonacci arc levels"""
        arc_levels = []

        # Calculate base radius (distance between center and reference points)
        time_diff = abs(center_point.index - reference_point.index)
        price_diff = abs(center_point.price - reference_point.price)

        # Normalize price and time for radius calculation
        # Simple approach: use Euclidean distance
        base_radius = np.sqrt(time_diff**2 + price_diff**2)

        if base_radius == 0:
            return arc_levels

        # Create arc levels using Fibonacci ratios
        for ratio in FIBONACCI_ARC_RATIOS:
            arc_radius = base_radius * ratio

            # Calculate current arc price (where arc intersects current time)
            current_time_diff = (data_length - 1) - center_point.index

            # Arc equation: (t - center_t)² + (p - center_p)² = radius²
            # Solve for p when t = current_time
            time_component = current_time_diff**2

            if arc_radius**2 >= time_component:
                price_component = np.sqrt(arc_radius**2 - time_component)

                # Two solutions (upper and lower arc)
                # Choose based on arc direction relative to trend
                if center_point.price > reference_point.price:
                    # Downward trend, use lower arc
                    current_arc_price = center_point.price - price_component
                    sr_type = "support"
                else:
                    # Upward trend, use upper arc
                    current_arc_price = center_point.price + price_component
                    sr_type = "resistance"
            else:
                # Arc doesn't reach current time (too small radius)
                current_arc_price = center_point.price
                sr_type = "neutral"

            # Calculate geometric strength based on ratio
            geometric_strength = 1.0 - abs(
                ratio - 0.618
            )  # Golden ratio gives max strength

            arc_levels.append(
                ArcLevel(
                    ratio=ratio,
                    radius=arc_radius,
                    center_price=center_point.price,
                    current_arc_price=current_arc_price,
                    support_resistance=sr_type,
                    geometric_strength=geometric_strength,
                )
            )

        return arc_levels

    def _find_active_arc(
        self, arc_levels: List[ArcLevel], current_price: float
    ) -> Optional[ArcLevel]:
        """Find the arc level closest to current price"""
        if not arc_levels:
            return None

        return min(
            arc_levels, key=lambda arc: abs(arc.current_arc_price - current_price)
        )

    def _calculate_price_time_confluence(
        self, arc_levels: List[ArcLevel], current_price: float, current_time: int
    ) -> float:
        """Calculate price-time confluence score"""
        if not arc_levels:
            return 0.0

        confluence_score = 0.0
        total_weight = 0.0

        for arc in arc_levels:
            # Calculate how close current price is to this arc
            price_distance = abs(arc.current_arc_price - current_price)
            relative_distance = (
                price_distance / current_price if current_price > 0 else 1.0
            )

            # Weight based on proximity (closer = higher weight)
            if relative_distance < 0.01:  # Within 1%
                weight = 1.0
            elif relative_distance < 0.02:  # Within 2%
                weight = 0.8
            elif relative_distance < 0.05:  # Within 5%
                weight = 0.5
            else:
                weight = 0.1

            confluence_score += weight * arc.geometric_strength
            total_weight += weight

        return confluence_score / total_weight if total_weight > 0 else 0.0

    def _determine_arc_support_resistance(
        self, arc_levels: List[ArcLevel], current_price: float
    ) -> str:
        """Determine overall arc support/resistance characteristic"""
        if not arc_levels:
            return "neutral"

        # Find arcs close to current price
        nearby_arcs = []
        for arc in arc_levels:
            price_distance = abs(arc.current_arc_price - current_price)
            relative_distance = (
                price_distance / current_price if current_price > 0 else 1.0
            )

            if relative_distance < 0.03:  # Within 3%
                nearby_arcs.append(arc)

        if not nearby_arcs:
            return "neutral"

        # Count support vs resistance arcs
        support_count = sum(
            1 for arc in nearby_arcs if arc.support_resistance == "support"
        )
        resistance_count = sum(
            1 for arc in nearby_arcs if arc.support_resistance == "resistance"
        )

        if support_count > resistance_count:
            return "support"
        elif resistance_count > support_count:
            return "resistance"
        else:
            return "neutral"

    def _calculate_geometric_strength(
        self, center_point: SwingPoint, reference_point: SwingPoint
    ) -> float:
        """Calculate geometric strength of the arc formation"""
        # Calculate strength based on swing point characteristics
        time_diff = abs(center_point.index - reference_point.index)
        price_diff = abs(center_point.price - reference_point.price)

        if time_diff == 0:
            return 0.0

        # Strength increases with price movement and time separation
        price_ratio = price_diff / max(center_point.price, reference_point.price)
        time_strength = min(1.0, time_diff / 20)  # Normalize to reasonable range

        # Combine factors
        geometric_strength = (price_ratio + time_strength) / 2

        return min(1.0, geometric_strength)
