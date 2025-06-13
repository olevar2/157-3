"""
Fibonacci Time Zones Indicator

The Fibonacci Time Zones indicator projects time-based intervals using Fibonacci sequence
numbers to identify potential turning points based on time analysis. These zones represent
periods where price action may change direction or experience significant volatility.

Mathematical Formula:
1. Identify base time period (usually from significant swing to swing)
2. Project forward using Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...
3. Time zones = Base_Time + (Base_Period × Fibonacci_Number)
4. Each zone represents a potential time-based support/resistance

Fibonacci Sequence: F(n) = F(n-1) + F(n-2), where F(0)=0, F(1)=1
First 12 numbers: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144

Interpretation:
- Time zones indicate periods of potential price reversals
- Higher Fibonacci numbers represent stronger time-based levels
- Confluence with price levels increases predictive power
- Used for timing entries, exits, and anticipating volatility

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

# Fibonacci sequence for time zones (first 15 numbers)
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

GOLDEN_RATIO = 1.6180339887498948482  # (1 + √5) / 2


@dataclass
class TimeZone:
    """Represents a Fibonacci time zone"""

    fibonacci_number: int
    time_index: int
    relative_time: int  # Periods from base
    zone_strength: float
    zone_type: str  # 'primary', 'secondary', 'minor'


@dataclass
class TimePrediction:
    """Represents a time-based prediction"""

    target_time: int
    fibonacci_number: int
    confidence: float
    prediction_type: str  # 'reversal', 'acceleration', 'consolidation'


@dataclass
class FibonacciTimeZoneResult:
    """Complete result structure for Fibonacci Time Zones analysis"""

    base_time: int
    time_zones: List[TimeZone]
    next_zone: Optional[TimeZone]
    current_zone_proximity: float  # 0-1, how close to nearest zone
    time_support_resistance: str  # 'support', 'resistance', 'neutral'
    time_confluence: float
    active_predictions: List[TimePrediction]


class FibonacciTimeZoneIndicator(StandardIndicatorInterface):
    """
    Fibonacci Time Zones Indicator

    Projects time-based Fibonacci intervals to identify potential turning points
    and periods of increased market activity based on mathematical time sequences.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "fibonacci"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        base_period: int = 21,
        max_zones: int = 12,
        zone_tolerance: int = 2,
        min_strength_threshold: float = 0.3,
        **kwargs,
    ):
        """
        Initialize Fibonacci Time Zones Indicator

        Args:
            base_period: Base period for time zone calculation (default: 21)
            max_zones: Maximum number of zones to project (default: 12)
            zone_tolerance: Tolerance for zone proximity in periods (default: 2)
            min_strength_threshold: Minimum strength for significant zones (default: 0.3)
        """
        # Mathematical constants with high precision
        self.PHI = (
            1.6180339887498948482045868343656  # Golden ratio (8+ decimal precision)
        )
        self.PHI_INV = 0.6180339887498948482045868343656  # 1/PHI (8+ decimal precision)

        super().__init__(
            base_period=base_period,
            max_zones=max_zones,
            zone_tolerance=zone_tolerance,
            min_strength_threshold=min_strength_threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate Fibonacci Time Zones analysis

        Args:
            data: DataFrame with price data or Series of prices

        Returns:
            pd.Series: Fibonacci time zones analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            temp_data = pd.DataFrame({"close": data})
            self.validate_input_data(temp_data)
            closes = data.values
            index = data.index
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                self.validate_input_data(data)
                closes = data["close"].values
                index = data.index
            else:
                raise IndicatorValidationError("DataFrame must contain 'close' column")
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        base_period = self.parameters.get("base_period", 21)
        max_zones = self.parameters.get("max_zones", 12)

        # Prepare results series
        # Prepare results DataFrame (CCI standard requires DataFrame)
        results = pd.DataFrame(index=index)
        results["fibonacci_time_zones"] = pd.Series(dtype=object, index=index)

        # Need sufficient data
        if len(closes) < base_period * 2:
            return results

        # Calculate for each point
        for i in range(base_period, len(closes)):
            try:
                # Calculate time zones from current position
                time_zones = self._calculate_time_zones(i, max_zones)

                # Find next upcoming zone
                next_zone = self._find_next_zone(i, time_zones)

                # Calculate proximity to nearest zone
                zone_proximity = self._calculate_zone_proximity(i, time_zones)

                # Determine time-based support/resistance
                time_sr = self._determine_time_support_resistance(i, time_zones, closes)

                # Calculate time confluence
                time_confluence = self._calculate_time_confluence(i, time_zones, closes)

                # Generate time-based predictions
                predictions = self._generate_predictions(i, time_zones, closes)

                # Create result
                result = FibonacciTimeZoneResult(
                    base_time=i - base_period,
                    time_zones=time_zones,
                    next_zone=next_zone,
                    current_zone_proximity=zone_proximity,
                    time_support_resistance=time_sr,
                    time_confluence=time_confluence,
                    active_predictions=predictions,
                )

                results.iloc[i, results.columns.get_loc("fibonacci_time_zones")] = (
                    result
                )

            except Exception:
                # Continue processing even if individual calculation fails
                continue

        # Store calculation details for debugging
        self._last_calculation = {
            "total_points": len(results),
            "valid_results": len([r for r in results if r is not None]),
            "base_period": base_period,
            "max_zones": max_zones,
        }

        return results

    def _calculate_time_zones(
        self, current_time: int, max_zones: int
    ) -> List[TimeZone]:
        """Calculate Fibonacci time zones from current position"""
        base_period = self.parameters.get("base_period", 21)
        base_time = current_time - base_period

        time_zones = []

        for i, fib_num in enumerate(FIBONACCI_SEQUENCE[:max_zones]):
            zone_time = base_time + (base_period * fib_num)

            # Calculate zone strength based on Fibonacci number significance
            if fib_num in [1, 2, 3, 5, 8]:
                zone_type = "primary"
                strength = 1.0
            elif fib_num in [13, 21, 34]:
                zone_type = "secondary"
                strength = 0.8
            else:
                zone_type = "minor"
                strength = 0.6

            # Adjust strength based on golden ratio relationships
            if i > 0:
                ratio = fib_num / FIBONACCI_SEQUENCE[i - 1]
                if abs(ratio - GOLDEN_RATIO) < 0.1:
                    strength *= 1.2

            time_zones.append(
                TimeZone(
                    fibonacci_number=fib_num,
                    time_index=zone_time,
                    relative_time=base_period * fib_num,
                    zone_strength=min(1.0, strength),
                    zone_type=zone_type,
                )
            )

        return time_zones

    def _find_next_zone(
        self, current_time: int, time_zones: List[TimeZone]
    ) -> Optional[TimeZone]:
        """Find the next upcoming time zone"""
        future_zones = [zone for zone in time_zones if zone.time_index > current_time]

        if future_zones:
            return min(future_zones, key=lambda z: z.time_index)

        return None

    def _calculate_zone_proximity(
        self, current_time: int, time_zones: List[TimeZone]
    ) -> float:
        """Calculate proximity to nearest time zone (0-1)"""
        zone_tolerance = self.parameters.get("zone_tolerance", 2)

        min_distance = float("inf")

        for zone in time_zones:
            distance = abs(current_time - zone.time_index)
            if distance < min_distance:
                min_distance = distance

        if min_distance <= zone_tolerance:
            return 1.0 - (min_distance / zone_tolerance)
        else:
            return 0.0

    def _determine_time_support_resistance(
        self, current_time: int, time_zones: List[TimeZone], closes: np.ndarray
    ) -> str:
        """Determine if current time represents support or resistance"""
        zone_tolerance = self.parameters.get("zone_tolerance", 2)

        # Check if we're near a significant time zone
        for zone in time_zones:
            if abs(current_time - zone.time_index) <= zone_tolerance:
                if zone.zone_strength >= 0.8:
                    # Analyze price behavior around this time
                    if (
                        current_time > zone_tolerance
                        and current_time < len(closes) - zone_tolerance
                    ):
                        before_avg = np.mean(
                            closes[current_time - zone_tolerance : current_time]
                        )
                        after_avg = np.mean(
                            closes[current_time : current_time + zone_tolerance]
                        )

                        if after_avg > before_avg * 1.01:
                            return "support"
                        elif after_avg < before_avg * 0.99:
                            return "resistance"

        return "neutral"

    def _calculate_time_confluence(
        self, current_time: int, time_zones: List[TimeZone], closes: np.ndarray
    ) -> float:
        """Calculate time confluence score"""
        zone_tolerance = self.parameters.get("zone_tolerance", 2)

        confluence_score = 0.0
        total_weight = 0.0

        for zone in time_zones:
            distance = abs(current_time - zone.time_index)
            if distance <= zone_tolerance * 2:
                # Weight by zone strength and proximity
                proximity_weight = max(0, 1 - distance / (zone_tolerance * 2))
                weight = zone.zone_strength * proximity_weight
                confluence_score += weight
                total_weight += zone.zone_strength

        return confluence_score / total_weight if total_weight > 0 else 0.0

    def _generate_predictions(
        self, current_time: int, time_zones: List[TimeZone], closes: np.ndarray
    ) -> List[TimePrediction]:
        """Generate time-based predictions"""
        predictions = []
        min_strength = self.parameters.get("min_strength_threshold", 0.3)

        # Look for upcoming significant zones
        for zone in time_zones:
            if (
                zone.time_index > current_time
                and zone.time_index < current_time + 50  # Within reasonable future
                and zone.zone_strength >= min_strength
            ):

                # Determine prediction type based on zone characteristics
                if zone.zone_type == "primary":
                    if zone.fibonacci_number in [5, 8, 13]:
                        pred_type = "reversal"
                        confidence = 0.8
                    else:
                        pred_type = "acceleration"
                        confidence = 0.6
                else:
                    pred_type = "consolidation"
                    confidence = 0.4

                predictions.append(
                    TimePrediction(
                        target_time=zone.time_index,
                        fibonacci_number=zone.fibonacci_number,
                        confidence=confidence * zone.zone_strength,
                        prediction_type=pred_type,
                    )
                )

        return predictions

    def validate_parameters(self) -> bool:
        """Validate Fibonacci Time Zones parameters"""
        base_period = self.parameters.get("base_period", 21)
        max_zones = self.parameters.get("max_zones", 12)
        zone_tolerance = self.parameters.get("zone_tolerance", 2)
        min_strength_threshold = self.parameters.get("min_strength_threshold", 0.3)
        period = self.parameters.get("period", 14)  # Support period parameter

        if not isinstance(base_period, int) or base_period < 5:
            raise IndicatorValidationError(
                f"base_period must be integer >= 5, got {base_period}"
            )

        # Validate period parameter if provided
        if not isinstance(period, int) or period < 5 or period > 500:
            raise IndicatorValidationError(
                f"period must be integer between 5 and 500, got {period}"
            )

        if (
            not isinstance(max_zones, int)
            or max_zones < 3
            or max_zones > len(FIBONACCI_SEQUENCE)
        ):
            raise IndicatorValidationError(
                f"max_zones must be integer 3-{len(FIBONACCI_SEQUENCE)}, got {max_zones}"
            )

        if not isinstance(zone_tolerance, int) or zone_tolerance < 1:
            raise IndicatorValidationError(
                f"zone_tolerance must be integer >= 1, got {zone_tolerance}"
            )

        if (
            not isinstance(min_strength_threshold, (int, float))
            or not 0 <= min_strength_threshold <= 1
        ):
            raise IndicatorValidationError(
                f"min_strength_threshold must be 0-1, got {min_strength_threshold}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Fibonacci Time Zones metadata"""
        return {
            "name": "Fibonacci Time Zones",
            "category": self.CATEGORY,
            "description": "Fibonacci Time Zones for time-based analysis and predictions",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series[FibonacciTimeZoneResult]",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
            "fibonacci_sequence": FIBONACCI_SEQUENCE,
            "golden_ratio": GOLDEN_RATIO,
        }

    def _get_required_columns(self) -> List[str]:
        """Required columns for calculation"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return self.parameters.get("base_period", 21) * 2

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "base_period" not in self.parameters:
            self.parameters["base_period"] = 21
        if "max_zones" not in self.parameters:
            self.parameters["max_zones"] = 12
        if "zone_tolerance" not in self.parameters:
            self.parameters["zone_tolerance"] = 2
        if "min_strength_threshold" not in self.parameters:
            self.parameters["min_strength_threshold"] = 0.3

    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        return self.get_metadata()

    @property
    def base_period(self) -> int:
        return self.parameters.get("base_period", 21)

    def get_next_time_zone(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Optional[TimeZone]:
        """Get the next upcoming time zone"""
        results = self.calculate(data)
        latest_result = None

        for result in reversed(results):
            if result is not None:
                latest_result = result
                break

        return latest_result.next_zone if latest_result else None


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return FibonacciTimeZoneIndicator
