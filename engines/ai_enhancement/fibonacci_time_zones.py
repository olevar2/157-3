"""
Platform3 Fibonacci Time Zones Indicator
=========================================

Individual implementation of Fibonacci Time Zones analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Fibonacci Sequence for Time Zones
FIBONACCI_TIME_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]


@dataclass
class TimeZone:
    """Represents a Fibonacci time zone"""

    period: int
    fibonacci_number: int
    projected_date: Optional[datetime]
    time_support_resistance: str  # 'support', 'resistance', 'neutral'
    confluence_strength: float


@dataclass
class FibonacciTimeZoneResult:
    """Result structure for Fibonacci Time Zone analysis"""

    base_time: datetime
    time_zones: List[TimeZone]
    next_zone: Optional[TimeZone]
    current_zone_position: float  # 0-1 position within current zone
    time_support_resistance: str
    time_confluence: float


class FibonacciTimeZones:
    """
    Fibonacci Time Zones Indicator

    Projects future time periods based on Fibonacci sequence
    to identify potential reversal time zones.
    """

    def __init__(self, swing_window: int = 10, max_projections: int = 8, **kwargs):
        """
        Initialize Fibonacci Time Zones indicator

        Args:
            swing_window: Number of periods to look for base time point
            max_projections: Maximum number of time zones to project
        """
        self.swing_window = swing_window
        self.max_projections = max_projections
        self.sensitivity = kwargs.get("sensitivity", 0.02)
        self.logger = logging.getLogger(__name__)

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FibonacciTimeZoneResult]:
        """
        Calculate Fibonacci Time Zones for given data.

        Args:
            data: Price data (DataFrame with OHLC, dict, or array)

        Returns:
            FibonacciTimeZoneResult with time zones and confluence analysis
        """
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                closes = data["close"].values
                dates = data.index if hasattr(data, "index") else None
            elif isinstance(data, dict):
                closes = np.array(data.get("close", []))
                dates = data.get("dates", None)
            elif isinstance(data, np.ndarray):
                closes = data.flatten()
                dates = None
            else:
                return None

            if len(closes) < self.swing_window:
                return None

            # Find significant time point (base for time zone calculation)
            base_time_index = self._find_base_time_point(closes)

            # Create base time reference
            base_time = self._create_base_time(dates, base_time_index)

            # Calculate time zones
            time_zones = self._calculate_time_zones(
                base_time, len(closes), base_time_index
            )

            # Find next upcoming time zone
            next_zone = self._find_next_zone(time_zones, len(closes))

            # Calculate current position within time zone cycle
            current_position = self._calculate_current_position(time_zones, len(closes))

            # Analyze time-based support/resistance
            time_sr = self._analyze_time_support_resistance(time_zones, closes)

            # Calculate time confluence
            time_confluence = self._calculate_time_confluence(time_zones, len(closes))

            return FibonacciTimeZoneResult(
                base_time=base_time,
                time_zones=time_zones,
                next_zone=next_zone,
                current_zone_position=current_position,
                time_support_resistance=time_sr,
                time_confluence=time_confluence,
            )

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci Time Zones: {e}")
            return None

    def _find_base_time_point(self, closes: np.ndarray) -> int:
        """Find significant time point to base Fibonacci projections on"""
        data_len = len(closes)

        # Look for significant swing point in early part of data
        search_start = max(0, data_len // 4)
        search_end = min(data_len - 1, 3 * data_len // 4)

        # Find the most significant price movement in the search range
        price_changes = np.abs(np.diff(closes[search_start : search_end + 1]))

        if len(price_changes) > 0:
            max_change_idx = np.argmax(price_changes) + search_start
            return max_change_idx
        else:
            return search_start

    def _create_base_time(self, dates, base_index: int) -> datetime:
        """Create base time reference for time zone calculations"""
        if dates is not None and hasattr(dates, "__getitem__"):
            try:
                if hasattr(dates[base_index], "to_pydatetime"):
                    return dates[base_index].to_pydatetime()
                elif isinstance(dates[base_index], datetime):
                    return dates[base_index]
                else:
                    return datetime.now() - timedelta(days=len(dates) - base_index)
            except (IndexError, AttributeError):
                pass

        # Default: create synthetic base time
        return datetime.now() - timedelta(days=50)

    def _calculate_time_zones(
        self, base_time: datetime, data_length: int, base_index: int
    ) -> List[TimeZone]:
        """Calculate Fibonacci time zones from base time"""
        time_zones = []

        # Calculate time zones using Fibonacci sequence
        for i, fib_number in enumerate(FIBONACCI_TIME_SEQUENCE[: self.max_projections]):
            # Calculate period offset from base
            period_offset = fib_number
            projected_period = base_index + period_offset

            # Calculate projected date
            try:
                projected_date = base_time + timedelta(days=period_offset)
            except (OverflowError, ValueError):
                projected_date = None

            # Determine support/resistance characteristic
            if i % 2 == 0:
                sr_type = "support"
            else:
                sr_type = "resistance"

            # Calculate confluence strength based on position in sequence
            confluence_strength = max(0.1, 1.0 - (i * 0.1))

            time_zones.append(
                TimeZone(
                    period=projected_period,
                    fibonacci_number=fib_number,
                    projected_date=projected_date,
                    time_support_resistance=sr_type,
                    confluence_strength=confluence_strength,
                )
            )

        return time_zones

    def _find_next_zone(
        self, time_zones: List[TimeZone], current_period: int
    ) -> Optional[TimeZone]:
        """Find the next upcoming time zone"""
        future_zones = [zone for zone in time_zones if zone.period > current_period]

        if future_zones:
            return min(future_zones, key=lambda z: z.period)
        return None

    def _calculate_current_position(
        self, time_zones: List[TimeZone], current_period: int
    ) -> float:
        """Calculate current position within the time zone cycle"""
        if not time_zones:
            return 0.0

        # Find the zone we're currently in or past
        past_zones = [zone for zone in time_zones if zone.period <= current_period]
        future_zones = [zone for zone in time_zones if zone.period > current_period]

        if not past_zones and not future_zones:
            return 0.0

        if not future_zones:
            return 1.0  # Past all calculated zones

        if not past_zones:
            # Before first zone
            next_zone = min(future_zones, key=lambda z: z.period)
            return current_period / next_zone.period

        # Between zones
        last_zone = max(past_zones, key=lambda z: z.period)
        next_zone = min(future_zones, key=lambda z: z.period)

        zone_range = next_zone.period - last_zone.period
        current_offset = current_period - last_zone.period

        if zone_range > 0:
            return current_offset / zone_range
        else:
            return 0.0

    def _analyze_time_support_resistance(
        self, time_zones: List[TimeZone], closes: np.ndarray
    ) -> str:
        """Analyze time-based support/resistance characteristics"""
        if not time_zones or len(closes) < 5:
            return "neutral"

        current_period = len(closes) - 1

        # Find zones near current time
        nearby_zones = []
        for zone in time_zones:
            time_distance = abs(zone.period - current_period)
            if time_distance <= 3:  # Within 3 periods
                nearby_zones.append(zone)

        if not nearby_zones:
            return "neutral"

        # Analyze recent price action around time zones
        support_count = sum(
            1 for zone in nearby_zones if zone.time_support_resistance == "support"
        )
        resistance_count = sum(
            1 for zone in nearby_zones if zone.time_support_resistance == "resistance"
        )

        if support_count > resistance_count:
            return "support"
        elif resistance_count > support_count:
            return "resistance"
        else:
            return "neutral"

    def _calculate_time_confluence(
        self, time_zones: List[TimeZone], current_period: int
    ) -> float:
        """Calculate time confluence score based on proximity to multiple zones"""
        if not time_zones:
            return 0.0

        confluence_score = 0.0
        total_weight = 0.0

        for zone in time_zones:
            # Calculate distance to this time zone
            time_distance = abs(zone.period - current_period)

            # Weight decreases with distance
            if time_distance == 0:
                weight = 1.0
            elif time_distance <= 2:
                weight = 0.8
            elif time_distance <= 5:
                weight = 0.5
            elif time_distance <= 10:
                weight = 0.2
            else:
                weight = 0.0

            confluence_score += weight * zone.confluence_strength
            total_weight += weight

        return confluence_score / total_weight if total_weight > 0 else 0.0
