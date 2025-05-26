"""
Gann Fan Analysis
Dynamic support/resistance levels using Gann fan methodology.

This module provides comprehensive Gann fan analysis including:
- Dynamic fan line calculations from pivot points
- Support and resistance level identification
- Fan line intersection analysis
- Trend strength assessment

Expected Benefits:
- Dynamic support/resistance levels
- Trend direction confirmation
- Price target identification
- Enhanced timing analysis
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass
class FanLine:
    """Gann fan line data"""
    angle_ratio: str  # e.g., "1x1", "2x1"
    angle_degrees: float
    slope: float
    start_point: Tuple[datetime, float]
    current_level: float
    line_type: str  # 'support', 'resistance'
    strength: float  # 0-1 confidence
    touch_count: int
    last_touch: Optional[datetime]


@dataclass
class FanIntersection:
    """Fan line intersection point"""
    time_point: datetime
    price_point: float
    line1_angle: str
    line2_angle: str
    intersection_type: str  # 'support_confluence', 'resistance_confluence'
    significance: float  # 0-1


@dataclass
class GannFanResult:
    """Gann fan analysis result"""
    symbol: str
    timestamp: datetime
    pivot_point: Tuple[datetime, float]
    fan_lines: List[FanLine]
    intersections: List[FanIntersection]
    current_price: float
    dominant_fan_line: Optional[FanLine]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    next_support: Optional[float]
    next_resistance: Optional[float]
    fan_strength: float  # Overall fan reliability


class GannFanAnalysis:
    """
    Gann Fan Analysis for dynamic support/resistance
    Implements W.D. Gann's fan line methodology
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Gann fan angles (same as angles calculator)
        self.fan_angles = {
            "8x1": 82.5,
            "4x1": 75.0,
            "3x1": 71.25,
            "2x1": 63.75,
            "1x1": 45.0,   # Most important
            "1x2": 26.25,
            "1x3": 18.75,
            "1x4": 15.0,
            "1x8": 7.5
        }

        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

        self.logger.info("Gann Fan Analysis initialized")

    async def analyze_gann_fan(
        self,
        symbol: str,
        price_data: List[Dict],
        pivot_point: Optional[Tuple[datetime, float]] = None
    ) -> GannFanResult:
        """
        Analyze Gann fan from a pivot point

        Args:
            symbol: Trading symbol
            price_data: List of OHLC data with timestamp
            pivot_point: Optional pivot point (time, price), auto-detected if None

        Returns:
            GannFanResult with complete fan analysis
        """
        start_time = time.perf_counter()

        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Find or use provided pivot point
            if pivot_point is None:
                pivot_point = await self._find_fan_pivot(df)

            current_price = df.iloc[-1]['close']

            # Generate fan lines
            fan_lines = await self._generate_fan_lines(df, pivot_point, current_price)

            # Find fan line intersections
            intersections = await self._find_fan_intersections(fan_lines, df)

            # Determine dominant fan line
            dominant_fan_line = await self._find_dominant_fan_line(fan_lines, current_price)

            # Determine trend direction
            trend_direction = await self._determine_fan_trend(fan_lines, current_price, pivot_point[1])

            # Find next support/resistance
            next_support, next_resistance = await self._find_next_fan_levels(fan_lines, current_price)

            # Calculate fan strength
            fan_strength = await self._calculate_fan_strength(fan_lines, df)

            result = GannFanResult(
                symbol=symbol,
                timestamp=datetime.now(),
                pivot_point=pivot_point,
                fan_lines=fan_lines,
                intersections=intersections,
                current_price=current_price,
                dominant_fan_line=dominant_fan_line,
                trend_direction=trend_direction,
                next_support=next_support,
                next_resistance=next_resistance,
                fan_strength=fan_strength
            )

            # Update performance metrics
            calculation_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time

            self.logger.debug(f"Gann fan analyzed for {symbol} in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing Gann fan for {symbol}: {e}")
            raise

    async def _find_fan_pivot(self, df: pd.DataFrame) -> Tuple[datetime, float]:
        """Find the most suitable pivot point for fan analysis"""

        # Look for significant swing points
        window = min(20, len(df) // 4)

        # Calculate swing highs and lows
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()

        # Find swing points
        swing_highs = df[(df['high'] == highs) &
                        (df['high'].shift(1) < df['high']) &
                        (df['high'].shift(-1) < df['high'])]

        swing_lows = df[(df['low'] == lows) &
                       (df['low'].shift(1) > df['low']) &
                       (df['low'].shift(-1) > df['low'])]

        # Combine and evaluate significance
        pivots = []
        current_price = df.iloc[-1]['close']

        for _, row in swing_highs.iterrows():
            price_significance = abs(row['high'] - current_price) / current_price
            time_significance = 1.0 - abs((df.iloc[-1]['timestamp'] - row['timestamp']).total_seconds()) / (len(df) * 3600)
            total_significance = (price_significance * 0.7) + (time_significance * 0.3)
            pivots.append((row['timestamp'], row['high'], total_significance))

        for _, row in swing_lows.iterrows():
            price_significance = abs(row['low'] - current_price) / current_price
            time_significance = 1.0 - abs((df.iloc[-1]['timestamp'] - row['timestamp']).total_seconds()) / (len(df) * 3600)
            total_significance = (price_significance * 0.7) + (time_significance * 0.3)
            pivots.append((row['timestamp'], row['low'], total_significance))

        if not pivots:
            # Fallback to middle point
            mid_idx = len(df) // 2
            return (df.iloc[mid_idx]['timestamp'], df.iloc[mid_idx]['close'])

        # Return most significant pivot
        pivots.sort(key=lambda x: x[2], reverse=True)
        return (pivots[0][0], pivots[0][1])

    async def _generate_fan_lines(
        self,
        df: pd.DataFrame,
        pivot_point: Tuple[datetime, float],
        current_price: float
    ) -> List[FanLine]:
        """Generate Gann fan lines from pivot point"""

        pivot_time, pivot_price = pivot_point
        fan_lines = []

        # Current time for calculations
        current_time = df.iloc[-1]['timestamp']
        time_diff_hours = (current_time - pivot_time).total_seconds() / 3600

        if time_diff_hours <= 0:
            time_diff_hours = 1

        for angle_name, angle_degrees in self.fan_angles.items():
            # Calculate slope
            slope_radians = math.radians(angle_degrees)
            slope = math.tan(slope_radians)

            # Calculate current level on this fan line
            if current_price > pivot_price:
                # Upward fan
                current_level = pivot_price + (slope * time_diff_hours)
                line_type = 'support' if current_level < current_price else 'resistance'
            else:
                # Downward fan
                current_level = pivot_price - (slope * time_diff_hours)
                line_type = 'resistance' if current_level > current_price else 'support'

            # Calculate strength and touches
            strength = await self._calculate_fan_line_strength(df, pivot_point, slope, angle_name)
            touch_count = await self._count_fan_line_touches(df, pivot_point, slope)

            fan_line = FanLine(
                angle_ratio=angle_name,
                angle_degrees=angle_degrees,
                slope=slope,
                start_point=pivot_point,
                current_level=current_level,
                line_type=line_type,
                strength=strength,
                touch_count=touch_count,
                last_touch=None  # Would need detailed analysis
            )

            fan_lines.append(fan_line)

        return fan_lines

    async def _calculate_fan_line_strength(
        self,
        df: pd.DataFrame,
        pivot_point: Tuple[datetime, float],
        slope: float,
        angle_name: str
    ) -> float:
        """Calculate the strength of a fan line"""

        # Key angles have higher base strength
        key_angles = ["1x1", "2x1", "1x2"]
        base_strength = 0.9 if angle_name in key_angles else 0.7

        # Calculate how well price respects this line
        respect_count = 0
        total_tests = 0

        pivot_time, pivot_price = pivot_point

        for _, row in df.iterrows():
            if row['timestamp'] <= pivot_time:
                continue

            # Calculate expected price on fan line at this time
            time_diff_hours = (row['timestamp'] - pivot_time).total_seconds() / 3600
            expected_price = pivot_price + (slope * time_diff_hours)

            # Check if price respected the line
            tolerance = expected_price * 0.002  # 0.2% tolerance

            if abs(row['low'] - expected_price) <= tolerance or abs(row['high'] - expected_price) <= tolerance:
                respect_count += 1

            total_tests += 1

        if total_tests == 0:
            return base_strength

        respect_ratio = respect_count / total_tests
        return min(1.0, base_strength * (0.5 + respect_ratio))

    async def _count_fan_line_touches(
        self,
        df: pd.DataFrame,
        pivot_point: Tuple[datetime, float],
        slope: float,
        tolerance: float = 0.002
    ) -> int:
        """Count touches on a fan line"""

        touches = 0
        pivot_time, pivot_price = pivot_point

        for _, row in df.iterrows():
            if row['timestamp'] <= pivot_time:
                continue

            time_diff_hours = (row['timestamp'] - pivot_time).total_seconds() / 3600
            expected_price = pivot_price + (slope * time_diff_hours)

            # Check for touches
            if (abs(row['low'] - expected_price) / expected_price <= tolerance or
                abs(row['high'] - expected_price) / expected_price <= tolerance):
                touches += 1

        return touches

    async def _find_fan_intersections(
        self,
        fan_lines: List[FanLine],
        df: pd.DataFrame
    ) -> List[FanIntersection]:
        """Find intersections between fan lines"""

        intersections = []

        # Compare each pair of fan lines
        for i, line1 in enumerate(fan_lines):
            for j, line2 in enumerate(fan_lines[i+1:], i+1):
                if line1.slope == line2.slope:
                    continue  # Parallel lines don't intersect

                # Calculate intersection point
                # Using y = mx + b format where b is adjusted for pivot point
                pivot_time, pivot_price = line1.start_point

                # Solve for intersection
                time_intersection = (pivot_price - pivot_price) / (line2.slope - line1.slope)
                price_intersection = pivot_price + (line1.slope * time_intersection)

                if time_intersection > 0:  # Future intersection
                    intersection_time = pivot_time + timedelta(hours=time_intersection)

                    # Determine intersection type
                    intersection_type = 'support_confluence' if price_intersection < df.iloc[-1]['close'] else 'resistance_confluence'

                    # Calculate significance
                    significance = min(line1.strength, line2.strength)

                    intersection = FanIntersection(
                        time_point=intersection_time,
                        price_point=price_intersection,
                        line1_angle=line1.angle_ratio,
                        line2_angle=line2.angle_ratio,
                        intersection_type=intersection_type,
                        significance=significance
                    )

                    intersections.append(intersection)

        # Sort by significance
        intersections.sort(key=lambda x: x.significance, reverse=True)

        return intersections[:10]  # Keep top 10

    async def _find_dominant_fan_line(
        self,
        fan_lines: List[FanLine],
        current_price: float
    ) -> Optional[FanLine]:
        """Find the most relevant fan line"""

        if not fan_lines:
            return None

        scored_lines = []

        for line in fan_lines:
            # Score based on proximity to current price
            proximity_score = 1 / (1 + abs(line.current_level - current_price) / current_price)

            # Combined score
            total_score = (line.strength * 0.6) + (proximity_score * 0.4)
            scored_lines.append((line, total_score))

        scored_lines.sort(key=lambda x: x[1], reverse=True)
        return scored_lines[0][0]

    async def _determine_fan_trend(
        self,
        fan_lines: List[FanLine],
        current_price: float,
        pivot_price: float
    ) -> str:
        """Determine trend direction based on fan analysis"""

        if current_price > pivot_price * 1.01:
            return 'bullish'
        elif current_price < pivot_price * 0.99:
            return 'bearish'
        else:
            return 'neutral'

    async def _find_next_fan_levels(
        self,
        fan_lines: List[FanLine],
        current_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find next support and resistance from fan lines"""

        next_support = None
        next_resistance = None

        for line in fan_lines:
            if line.line_type == 'support' and line.current_level < current_price:
                if next_support is None or line.current_level > next_support:
                    next_support = line.current_level
            elif line.line_type == 'resistance' and line.current_level > current_price:
                if next_resistance is None or line.current_level < next_resistance:
                    next_resistance = line.current_level

        return next_support, next_resistance

    async def _calculate_fan_strength(
        self,
        fan_lines: List[FanLine],
        df: pd.DataFrame
    ) -> float:
        """Calculate overall fan strength"""

        if not fan_lines:
            return 0.0

        avg_strength = sum(line.strength for line in fan_lines) / len(fan_lines)
        total_touches = sum(line.touch_count for line in fan_lines)
        touch_factor = min(1.0, total_touches / 20)  # Normalize

        return min(1.0, (avg_strength * 0.7) + (touch_factor * 0.3))

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the analyzer"""

        if self.calculation_count == 0:
            return {
                'calculations_performed': 0,
                'average_calculation_time': 0.0,
                'total_calculation_time': 0.0
            }

        return {
            'calculations_performed': self.calculation_count,
            'average_calculation_time': self.total_calculation_time / self.calculation_count,
            'total_calculation_time': self.total_calculation_time
        }
