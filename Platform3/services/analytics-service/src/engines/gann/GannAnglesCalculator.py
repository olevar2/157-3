"""
Gann Angles Calculator
Complete Gann analysis toolkit for precise geometric price analysis.

This module provides comprehensive Gann angle calculations including:
- 1x1, 2x1, 3x1, 4x1, 8x1 angle calculations
- Dynamic Gann fan analysis
- Time-price cycle detection
- Mathematical precision in forecasting

Expected Benefits:
- Precise geometric price analysis
- Time-based cycle predictions
- Dynamic support/resistance levels
- Mathematical precision in forecasting
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import math


@dataclass
class GannAngle:
    """Gann angle data structure"""
    angle_ratio: str  # e.g., "1x1", "2x1", "3x1"
    angle_degrees: float
    slope: float
    support_level: float
    resistance_level: float
    strength: float  # 0-1 confidence
    last_touch: Optional[datetime]
    touch_count: int


@dataclass
class GannFanLevel:
    """Gann fan level data"""
    price_level: float
    angle_type: str
    direction: str  # 'up' or 'down'
    strength: float
    distance_from_price: float


@dataclass
class GannAnglesResult:
    """Gann angles analysis result"""
    symbol: str
    timestamp: datetime
    pivot_point: Tuple[datetime, float]  # (time, price)
    angles: List[GannAngle]
    fan_levels: List[GannFanLevel]
    current_price: float
    dominant_angle: Optional[GannAngle]
    next_support: Optional[float]
    next_resistance: Optional[float]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    analysis_confidence: float


class GannAnglesCalculator:
    """
    Gann Angles Calculator for geometric price analysis
    Implements W.D. Gann's geometric trading methods
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Gann angle ratios and their degrees
        self.gann_angles = {
            "8x1": 82.5,   # 8 units price to 1 unit time
            "4x1": 75.0,   # 4 units price to 1 unit time
            "3x1": 71.25,  # 3 units price to 1 unit time
            "2x1": 63.75,  # 2 units price to 1 unit time
            "1x1": 45.0,   # 1 unit price to 1 unit time (most important)
            "1x2": 26.25,  # 1 unit price to 2 units time
            "1x3": 18.75,  # 1 unit price to 3 units time
            "1x4": 15.0,   # 1 unit price to 4 units time
            "1x8": 7.5     # 1 unit price to 8 units time
        }

        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

        self.logger.info("Gann Angles Calculator initialized")

    async def calculate_gann_angles(
        self,
        symbol: str,
        price_data: List[Dict],
        pivot_point: Optional[Tuple[datetime, float]] = None
    ) -> GannAnglesResult:
        """
        Calculate Gann angles from a pivot point

        Args:
            symbol: Trading symbol
            price_data: List of OHLC data with timestamp
            pivot_point: Optional pivot point (time, price), auto-detected if None

        Returns:
            GannAnglesResult with complete analysis
        """
        start_time = time.perf_counter()

        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Find or use provided pivot point
            if pivot_point is None:
                pivot_point = await self._find_significant_pivot(df)

            pivot_time, pivot_price = pivot_point
            current_price = df.iloc[-1]['close']

            # Calculate all Gann angles
            angles = await self._calculate_all_angles(df, pivot_point, current_price)

            # Generate fan levels
            fan_levels = await self._generate_fan_levels(angles, current_price, pivot_price)

            # Determine dominant angle and trend
            dominant_angle = await self._find_dominant_angle(angles, current_price)
            trend_direction = await self._determine_trend_direction(angles, current_price, pivot_price)

            # Find next support/resistance
            next_support, next_resistance = await self._find_next_levels(fan_levels, current_price)

            # Calculate analysis confidence
            confidence = await self._calculate_analysis_confidence(angles, df)

            result = GannAnglesResult(
                symbol=symbol,
                timestamp=datetime.now(),
                pivot_point=pivot_point,
                angles=angles,
                fan_levels=fan_levels,
                current_price=current_price,
                dominant_angle=dominant_angle,
                next_support=next_support,
                next_resistance=next_resistance,
                trend_direction=trend_direction,
                analysis_confidence=confidence
            )

            # Update performance metrics
            calculation_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time

            self.logger.debug(f"Gann angles calculated for {symbol} in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating Gann angles for {symbol}: {e}")
            raise

    async def _find_significant_pivot(self, df: pd.DataFrame) -> Tuple[datetime, float]:
        """Find the most significant pivot point for Gann analysis"""

        # Look for swing highs and lows in recent data
        window = min(20, len(df) // 4)  # Use 20 periods or 1/4 of data

        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()

        # Find swing highs (local maxima)
        swing_highs = df[(df['high'] == highs) & (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])]

        # Find swing lows (local minima)
        swing_lows = df[(df['low'] == lows) & (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])]

        # Combine and sort by significance (price range)
        pivots = []

        for _, row in swing_highs.iterrows():
            significance = abs(row['high'] - df['close'].iloc[-1]) / df['close'].iloc[-1]
            pivots.append((row['timestamp'], row['high'], significance, 'high'))

        for _, row in swing_lows.iterrows():
            significance = abs(row['low'] - df['close'].iloc[-1]) / df['close'].iloc[-1]
            pivots.append((row['timestamp'], row['low'], significance, 'low'))

        if not pivots:
            # Fallback to recent significant point
            mid_point = len(df) // 2
            return (df.iloc[mid_point]['timestamp'], df.iloc[mid_point]['close'])

        # Sort by significance and recency
        pivots.sort(key=lambda x: (x[2], -abs((df.iloc[-1]['timestamp'] - x[0]).total_seconds())), reverse=True)

        return (pivots[0][0], pivots[0][1])

    async def _calculate_all_angles(
        self,
        df: pd.DataFrame,
        pivot_point: Tuple[datetime, float],
        current_price: float
    ) -> List[GannAngle]:
        """Calculate all Gann angles from pivot point"""

        pivot_time, pivot_price = pivot_point
        angles = []

        # Time difference in hours for scaling
        current_time = df.iloc[-1]['timestamp']
        time_diff_hours = (current_time - pivot_time).total_seconds() / 3600

        if time_diff_hours <= 0:
            time_diff_hours = 1  # Minimum time difference

        for angle_name, angle_degrees in self.gann_angles.items():
            # Calculate slope based on Gann angle
            slope_radians = math.radians(angle_degrees)
            slope = math.tan(slope_radians)

            # Calculate support and resistance levels
            # For upward angles
            resistance_level = pivot_price + (slope * time_diff_hours)
            support_level = pivot_price - (slope * time_diff_hours)

            # Calculate strength based on price proximity
            price_distance = min(
                abs(current_price - resistance_level),
                abs(current_price - support_level)
            )
            max_distance = abs(resistance_level - support_level)
            strength = max(0, 1 - (price_distance / max_distance)) if max_distance > 0 else 0

            # Count touches (simplified - check recent price action near levels)
            touch_count = await self._count_level_touches(df, resistance_level, support_level)

            angle = GannAngle(
                angle_ratio=angle_name,
                angle_degrees=angle_degrees,
                slope=slope,
                support_level=support_level,
                resistance_level=resistance_level,
                strength=strength,
                last_touch=None,  # Would need more complex analysis
                touch_count=touch_count
            )

            angles.append(angle)

        return angles

    async def _count_level_touches(
        self,
        df: pd.DataFrame,
        resistance_level: float,
        support_level: float,
        tolerance: float = 0.001
    ) -> int:
        """Count how many times price touched the Gann levels"""

        touches = 0

        for _, row in df.iterrows():
            high_price = row['high']
            low_price = row['low']

            # Check resistance touch
            if abs(high_price - resistance_level) / resistance_level <= tolerance:
                touches += 1

            # Check support touch
            if abs(low_price - support_level) / support_level <= tolerance:
                touches += 1

        return touches

    async def _generate_fan_levels(
        self,
        angles: List[GannAngle],
        current_price: float,
        pivot_price: float
    ) -> List[GannFanLevel]:
        """Generate Gann fan levels for visualization and trading"""

        fan_levels = []
        direction = 'up' if current_price > pivot_price else 'down'

        for angle in angles:
            # Resistance level
            resistance_distance = abs(angle.resistance_level - current_price)
            fan_levels.append(GannFanLevel(
                price_level=angle.resistance_level,
                angle_type=angle.angle_ratio,
                direction=direction,
                strength=angle.strength,
                distance_from_price=resistance_distance
            ))

            # Support level
            support_distance = abs(angle.support_level - current_price)
            fan_levels.append(GannFanLevel(
                price_level=angle.support_level,
                angle_type=angle.angle_ratio,
                direction=direction,
                strength=angle.strength,
                distance_from_price=support_distance
            ))

        # Sort by distance from current price
        fan_levels.sort(key=lambda x: x.distance_from_price)

        return fan_levels

    async def _find_dominant_angle(
        self,
        angles: List[GannAngle],
        current_price: float
    ) -> Optional[GannAngle]:
        """Find the most relevant Gann angle for current price action"""

        if not angles:
            return None

        # Sort by strength and proximity to current price
        scored_angles = []

        for angle in angles:
            # Calculate proximity score
            resistance_proximity = abs(current_price - angle.resistance_level) / current_price
            support_proximity = abs(current_price - angle.support_level) / current_price
            proximity_score = 1 / (1 + min(resistance_proximity, support_proximity))

            # Combined score
            total_score = (angle.strength * 0.6) + (proximity_score * 0.4)
            scored_angles.append((angle, total_score))

        # Return angle with highest score
        scored_angles.sort(key=lambda x: x[1], reverse=True)
        return scored_angles[0][0]

    async def _determine_trend_direction(
        self,
        angles: List[GannAngle],
        current_price: float,
        pivot_price: float
    ) -> str:
        """Determine overall trend direction based on Gann analysis"""

        if current_price > pivot_price * 1.01:  # 1% threshold
            return 'bullish'
        elif current_price < pivot_price * 0.99:  # 1% threshold
            return 'bearish'
        else:
            return 'neutral'

    async def _find_next_levels(
        self,
        fan_levels: List[GannFanLevel],
        current_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find next support and resistance levels"""

        next_support = None
        next_resistance = None

        for level in fan_levels:
            if level.price_level < current_price and (next_support is None or level.price_level > next_support):
                next_support = level.price_level
            elif level.price_level > current_price and (next_resistance is None or level.price_level < next_resistance):
                next_resistance = level.price_level

        return next_support, next_resistance

    async def _calculate_analysis_confidence(
        self,
        angles: List[GannAngle],
        df: pd.DataFrame
    ) -> float:
        """Calculate confidence in the Gann analysis"""

        if not angles:
            return 0.0

        # Base confidence on angle strengths and data quality
        avg_strength = sum(angle.strength for angle in angles) / len(angles)
        data_quality = min(1.0, len(df) / 100)  # More data = higher confidence

        # Touch count factor
        total_touches = sum(angle.touch_count for angle in angles)
        touch_factor = min(1.0, total_touches / 10)  # Normalize to 0-1

        confidence = (avg_strength * 0.5) + (data_quality * 0.3) + (touch_factor * 0.2)
        return min(1.0, confidence)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the calculator"""

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
