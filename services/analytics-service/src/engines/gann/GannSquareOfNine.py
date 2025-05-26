"""
Gann Square of Nine
Price/time predictions using W.D. Gann's Square of 9 algorithm.

This module provides comprehensive Square of Nine analysis including:
- Price level calculations using square root mathematics
- Time cycle predictions based on square relationships
- Support and resistance level identification
- Mathematical precision in forecasting

Expected Benefits:
- Precise price target calculations
- Time-based cycle predictions
- Mathematical support/resistance levels
- Enhanced forecasting accuracy
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
class SquareLevel:
    """Square of Nine level data"""
    level: float
    square_root: float
    angle: float  # 0-360 degrees
    level_type: str  # 'support', 'resistance', 'neutral'
    strength: float  # 0-1 confidence
    distance_from_price: float


@dataclass
class PriceTimeTarget:
    """Price and time target from Square of Nine"""
    price_target: float
    time_target: datetime
    confidence: float
    calculation_method: str
    supporting_levels: List[SquareLevel]


@dataclass
class SquareOfNineResult:
    """Square of Nine analysis result"""
    symbol: str
    timestamp: datetime
    base_price: float
    current_price: float
    square_root_base: float
    levels: List[SquareLevel]
    price_targets: List[PriceTimeTarget]
    next_support: Optional[float]
    next_resistance: Optional[float]
    dominant_cycle: Optional[int]
    analysis_confidence: float


class GannSquareOfNine:
    """
    Gann Square of Nine Calculator
    Implements W.D. Gann's Square of 9 mathematical trading method
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Square of Nine angles (degrees)
        self.square_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        # Common Gann cycles
        self.gann_cycles = [7, 14, 21, 30, 45, 60, 90, 120, 144, 180, 360]

        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

        self.logger.info("Gann Square of Nine initialized")

    async def calculate_square_of_nine(
        self,
        symbol: str,
        current_price: float,
        base_price: Optional[float] = None,
        time_data: Optional[List[Dict]] = None
    ) -> SquareOfNineResult:
        """
        Calculate Square of Nine levels and targets

        Args:
            symbol: Trading symbol
            current_price: Current market price
            base_price: Base price for calculations (uses current if None)
            time_data: Optional time series data for cycle analysis

        Returns:
            SquareOfNineResult with complete analysis
        """
        start_time = time.perf_counter()

        try:
            # Use current price as base if not provided
            if base_price is None:
                base_price = current_price

            # Calculate square root of base price
            square_root_base = math.sqrt(base_price)

            # Generate Square of Nine levels
            levels = await self._generate_square_levels(base_price, current_price, square_root_base)

            # Calculate price and time targets
            price_targets = await self._calculate_price_time_targets(
                base_price, current_price, square_root_base, time_data
            )

            # Find next support and resistance
            next_support, next_resistance = await self._find_next_levels(levels, current_price)

            # Determine dominant cycle
            dominant_cycle = await self._find_dominant_cycle(time_data) if time_data else None

            # Calculate analysis confidence
            confidence = await self._calculate_analysis_confidence(levels, price_targets)

            result = SquareOfNineResult(
                symbol=symbol,
                timestamp=datetime.now(),
                base_price=base_price,
                current_price=current_price,
                square_root_base=square_root_base,
                levels=levels,
                price_targets=price_targets,
                next_support=next_support,
                next_resistance=next_resistance,
                dominant_cycle=dominant_cycle,
                analysis_confidence=confidence
            )

            # Update performance metrics
            calculation_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time

            self.logger.debug(f"Square of Nine calculated for {symbol} in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating Square of Nine for {symbol}: {e}")
            raise

    async def _generate_square_levels(
        self,
        base_price: float,
        current_price: float,
        square_root_base: float
    ) -> List[SquareLevel]:
        """Generate Square of Nine price levels"""

        levels = []

        # Generate levels for each angle
        for angle in self.square_angles:
            # Calculate angle in radians
            angle_radians = math.radians(angle)

            # Calculate level using square root + angle
            level_increment = angle / 360.0  # Normalize angle to 0-1

            # Generate levels above and below base
            for direction in [-1, 1]:
                for multiplier in [0.5, 1, 1.5, 2, 2.5, 3]:
                    new_square_root = square_root_base + (direction * multiplier * level_increment)
                    level_price = new_square_root ** 2

                    if level_price <= 0:
                        continue

                    # Determine level type based on current price
                    if level_price > current_price * 1.001:  # 0.1% threshold
                        level_type = 'resistance'
                    elif level_price < current_price * 0.999:  # 0.1% threshold
                        level_type = 'support'
                    else:
                        level_type = 'neutral'

                    # Calculate strength based on mathematical significance
                    strength = self._calculate_level_strength(angle, multiplier)

                    # Calculate distance from current price
                    distance = abs(level_price - current_price)

                    level = SquareLevel(
                        level=level_price,
                        square_root=new_square_root,
                        angle=angle,
                        level_type=level_type,
                        strength=strength,
                        distance_from_price=distance
                    )

                    levels.append(level)

        # Sort by distance from current price
        levels.sort(key=lambda x: x.distance_from_price)

        # Keep only the most relevant levels (closest 20)
        return levels[:20]

    def _calculate_level_strength(self, angle: float, multiplier: float) -> float:
        """Calculate the strength of a Square of Nine level"""

        # Key angles have higher strength
        key_angles = [0, 90, 180, 270, 360]
        angle_strength = 1.0 if angle in key_angles else 0.7

        # Whole number multipliers have higher strength
        multiplier_strength = 1.0 if multiplier == int(multiplier) else 0.8

        # Combine factors
        return min(1.0, angle_strength * multiplier_strength)

    async def _calculate_price_time_targets(
        self,
        base_price: float,
        current_price: float,
        square_root_base: float,
        time_data: Optional[List[Dict]]
    ) -> List[PriceTimeTarget]:
        """Calculate price and time targets using Square of Nine"""

        targets = []

        # Calculate targets for each Gann cycle
        for cycle in self.gann_cycles[:5]:  # Use first 5 cycles
            # Price target calculation
            cycle_increment = cycle / 360.0
            target_square_root = square_root_base + cycle_increment
            price_target = target_square_root ** 2

            # Time target calculation (simplified)
            time_target = datetime.now() + timedelta(days=cycle)

            # Calculate confidence based on cycle significance
            confidence = self._calculate_target_confidence(cycle, price_target, current_price)

            target = PriceTimeTarget(
                price_target=price_target,
                time_target=time_target,
                confidence=confidence,
                calculation_method=f"Square of Nine - {cycle} cycle",
                supporting_levels=[]  # Would be populated with related levels
            )

            targets.append(target)

        # Sort by confidence
        targets.sort(key=lambda x: x.confidence, reverse=True)

        return targets

    def _calculate_target_confidence(self, cycle: int, price_target: float, current_price: float) -> float:
        """Calculate confidence in a price/time target"""

        # Key cycles have higher confidence
        key_cycles = [30, 60, 90, 180, 360]
        cycle_confidence = 0.9 if cycle in key_cycles else 0.7

        # Reasonable price targets have higher confidence
        price_change = abs(price_target - current_price) / current_price
        price_confidence = max(0.3, 1.0 - price_change)  # Lower confidence for extreme targets

        return min(1.0, cycle_confidence * price_confidence)

    async def _find_next_levels(
        self,
        levels: List[SquareLevel],
        current_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find next support and resistance levels"""

        next_support = None
        next_resistance = None

        for level in levels:
            if level.level_type == 'support' and level.level < current_price:
                if next_support is None or level.level > next_support:
                    next_support = level.level
            elif level.level_type == 'resistance' and level.level > current_price:
                if next_resistance is None or level.level < next_resistance:
                    next_resistance = level.level

        return next_support, next_resistance

    async def _find_dominant_cycle(self, time_data: List[Dict]) -> Optional[int]:
        """Find the dominant time cycle in the data"""

        if not time_data or len(time_data) < 30:
            return None

        # Simple cycle detection using price peaks
        df = pd.DataFrame(time_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Find peaks in price data
        prices = df['close'].values
        peaks = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(i)

        if len(peaks) < 3:
            return None

        # Calculate average distance between peaks
        distances = []
        for i in range(1, len(peaks)):
            distances.append(peaks[i] - peaks[i-1])

        if not distances:
            return None

        avg_distance = sum(distances) / len(distances)

        # Find closest Gann cycle
        closest_cycle = min(self.gann_cycles, key=lambda x: abs(x - avg_distance))

        return closest_cycle

    async def _calculate_analysis_confidence(
        self,
        levels: List[SquareLevel],
        price_targets: List[PriceTimeTarget]
    ) -> float:
        """Calculate overall confidence in the Square of Nine analysis"""

        if not levels and not price_targets:
            return 0.0

        # Level confidence
        level_confidence = 0.0
        if levels:
            avg_level_strength = sum(level.strength for level in levels) / len(levels)
            level_confidence = avg_level_strength

        # Target confidence
        target_confidence = 0.0
        if price_targets:
            avg_target_confidence = sum(target.confidence for target in price_targets) / len(price_targets)
            target_confidence = avg_target_confidence

        # Combined confidence
        if levels and price_targets:
            return (level_confidence * 0.6) + (target_confidence * 0.4)
        elif levels:
            return level_confidence
        else:
            return target_confidence

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
