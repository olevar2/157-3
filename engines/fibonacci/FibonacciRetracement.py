# -*- coding: utf-8 -*-
"""
Fibonacci Retracement
Multi-level retracement calculations for support and resistance analysis.

This module provides comprehensive Fibonacci retracement analysis including:
- Standard Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Custom retracement levels
- Dynamic support/resistance identification
- Level strength assessment

Expected Benefits:
- Precise support/resistance levels
- Enhanced entry/exit timing
- Dynamic level strength analysis
- Mathematical precision in retracements
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass
class FibonacciLevel:
    """Fibonacci retracement level data"""
    level_percentage: float  # e.g., 0.618 for 61.8%
    price_level: float
    level_type: str  # 'support', 'resistance', 'neutral'
    strength: float  # 0-1 confidence
    touch_count: int
    last_touch: Optional[datetime]
    distance_from_current: float


@dataclass
class FibonacciZone:
    """Fibonacci confluence zone"""
    zone_start: float
    zone_end: float
    zone_center: float
    zone_strength: float
    contributing_levels: List[FibonacciLevel]
    zone_type: str  # 'support_zone', 'resistance_zone'


@dataclass
class RetracementResult:
    """Fibonacci retracement analysis result"""
    symbol: str
    timestamp: datetime
    swing_high: Tuple[datetime, float]
    swing_low: Tuple[datetime, float]
    current_price: float
    retracement_direction: str  # 'bullish', 'bearish'
    fibonacci_levels: List[FibonacciLevel]
    confluence_zones: List[FibonacciZone]
    next_support: Optional[float]
    next_resistance: Optional[float]
    current_retracement_level: float  # Current price as % of retracement
    analysis_confidence: float


class FibonacciRetracement:
    """
    Fibonacci Retracement Calculator
    Implements comprehensive Fibonacci retracement analysis
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Standard Fibonacci retracement levels
        self.fibonacci_levels = [
            0.0,     # 0% - Swing high/low
            0.236,   # 23.6%
            0.382,   # 38.2%
            0.500,   # 50% - Not technically Fibonacci but widely used
            0.618,   # 61.8% - Golden ratio
            0.786,   # 78.6%
            1.0      # 100% - Swing low/high
        ]

        # Extended levels for deeper retracements
        self.extended_levels = [
            1.272,   # 127.2%
            1.414,   # 141.4%
            1.618,   # 161.8%
            2.0,     # 200%
            2.618    # 261.8%
        ]

        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

        self.logger.info("Fibonacci Retracement Calculator initialized")

    async def calculate_retracement(
        self,
        symbol: str,
        price_data: List[Dict],
        swing_high: Optional[Tuple[datetime, float]] = None,
        swing_low: Optional[Tuple[datetime, float]] = None,
        include_extended: bool = False
    ) -> RetracementResult:
        """
        Calculate Fibonacci retracement levels

        Args:
            symbol: Trading symbol
            price_data: List of OHLC data with timestamp
            swing_high: Optional swing high (time, price), auto-detected if None
            swing_low: Optional swing low (time, price), auto-detected if None
            include_extended: Include extended Fibonacci levels

        Returns:
            RetracementResult with complete retracement analysis
        """
        start_time = time.perf_counter()

        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Find swing points if not provided
            if swing_high is None or swing_low is None:
                swing_high, swing_low = await self._find_swing_points(df)

            current_price = df.iloc[-1]['close']

            # Determine retracement direction
            retracement_direction = 'bullish' if swing_low[0] > swing_high[0] else 'bearish'

            # Calculate Fibonacci levels
            levels_to_use = self.fibonacci_levels
            if include_extended:
                levels_to_use = self.fibonacci_levels + self.extended_levels

            fibonacci_levels = await self._calculate_fibonacci_levels(
                swing_high, swing_low, current_price, levels_to_use, df
            )

            # Find confluence zones
            confluence_zones = await self._find_confluence_zones(fibonacci_levels)

            # Find next support/resistance
            next_support, next_resistance = await self._find_next_levels(fibonacci_levels, current_price)

            # Calculate current retracement level
            current_retracement = await self._calculate_current_retracement(
                swing_high, swing_low, current_price
            )

            # Calculate analysis confidence
            confidence = await self._calculate_analysis_confidence(fibonacci_levels, df)

            result = RetracementResult(
                symbol=symbol,
                timestamp=datetime.now(),
                swing_high=swing_high,
                swing_low=swing_low,
                current_price=current_price,
                retracement_direction=retracement_direction,
                fibonacci_levels=fibonacci_levels,
                confluence_zones=confluence_zones,
                next_support=next_support,
                next_resistance=next_resistance,
                current_retracement_level=current_retracement,
                analysis_confidence=confidence
            )

            # Update performance metrics
            calculation_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time

            self.logger.debug(f"Fibonacci retracement calculated for {symbol} in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracement for {symbol}: {e}")
            raise

    def calculate(self, data: Union[Dict, List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Standard calculate method for BaseIndicator compatibility

        Args:
            data: Market data in dict format with high, low, close arrays
                 or list of OHLC dictionaries, or pandas DataFrame

        Returns:
            Dict containing Fibonacci retracement levels and analysis
        """
        start_time = time.time()

        try:
            # Convert data to standard format
            if isinstance(data, pd.DataFrame):
                high_values = data['high'].tolist()
                low_values = data['low'].tolist()
                close_values = data['close'].tolist()
            elif isinstance(data, dict):
                high_values = data.get('high', [])
                low_values = data.get('low', [])
                close_values = data.get('close', [])
            else:
                # Assume list of dicts
                high_values = [d.get('high', 0) for d in data]
                low_values = [d.get('low', 0) for d in data]
                close_values = [d.get('close', 0) for d in data]

            if not high_values or not low_values:
                return {"error": "Insufficient data for Fibonacci calculation"}

            # Find swing high and low
            swing_high = max(high_values)
            swing_low = min(low_values)
            current_price = close_values[-1] if close_values else swing_high

            # Calculate range
            price_range = swing_high - swing_low

            if price_range == 0:
                return {"error": "No price range available for Fibonacci calculation"}

            # Calculate Fibonacci levels
            fibonacci_levels = {}
            for level in self.fibonacci_levels:
                price_level = swing_high - (price_range * level)
                fibonacci_levels[f"{level * 100:.1f}%"] = round(price_level, 5)

            # Determine trend direction
            trend_direction = "bullish" if current_price > swing_low + (price_range * 0.5) else "bearish"

            # Calculate current retracement percentage
            if trend_direction == "bullish":
                current_retracement = ((swing_high - current_price) / price_range) * 100
            else:
                current_retracement = ((current_price - swing_low) / price_range) * 100

            # Find next support/resistance levels
            next_support = None
            next_resistance = None

            for level_name, level_price in fibonacci_levels.items():
                if trend_direction == "bullish" and level_price < current_price:
                    if next_support is None or level_price > next_support:
                        next_support = level_price
                elif trend_direction == "bearish" and level_price > current_price:
                    if next_resistance is None or level_price < next_resistance:
                        next_resistance = level_price

            # Performance tracking
            calculation_time = time.time() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time

            result = {
                "symbol": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "swing_high": swing_high,
                "swing_low": swing_low,
                "current_price": current_price,
                "price_range": price_range,
                "trend_direction": trend_direction,
                "fibonacci_levels": fibonacci_levels,
                "current_retracement_pct": round(current_retracement, 2),
                "next_support": next_support,
                "next_resistance": next_resistance,
                "calculation_time_ms": round(calculation_time * 1000, 2),
                "total_calculations": self.calculation_count
            }

            self.logger.info(f"Fibonacci retracement calculated successfully in {calculation_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracement: {e}")
            return {"error": str(e)}

    async def _find_swing_points(self, df: pd.DataFrame) -> Tuple[Tuple[datetime, float], Tuple[datetime, float]]:
        """Find significant swing high and low points"""

        # Look for swing points in recent data
        lookback = min(50, len(df))
        recent_df = df.tail(lookback)

        # Find highest high and lowest low
        max_idx = recent_df['high'].idxmax()
        min_idx = recent_df['low'].idxmin()

        swing_high = (recent_df.loc[max_idx, 'timestamp'], recent_df.loc[max_idx, 'high'])
        swing_low = (recent_df.loc[min_idx, 'timestamp'], recent_df.loc[min_idx, 'low'])

        return swing_high, swing_low

    async def _calculate_fibonacci_levels(
        self,
        swing_high: Tuple[datetime, float],
        swing_low: Tuple[datetime, float],
        current_price: float,
        levels: List[float],
        df: pd.DataFrame
    ) -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement levels"""

        high_price = swing_high[1]
        low_price = swing_low[1]
        price_range = high_price - low_price

        fibonacci_levels = []

        for level_pct in levels:
            # Calculate price level
            if swing_high[0] > swing_low[0]:  # Bullish retracement
                price_level = high_price - (price_range * level_pct)
            else:  # Bearish retracement
                price_level = low_price + (price_range * level_pct)

            # Determine level type
            if price_level > current_price:
                level_type = 'resistance'
            elif price_level < current_price:
                level_type = 'support'
            else:
                level_type = 'neutral'

            # Calculate level strength and touches
            strength = await self._calculate_level_strength(price_level, df, level_pct)
            touch_count = await self._count_level_touches(price_level, df)

            # Calculate distance from current price
            distance = abs(price_level - current_price)

            fib_level = FibonacciLevel(
                level_percentage=level_pct,
                price_level=price_level,
                level_type=level_type,
                strength=strength,
                touch_count=touch_count,
                last_touch=None,  # Would need detailed analysis
                distance_from_current=distance
            )

            fibonacci_levels.append(fib_level)

        # Sort by distance from current price
        fibonacci_levels.sort(key=lambda x: x.distance_from_current)

        return fibonacci_levels

    async def _calculate_level_strength(
        self,
        price_level: float,
        df: pd.DataFrame,
        level_percentage: float
    ) -> float:
        """Calculate the strength of a Fibonacci level"""

        # Key Fibonacci levels have higher base strength
        key_levels = [0.382, 0.500, 0.618]
        base_strength = 0.9 if level_percentage in key_levels else 0.7

        # Calculate how often price respects this level
        respect_count = 0
        total_tests = 0
        tolerance = price_level * 0.002  # 0.2% tolerance

        for _, row in df.iterrows():
            # Check if price tested this level
            if (row['low'] <= price_level + tolerance and
                row['high'] >= price_level - tolerance):
                total_tests += 1

                # Check if price respected the level (bounced)
                if (row['close'] > price_level and row['low'] <= price_level) or \
                   (row['close'] < price_level and row['high'] >= price_level):
                    respect_count += 1

        if total_tests == 0:
            return base_strength

        respect_ratio = respect_count / total_tests
        return min(1.0, base_strength * (0.5 + respect_ratio))

    async def _count_level_touches(
        self,
        price_level: float,
        df: pd.DataFrame,
        tolerance: float = 0.002
    ) -> int:
        """Count how many times price touched a Fibonacci level"""

        touches = 0
        tolerance_amount = price_level * tolerance

        for _, row in df.iterrows():
            # Check if high or low touched the level
            if (abs(row['high'] - price_level) <= tolerance_amount or
                abs(row['low'] - price_level) <= tolerance_amount):
                touches += 1

        return touches

    async def _find_confluence_zones(
        self,
        fibonacci_levels: List[FibonacciLevel]
    ) -> List[FibonacciZone]:
        """Find confluence zones where multiple Fibonacci levels cluster"""

        zones = []

        # Group levels that are close together
        for i, level1 in enumerate(fibonacci_levels):
            close_levels = [level1]

            for j, level2 in enumerate(fibonacci_levels[i+1:], i+1):
                # Check if levels are within 1% of each other
                price_diff = abs(level1.price_level - level2.price_level)
                avg_price = (level1.price_level + level2.price_level) / 2

                if price_diff / avg_price <= 0.01:  # Within 1%
                    close_levels.append(level2)

            # Create zone if multiple levels cluster
            if len(close_levels) >= 2:
                prices = [level.price_level for level in close_levels]
                zone_start = min(prices)
                zone_end = max(prices)
                zone_center = sum(prices) / len(prices)

                # Calculate zone strength
                zone_strength = sum(level.strength for level in close_levels) / len(close_levels)

                # Determine zone type
                zone_type = 'support_zone' if close_levels[0].level_type == 'support' else 'resistance_zone'

                zone = FibonacciZone(
                    zone_start=zone_start,
                    zone_end=zone_end,
                    zone_center=zone_center,
                    zone_strength=zone_strength,
                    contributing_levels=close_levels,
                    zone_type=zone_type
                )

                zones.append(zone)

        # Sort by strength
        zones.sort(key=lambda x: x.zone_strength, reverse=True)

        return zones[:5]  # Keep top 5 zones

    async def _find_next_levels(
        self,
        fibonacci_levels: List[FibonacciLevel],
        current_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Find next support and resistance levels"""

        next_support = None
        next_resistance = None

        for level in fibonacci_levels:
            if level.level_type == 'support' and level.price_level < current_price:
                if next_support is None or level.price_level > next_support:
                    next_support = level.price_level
            elif level.level_type == 'resistance' and level.price_level > current_price:
                if next_resistance is None or level.price_level < next_resistance:
                    next_resistance = level.price_level

        return next_support, next_resistance

    async def _calculate_current_retracement(
        self,
        swing_high: Tuple[datetime, float],
        swing_low: Tuple[datetime, float],
        current_price: float
    ) -> float:
        """Calculate current price as percentage of retracement"""

        high_price = swing_high[1]
        low_price = swing_low[1]
        price_range = high_price - low_price

        if price_range == 0:
            return 0.0

        if swing_high[0] > swing_low[0]:  # Bullish retracement
            retracement = (high_price - current_price) / price_range
        else:  # Bearish retracement
            retracement = (current_price - low_price) / price_range

        return max(0.0, min(1.0, retracement))

    async def _calculate_analysis_confidence(
        self,
        fibonacci_levels: List[FibonacciLevel],
        df: pd.DataFrame
    ) -> float:
        """Calculate overall confidence in Fibonacci analysis"""

        if not fibonacci_levels:
            return 0.0

        # Base confidence on level strengths
        avg_strength = sum(level.strength for level in fibonacci_levels) / len(fibonacci_levels)

        # Data quality factor
        data_quality = min(1.0, len(df) / 100)

        # Touch count factor
        total_touches = sum(level.touch_count for level in fibonacci_levels)
        touch_factor = min(1.0, total_touches / 20)

        confidence = (avg_strength * 0.6) + (data_quality * 0.3) + (touch_factor * 0.1)

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
