#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gann Angles Calculator - W.D. Gann's Geometric Analysis Tool
Platform3 Phase 3 - Enhanced Gann Analysis

The Gann Angles Calculator implements W.D. Gann's famous 1x1, 2x1, 1x2 angle lines
and related geometric price/time relationships for market analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from engines.indicator_base import (
    IndicatorBase,
    IndicatorResult,
    IndicatorType,
    TimeFrame,
)


@dataclass
class GannAngle:
    """Represents a single Gann angle line"""

    ratio: str  # '1x1', '2x1', '1x2', etc.
    angle_degrees: float
    slope: float
    start_point: Tuple[float, float]  # (time, price)
    current_value: float
    is_support: bool
    strength: float  # 0-100


@dataclass
class GannCalculationResult:
    """Results from Gann angles calculation"""

    angles: List[GannAngle]
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float
    primary_angle: Optional[GannAngle]


class GannAnglesCalculator(IndicatorBase):
    """
    Gann Angles Calculator for geometric price/time analysis

    Features:
    - Classic Gann angles (1x1, 2x1, 1x2, 4x1, 1x4, 8x1, 1x8)    - Dynamic angle calculation from swing points
    - Support/resistance level identification
    - Trend strength assessment
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        swing_lookback: int = 20,
        min_swing_strength: float = 0.5,
        angle_tolerance: float = 2.0,
        **kwargs
    ):
        """
        Initialize Gann Angles Calculator

        Args:
            config: Optional configuration dictionary
            swing_lookback: Periods to look back for swing points
            min_swing_strength: Minimum strength for valid swing points
            angle_tolerance: Tolerance in degrees for angle validation
        """
        super().__init__(
            config=config,
            name="GannAnglesCalculator",
            indicator_type=IndicatorType.GANN,
            timeframe=TimeFrame.H1,
            lookback_periods=swing_lookback,
            parameters={
                "swing_lookback": swing_lookback,
                "min_swing_strength": min_swing_strength,
                "angle_tolerance": angle_tolerance,
            },
            **kwargs
        )
        self.swing_lookback = swing_lookback
        self.min_swing_strength = min_swing_strength
        self.angle_tolerance = angle_tolerance

        # Standard Gann angle ratios
        self.gann_ratios = {
            "8x1": 82.5,  # 8 price units to 1 time unit
            "4x1": 75.0,  # 4 price units to 1 time unit
            "3x1": 71.25,  # 3 price units to 1 time unit
            "2x1": 63.75,  # 2 price units to 1 time unit
            "1x1": 45.0,  # 1 price unit to 1 time unit (main trend line)
            "1x2": 26.25,  # 1 price unit to 2 time units
            "1x3": 18.75,  # 1 price unit to 3 time units
            "1x4": 15.0,  # 1 price unit to 4 time units
            "1x8": 7.5,  # 1 price unit to 8 time units
        }

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate Gann angles from price data

        Args:
            data: DataFrame with OHLCV data

        Returns:
            IndicatorResult containing calculated Gann angles
        """
        try:
            if len(data) < self.swing_lookback:
                return IndicatorResult(
                    timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                    indicator_name="GannAnglesCalculator",
                    indicator_type=IndicatorType.GANN,
                    timeframe=TimeFrame.D1,
                    value=[],
                    raw_data={"error": "Insufficient data for Gann angle calculation"},
                )

            # Find significant swing points
            swing_highs, swing_lows = self._find_swing_points(data)

            # Calculate Gann angles from swing points
            gann_angles = []

            # Calculate angles from most recent significant swing low (uptrend angles)
            if swing_lows:
                recent_low = swing_lows[-1]
                up_angles = self._calculate_angles_from_point(
                    data, recent_low, direction="up"
                )
                gann_angles.extend(up_angles)

            # Calculate angles from most recent significant swing high (downtrend angles)
            if swing_highs:
                recent_high = swing_highs[-1]
                down_angles = self._calculate_angles_from_point(
                    data, recent_high, direction="down"
                )
                gann_angles.extend(down_angles)

            # Evaluate current angle relationships
            current_analysis = self._analyze_current_position(data, gann_angles)

            return IndicatorResult(
                timestamp=data.index[-1],
                indicator_name="GannAnglesCalculator",
                indicator_type=IndicatorType.GANN,
                timeframe=TimeFrame.D1,
                value=gann_angles,
                raw_data={
                    "total_angles": len(gann_angles),
                    "swing_highs": len(swing_highs),
                    "swing_lows": len(swing_lows),
                    "current_analysis": current_analysis,
                },
            )

        except Exception as e:
            return IndicatorResult(
                timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                indicator_name="GannAnglesCalculator",
                indicator_type=IndicatorType.GANN,
                timeframe=TimeFrame.D1,
                value=[],
                raw_data={"error": str(e)},
            )

    def _find_swing_points(
        self, data: pd.DataFrame
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find significant swing highs and lows"""
        highs = []
        lows = []

        high_prices = data["high"].values
        low_prices = data["low"].values

        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            # Check for swing high
            if self._is_swing_high(high_prices, i):
                highs.append((i, high_prices[i]))

            # Check for swing low
            if self._is_swing_low(low_prices, i):
                lows.append((i, low_prices[i]))

        return highs, lows

    def _is_swing_high(self, prices: np.ndarray, index: int) -> bool:
        """Check if index represents a swing high"""
        current_price = prices[index]

        # Check if current price is higher than surrounding prices
        left_max = np.max(prices[index - self.swing_lookback : index])
        right_max = np.max(prices[index + 1 : index + self.swing_lookback + 1])

        return current_price > left_max and current_price > right_max

    def _is_swing_low(self, prices: np.ndarray, index: int) -> bool:
        """Check if index represents a swing low"""
        current_price = prices[index]

        # Check if current price is lower than surrounding prices
        left_min = np.min(prices[index - self.swing_lookback : index])
        right_min = np.min(prices[index + 1 : index + self.swing_lookback + 1])

        return current_price < left_min and current_price < right_min

    def _calculate_angles_from_point(
        self, data: pd.DataFrame, start_point: Tuple[int, float], direction: str
    ) -> List[GannAngle]:
        """Calculate Gann angles from a given starting point"""
        angles = []
        start_index, start_price = start_point
        current_index = len(data) - 1
        time_diff = current_index - start_index

        if time_diff <= 0:
            return angles

        # Calculate unit scale (price movement per time unit)
        price_range = data["high"].max() - data["low"].min()
        time_range = len(data)
        unit_scale = price_range / time_range

        for ratio_name, angle_degrees in self.gann_ratios.items():
            # Calculate angle slope based on ratio
            if "x" in ratio_name:
                price_units, time_units = map(float, ratio_name.split("x"))
                slope = (price_units / time_units) * unit_scale

                if direction == "down":
                    slope = -slope

                # Calculate current value of this angle line
                current_value = start_price + (slope * time_diff)

                # Determine if this is acting as support or resistance
                current_price = data["close"].iloc[-1]
                is_support = (direction == "up" and current_price > current_value) or (
                    direction == "down" and current_price < current_value
                )

                # Calculate strength based on how close price is to the angle
                price_distance = abs(current_price - current_value)
                strength = max(0, 100 - (price_distance / (price_range * 0.1)) * 100)

                angle = GannAngle(
                    ratio=ratio_name,
                    angle_degrees=(
                        angle_degrees if direction == "up" else 180 - angle_degrees
                    ),
                    slope=slope,
                    start_point=(start_index, start_price),
                    current_value=current_value,
                    is_support=is_support,
                    strength=strength,
                )

                angles.append(angle)

        return angles

    def _analyze_current_position(
        self, data: pd.DataFrame, gann_angles: List[GannAngle]
    ) -> Dict[str, Any]:
        """Analyze current price position relative to Gann angles"""
        if not gann_angles:
            return {"analysis": "No angles available"}

        current_price = data["close"].iloc[-1]

        # Find nearest support and resistance angles
        support_angles = [
            a for a in gann_angles if a.is_support and a.current_value < current_price
        ]
        resistance_angles = [
            a
            for a in gann_angles
            if not a.is_support and a.current_value > current_price
        ]

        nearest_support = (
            max(support_angles, key=lambda x: x.current_value)
            if support_angles
            else None
        )
        nearest_resistance = (
            min(resistance_angles, key=lambda x: x.current_value)
            if resistance_angles
            else None
        )

        # Find strongest angle influence
        strongest_angle = (
            max(gann_angles, key=lambda x: x.strength) if gann_angles else None
        )

        analysis = {
            "current_price": current_price,
            "nearest_support": (
                {
                    "ratio": nearest_support.ratio,
                    "level": nearest_support.current_value,
                    "strength": nearest_support.strength,
                }
                if nearest_support
                else None
            ),
            "nearest_resistance": (
                {
                    "ratio": nearest_resistance.ratio,
                    "level": nearest_resistance.current_value,
                    "strength": nearest_resistance.strength,
                }
                if nearest_resistance
                else None
            ),
            "strongest_influence": (
                {
                    "ratio": strongest_angle.ratio,
                    "level": strongest_angle.current_value,
                    "strength": strongest_angle.strength,
                    "type": "support" if strongest_angle.is_support else "resistance",
                }
                if strongest_angle
                else None
            ),
            "total_active_angles": len(gann_angles),
        }

        return analysis
