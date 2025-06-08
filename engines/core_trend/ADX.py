# -*- coding: utf-8 -*-
"""
Average Directional Index (ADX) Indicator
Advanced trend strength analysis for forex trading

This module implements the ADX indicator with directional movement indicators (DI+ and DI-)
for comprehensive trend analysis. Optimized for scalping, day trading, and swing trading
strategies with adaptive parameters and signal generation.

Features:
- ADX trend strength calculation
- Directional Movement Index (DI+ and DI-)
- Adaptive period adjustment
- Trend strength classification
- Signal generation and filtering
- Multi-timeframe analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendStrength(Enum):
    """ADX trend strength levels"""
    NO_TREND = "no_trend"          # ADX < 20
    WEAK_TREND = "weak_trend"      # 20 <= ADX < 25
    MODERATE_TREND = "moderate_trend"  # 25 <= ADX < 40
    STRONG_TREND = "strong_trend"  # 40 <= ADX < 60
    VERY_STRONG_TREND = "very_strong_trend"  # ADX >= 60

class TrendDirection(Enum):
    """Trend direction based on DI+ and DI-"""
    BULLISH = "bullish"    # DI+ > DI-
    BEARISH = "bearish"    # DI- > DI+
    NEUTRAL = "neutral"    # DI+ ~= DI-

class ADXSignalType(Enum):
    """Types of ADX signals"""
    TREND_STRENGTHENING = "trend_strengthening"
    TREND_WEAKENING = "trend_weakening"
    TREND_REVERSAL = "trend_reversal"
    DIRECTIONAL_CHANGE = "directional_change"
    NO_SIGNAL = "no_signal"

@dataclass
class ADXSignal:
    """ADX signal information"""
    signal_type: ADXSignalType
    direction: TrendDirection
    strength: TrendStrength
    confidence: float
    timestamp: datetime
    adx_value: float
    di_plus: float
    di_minus: float

@dataclass
class ADXResult:
    """Complete ADX analysis result"""
    adx: float
    di_plus: float
    di_minus: float
    trend_strength: TrendStrength
    trend_direction: TrendDirection
    signal: ADXSignal
    dx_values: List[float]
    timestamp: datetime

class ADX:
    """
    Average Directional Index (ADX) Indicator

    Calculates trend strength and direction using ADX, DI+, and DI- indicators.
    Provides comprehensive trend analysis for forex trading strategies.
    """

    def __init__(self, period: int = 14, adaptive: bool = True, smoothing_period: int = 14):
        """
        Initialize ADX indicator

        Args:
            period: Period for DI calculation (default: 14)
            adaptive: Enable adaptive period adjustment (default: True)
            smoothing_period: Period for ADX smoothing (default: 14)
        """
        self.period = period
        self.adaptive = adaptive
        self.smoothing_period = smoothing_period

        # Internal state
        self.tr_values = []
        self.dm_plus_values = []
        self.dm_minus_values = []
        self.dx_values = []
        self.adx_values = []

        # Performance tracking
        self.calculation_count = 0
        self.signal_count = 0

        logger.info(f"ADX initialized with period={period}, adaptive={adaptive}")

    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> ADXResult:
        """
        Calculate ADX indicator

        Args:
            high: High prices array
            low: Low prices array
            close: Close prices array

        Returns:
            ADXResult with complete analysis
        """
        try:
            if len(high) < self.period + self.smoothing_period:
                raise ValueError(f"Insufficient data: need at least {self.period + self.smoothing_period} periods")

            # Calculate True Range (TR)
            tr = self._calculate_true_range(high, low, close)

            # Calculate Directional Movement (DM+ and DM-)
            dm_plus, dm_minus = self._calculate_directional_movement(high, low)

            # Calculate smoothed TR and DM values
            atr = self._smooth_values(tr, self.period)
            smoothed_dm_plus = self._smooth_values(dm_plus, self.period)
            smoothed_dm_minus = self._smooth_values(dm_minus, self.period)

            # Calculate Directional Indicators (DI+ and DI-)
            di_plus = 100 * (smoothed_dm_plus / atr)
            di_minus = 100 * (smoothed_dm_minus / atr)

            # Calculate Directional Index (DX)
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            dx = np.nan_to_num(dx, nan=0.0)

            # Calculate ADX (smoothed DX)
            adx = self._smooth_values(dx, self.smoothing_period)

            # Get current values
            current_adx = adx[-1] if len(adx) > 0 else 0.0
            current_di_plus = di_plus[-1] if len(di_plus) > 0 else 0.0
            current_di_minus = di_minus[-1] if len(di_minus) > 0 else 0.0

            # Determine trend strength and direction
            trend_strength = self._classify_trend_strength(current_adx)
            trend_direction = self._determine_trend_direction(current_di_plus, current_di_minus)

            # Generate signal
            signal = self._generate_signal(adx, di_plus, di_minus, trend_strength, trend_direction)

            # Update performance tracking
            self.calculation_count += 1
            if signal.signal_type != ADXSignalType.NO_SIGNAL:
                self.signal_count += 1

            return ADXResult(
                adx=current_adx,
                di_plus=current_di_plus,
                di_minus=current_di_minus,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                signal=signal,
                dx_values=dx.tolist()[-20:],  # Keep last 20 values
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            raise

    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        return np.maximum(tr1, np.maximum(tr2, tr3))

    def _calculate_directional_movement(self, high: np.ndarray, low: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Directional Movement (DM+ and DM-)"""
        high_diff = np.diff(high, prepend=high[0])
        low_diff = -np.diff(low, prepend=low[0])

        dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        return dm_plus, dm_minus

    def _smooth_values(self, values: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing to values"""
        if len(values) < period:
            return np.array([])

        smoothed = np.zeros(len(values))
        smoothed[period-1] = np.mean(values[:period])

        for i in range(period, len(values)):
            smoothed[i] = (smoothed[i-1] * (period - 1) + values[i]) / period

        return smoothed[period-1:]

    def _classify_trend_strength(self, adx_value: float) -> TrendStrength:
        """Classify trend strength based on ADX value"""
        if adx_value < 20:
            return TrendStrength.NO_TREND
        elif adx_value < 25:
            return TrendStrength.WEAK_TREND
        elif adx_value < 40:
            return TrendStrength.MODERATE_TREND
        elif adx_value < 60:
            return TrendStrength.STRONG_TREND
        else:
            return TrendStrength.VERY_STRONG_TREND

    def _determine_trend_direction(self, di_plus: float, di_minus: float) -> TrendDirection:
        """Determine trend direction based on DI+ and DI-"""
        diff = abs(di_plus - di_minus)

        if diff < 2:  # Very close values
            return TrendDirection.NEUTRAL
        elif di_plus > di_minus:
            return TrendDirection.BULLISH
        else:
            return TrendDirection.BEARISH

    def _generate_signal(
        self,
        adx_values: np.ndarray,
        di_plus_values: np.ndarray,
        di_minus_values: np.ndarray,
        trend_strength: TrendStrength,
        trend_direction: TrendDirection
    ) -> ADXSignal:
        """Generate ADX trading signal"""
        try:
            if len(adx_values) < 3:
                return ADXSignal(
                    signal_type=ADXSignalType.NO_SIGNAL,
                    direction=trend_direction,
                    strength=trend_strength,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    adx_value=adx_values[-1] if len(adx_values) > 0 else 0.0,
                    di_plus=di_plus_values[-1] if len(di_plus_values) > 0 else 0.0,
                    di_minus=di_minus_values[-1] if len(di_minus_values) > 0 else 0.0
                )

            current_adx = adx_values[-1]
            prev_adx = adx_values[-2]
            prev2_adx = adx_values[-3]

            current_di_plus = di_plus_values[-1]
            current_di_minus = di_minus_values[-1]
            prev_di_plus = di_plus_values[-2]
            prev_di_minus = di_minus_values[-2]

            signal_type = ADXSignalType.NO_SIGNAL
            confidence = 0.0

            # Check for trend strengthening
            if current_adx > prev_adx > prev2_adx and current_adx > 25:
                signal_type = ADXSignalType.TREND_STRENGTHENING
                confidence = min(0.9, current_adx / 60)

            # Check for trend weakening
            elif current_adx < prev_adx < prev2_adx and prev_adx > 25:
                signal_type = ADXSignalType.TREND_WEAKENING
                confidence = min(0.8, (60 - current_adx) / 60)

            # Check for directional change
            elif ((prev_di_plus > prev_di_minus and current_di_plus < current_di_minus) or
                  (prev_di_plus < prev_di_minus and current_di_plus > current_di_minus)):
                if abs(current_di_plus - current_di_minus) > 5:  # Significant difference
                    signal_type = ADXSignalType.DIRECTIONAL_CHANGE
                    confidence = min(0.8, abs(current_di_plus - current_di_minus) / 30)

            # Check for trend reversal
            elif (current_adx < 20 and prev_adx > 30) or (current_adx > 30 and prev_adx < 20):
                signal_type = ADXSignalType.TREND_REVERSAL
                confidence = min(0.7, abs(current_adx - prev_adx) / 40)

            return ADXSignal(
                signal_type=signal_type,
                direction=trend_direction,
                strength=trend_strength,
                confidence=confidence,
                timestamp=datetime.now(),
                adx_value=current_adx,
                di_plus=current_di_plus,
                di_minus=current_di_minus
            )

        except Exception as e:
            logger.error(f"Error generating ADX signal: {e}")
            return ADXSignal(
                signal_type=ADXSignalType.NO_SIGNAL,
                direction=TrendDirection.NEUTRAL,
                strength=TrendStrength.NO_TREND,
                confidence=0.0,
                timestamp=datetime.now(),
                adx_value=0.0,
                di_plus=0.0,
                di_minus=0.0
            )

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'calculation_count': self.calculation_count,
            'signal_count': self.signal_count,
            'signal_rate': self.signal_count / max(1, self.calculation_count),
            'period': self.period,
            'smoothing_period': self.smoothing_period
        }
