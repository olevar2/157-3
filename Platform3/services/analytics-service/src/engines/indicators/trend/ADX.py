"""
Average Directional Index (ADX) Indicator Module

This module provides comprehensive ADX analysis for forex trading, including trend strength
measurement, directional movement analysis, and trading signal generation.
Optimized for scalping (M1-M5), day trading (M15-H1), and swing trading (H4) strategies.

Features:
- ADX trend strength calculation
- +DI and -DI directional indicators
- DX (Directional Movement Index) calculation
- Trend strength classification
- Crossover signal detection
- Multi-timeframe analysis
- Session-based trend analysis

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendStrength(Enum):
    """Trend strength classification based on ADX values"""
    NO_TREND = "no_trend"          # ADX < 20
    WEAK_TREND = "weak_trend"      # 20 <= ADX < 25
    MODERATE_TREND = "moderate_trend"  # 25 <= ADX < 40
    STRONG_TREND = "strong_trend"  # 40 <= ADX < 60
    VERY_STRONG_TREND = "very_strong_trend"  # ADX >= 60

class TrendDirection(Enum):
    """Trend direction based on DI comparison"""
    BULLISH = "bullish"    # +DI > -DI
    BEARISH = "bearish"    # -DI > +DI
    NEUTRAL = "neutral"    # +DI ≈ -DI

@dataclass
class ADXResults:
    """Container for ADX analysis results"""
    adx: float
    plus_di: float
    minus_di: float
    dx: float
    trend_strength: TrendStrength
    trend_direction: TrendDirection
    di_spread: float
    adx_slope: float
    signal_strength: float
    crossover_signal: Optional[str]
    support_resistance_level: float

class ADXIndicator:
    """
    Average Directional Index (ADX) Indicator

    Provides comprehensive trend analysis using ADX, +DI, and -DI indicators
    for forex trading strategy development and signal generation.
    """

    def __init__(self,
                 period: int = 14,
                 smoothing_period: int = 14,
                 signal_threshold: float = 25.0):
        """
        Initialize ADX Indicator

        Args:
            period: Period for DI calculation
            smoothing_period: Period for ADX smoothing
            signal_threshold: Minimum ADX value for trend signals
        """
        self.period = period
        self.smoothing_period = smoothing_period
        self.signal_threshold = signal_threshold

        # Internal state for calculations
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.close_history: List[float] = []
        self.adx_history: List[float] = []
        self.plus_di_history: List[float] = []
        self.minus_di_history: List[float] = []

        logger.info(f"ADX Indicator initialized: period={period}, smoothing={smoothing_period}")

    def _calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        """
        Calculate True Range for current period

        Args:
            high: Current period high
            low: Current period low
            prev_close: Previous period close

        Returns:
            True Range value
        """
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        return max(tr1, tr2, tr3)

    def _calculate_directional_movement(self,
                                      current_high: float,
                                      current_low: float,
                                      prev_high: float,
                                      prev_low: float) -> Tuple[float, float]:
        """
        Calculate Directional Movement (+DM and -DM)

        Args:
            current_high: Current period high
            current_low: Current period low
            prev_high: Previous period high
            prev_low: Previous period low

        Returns:
            Tuple of (+DM, -DM)
        """
        up_move = current_high - prev_high
        down_move = prev_low - current_low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        return plus_dm, minus_dm

    def _smooth_values(self, values: List[float], period: int) -> float:
        """
        Apply Wilder's smoothing to values

        Args:
            values: List of values to smooth
            period: Smoothing period

        Returns:
            Smoothed value
        """
        if len(values) < period:
            return np.mean(values) if values else 0.0

        # Wilder's smoothing: (previous_smooth * (period - 1) + current_value) / period
        if len(values) == period:
            return np.mean(values)
        else:
            previous_smooth = self._smooth_values(values[:-1], period)
            current_value = values[-1]
            return (previous_smooth * (period - 1) + current_value) / period

    def _calculate_di(self, dm_values: List[float], tr_values: List[float]) -> float:
        """
        Calculate Directional Indicator (DI)

        Args:
            dm_values: Directional Movement values
            tr_values: True Range values

        Returns:
            DI value (0-100)
        """
        if len(dm_values) < self.period or len(tr_values) < self.period:
            return 0.0

        smoothed_dm = self._smooth_values(dm_values[-self.period:], self.period)
        smoothed_tr = self._smooth_values(tr_values[-self.period:], self.period)

        if smoothed_tr == 0:
            return 0.0

        di = (smoothed_dm / smoothed_tr) * 100
        return di

    def _calculate_dx(self, plus_di: float, minus_di: float) -> float:
        """
        Calculate Directional Movement Index (DX)

        Args:
            plus_di: +DI value
            minus_di: -DI value

        Returns:
            DX value (0-100)
        """
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        di_diff = abs(plus_di - minus_di)
        dx = (di_diff / di_sum) * 100

        return dx

    def _classify_trend_strength(self, adx: float) -> TrendStrength:
        """
        Classify trend strength based on ADX value

        Args:
            adx: ADX value

        Returns:
            TrendStrength classification
        """
        if adx < 20:
            return TrendStrength.NO_TREND
        elif adx < 25:
            return TrendStrength.WEAK_TREND
        elif adx < 40:
            return TrendStrength.MODERATE_TREND
        elif adx < 60:
            return TrendStrength.STRONG_TREND
        else:
            return TrendStrength.VERY_STRONG_TREND

    def _determine_trend_direction(self, plus_di: float, minus_di: float) -> TrendDirection:
        """
        Determine trend direction based on DI comparison

        Args:
            plus_di: +DI value
            minus_di: -DI value

        Returns:
            TrendDirection classification
        """
        di_diff = abs(plus_di - minus_di)

        if di_diff < 2.0:  # Very close values
            return TrendDirection.NEUTRAL
        elif plus_di > minus_di:
            return TrendDirection.BULLISH
        else:
            return TrendDirection.BEARISH

    def _calculate_adx_slope(self) -> float:
        """
        Calculate ADX slope (rate of change)

        Returns:
            ADX slope value
        """
        if len(self.adx_history) < 3:
            return 0.0

        # Calculate slope over last 3 periods
        recent_adx = self.adx_history[-3:]
        x = np.arange(len(recent_adx))
        slope = np.polyfit(x, recent_adx, 1)[0]

        return slope

    def _detect_crossover_signals(self) -> Optional[str]:
        """
        Detect DI crossover signals

        Returns:
            Crossover signal type or None
        """
        if len(self.plus_di_history) < 2 or len(self.minus_di_history) < 2:
            return None

        current_plus_di = self.plus_di_history[-1]
        current_minus_di = self.minus_di_history[-1]
        prev_plus_di = self.plus_di_history[-2]
        prev_minus_di = self.minus_di_history[-2]

        # Bullish crossover: +DI crosses above -DI
        if (prev_plus_di <= prev_minus_di and
            current_plus_di > current_minus_di and
            self.adx_history[-1] > self.signal_threshold):
            return "bullish_crossover"

        # Bearish crossover: -DI crosses above +DI
        elif (prev_minus_di <= prev_plus_di and
              current_minus_di > current_plus_di and
              self.adx_history[-1] > self.signal_threshold):
            return "bearish_crossover"

        return None

    def _calculate_signal_strength(self, adx: float, di_spread: float) -> float:
        """
        Calculate overall signal strength

        Args:
            adx: ADX value
            di_spread: Spread between +DI and -DI

        Returns:
            Signal strength (0-1)
        """
        # Normalize ADX (0-100 to 0-1)
        adx_normalized = min(1.0, adx / 100.0)

        # Normalize DI spread (0-100 to 0-1)
        di_spread_normalized = min(1.0, di_spread / 100.0)

        # Combine ADX and DI spread for signal strength
        signal_strength = (adx_normalized * 0.7) + (di_spread_normalized * 0.3)

        return signal_strength

    def update(self, high: float, low: float, close: float) -> ADXResults:
        """
        Update ADX calculation with new price data

        Args:
            high: Current period high
            low: Current period low
            close: Current period close

        Returns:
            ADXResults object with current analysis
        """
        try:
            # Add to history
            self.high_history.append(high)
            self.low_history.append(low)
            self.close_history.append(close)

            # Need at least 2 periods for calculations
            if len(self.close_history) < 2:
                return self._create_empty_results()

            # Calculate True Range and Directional Movement
            tr_values = []
            plus_dm_values = []
            minus_dm_values = []

            for i in range(1, len(self.close_history)):
                # True Range
                tr = self._calculate_true_range(
                    self.high_history[i],
                    self.low_history[i],
                    self.close_history[i-1]
                )
                tr_values.append(tr)

                # Directional Movement
                plus_dm, minus_dm = self._calculate_directional_movement(
                    self.high_history[i],
                    self.low_history[i],
                    self.high_history[i-1],
                    self.low_history[i-1]
                )
                plus_dm_values.append(plus_dm)
                minus_dm_values.append(minus_dm)

            # Calculate +DI and -DI
            plus_di = self._calculate_di(plus_dm_values, tr_values)
            minus_di = self._calculate_di(minus_dm_values, tr_values)

            # Calculate DX
            dx = self._calculate_dx(plus_di, minus_di)

            # Update DI history
            self.plus_di_history.append(plus_di)
            self.minus_di_history.append(minus_di)

            # Calculate ADX (smoothed DX)
            dx_values = []
            for i in range(len(self.plus_di_history)):
                if i < len(self.minus_di_history):
                    dx_val = self._calculate_dx(self.plus_di_history[i], self.minus_di_history[i])
                    dx_values.append(dx_val)

            adx = self._smooth_values(dx_values, self.smoothing_period)
            self.adx_history.append(adx)

            # Maintain history size
            max_history = max(100, self.period * 3)
            for history in [self.high_history, self.low_history, self.close_history,
                           self.adx_history, self.plus_di_history, self.minus_di_history]:
                if len(history) > max_history:
                    history[:] = history[-max_history:]

            # Analyze results
            trend_strength = self._classify_trend_strength(adx)
            trend_direction = self._determine_trend_direction(plus_di, minus_di)
            di_spread = abs(plus_di - minus_di)
            adx_slope = self._calculate_adx_slope()
            crossover_signal = self._detect_crossover_signals()
            signal_strength = self._calculate_signal_strength(adx, di_spread)

            # Calculate support/resistance level (simplified)
            support_resistance_level = close  # Could be enhanced with more sophisticated calculation

            result = ADXResults(
                adx=adx,
                plus_di=plus_di,
                minus_di=minus_di,
                dx=dx,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                di_spread=di_spread,
                adx_slope=adx_slope,
                signal_strength=signal_strength,
                crossover_signal=crossover_signal,
                support_resistance_level=support_resistance_level
            )

            logger.debug(f"ADX updated: ADX={adx:.2f}, +DI={plus_di:.2f}, -DI={minus_di:.2f}, "
                        f"Strength={trend_strength.value}, Direction={trend_direction.value}")

            return result

        except Exception as e:
            logger.error(f"Error updating ADX: {str(e)}")
            raise

    def _create_empty_results(self) -> ADXResults:
        """Create empty ADX results for insufficient data"""
        return ADXResults(
            adx=0.0,
            plus_di=0.0,
            minus_di=0.0,
            dx=0.0,
            trend_strength=TrendStrength.NO_TREND,
            trend_direction=TrendDirection.NEUTRAL,
            di_spread=0.0,
            adx_slope=0.0,
            signal_strength=0.0,
            crossover_signal=None,
            support_resistance_level=0.0
        )

    def analyze_batch(self,
                     highs: List[float],
                     lows: List[float],
                     closes: List[float]) -> List[ADXResults]:
        """
        Analyze batch of price data

        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices

        Returns:
            List of ADXResults for each period
        """
        results = []

        for high, low, close in zip(highs, lows, closes):
            result = self.update(high, low, close)
            results.append(result)

        return results

    def get_trading_signals(self, results: ADXResults) -> Dict[str, Union[str, float, bool]]:
        """
        Generate trading signals based on ADX analysis

        Args:
            results: ADXResults from analysis

        Returns:
            Dictionary with trading signals and recommendations
        """
        signals = {
            "trend_strength": results.trend_strength.value,
            "trend_direction": results.trend_direction.value,
            "signal_strength": results.signal_strength,
            "adx_value": results.adx,
            "crossover_signal": results.crossover_signal,
            "trending_market": results.adx >= self.signal_threshold
        }

        # Trading recommendations based on ADX analysis
        if results.trend_strength == TrendStrength.VERY_STRONG_TREND:
            signals["trading_action"] = "strong_trend_following"
            signals["position_size"] = "large"
            signals["timeframe_preference"] = "H1-H4"
        elif results.trend_strength == TrendStrength.STRONG_TREND:
            signals["trading_action"] = "trend_following"
            signals["position_size"] = "normal"
            signals["timeframe_preference"] = "M15-H1"
        elif results.trend_strength == TrendStrength.MODERATE_TREND:
            signals["trading_action"] = "cautious_trend_following"
            signals["position_size"] = "small"
            signals["timeframe_preference"] = "M15-H1"
        elif results.trend_strength == TrendStrength.WEAK_TREND:
            signals["trading_action"] = "range_trading"
            signals["position_size"] = "small"
            signals["timeframe_preference"] = "M5-M15"
        else:  # NO_TREND
            signals["trading_action"] = "avoid_trending_strategies"
            signals["position_size"] = "minimal"
            signals["timeframe_preference"] = "M1-M5"

        # Direction-based signals
        if results.trend_direction == TrendDirection.BULLISH and results.adx >= self.signal_threshold:
            signals["bias"] = "long"
            signals["entry_strategy"] = "buy_pullbacks"
        elif results.trend_direction == TrendDirection.BEARISH and results.adx >= self.signal_threshold:
            signals["bias"] = "short"
            signals["entry_strategy"] = "sell_rallies"
        else:
            signals["bias"] = "neutral"
            signals["entry_strategy"] = "range_bound"

        # Crossover signals
        if results.crossover_signal == "bullish_crossover":
            signals["immediate_signal"] = "buy"
            signals["signal_confidence"] = "high" if results.signal_strength > 0.7 else "medium"
        elif results.crossover_signal == "bearish_crossover":
            signals["immediate_signal"] = "sell"
            signals["signal_confidence"] = "high" if results.signal_strength > 0.7 else "medium"
        else:
            signals["immediate_signal"] = "hold"
            signals["signal_confidence"] = "low"

        # ADX slope analysis
        if results.adx_slope > 1.0:
            signals["trend_momentum"] = "increasing"
        elif results.adx_slope < -1.0:
            signals["trend_momentum"] = "decreasing"
        else:
            signals["trend_momentum"] = "stable"

        return signals

# Create alias for expected class name
ADX = ADXIndicator
