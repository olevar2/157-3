"""
Moving Average Convergence Divergence (MACD) - Real Implementation
A trend-following momentum indicator that shows the relationship between two moving averages
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from ..base_indicator import StandardIndicatorInterface as BaseIndicator
except ImportError:
    # Fallback if base_indicator is not available
    class BaseIndicator:
        def __init__(self, **kwargs):
            pass


class MovingAverageConvergenceDivergenceIndicator(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) - Real Implementation

    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD Line
    Histogram = MACD Line - Signal Line
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.logger = logging.getLogger(__name__)

    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return []

        ema_values = []
        multiplier = 2 / (period + 1)

        # Start with SMA for first value
        sma = np.mean(data[:period])
        ema_values.append(sma)

        # Calculate EMA for remaining values
        for i in range(period, len(data)):
            ema = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return ema_values

    def calculate(self, data) -> Optional[Dict]:
        """Calculate MACD"""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "close" in data.columns:
                    prices = data["close"].values
                else:
                    return None
            elif isinstance(data, dict):
                prices = np.array(data.get("close", []))
            elif isinstance(data, (list, np.ndarray)):
                prices = np.array(data)
            else:
                return None

            if len(prices) < self.slow_period + self.signal_period:
                return None

            # Calculate EMAs
            fast_ema = self._calculate_ema(prices, self.fast_period)
            slow_ema = self._calculate_ema(prices, self.slow_period)

            if not fast_ema or not slow_ema:
                return None

            # Align EMAs (slow EMA starts later)
            ema_start_offset = self.slow_period - self.fast_period
            aligned_fast_ema = fast_ema[ema_start_offset:]
            aligned_slow_ema = slow_ema

            # Calculate MACD line
            min_length = min(len(aligned_fast_ema), len(aligned_slow_ema))
            macd_line = []
            for i in range(min_length):
                macd_value = aligned_fast_ema[i] - aligned_slow_ema[i]
                macd_line.append(macd_value)

            if len(macd_line) < self.signal_period:
                return None

            # Calculate signal line (EMA of MACD line)
            signal_line = self._calculate_ema(macd_line, self.signal_period)

            if not signal_line:
                return None

            # Calculate histogram
            histogram = []
            signal_start_offset = len(macd_line) - len(signal_line)
            for i in range(len(signal_line)):
                hist_value = macd_line[signal_start_offset + i] - signal_line[i]
                histogram.append(hist_value)

            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1]

            # Generate signals
            macd_above_signal = current_macd > current_signal
            macd_above_zero = current_macd > 0
            histogram_positive = current_histogram > 0

            # Determine trend and signal
            if macd_above_signal and macd_above_zero:
                signal = "bullish"
                trend = "bullish"
            elif not macd_above_signal and not macd_above_zero:
                signal = "bearish"
                trend = "bearish"
            elif macd_above_signal:
                signal = "bullish"
                trend = "neutral_bullish"
            elif not macd_above_signal:
                signal = "bearish"
                trend = "neutral_bearish"
            else:
                signal = "neutral"
                trend = "sideways"

            # Calculate signal strength based on histogram magnitude
            if len(histogram) >= 5:
                recent_hist = histogram[-5:]
                avg_magnitude = np.mean(np.abs(recent_hist))
                current_magnitude = abs(current_histogram)

                if avg_magnitude > 0:
                    strength = min(100, (current_magnitude / avg_magnitude) * 50)
                else:
                    strength = 50
            else:
                strength = 50

            # Calculate confidence based on signal consistency
            if len(histogram) >= 3:
                recent_hist = histogram[-3:]
                if all(h > 0 for h in recent_hist) or all(h < 0 for h in recent_hist):
                    confidence = 80
                else:
                    confidence = 50
            else:
                confidence = 60

            # Check for crossovers
            crossover_signal = "none"
            if len(histogram) >= 2:
                prev_histogram = histogram[-2]
                if prev_histogram <= 0 and current_histogram > 0:
                    crossover_signal = "bullish_crossover"
                elif prev_histogram >= 0 and current_histogram < 0:
                    crossover_signal = "bearish_crossover"

            return {
                "macd_line": float(current_macd),
                "signal_line": float(current_signal),
                "histogram": float(current_histogram),
                "signal": signal,
                "trend": trend,
                "strength": float(strength),
                "confidence": confidence,
                "macd_above_signal": bool(macd_above_signal),
                "macd_above_zero": bool(macd_above_zero),
                "histogram_positive": bool(histogram_positive),
                "crossover": crossover_signal,
                "macd_series": [float(x) for x in macd_line[-10:]],
                "signal_series": [float(x) for x in signal_line[-10:]],
                "histogram_series": [float(x) for x in histogram[-10:]],
                "parameters": {
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                    "signal_period": self.signal_period,
                },
            }

        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return None

    def get_signals(self, data) -> Dict:
        """Get trading signals from MACD"""
        result = self.calculate(data)
        if not result:
            return {"action": "hold", "reason": "insufficient_data"}

        crossover = result["crossover"]
        histogram = result["histogram"]
        macd_above_signal = result["macd_above_signal"]

        if crossover == "bullish_crossover":
            return {
                "action": "buy",
                "reason": "bullish_crossover",
                "confidence": result["confidence"],
                "histogram": histogram,
            }
        elif crossover == "bearish_crossover":
            return {
                "action": "sell",
                "reason": "bearish_crossover",
                "confidence": result["confidence"],
                "histogram": histogram,
            }
        elif macd_above_signal and histogram > 0:
            return {
                "action": "hold_long",
                "reason": "bullish_momentum",
                "confidence": result["confidence"],
                "histogram": histogram,
            }
        elif not macd_above_signal and histogram < 0:
            return {
                "action": "hold_short",
                "reason": "bearish_momentum",
                "confidence": result["confidence"],
                "histogram": histogram,
            }
        else:
            return {
                "action": "hold",
                "reason": "neutral_momentum",
                "confidence": result["confidence"],
                "histogram": histogram,
            }

    def get_metadata(self) -> Dict:
        """Get metadata for the MACD indicator"""
        return {
            "name": "Moving Average Convergence Divergence",
            "description": "Trend-following momentum indicator showing relationship between two moving averages",
            "category": "momentum",
            "subcategory": "trend_following",
            "type": "real_implementation",
            "version": "1.0",
            "minimum_periods": self.slow_period + self.signal_period,
            "maximum_periods": None,
            "data_requirements": ["close"],
            "output_range": "unbounded",
            "interpretation": {
                "macd_line": "EMA(fast) - EMA(slow)",
                "signal_line": "EMA of MACD line",
                "histogram": "MACD line - Signal line",
                "bullish_crossover": "MACD crosses above signal line",
                "bearish_crossover": "MACD crosses below signal line",
            },
            "trading_rules": {
                "buy_signals": ["bullish_crossover", "histogram_increasing"],
                "sell_signals": ["bearish_crossover", "histogram_decreasing"],
                "divergence": "Price vs MACD divergence indicates potential reversal",
            },
            "parameters": {
                "fast_period": {
                    "description": "Fast EMA period",
                    "default": 12,
                    "range": [5, 20],
                    "type": "int",
                },
                "slow_period": {
                    "description": "Slow EMA period",
                    "default": 26,
                    "range": [20, 50],
                    "type": "int",
                },
                "signal_period": {
                    "description": "Signal line EMA period",
                    "default": 9,
                    "range": [5, 15],
                    "type": "int",
                },
            },
        }

    def validate_parameters(self) -> bool:
        """Validate MACD indicator parameters"""
        try:
            if not all(
                isinstance(p, int)
                for p in [self.fast_period, self.slow_period, self.signal_period]
            ):
                self.logger.error("MACD periods must be integers")
                return False

            if (
                self.fast_period <= 0
                or self.slow_period <= 0
                or self.signal_period <= 0
            ):
                self.logger.error("MACD periods must be positive")
                return False

            if self.fast_period >= self.slow_period:
                self.logger.error("Fast period must be less than slow period")
                return False

            if self.signal_period > self.fast_period:
                self.logger.warning("Signal period is greater than fast period")

            return True

        except Exception as e:
            self.logger.error(f"Error validating MACD parameters: {e}")
            return False
