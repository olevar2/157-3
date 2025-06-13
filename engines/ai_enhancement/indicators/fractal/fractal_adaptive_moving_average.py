"""
Platform3 Fractal Adaptive Moving Average (FRAMA) Indicator
Real implementation of FRAMA that adapts smoothing based on fractal dimension.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class FractalAdaptiveResult:
    """Result structure for Fractal Adaptive Moving Average"""

    frama_value: float
    fractal_dimension: float
    smoothing_factor: float
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    adaptation_speed: float
    signal_strength: float


class FractalAdaptiveMovingAverage:
    """
    Real Fractal Adaptive Moving Average (FRAMA) Implementation

    Adaptive moving average that adjusts smoothing based on fractal dimension.
    Provides faster response in trending markets and slower in choppy markets.
    """

    def __init__(self, period=20, fast_limit=0.67, slow_limit=0.03, **kwargs):
        """
        Initialize FRAMA indicator.

        Args:
            period: Period for fractal dimension calculation
            fast_limit: Fast smoothing limit (trending markets)
            slow_limit: Slow smoothing limit (choppy markets)
        """
        self.period = period
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

        # Internal state
        self.frama_values = []

    def calculate_fractal_dimension(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate fractal dimension for FRAMA."""
        try:
            n = len(highs)
            if n < 2:
                return 1.5

            # Calculate the length of the price path
            total_length = 0
            for i in range(1, n):
                high_diff = abs(highs[i] - highs[i - 1])
                low_diff = abs(lows[i] - lows[i - 1])
                total_length += max(high_diff, low_diff)

            # Calculate straight line distance
            straight_distance = abs(highs[-1] - highs[0]) + abs(lows[-1] - lows[0])

            if straight_distance == 0:
                return 1.5

            # Fractal dimension
            dimension = np.log(total_length / straight_distance) / np.log(n)

            return np.clip(dimension, 1.0, 2.0)

        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.5

    def calculate(
        self, data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Optional[FractalAdaptiveResult]:
        """Calculate FRAMA for given data."""
        try:
            # Parse input data
            if isinstance(data, pd.DataFrame):
                if "high" in data.columns and "low" in data.columns:
                    highs = data["high"].values
                    lows = data["low"].values
                    closes = data["close"].values if "close" in data.columns else highs
                else:
                    prices = data.iloc[:, 0].values
                    highs = lows = closes = prices
            elif isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 3:
                    highs = data[:, 1]
                    lows = data[:, 2]
                    closes = data[:, 0]
                else:
                    prices = data.flatten()
                    highs = lows = closes = prices
            elif isinstance(data, dict):
                highs = np.array(data.get("high", data.get("close", [])))
                lows = np.array(data.get("low", data.get("close", [])))
                closes = np.array(data.get("close", []))
            elif isinstance(data, (list, tuple)):
                prices = np.array(data)
                highs = lows = closes = prices
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None

            if len(closes) < self.period:
                return None

            # Calculate fractal dimension for recent period
            recent_highs = highs[-self.period :]
            recent_lows = lows[-self.period :]

            fractal_dim = self.calculate_fractal_dimension(recent_highs, recent_lows)

            # Calculate smoothing factor
            # Higher dimension (choppy) -> slower smoothing
            # Lower dimension (trending) -> faster smoothing
            alpha = (fractal_dim - 1.0) * (
                self.slow_limit - self.fast_limit
            ) + self.fast_limit
            alpha = np.clip(alpha, self.slow_limit, self.fast_limit)

            # Calculate FRAMA
            current_price = closes[-1]
            if len(self.frama_values) == 0:
                frama_value = current_price
            else:
                prev_frama = self.frama_values[-1]
                frama_value = alpha * current_price + (1 - alpha) * prev_frama

            self.frama_values.append(frama_value)

            # Keep only recent values
            if len(self.frama_values) > self.period * 2:
                self.frama_values = self.frama_values[-self.period :]

            # Determine trend direction
            if len(self.frama_values) >= 2:
                if frama_value > self.frama_values[-2]:
                    trend_direction = "bullish"
                elif frama_value < self.frama_values[-2]:
                    trend_direction = "bearish"
                else:
                    trend_direction = "neutral"
            else:
                trend_direction = "neutral"

            # Adaptation speed (how fast the indicator is adapting)
            adaptation_speed = alpha / self.fast_limit

            # Signal strength based on price vs FRAMA distance
            price_distance = abs(current_price - frama_value) / current_price
            signal_strength = min(1.0, price_distance * 10)

            return FractalAdaptiveResult(
                frama_value=float(frama_value),
                fractal_dimension=float(fractal_dim),
                smoothing_factor=float(alpha),
                trend_direction=trend_direction,
                adaptation_speed=float(adaptation_speed),
                signal_strength=float(signal_strength),
            )

        except Exception as e:
            self.logger.error(f"Error calculating FRAMA: {e}")
            return None

    def get_confluence_analysis(
        self, price: float, frama_result: FractalAdaptiveResult
    ) -> Dict[str, Any]:
        """
        Analyze confluence between price and FRAMA.

        Args:
            price: Current price
            frama_result: FRAMA calculation result

        Returns:
            Dictionary containing confluence analysis
        """
        try:
            confluence = {
                "trend_confirmation": False,
                "adaptation_level": "medium",
                "signal_quality": "weak",
                "recommendation": "neutral",
            }

            # Trend confirmation
            price_above_frama = price > frama_result.frama_value
            if price_above_frama and frama_result.trend_direction == "bullish":
                confluence["trend_confirmation"] = True
                confluence["recommendation"] = "bullish"
            elif not price_above_frama and frama_result.trend_direction == "bearish":
                confluence["trend_confirmation"] = True
                confluence["recommendation"] = "bearish"

            # Adaptation level
            if frama_result.adaptation_speed > 0.7:
                confluence["adaptation_level"] = "high"
            elif frama_result.adaptation_speed < 0.3:
                confluence["adaptation_level"] = "low"

            # Signal quality
            if frama_result.signal_strength > 0.6:
                confluence["signal_quality"] = "strong"
            elif frama_result.signal_strength > 0.3:
                confluence["signal_quality"] = "medium"

            return confluence

        except Exception as e:
            self.logger.error(f"Error in confluence analysis: {e}")
            return {"error": str(e)}

    def generate_signals(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on FRAMA analysis.

        Args:
            data: Price data for analysis

        Returns:
            List of signal dictionaries
        """
        signals = []

        try:
            # Calculate FRAMA
            result = self.calculate(data)
            if not result:
                return signals

            # Extract current price
            if isinstance(data, pd.DataFrame):
                current_price = (
                    data["close"].iloc[-1]
                    if "close" in data.columns
                    else data.iloc[-1, 0]
                )
            else:
                current_price = float(
                    data[-1] if isinstance(data, (list, np.ndarray)) else data
                )

            # Get confluence analysis
            confluence = self.get_confluence_analysis(current_price, result)

            # Generate signal based on confluence
            if confluence["trend_confirmation"] and confluence["signal_quality"] in [
                "medium",
                "strong",
            ]:
                signal = {
                    "type": "frama_signal",
                    "direction": confluence["recommendation"],
                    "strength": confluence["signal_quality"],
                    "confidence": result.signal_strength,
                    "frama_value": result.frama_value,
                    "fractal_dimension": result.fractal_dimension,
                    "adaptation_speed": result.adaptation_speed,
                    "price": current_price,
                }
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
