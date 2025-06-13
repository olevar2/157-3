"""
Relative Strength Index (RSI) - Real Implementation
Momentum oscillator that measures the speed and change of price movements
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from ..base_indicator import IndicatorMetadata
    from ..base_indicator import StandardIndicatorInterface as BaseIndicator
except ImportError:
    # Fallback if base_indicator is not available
    from dataclasses import dataclass
    from typing import Any, List

    @dataclass
    class IndicatorMetadata:
        name: str
        category: str
        description: str
        parameters: Dict[str, Any]
        input_requirements: List[str]
        output_type: str
        version: str = "1.0.0"
        author: str = "Platform3"
        trading_grade: bool = True
        performance_tier: str = "standard"
        min_data_points: int = 1
        max_lookback_period: int = 1000

    class BaseIndicator:
        def __init__(self, **kwargs):
            self.parameters = kwargs
            self.logger = logging.getLogger(__name__)


class RelativeStrengthIndexIndicator(BaseIndicator):
    """
    Relative Strength Index (RSI) - Real Implementation

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """

    def __init__(self, period=14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.logger = logging.getLogger(__name__)

    def calculate(self, data) -> Optional[Dict]:
        """Calculate Relative Strength Index"""
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

            if len(prices) < self.period + 1:
                return None

            # Calculate price changes
            price_changes = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i - 1]
                price_changes.append(change)

            # Separate gains and losses
            gains = []
            losses = []
            for change in price_changes:
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                elif change < 0:
                    gains.append(0)
                    losses.append(abs(change))
                else:
                    gains.append(0)
                    losses.append(0)

            if len(gains) < self.period:
                return None

            # Calculate RSI using Wilder's smoothing method
            rsi_values = []

            # First RSI calculation (simple average)
            initial_avg_gain = np.mean(gains[: self.period])
            initial_avg_loss = np.mean(losses[: self.period])

            if initial_avg_loss == 0:
                rsi = 100
            else:
                rs = initial_avg_gain / initial_avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)
            avg_gain = initial_avg_gain
            avg_loss = initial_avg_loss

            # Subsequent RSI calculations (exponential smoothing)
            for i in range(self.period, len(gains)):
                avg_gain = ((avg_gain * (self.period - 1)) + gains[i]) / self.period
                avg_loss = ((avg_loss * (self.period - 1)) + losses[i]) / self.period

                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                rsi_values.append(rsi)

            if not rsi_values:
                return None

            current_rsi = rsi_values[-1]

            # Generate signals based on RSI levels
            overbought = current_rsi > 70
            oversold = current_rsi < 30

            if current_rsi > 70:
                signal = "bearish"  # Overbought
                trend = "overbought"
            elif current_rsi < 30:
                signal = "bullish"  # Oversold
                trend = "oversold"
            elif current_rsi > 50:
                signal = "bullish"
                trend = "bullish"
            elif current_rsi < 50:
                signal = "bearish"
                trend = "bearish"
            else:
                signal = "neutral"
                trend = "sideways"

            # Calculate signal strength based on distance from center line (50)
            strength = abs(current_rsi - 50) * 2  # Scale to 0-100

            # Calculate confidence based on RSI level extremes
            if current_rsi > 80 or current_rsi < 20:
                confidence = 90
            elif current_rsi > 70 or current_rsi < 30:
                confidence = 75
            elif current_rsi > 60 or current_rsi < 40:
                confidence = 60
            else:
                confidence = 45  # Check for divergence patterns (simplified)
            divergence_signal = "none"
            if len(rsi_values) >= 5:
                # More complex divergence analysis would require price comparison
                divergence_signal = "none"  # Placeholder

            # Check for RSI crossovers
            crossover_signal = "none"
            if len(rsi_values) >= 2:
                prev_rsi = rsi_values[-2]
                if prev_rsi <= 30 and current_rsi > 30:
                    crossover_signal = "bullish_oversold_exit"
                elif prev_rsi >= 70 and current_rsi < 70:
                    crossover_signal = "bearish_overbought_exit"
                elif prev_rsi <= 50 and current_rsi > 50:
                    crossover_signal = "bullish_centerline_cross"
                elif prev_rsi >= 50 and current_rsi < 50:
                    crossover_signal = "bearish_centerline_cross"

            return {
                "value": float(current_rsi),
                "signal": signal,
                "trend": trend,
                "strength": float(strength),
                "confidence": confidence,
                "overbought": bool(overbought),
                "oversold": bool(oversold),
                "divergence": divergence_signal,
                "crossover": crossover_signal,
                "rsi_series": [float(x) for x in rsi_values[-10:]],  # Last 10 values
                "period": self.period,
                "levels": {
                    "overbought": 70,
                    "oversold": 30,
                    "centerline": 50,
                    "extreme_overbought": 80,
                    "extreme_oversold": 20,
                },
                "current_zone": self._get_rsi_zone(current_rsi),
            }

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    def _get_rsi_zone(self, rsi_value):
        """Determine which RSI zone the current value is in"""
        if rsi_value >= 80:
            return "extreme_overbought"
        elif rsi_value >= 70:
            return "overbought"
        elif rsi_value >= 50:
            return "bullish"
        elif rsi_value >= 30:
            return "bearish"
        elif rsi_value >= 20:
            return "oversold"
        else:
            return "extreme_oversold"

    def get_signals(self, data) -> Dict:
        """Get trading signals from RSI"""
        result = self.calculate(data)
        if not result:
            return {"action": "hold", "reason": "insufficient_data"}

        rsi = result["value"]
        crossover = result["crossover"]

        if crossover == "bullish_oversold_exit":
            return {
                "action": "buy",
                "reason": "oversold_recovery",
                "confidence": result["confidence"],
                "rsi_value": rsi,
            }
        elif crossover == "bearish_overbought_exit":
            return {
                "action": "sell",
                "reason": "overbought_correction",
                "confidence": result["confidence"],
                "rsi_value": rsi,
            }
        elif rsi < 20:
            return {
                "action": "buy",
                "reason": "extreme_oversold",
                "confidence": 90,
                "rsi_value": rsi,
            }
        elif rsi > 80:
            return {
                "action": "sell",
                "reason": "extreme_overbought",
                "confidence": 90,
                "rsi_value": rsi,
            }
        elif rsi < 30:
            return {
                "action": "buy_signal",
                "reason": "oversold_condition",
                "confidence": result["confidence"],
                "rsi_value": rsi,
            }
        elif rsi > 70:
            return {
                "action": "sell_signal",
                "reason": "overbought_condition",
                "confidence": result["confidence"],
                "rsi_value": rsi,
            }
        else:
            return {
                "action": "hold",
                "reason": "neutral_zone",
                "confidence": result["confidence"],
                "rsi_value": rsi,
            }

    def get_metadata(self) -> IndicatorMetadata:
        """Get metadata for the RSI indicator"""
        return IndicatorMetadata(
            name="Relative Strength Index",
            category="momentum",
            description="Momentum oscillator that measures the speed and magnitude of recent price changes",
            parameters={
                "period": {
                    "description": "Number of periods for RSI calculation",
                    "default": 14,
                    "range": [2, 50],
                    "type": "int",
                }
            },
            input_requirements=["close"],
            output_type="single_series",
            version="1.0.0",
            author="Platform3",
            trading_grade=True,
            performance_tier="standard",
            min_data_points=self.period + 1,
            max_lookback_period=1000,
        )

    def validate_parameters(self) -> bool:
        """Validate RSI indicator parameters"""
        try:
            if not isinstance(self.period, int):
                self.logger.error("RSI period must be an integer")
                return False

            if self.period < 2:
                self.logger.error("RSI period must be at least 2")
                return False

            if self.period > 50:
                self.logger.warning("RSI period > 50 is unusually high")

            return True

        except Exception as e:
            self.logger.error(f"Error validating RSI parameters: {e}")
            return False
