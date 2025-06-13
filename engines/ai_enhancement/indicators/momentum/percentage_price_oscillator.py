"""
Percentage Price Oscillator (PPO) Indicator
Trading-grade implementation for Platform3

The Percentage Price Oscillator is a momentum oscillator that measures the percentage
difference between two moving averages. It's similar to MACD but uses percentages
instead of absolute values, making it easier to compare across different price levels.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Try absolute import first, fall back to relative
try:
    from engines.ai_enhancement.indicators.base_indicator import (
        IndicatorMetadata,
        IndicatorValidationError,
        StandardIndicatorInterface,
    )
except ImportError:
    from ..base_indicator import (
        IndicatorMetadata,
        IndicatorValidationError,
        StandardIndicatorInterface,
    )

logger = logging.getLogger(__name__)


class PercentagePriceOscillatorIndicator(StandardIndicatorInterface):
    """
    Percentage Price Oscillator (PPO) - Momentum Oscillator

    Formula:
    PPO = ((Fast EMA - Slow EMA) / Slow EMA) × 100
    Signal Line = EMA of PPO (typically 9 periods)
    Histogram = PPO - Signal Line

    The PPO oscillates around zero:
    - Positive values indicate upward momentum
    - Negative values indicate downward momentum
    - Crossovers of PPO and signal line generate trading signals
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs,
    ):
        """
        Initialize Percentage Price Oscillator

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        """
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            **kwargs,
        )

    def validate_parameters(self) -> bool:
        """Validate PPO parameters"""
        fast_period = self.parameters.get("fast_period", 12)
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)

        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise IndicatorValidationError("All periods must be positive integers")

        if fast_period >= slow_period:
            raise IndicatorValidationError("Fast period must be less than slow period")

        return True

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Percentage Price Oscillator

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with PPO, Signal, and Histogram columns
        """
        self.validate_parameters()

        if len(data) < self.parameters["slow_period"]:
            raise IndicatorValidationError(
                f"Insufficient data: need at least {self.parameters['slow_period']} periods"
            )

        close_prices = data["close"].copy()

        # Calculate EMAs
        fast_ema = close_prices.ewm(span=self.parameters["fast_period"]).mean()
        slow_ema = close_prices.ewm(span=self.parameters["slow_period"]).mean()

        # Calculate PPO
        ppo = ((fast_ema - slow_ema) / slow_ema) * 100

        # Calculate Signal Line
        signal = ppo.ewm(span=self.parameters["signal_period"]).mean()

        # Calculate Histogram
        histogram = ppo - signal

        result = pd.DataFrame(
            {
                "ppo": ppo,
                "signal": signal,
                "histogram": histogram,
            },
            index=data.index,
        )

        return result

    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Percentage Price Oscillator",
            category=self.CATEGORY,
            description="Momentum oscillator measuring percentage difference between EMAs",
            parameters=self.parameters,
            input_requirements=["close"],
            output_type="multi_series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=max(
                self.parameters.get("slow_period", 26),
                self.parameters.get("signal_period", 9),
            ),
        )

    def interpret_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Interpret PPO signals

        Args:
            data: DataFrame with PPO calculations

        Returns:
            List of signal dictionaries
        """
        signals = []
        ppo = data["ppo"]
        signal_line = data["signal"]
        histogram = data["histogram"]

        for i in range(1, len(data)):
            current_ppo = ppo.iloc[i]
            prev_ppo = ppo.iloc[i - 1]
            current_signal = signal_line.iloc[i]
            prev_signal = signal_line.iloc[i - 1]
            current_histogram = histogram.iloc[i]

            # PPO crosses above signal line (bullish)
            if prev_ppo <= prev_signal and current_ppo > current_signal:
                signals.append(
                    {
                        "timestamp": data.index[i],
                        "type": "bullish_crossover",
                        "strength": abs(current_histogram),
                        "description": "PPO crossed above signal line",
                    }
                )

            # PPO crosses below signal line (bearish)
            elif prev_ppo >= prev_signal and current_ppo < current_signal:
                signals.append(
                    {
                        "timestamp": data.index[i],
                        "type": "bearish_crossover",
                        "strength": abs(current_histogram),
                        "description": "PPO crossed below signal line",
                    }
                )

            # PPO crosses above zero (bullish momentum)
            elif prev_ppo <= 0 and current_ppo > 0:
                signals.append(
                    {
                        "timestamp": data.index[i],
                        "type": "bullish_momentum",
                        "strength": current_ppo,
                        "description": "PPO crossed above zero line",
                    }
                )

            # PPO crosses below zero (bearish momentum)
            elif prev_ppo >= 0 and current_ppo < 0:
                signals.append(
                    {
                        "timestamp": data.index[i],
                        "type": "bearish_momentum",
                        "strength": abs(current_ppo),
                        "description": "PPO crossed below zero line",
                    }
                )

        return signals

    def optimize_parameters(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Optimize PPO parameters

        Args:
            data: Historical price data
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary with optimized parameters
        """
        best_params = self.parameters.copy()
        best_score = 0

        # Parameter ranges for optimization
        fast_range = range(8, 16)
        slow_range = range(20, 35)
        signal_range = range(6, 12)

        for fast in fast_range:
            for slow in slow_range:
                for signal in signal_range:
                    if fast >= slow:
                        continue

                    # Test parameters
                    test_params = {
                        "fast_period": fast,
                        "slow_period": slow,
                        "signal_period": signal,
                    }

                    try:
                        temp_indicator = PercentagePriceOscillatorIndicator(
                            **test_params
                        )
                        result = temp_indicator.calculate(data)
                        signals = temp_indicator.interpret_signals(result)

                        # Simple scoring based on signal frequency and strength
                        score = (
                            len(signals)
                            * np.mean([s.get("strength", 0) for s in signals])
                            if signals
                            else 0
                        )

                        if score > best_score:
                            best_score = score
                            best_params = test_params

                    except Exception:
                        continue

        return best_params

    def get_trading_rules(self) -> Dict[str, Any]:
        """
        Get PPO trading rules

        Returns:
            Dictionary with trading rules and guidelines
        """
        return {
            "entry_signals": {
                "bullish": [
                    "PPO crosses above signal line",
                    "PPO crosses above zero line",
                    "Histogram turns positive and increasing",
                ],
                "bearish": [
                    "PPO crosses below signal line",
                    "PPO crosses below zero line",
                    "Histogram turns negative and decreasing",
                ],
            },
            "exit_signals": {
                "profit_taking": [
                    "PPO divergence with price",
                    "Extreme PPO values (>±3%)",
                    "Histogram momentum weakening",
                ],
                "stop_loss": [
                    "PPO crosses in opposite direction",
                    "Strong adverse momentum",
                ],
            },
            "risk_management": {
                "position_sizing": "Based on histogram strength",
                "max_risk": "2% per trade",
                "confirmation": "Use with trend indicators",
            },
        }


def create_sample_data():
    """Create sample data for testing"""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices + np.random.randn(100) * 0.1,
            "high": prices + abs(np.random.randn(100) * 0.5),
            "low": prices - abs(np.random.randn(100) * 0.5),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )


if __name__ == "__main__":
    # Example usage
    indicator = PercentagePriceOscillatorIndicator()

    # Create sample data
    sample_data = create_sample_data()

    # Calculate PPO
    result = indicator.calculate(sample_data)

    print("Percentage Price Oscillator Results:")
    print(result.tail())

    # Get signals
    signals = indicator.interpret_signals(result)
    print(f"\nGenerated {len(signals)} signals")

    # Get metadata
    metadata = indicator.get_metadata()
    print(f"\nIndicator: {metadata.name} v{metadata.version}")
