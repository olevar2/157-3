"""
Aroon Indicator

The Aroon indicator measures the time since the highest high and lowest low
within a given period, helping to identify trend strength and potential reversals.

Formula:
Aroon Up = ((period - periods since highest high) / period) * 100
Aroon Down = ((period - periods since lowest low) / period) * 100
Aroon Oscillator = Aroon Up - Aroon Down

Interpretation:
- Aroon Up > 70 and Aroon Down < 30: Strong uptrend
- Aroon Down > 70 and Aroon Up < 30: Strong downtrend
- Aroon Up and Down both < 50: Consolidation phase
- Aroon Up crosses above Aroon Down: Potential bullish signal
- Aroon Down crosses above Aroon Up: Potential bearish signal

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
from indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
    IndicatorMetadata,
)


class AroonIndicator(StandardIndicatorInterface):
    """
    Aroon Indicator for trend strength and direction analysis
    
    The Aroon indicator helps identify trend changes and measure trend strength
    by calculating how recently the highest highs and lowest lows occurred.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        signal_threshold: float = 70.0,
        **kwargs,
    ):
        """
        Initialize Aroon indicator

        Args:
            period: Period for Aroon calculation (default: 20)
            signal_threshold: Threshold for strong trend signals (default: 70.0)
        """
        super().__init__(
            period=period,
            signal_threshold=signal_threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Aroon indicator

        Args:
            data: DataFrame with 'high' and 'low' columns

        Returns:
            pd.DataFrame: DataFrame with columns 'aroon_up', 'aroon_down', 'aroon_oscillator'
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "Aroon indicator requires DataFrame with 'high' and 'low' columns"
            )

        period = self.parameters.get("period", 20)
        
        high_series = data["high"]
        low_series = data["low"]
        
        # Initialize result arrays
        aroon_up = pd.Series(index=data.index, dtype=float)
        aroon_down = pd.Series(index=data.index, dtype=float)
        
        # Calculate Aroon for each position where we have enough data
        for i in range(period - 1, len(data)):
            # Get the window of high and low prices
            high_window = high_series.iloc[i - period + 1:i + 1]
            low_window = low_series.iloc[i - period + 1:i + 1]
            
            # Find positions of highest high and lowest low within the window
            highest_high_pos = high_window.idxmax()
            lowest_low_pos = low_window.idxmin()
            
            # Calculate periods since highest high and lowest low
            periods_since_high = i - high_window.index.get_loc(highest_high_pos)
            periods_since_low = i - low_window.index.get_loc(lowest_low_pos)
            
            # Calculate Aroon Up and Aroon Down
            aroon_up.iloc[i] = ((period - periods_since_high) / period) * 100
            aroon_down.iloc[i] = ((period - periods_since_low) / period) * 100

        # Calculate Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down
        
        # Create result DataFrame
        result = pd.DataFrame({
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }, index=data.index)
        
        # Store calculation details for analysis
        self._last_calculation = {
            "aroon_up": aroon_up,
            "aroon_down": aroon_down,
            "aroon_oscillator": aroon_oscillator,
            "period": period,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate Aroon parameters"""
        period = self.parameters.get("period", 20)
        signal_threshold = self.parameters.get("signal_threshold", 70.0)

        if not isinstance(period, int) or period < 2:
            raise IndicatorValidationError(
                f"period must be integer >= 2, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(signal_threshold, (int, float)) or signal_threshold < 0 or signal_threshold > 100:
            raise IndicatorValidationError(
                f"signal_threshold must be between 0 and 100, got {signal_threshold}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return Aroon metadata"""
        return IndicatorMetadata(
            name="Aroon",
            category=self.CATEGORY,
            description="Aroon Indicator - Measures trend strength and identifies potential reversals",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """Aroon requires high and low prices"""
        return ["high", "low"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Aroon calculation"""
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "signal_threshold" not in self.parameters:
            self.parameters["signal_threshold"] = 70.0

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def signal_threshold(self) -> float:
        """Signal threshold for backward compatibility"""
        return self.parameters.get("signal_threshold", 70.0)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required"""
        return self.parameters.get("period", 20)

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "Aroon",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }

    def get_signals(self, aroon_data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Aroon values

        Args:
            aroon_data: DataFrame with aroon_up and aroon_down columns

        Returns:
            pd.Series: Trading signals ("bullish", "bearish", "neutral")
        """
        threshold = self.signal_threshold
        signals = pd.Series(index=aroon_data.index, dtype=str)
        
        for i in range(len(aroon_data)):
            aroon_up = aroon_data['aroon_up'].iloc[i]
            aroon_down = aroon_data['aroon_down'].iloc[i]
            
            if pd.isna(aroon_up) or pd.isna(aroon_down):
                signals.iloc[i] = "neutral"
            elif aroon_up > threshold and aroon_down < (100 - threshold):
                signals.iloc[i] = "bullish"
            elif aroon_down > threshold and aroon_up < (100 - threshold):
                signals.iloc[i] = "bearish"
            else:
                signals.iloc[i] = "neutral"
        
        return signals

    def detect_crossovers(self, aroon_data: pd.DataFrame) -> pd.Series:
        """
        Detect Aroon crossover signals

        Args:
            aroon_data: DataFrame with aroon_up and aroon_down columns

        Returns:
            pd.Series: Crossover signals ("bullish_crossover", "bearish_crossover", "none")
        """
        crossovers = pd.Series(index=aroon_data.index, dtype=str)
        crossovers[:] = "none"
        
        aroon_up = aroon_data['aroon_up']
        aroon_down = aroon_data['aroon_down']
        
        # Detect crossovers
        up_cross_down = (aroon_up > aroon_down) & (aroon_up.shift(1) <= aroon_down.shift(1))
        down_cross_up = (aroon_down > aroon_up) & (aroon_down.shift(1) <= aroon_up.shift(1))
        
        crossovers[up_cross_down] = "bullish_crossover"
        crossovers[down_cross_up] = "bearish_crossover"
        
        return crossovers


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return AroonIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic price data with trend changes
    close_prices = [base_price]
    for i in range(n_points - 1):
        # Add trend and volatility
        trend = np.sin(i / 30) * 0.5  # Cyclical trend
        change = np.random.randn() * 0.8 + trend
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame({
        "close": close_prices,
        "high": [c + abs(np.random.randn() * 0.8) for c in close_prices],
        "low": [c - abs(np.random.randn() * 0.8) for c in close_prices],
    })

    # Calculate Aroon
    aroon = AroonIndicator(period=14)
    result = aroon.calculate(data)

    # Generate signals
    signals = aroon.get_signals(result)
    crossovers = aroon.detect_crossovers(result)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # Aroon chart
    ax2.plot(result['aroon_up'].values, label="Aroon Up", color="green", linewidth=2)
    ax2.plot(result['aroon_down'].values, label="Aroon Down", color="red", linewidth=2)
    ax2.plot(result['aroon_oscillator'].values, label="Aroon Oscillator", color="purple", alpha=0.7)
    ax2.axhline(y=70, color="green", linestyle="--", alpha=0.5, label="Strong Trend Threshold")
    ax2.axhline(y=30, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="Neutral Line")
    ax2.set_title("Aroon Indicator")
    ax2.set_ylim(-100, 100)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print("Aroon calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Parameters: {aroon.parameters}")
    
    # Show latest values
    latest_idx = result.dropna().index[-1]
    print(f"\nLatest Aroon values:")
    print(f"Aroon Up: {result.loc[latest_idx, 'aroon_up']:.2f}")
    print(f"Aroon Down: {result.loc[latest_idx, 'aroon_down']:.2f}")
    print(f"Aroon Oscillator: {result.loc[latest_idx, 'aroon_oscillator']:.2f}")
    print(f"Signal: {signals.loc[latest_idx]}")
    
    # Count signals
    signal_counts = signals.value_counts()
    print(f"\nSignal distribution:")
    for signal, count in signal_counts.items():
        print(f"{signal}: {count}")
    
    # Count crossovers
    crossover_counts = crossovers.value_counts()
    print(f"\nCrossover distribution:")
    for crossover, count in crossover_counts.items():
        print(f"{crossover}: {count}")