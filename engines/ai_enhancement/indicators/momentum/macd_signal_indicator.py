"""
MACD Signal Indicator

The MACD (Moving Average Convergence Divergence) Signal is an enhanced version of the traditional MACD
that focuses on signal line crossovers and momentum strength analysis for high-precision trading decisions.

The MACD Signal provides:
1. Signal line crossover detection with strength measurement
2. Histogram momentum analysis with acceleration tracking
3. Adaptive signal filtering based on market volatility
4. Real-time momentum strength scoring with confidence levels

Formula:
1. MACD Line = EMA(close, fast_period) - EMA(close, slow_period)
2. Signal Line = EMA(MACD Line, signal_period)
3. Histogram = MACD Line - Signal Line
4. Signal Strength = |Histogram| / ATR(close, volatility_period)

Interpretation:
- Signal crossover above: Bullish momentum with strength measurement
- Signal crossover below: Bearish momentum with strength measurement
- Histogram acceleration: Momentum change rate analysis
- Strength score: Confidence level in signal validity

Author: Platform3 AI Framework
Created: 2025-06-10
Version: 1.0.0
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class MACDSignalIndicator(StandardIndicatorInterface):
    """
    MACD Signal Indicator

    Enhanced MACD implementation with signal strength analysis,
    adaptive filtering, and real-time momentum confidence scoring.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "core_momentum"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        volatility_period: int = 14,
        strength_threshold: float = 1.0,
        **kwargs,
    ):
        """
        Initialize MACD Signal indicator

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            volatility_period: ATR period for strength calculation (default: 14)
            strength_threshold: Minimum strength for valid signals (default: 1.0)
        """
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            volatility_period=volatility_period,
            strength_threshold=strength_threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate MACD Signal indicator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns, or Series of prices

        Returns:
            pd.DataFrame: MACD components with columns:
                - macd: MACD line
                - signal: Signal line
                - histogram: MACD histogram
                - strength: Signal strength score
                - crossover: Crossover signals (+1 bullish, -1 bearish, 0 none)
        """
        # Handle input data
        if isinstance(data, pd.Series):
            close_prices = data
            # Create synthetic high/low for ATR calculation
            high_prices = close_prices
            low_prices = close_prices
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                close_prices = data["close"]
                high_prices = data.get("high", close_prices)
                low_prices = data.get("low", close_prices)
                self.validate_input_data(data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        fast_period = self.parameters.get("fast_period", 12)
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)
        volatility_period = self.parameters.get("volatility_period", 14)
        strength_threshold = self.parameters.get("strength_threshold", 1.0)

        # Calculate EMAs
        ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate Histogram
        histogram = macd_line - signal_line

        # Calculate ATR for strength measurement
        high_low = high_prices - low_prices
        high_close = np.abs(high_prices - close_prices.shift())
        low_close = np.abs(low_prices - close_prices.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=volatility_period, min_periods=1).mean()

        # Calculate signal strength
        strength = np.abs(histogram) / (atr + 1e-10)  # Avoid division by zero

        # Detect crossovers
        crossover = pd.Series(0, index=close_prices.index)
        
        # Bullish crossover: MACD crosses above signal
        bullish_cross = (macd_line > signal_line) & (macd_line.shift() <= signal_line.shift())
        bearish_cross = (macd_line < signal_line) & (macd_line.shift() >= signal_line.shift())
        
        crossover[bullish_cross & (strength >= strength_threshold)] = 1
        crossover[bearish_cross & (strength >= strength_threshold)] = -1

        # Create result DataFrame
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'strength': strength,
            'crossover': crossover
        }, index=close_prices.index)

        # Store calculation details for analysis
        self._last_calculation = {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
            "strength": strength,
            "crossover": crossover,
            "atr": atr,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "volatility_period": volatility_period,
            "strength_threshold": strength_threshold,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate MACD Signal parameters"""
        fast_period = self.parameters.get("fast_period", 12)
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)
        volatility_period = self.parameters.get("volatility_period", 14)
        strength_threshold = self.parameters.get("strength_threshold", 1.0)

        if not isinstance(fast_period, int) or fast_period < 1:
            raise IndicatorValidationError(
                f"fast_period must be positive integer, got {fast_period}"
            )

        if not isinstance(slow_period, int) or slow_period < 1:
            raise IndicatorValidationError(
                f"slow_period must be positive integer, got {slow_period}"
            )

        if not isinstance(signal_period, int) or signal_period < 1:
            raise IndicatorValidationError(
                f"signal_period must be positive integer, got {signal_period}"
            )

        if not isinstance(volatility_period, int) or volatility_period < 1:
            raise IndicatorValidationError(
                f"volatility_period must be positive integer, got {volatility_period}"
            )

        if fast_period >= slow_period:
            raise IndicatorValidationError(
                f"fast_period must be < slow_period, got fast={fast_period}, slow={slow_period}"
            )

        if not isinstance(strength_threshold, (int, float)) or strength_threshold < 0:
            raise IndicatorValidationError(
                f"strength_threshold must be non-negative number, got {strength_threshold}"
            )

        if any(period > 1000 for period in [fast_period, slow_period, signal_period, volatility_period]):
            raise IndicatorValidationError(
                "Period values too large, maximum 1000"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return MACD Signal metadata as dictionary for compatibility"""
        return {
            "name": "MACD_Signal",
            "category": self.CATEGORY,
            "description": "MACD Signal - Enhanced momentum indicator with strength analysis and adaptive filtering",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """MACD Signal requires close prices, optionally high/low for strength calculation"""
        return ["close"]  # High/low optional for enhanced strength calculation

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for MACD Signal calculation"""
        slow_period = self.parameters.get("slow_period", 26)
        signal_period = self.parameters.get("signal_period", 9)
        return max(slow_period + signal_period, 50)  # Extra buffer for stability

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "fast_period" not in self.parameters:
            self.parameters["fast_period"] = 12
        if "slow_period" not in self.parameters:
            self.parameters["slow_period"] = 26
        if "signal_period" not in self.parameters:
            self.parameters["signal_period"] = 9
        if "volatility_period" not in self.parameters:
            self.parameters["volatility_period"] = 14
        if "strength_threshold" not in self.parameters:
            self.parameters["strength_threshold"] = 1.0

    # Property accessors for backward compatibility
    @property
    def fast_period(self) -> int:
        """Fast period for backward compatibility"""
        return self.parameters.get("fast_period", 12)

    @property
    def slow_period(self) -> int:
        """Slow period for backward compatibility"""
        return self.parameters.get("slow_period", 26)

    @property
    def signal_period(self) -> int:
        """Signal period for backward compatibility"""
        return self.parameters.get("signal_period", 9)

    @property
    def volatility_period(self) -> int:
        """Volatility period for backward compatibility"""
        return self.parameters.get("volatility_period", 14)

    @property
    def strength_threshold(self) -> float:
        """Strength threshold for backward compatibility"""
        return self.parameters.get("strength_threshold", 1.0)

    def get_signal(self, macd_data: pd.Series) -> str:
        """
        Get trading signal based on MACD analysis

        Args:
            macd_data: Series with MACD values (last row from calculate result)

        Returns:
            str: "strong_bullish", "bullish", "strong_bearish", "bearish", or "neutral"
        """
        if len(macd_data) < 4:
            return "neutral"

        macd = macd_data.get('macd', 0)
        signal = macd_data.get('signal', 0)
        histogram = macd_data.get('histogram', 0)
        strength = macd_data.get('strength', 0)
        crossover = macd_data.get('crossover', 0)

        # Strong signals with crossover confirmation
        if crossover == 1 and strength >= self.strength_threshold * 1.5:
            return "strong_bullish"
        elif crossover == -1 and strength >= self.strength_threshold * 1.5:
            return "strong_bearish"
        elif macd > signal and histogram > 0:
            return "bullish"
        elif macd < signal and histogram < 0:
            return "bearish"
        else:
            return "neutral"

    def get_momentum_strength(self, macd_data: pd.Series) -> float:
        """
        Calculate momentum strength score

        Args:
            macd_data: Series with MACD values (last row from calculate result)

        Returns:
            float: Momentum strength score (0-100)
        """
        strength = macd_data.get('strength', 0)
        histogram = abs(macd_data.get('histogram', 0))
        
        # Normalize strength to 0-100 scale
        normalized_strength = min(strength * 20, 100)
        
        # Weight by histogram magnitude
        if histogram > 0:
            normalized_strength *= (1 + min(histogram * 10, 1))
        
        return min(normalized_strength, 100)


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return MACDSignalIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic OHLC data with trending behavior
    close_prices = [base_price]
    for i in range(n_points - 1):
        # Add trending behavior for MACD to detect
        trend = np.sin(i / 30) * 1.5
        change = np.random.randn() * 0.5 + trend * 0.2
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame({
        "close": close_prices,
        "high": [c + abs(np.random.randn() * 0.3) for c in close_prices],
        "low": [c - abs(np.random.randn() * 0.3) for c in close_prices],
    })

    # Calculate MACD Signal
    macd_signal = MACDSignalIndicator()
    result = macd_signal.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # MACD and Signal lines
    ax2.plot(result["macd"].values, label="MACD", color="blue", linewidth=2)
    ax2.plot(result["signal"].values, label="Signal", color="red", linewidth=2)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("MACD and Signal Lines")
    ax2.legend()
    ax2.grid(True)

    # Histogram and Strength
    ax3.bar(range(len(result)), result["histogram"].values, label="Histogram", 
            color=["green" if h >= 0 else "red" for h in result["histogram"].values], alpha=0.6)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(result["strength"].values, label="Strength", color="purple", linewidth=2)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.set_title("MACD Histogram and Signal Strength")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("MACD Signal calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"MACD Signal parameters: {macd_signal.parameters}")
    
    # Show recent signals
    recent_data = result.iloc[-5:]
    print("\nRecent MACD Signal Analysis:")
    for idx, row in recent_data.iterrows():
        signal = macd_signal.get_signal(row)
        strength = macd_signal.get_momentum_strength(row)
        print(f"Point {idx}: MACD={row['macd']:.4f}, Signal={row['signal']:.4f}, "
              f"Histogram={row['histogram']:.4f}, Strength={strength:.1f}, Signal={signal}")