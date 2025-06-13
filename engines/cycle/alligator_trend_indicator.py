"""
Alligator Trend Indicator

The Alligator is a technical analysis indicator created by Bill Williams that uses three
smoothed moving averages set at 5, 8, and 13 periods, with additional smoothing factors.
It's designed to identify the absence of a trend and help determine when a market is
dormant (sleeping) or trending.

The Alligator consists of three lines:
1. Jaw (Blue line): 13-period SMA, moved 8 bars forward
2. Teeth (Red line): 8-period SMA, moved 5 bars forward  
3. Lips (Green line): 5-period SMA, moved 3 bars forward

When all three lines are intertwined, the market is considered to be sleeping.
When they diverge, the market is considered to be awakening or trending.

Formulas:
- Jaw = SMA(HL/2, 13) shifted forward by 8
- Teeth = SMA(HL/2, 8) shifted forward by 5
- Lips = SMA(HL/2, 5) shifted forward by 3

Where HL/2 = (High + Low) / 2

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class AlligatorTrendIndicator(StandardIndicatorInterface):
    """
    Alligator Trend Indicator
    
    A trend-following indicator that uses three smoothed moving averages
    to identify trend direction and market dormancy/activity states.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        jaw_period: int = 13,
        jaw_shift: int = 8,
        teeth_period: int = 8,
        teeth_shift: int = 5,
        lips_period: int = 5,
        lips_shift: int = 3,
        source: str = "hl2",  # hl2, close, ohlc4
        **kwargs,
    ):
        """
        Initialize Alligator Trend indicator

        Args:
            jaw_period: Period for Jaw line calculation (default: 13)
            jaw_shift: Forward shift for Jaw line (default: 8)
            teeth_period: Period for Teeth line calculation (default: 8)
            teeth_shift: Forward shift for Teeth line (default: 5)
            lips_period: Period for Lips line calculation (default: 5)
            lips_shift: Forward shift for Lips line (default: 3)
            source: Price source ('hl2', 'close', 'ohlc4') (default: 'hl2')
        """
        super().__init__(
            jaw_period=jaw_period,
            jaw_shift=jaw_shift,
            teeth_period=teeth_period,
            teeth_shift=teeth_shift,
            lips_period=lips_period,
            lips_shift=lips_shift,
            source=source,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Alligator Trend indicator

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Alligator lines (jaw, teeth, lips) and trend signals
        """
        # Handle input data and get source price
        if isinstance(data, pd.Series):
            price = data
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            source = self.parameters.get("source", "hl2")
            
            if source == "hl2":
                if "high" in data.columns and "low" in data.columns:
                    price = (data["high"] + data["low"]) / 2
                else:
                    price = data["close"]  # Fallback to close
            elif source == "ohlc4":
                if all(col in data.columns for col in ["open", "high", "low", "close"]):
                    price = (data["open"] + data["high"] + data["low"] + data["close"]) / 4
                else:
                    price = data["close"]  # Fallback to close
            else:  # source == "close" or fallback
                price = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        jaw_period = self.parameters.get("jaw_period", 13)
        jaw_shift = self.parameters.get("jaw_shift", 8)
        teeth_period = self.parameters.get("teeth_period", 8)
        teeth_shift = self.parameters.get("teeth_shift", 5)
        lips_period = self.parameters.get("lips_period", 5)
        lips_shift = self.parameters.get("lips_shift", 3)

        # Calculate the three smoothed moving averages (Alligator lines)
        jaw_sma = price.rolling(window=jaw_period, min_periods=jaw_period).mean()
        teeth_sma = price.rolling(window=teeth_period, min_periods=teeth_period).mean()
        lips_sma = price.rolling(window=lips_period, min_periods=lips_period).mean()

        # Apply forward shifts (displacement)
        # Note: In practice, forward shift means the line represents future values
        # For historical analysis, we shift backward to see the effect
        jaw = jaw_sma.shift(jaw_shift)
        teeth = teeth_sma.shift(teeth_shift)
        lips = lips_sma.shift(lips_shift)

        # Calculate trend signals
        trend_signal = self._calculate_trend_signals(jaw, teeth, lips, price)
        
        # Calculate alligator state (sleeping/awakening/hunting)
        alligator_state = self._calculate_alligator_state(jaw, teeth, lips)
        
        # Calculate momentum and strength indicators
        momentum = self._calculate_momentum(jaw, teeth, lips, price)
        strength = self._calculate_strength(jaw, teeth, lips)

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["jaw"] = jaw
        result["teeth"] = teeth
        result["lips"] = lips
        result["trend_signal"] = trend_signal
        result["alligator_state"] = alligator_state
        result["momentum"] = momentum
        result["strength"] = strength

        # Store calculation details for analysis
        self._last_calculation = {
            "price": price,
            "jaw_sma": jaw_sma,
            "teeth_sma": teeth_sma,
            "lips_sma": lips_sma,
            "jaw": jaw,
            "teeth": teeth,
            "lips": lips,
            "parameters": self.parameters,
        }

        return result

    def _calculate_trend_signals(self, jaw: pd.Series, teeth: pd.Series, lips: pd.Series, price: pd.Series) -> pd.Series:
        """Calculate trend direction signals"""
        signals = pd.Series(0, index=price.index)  # 0 = neutral
        
        # Bullish when lips > teeth > jaw and price > lips
        bullish_condition = (lips > teeth) & (teeth > jaw) & (price > lips)
        signals.loc[bullish_condition] = 1
        
        # Bearish when lips < teeth < jaw and price < lips
        bearish_condition = (lips < teeth) & (teeth < jaw) & (price < lips)
        signals.loc[bearish_condition] = -1
        
        return signals

    def _calculate_alligator_state(self, jaw: pd.Series, teeth: pd.Series, lips: pd.Series) -> pd.Series:
        """Calculate alligator state (sleeping, awakening, hunting)"""
        states = pd.Series("sleeping", index=jaw.index)
        
        # Calculate the range between lines as percentage of price
        max_line = pd.concat([jaw, teeth, lips], axis=1).max(axis=1)
        min_line = pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
        line_spread = (max_line - min_line) / max_line * 100
        
        # Awakening when lines start to diverge (moderate spread)
        awakening_condition = (line_spread > 0.5) & (line_spread <= 2.0)
        states.loc[awakening_condition] = "awakening"
        
        # Hunting when lines are well separated (high spread)
        hunting_condition = line_spread > 2.0
        states.loc[hunting_condition] = "hunting"
        
        return states

    def _calculate_momentum(self, jaw: pd.Series, teeth: pd.Series, lips: pd.Series, price: pd.Series) -> pd.Series:
        """Calculate momentum based on price position relative to alligator lines"""
        # Distance from price to average of three lines
        alligator_center = (jaw + teeth + lips) / 3
        momentum = (price - alligator_center) / alligator_center * 100
        
        return momentum

    def _calculate_strength(self, jaw: pd.Series, teeth: pd.Series, lips: pd.Series) -> pd.Series:
        """Calculate trend strength based on line separation and alignment"""
        # Calculate line separation as strength indicator
        max_line = pd.concat([jaw, teeth, lips], axis=1).max(axis=1)
        min_line = pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
        line_spread = (max_line - min_line) / max_line * 100
        
        # Normalize strength to 0-100 scale
        strength = np.clip(line_spread * 10, 0, 100)
        
        return strength

    def validate_parameters(self) -> bool:
        """Validate Alligator parameters"""
        jaw_period = self.parameters.get("jaw_period", 13)
        jaw_shift = self.parameters.get("jaw_shift", 8)
        teeth_period = self.parameters.get("teeth_period", 8)
        teeth_shift = self.parameters.get("teeth_shift", 5)
        lips_period = self.parameters.get("lips_period", 5)
        lips_shift = self.parameters.get("lips_shift", 3)
        source = self.parameters.get("source", "hl2")

        # Validate periods
        for period_name, period_value in [
            ("jaw_period", jaw_period),
            ("teeth_period", teeth_period),
            ("lips_period", lips_period),
        ]:
            if not isinstance(period_value, int) or period_value < 1:
                raise IndicatorValidationError(
                    f"{period_name} must be positive integer, got {period_value}"
                )
            if period_value > 200:
                raise IndicatorValidationError(
                    f"{period_name} too large, maximum 200, got {period_value}"
                )

        # Validate shifts
        for shift_name, shift_value in [
            ("jaw_shift", jaw_shift),
            ("teeth_shift", teeth_shift),
            ("lips_shift", lips_shift),
        ]:
            if not isinstance(shift_value, int) or shift_value < 0:
                raise IndicatorValidationError(
                    f"{shift_name} must be non-negative integer, got {shift_value}"
                )
            if shift_value > 50:
                raise IndicatorValidationError(
                    f"{shift_name} too large, maximum 50, got {shift_value}"
                )

        # Validate source
        valid_sources = ["hl2", "close", "ohlc4"]
        if source not in valid_sources:
            raise IndicatorValidationError(
                f"source must be one of {valid_sources}, got {source}"
            )

        # Validate period hierarchy (jaw > teeth > lips typically)
        if not (jaw_period >= teeth_period >= lips_period):
            raise IndicatorValidationError(
                f"Period hierarchy violated: jaw_period({jaw_period}) >= teeth_period({teeth_period}) >= lips_period({lips_period})"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Alligator metadata"""
        return {
            "name": "AlligatorTrend",
            "category": self.CATEGORY,
            "description": "Alligator Trend Indicator - Uses three smoothed moving averages to identify trend direction and market states",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": ["jaw", "teeth", "lips", "trend_signal", "alligator_state", "momentum", "strength"],
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Required columns depend on source parameter"""
        source = self.parameters.get("source", "hl2")
        if source == "hl2":
            return ["high", "low"]
        elif source == "ohlc4":
            return ["open", "high", "low", "close"]
        else:  # close
            return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        jaw_period = self.parameters.get("jaw_period", 13)
        jaw_shift = self.parameters.get("jaw_shift", 8)
        return jaw_period + jaw_shift  # Need enough for calculation + shift

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "jaw_period": 13,
            "jaw_shift": 8,
            "teeth_period": 8,
            "teeth_shift": 5,
            "lips_period": 5,
            "lips_shift": 3,
            "source": "hl2",
        }
        
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    # Backward compatibility properties
    @property
    def jaw_period(self) -> int:
        return self.parameters.get("jaw_period", 13)

    @property
    def teeth_period(self) -> int:
        return self.parameters.get("teeth_period", 8)

    @property
    def lips_period(self) -> int:
        return self.parameters.get("lips_period", 5)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods for calculation"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "AlligatorTrend",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }

    def get_signal(self, trend_signal: float, alligator_state: str, momentum: float) -> str:
        """
        Get trading signal based on Alligator analysis

        Args:
            trend_signal: Current trend signal value (-1, 0, 1)
            alligator_state: Current alligator state
            momentum: Current momentum value

        Returns:
            str: Trading signal
        """
        if alligator_state == "sleeping":
            return "no_trade"
        elif alligator_state == "awakening":
            if trend_signal > 0 and momentum > 0:
                return "prepare_long"
            elif trend_signal < 0 and momentum < 0:
                return "prepare_short"
            else:
                return "wait"
        else:  # hunting
            if trend_signal > 0 and momentum > 2:
                return "strong_long"
            elif trend_signal > 0 and momentum > 0:
                return "long"
            elif trend_signal < 0 and momentum < -2:
                return "strong_short"
            elif trend_signal < 0 and momentum < 0:
                return "short"
            else:
                return "neutral"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return AlligatorTrendIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample data
    np.random.seed(42)
    n_points = 200
    
    # Create realistic OHLC data with trend changes
    dates = pd.date_range(start="2024-01-01", periods=n_points, freq="1H")
    base_price = 100
    
    prices = [base_price]
    for i in range(n_points - 1):
        # Add trend with cycles
        trend = np.sin(i / 30) * 0.5
        noise = np.random.randn() * 0.3
        change = trend + noise
        prices.append(prices[-1] * (1 + change / 100))
    
    # Create OHLC DataFrame
    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["high"] = [p * (1 + abs(np.random.randn() * 0.005)) for p in prices]
    data["low"] = [p * (1 - abs(np.random.randn() * 0.005)) for p in prices]
    data["open"] = data["close"].shift(1).fillna(prices[0])

    # Calculate Alligator
    alligator = AlligatorTrendIndicator()
    result = alligator.calculate(data)

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Price and Alligator lines
    ax1.plot(data.index, data["close"], label="Close Price", color="black", linewidth=1)
    ax1.plot(result.index, result["jaw"], label="Jaw (13)", color="blue", linewidth=2)
    ax1.plot(result.index, result["teeth"], label="Teeth (8)", color="red", linewidth=2)
    ax1.plot(result.index, result["lips"], label="Lips (5)", color="green", linewidth=2)
    ax1.set_title("Alligator Trend Indicator")
    ax1.legend()
    ax1.grid(True)

    # Trend signals
    ax2.plot(result.index, result["trend_signal"], label="Trend Signal", color="purple", linewidth=2)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_title("Trend Signals")
    ax2.set_ylabel("Signal (-1=Bear, 0=Neutral, 1=Bull)")
    ax2.legend()
    ax2.grid(True)

    # Momentum
    ax3.plot(result.index, result["momentum"], label="Momentum", color="orange", linewidth=2)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.set_title("Momentum")
    ax3.set_ylabel("Momentum %")
    ax3.legend()
    ax3.grid(True)

    # Strength
    ax4.plot(result.index, result["strength"], label="Trend Strength", color="red", linewidth=2)
    ax4.set_title("Trend Strength")
    ax4.set_ylabel("Strength (0-100)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    print("Alligator Trend calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Parameters: {alligator.parameters}")
    
    # Show recent values
    recent = result.tail(5)
    print("\nRecent values:")
    print(recent)
    
    # Show state distribution
    state_counts = result["alligator_state"].value_counts()
    print(f"\nAlligator state distribution:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} ({count/len(result)*100:.1f}%)")