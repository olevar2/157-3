"""
Average True Range (ATR) Indicator

The Average True Range is a volatility indicator that measures the degree of price volatility.
It does not indicate price direction but measures volatility that arises due to gaps and limit moves.

Formula:
1. True Range (TR) = max(high - low, |high - previous_close|, |low - previous_close|)
2. ATR = Moving Average of TR over period (typically Exponential Moving Average)

The ATR is primarily used for:
- Setting stop-loss levels
- Position sizing based on volatility
- Confirming breakouts (high ATR suggests strong moves)
- Volatility-based indicators and systems

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


class AverageTrueRangeIndicator(StandardIndicatorInterface):
    """
    Average True Range (ATR) Indicator for volatility measurement
    
    ATR measures the average range of price movement over a specified period,
    providing insight into market volatility and helping with risk management.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"  # ATR is often categorized as volatility but used for trend analysis
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        smoothing_method: str = "ema",  # "ema", "sma", "wilder"
        multiplier: float = 1.0,
        **kwargs,
    ):
        """
        Initialize ATR indicator

        Args:
            period: Period for ATR calculation (default: 14)
            smoothing_method: Method for averaging TR ("ema", "sma", "wilder")
            multiplier: Multiplier for ATR values (default: 1.0)
        """
        super().__init__(
            period=period,
            smoothing_method=smoothing_method,
            multiplier=multiplier,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate ATR indicator

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.Series: ATR values with same index as input data
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "ATR indicator requires DataFrame with 'high', 'low', 'close' columns"
            )

        period = self.parameters.get("period", 14)
        smoothing_method = self.parameters.get("smoothing_method", "ema")
        multiplier = self.parameters.get("multiplier", 1.0)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range
        # TR = max(high - low, |high - previous_close|, |low - previous_close|)
        high_low = high - low
        high_close_prev = abs(high - close.shift(1))
        low_close_prev = abs(low - close.shift(1))
        
        # Take maximum of the three values
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using specified smoothing method
        if smoothing_method == "sma":
            atr = true_range.rolling(window=period, min_periods=period).mean()
        elif smoothing_method == "ema":
            atr = true_range.ewm(span=period, min_periods=period).mean()
        elif smoothing_method == "wilder":
            # Wilder's smoothing: ATR = (previous_ATR * (period-1) + current_TR) / period
            atr = pd.Series(index=data.index, dtype=float)
            
            # First ATR value is SMA of first 'period' TR values
            first_atr = true_range.iloc[:period].mean()
            atr.iloc[period-1] = first_atr
            
            # Subsequent values use Wilder's formula
            for i in range(period, len(data)):
                if pd.notna(atr.iloc[i-1]) and pd.notna(true_range.iloc[i]):
                    atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + true_range.iloc[i]) / period
        else:
            raise IndicatorValidationError(
                f"Invalid smoothing_method: {smoothing_method}. Must be 'sma', 'ema', or 'wilder'"
            )
        
        # Apply multiplier
        atr_result = atr * multiplier
        
        # Store calculation details for analysis
        self._last_calculation = {
            "true_range": true_range,
            "atr": atr_result,
            "period": period,
            "smoothing_method": smoothing_method,
            "multiplier": multiplier,
        }

        return pd.Series(atr_result, index=data.index, name="ATR")

    def validate_parameters(self) -> bool:
        """Validate ATR parameters"""
        period = self.parameters.get("period", 14)
        smoothing_method = self.parameters.get("smoothing_method", "ema")
        multiplier = self.parameters.get("multiplier", 1.0)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        valid_methods = ["sma", "ema", "wilder"]
        if smoothing_method not in valid_methods:
            raise IndicatorValidationError(
                f"smoothing_method must be one of {valid_methods}, got {smoothing_method}"
            )

        if not isinstance(multiplier, (int, float)) or multiplier <= 0:
            raise IndicatorValidationError(
                f"multiplier must be positive number, got {multiplier}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return ATR metadata"""
        return IndicatorMetadata(
            name="ATR",
            category=self.CATEGORY,
            description="Average True Range - Measures price volatility and market uncertainty",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="Series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """ATR requires high, low, and close prices"""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for ATR calculation"""
        return self.parameters.get("period", 14) + 1  # +1 for previous close calculation

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
        if "smoothing_method" not in self.parameters:
            self.parameters["smoothing_method"] = "ema"
        if "multiplier" not in self.parameters:
            self.parameters["multiplier"] = 1.0

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 14)

    @property
    def smoothing_method(self) -> str:
        """Smoothing method for backward compatibility"""
        return self.parameters.get("smoothing_method", "ema")

    @property
    def multiplier(self) -> float:
        """Multiplier for backward compatibility"""
        return self.parameters.get("multiplier", 1.0)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required"""
        return self.parameters.get("period", 14) + 1

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "ATR",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }

    def get_volatility_levels(self, atr_values: pd.Series) -> Dict[str, float]:
        """
        Calculate volatility levels based on ATR values

        Args:
            atr_values: Series of ATR values

        Returns:
            Dict: Volatility level thresholds
        """
        valid_atr = atr_values.dropna()
        
        if len(valid_atr) == 0:
            return {"low": 0, "normal": 0, "high": 0, "extreme": 0}
        
        # Calculate percentiles for volatility classification
        percentiles = valid_atr.quantile([0.25, 0.5, 0.75, 0.9])
        
        return {
            "low": percentiles[0.25],
            "normal": percentiles[0.5],
            "high": percentiles[0.75],
            "extreme": percentiles[0.9],
        }

    def classify_volatility(self, atr_value: float, atr_values: pd.Series) -> str:
        """
        Classify current volatility level

        Args:
            atr_value: Current ATR value
            atr_values: Historical ATR values for comparison

        Returns:
            str: Volatility classification ("low", "normal", "high", "extreme")
        """
        levels = self.get_volatility_levels(atr_values)
        
        if atr_value >= levels["extreme"]:
            return "extreme"
        elif atr_value >= levels["high"]:
            return "high"
        elif atr_value >= levels["normal"]:
            return "normal"
        else:
            return "low"

    def calculate_stop_loss_levels(self, price: float, atr_value: float, direction: str = "long") -> Dict[str, float]:
        """
        Calculate stop-loss levels based on ATR

        Args:
            price: Current price
            atr_value: Current ATR value
            direction: Position direction ("long" or "short")

        Returns:
            Dict: Stop-loss levels at different ATR multiples
        """
        multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
        stops = {}
        
        for mult in multipliers:
            if direction.lower() == "long":
                stops[f"stop_{mult}x"] = price - (atr_value * mult)
            else:  # short
                stops[f"stop_{mult}x"] = price + (atr_value * mult)
        
        return stops


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return AverageTrueRangeIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data with varying volatility
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate price data with varying volatility periods
    close_prices = [base_price]
    volatility_factor = 1.0
    
    for i in range(n_points - 1):
        # Vary volatility over time
        if i % 50 == 0:
            volatility_factor = np.random.uniform(0.5, 2.0)
        
        change = np.random.randn() * volatility_factor
        close_prices.append(close_prices[-1] + change)

    # Create OHLC from close prices
    data = pd.DataFrame({
        "close": close_prices,
        "high": [c + abs(np.random.randn() * 1.2) for c in close_prices],
        "low": [c - abs(np.random.randn() * 1.2) for c in close_prices],
    })
    data["open"] = data["close"].shift(1).fillna(data["close"].iloc[0])

    # Calculate ATR with different methods
    atr_ema = AverageTrueRangeIndicator(period=14, smoothing_method="ema")
    atr_sma = AverageTrueRangeIndicator(period=14, smoothing_method="sma")
    atr_wilder = AverageTrueRangeIndicator(period=14, smoothing_method="wilder")
    
    result_ema = atr_ema.calculate(data)
    result_sma = atr_sma.calculate(data)
    result_wilder = atr_wilder.calculate(data)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Price chart with volatility bands
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    
    # Add ATR-based bands around price
    latest_atr = result_ema.dropna().iloc[-1]
    upper_band = data["close"] + latest_atr
    lower_band = data["close"] - latest_atr
    ax1.fill_between(range(len(data)), upper_band, lower_band, alpha=0.2, color="gray", label="ATR Bands")
    
    ax1.set_title("Price Data with ATR Volatility Bands")
    ax1.legend()
    ax1.grid(True)

    # ATR chart
    ax2.plot(result_ema.values, label="ATR (EMA)", color="red", linewidth=2)
    ax2.plot(result_sma.values, label="ATR (SMA)", color="green", linewidth=1, alpha=0.7)
    ax2.plot(result_wilder.values, label="ATR (Wilder)", color="orange", linewidth=1, alpha=0.7)
    ax2.set_title("Average True Range (ATR) - Different Smoothing Methods")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print("ATR calculation completed successfully!")
    print(f"Data points: {len(result_ema)}")
    print(f"Parameters: {atr_ema.parameters}")
    
    # Show latest values
    latest_idx = result_ema.dropna().index[-1]
    print(f"\nLatest ATR values:")
    print(f"ATR (EMA): {result_ema.loc[latest_idx]:.4f}")
    print(f"ATR (SMA): {result_sma.loc[latest_idx]:.4f}")
    print(f"ATR (Wilder): {result_wilder.loc[latest_idx]:.4f}")
    
    # Volatility analysis
    current_price = data["close"].iloc[-1]
    current_atr = result_ema.iloc[-1]
    volatility_class = atr_ema.classify_volatility(current_atr, result_ema)
    
    print(f"\nVolatility Analysis:")
    print(f"Current Price: {current_price:.2f}")
    print(f"Current ATR: {current_atr:.4f}")
    print(f"Volatility Level: {volatility_class}")
    
    # Stop-loss examples
    stop_levels = atr_ema.calculate_stop_loss_levels(current_price, current_atr, "long")
    print(f"\nLong Position Stop-Loss Levels:")
    for level, price in stop_levels.items():
        print(f"{level}: {price:.2f}")
    
    # Statistics
    valid_atr = result_ema.dropna()
    print(f"\nATR Statistics:")
    print(f"Min: {valid_atr.min():.4f}")
    print(f"Max: {valid_atr.max():.4f}")
    print(f"Mean: {valid_atr.mean():.4f}")
    print(f"Std: {valid_atr.std():.4f}")