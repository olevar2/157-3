"""
On-Balance Volume (OBV)

On-Balance Volume relates volume to price change and is used to measure buying 
and selling pressure. It's a momentum indicator that uses volume flow to predict 
changes in stock price.

Formula:
If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume  
If Close = Previous Close: OBV = Previous OBV

Interpretation:
- Rising OBV: Buying pressure (bullish)
- Falling OBV: Selling pressure (bearish)
- OBV divergence from price: Potential reversal signal
- OBV confirmation of price trend: Strong trend continuation

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


class OnBalanceVolume(StandardIndicatorInterface):
    """
    On-Balance Volume for momentum and trend confirmation analysis
    
    Combines price direction with volume to create a running total that
    indicates buying and selling pressure in the market.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "volume"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        smoothing_period: int = 20,
        signal_period: int = 10,
        divergence_lookback: int = 20,
        volume_threshold: float = 1.5,
        **kwargs,
    ):
        """
        Initialize On-Balance Volume indicator

        Args:
            smoothing_period: Period for OBV smoothing (default: 20)
            signal_period: Period for signal line calculation (default: 10)  
            divergence_lookback: Lookback period for divergence detection (default: 20)
            volume_threshold: Relative volume threshold for significant moves (default: 1.5)
        """
        super().__init__(
            smoothing_period=smoothing_period,
            signal_period=signal_period,
            divergence_lookback=divergence_lookback,
            volume_threshold=volume_threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate On-Balance Volume

        Args:
            data: DataFrame with 'close' and 'volume' columns

        Returns:
            pd.DataFrame: DataFrame with OBV, signals, and analysis
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "OBV requires DataFrame with 'close' and 'volume' columns"
            )

        smoothing_period = self.parameters.get("smoothing_period", 20)
        signal_period = self.parameters.get("signal_period", 10)
        divergence_lookback = self.parameters.get("divergence_lookback", 20)
        volume_threshold = self.parameters.get("volume_threshold", 1.5)
        
        close = data["close"]
        volume = data["volume"]
        
        # Calculate price direction
        price_direction = close.diff()
        
        # Initialize OBV series
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = 0  # Start with 0
        
        # Calculate OBV
        for i in range(1, len(data)):
            if price_direction.iloc[i] > 0:
                # Price up: add volume
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_direction.iloc[i] < 0:
                # Price down: subtract volume
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                # Price unchanged: OBV unchanged
                obv.iloc[i] = obv.iloc[i-1]
        
        # Calculate OBV moving averages for signals
        obv_sma = obv.rolling(window=smoothing_period, min_periods=1).mean()
        obv_signal = obv.rolling(window=signal_period, min_periods=1).mean()
        
        # Calculate OBV rate of change
        obv_roc = obv.pct_change(periods=signal_period)
        obv_momentum = obv.diff(periods=signal_period)
        
        # Volume analysis
        volume_ma = volume.rolling(window=smoothing_period).mean()
        relative_volume = volume / volume_ma
        
        # Detect high-volume moves
        high_volume_up = (price_direction > 0) & (relative_volume > volume_threshold)
        high_volume_down = (price_direction < 0) & (relative_volume > volume_threshold)
        
        # Calculate OBV trend strength
        obv_trend_up = obv > obv_sma
        obv_trend_strength = abs(obv - obv_sma) / obv_sma.rolling(window=smoothing_period).std()
        
        # Detect divergences between price and OBV
        price_trend = close.rolling(window=divergence_lookback).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
        )
        obv_trend = obv.rolling(window=divergence_lookback).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
        )
        
        # Bullish divergence: price down, OBV up
        bullish_divergence = (price_trend < 0) & (obv_trend > 0)
        # Bearish divergence: price up, OBV down  
        bearish_divergence = (price_trend > 0) & (obv_trend < 0)
        
        # Advanced pattern detection
        # OBV breakout signals
        obv_breakout_up = (obv > obv.rolling(window=20).max().shift(1)) & (relative_volume > 1.2)
        obv_breakout_down = (obv < obv.rolling(window=20).min().shift(1)) & (relative_volume > 1.2)
        
        # OBV crossover signals
        obv_cross_up = (obv > obv_sma) & (obv.shift(1) <= obv_sma.shift(1))
        obv_cross_down = (obv < obv_sma) & (obv.shift(1) >= obv_sma.shift(1))
        
        # Institutional accumulation/distribution detection
        # Large volume with sustained OBV direction
        accumulation_pattern = (
            (obv_trend > 0) & 
            (relative_volume.rolling(window=5).mean() > 1.3) &
            (obv_roc > obv_roc.quantile(0.7))
        )
        
        distribution_pattern = (
            (obv_trend < 0) & 
            (relative_volume.rolling(window=5).mean() > 1.3) &
            (obv_roc < obv_roc.quantile(0.3))
        )
        
        # Calculate signal strength and quality
        trend_consistency = obv_trend.rolling(window=5).mean()  # How consistent is the trend
        volume_consistency = (relative_volume > 1.0).rolling(window=5).mean()
        signal_quality = trend_consistency * volume_consistency
        
        # Generate comprehensive trading signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "neutral"
        
        # Strong signals with high volume confirmation
        strong_bullish = (
            (obv_cross_up | obv_breakout_up | accumulation_pattern) & 
            (relative_volume > volume_threshold) &
            (signal_quality > 0.6)
        )
        strong_bearish = (
            (obv_cross_down | obv_breakout_down | distribution_pattern) & 
            (relative_volume > volume_threshold) &
            (signal_quality > 0.6)
        )
        
        signals[strong_bullish] = "strong_buy"
        signals[strong_bearish] = "strong_sell"
        
        # Regular signals
        signals[(obv_trend_up & (relative_volume > 1.0)) & ~strong_bullish] = "buy"
        signals[(~obv_trend_up & (relative_volume > 1.0)) & ~strong_bearish] = "sell"
        
        # Divergence signals (highest priority)
        signals[bullish_divergence] = "bullish_divergence"
        signals[bearish_divergence] = "bearish_divergence"
        
        # Create result DataFrame
        result = pd.DataFrame({
            'obv': obv,
            'obv_sma': obv_sma,
            'obv_signal': obv_signal,
            'obv_roc': obv_roc,
            'obv_momentum': obv_momentum,
            'relative_volume': relative_volume,
            'obv_trend_strength': obv_trend_strength,
            'signal_quality': signal_quality,
            'high_volume_up': high_volume_up.astype(int),
            'high_volume_down': high_volume_down.astype(int),
            'bullish_divergence': bullish_divergence.astype(int),
            'bearish_divergence': bearish_divergence.astype(int),
            'obv_breakout_up': obv_breakout_up.astype(int),
            'obv_breakout_down': obv_breakout_down.astype(int),
            'accumulation_pattern': accumulation_pattern.astype(int),
            'distribution_pattern': distribution_pattern.astype(int),
            'signals': signals
        }, index=data.index)
        
        # Store calculation details for analysis
        self._last_calculation = {
            "obv": obv,
            "obv_final": obv.iloc[-1] if len(obv) > 0 else 0,
            "signals": signals,
            "smoothing_period": smoothing_period,
            "trend_direction": "up" if obv_trend.iloc[-1] > 0 else "down" if obv_trend.iloc[-1] < 0 else "sideways",
            "divergence_count": (bullish_divergence.sum() + bearish_divergence.sum()),
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate OBV parameters"""
        smoothing_period = self.parameters.get("smoothing_period", 20)
        signal_period = self.parameters.get("signal_period", 10)
        divergence_lookback = self.parameters.get("divergence_lookback", 20)
        volume_threshold = self.parameters.get("volume_threshold", 1.5)

        if not isinstance(smoothing_period, int) or smoothing_period < 1:
            raise IndicatorValidationError(
                f"smoothing_period must be integer >= 1, got {smoothing_period}"
            )

        if not isinstance(signal_period, int) or signal_period < 1:
            raise IndicatorValidationError(
                f"signal_period must be integer >= 1, got {signal_period}"
            )

        if smoothing_period > 200 or signal_period > 200:
            raise IndicatorValidationError(
                f"Periods too large, maximum 200"
            )

        if not isinstance(divergence_lookback, int) or divergence_lookback < 5:
            raise IndicatorValidationError(
                f"divergence_lookback must be integer >= 5, got {divergence_lookback}"
            )

        if not isinstance(volume_threshold, (int, float)) or volume_threshold <= 0:
            raise IndicatorValidationError(
                f"volume_threshold must be positive number, got {volume_threshold}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return OBV metadata"""
        return IndicatorMetadata(
            name="OnBalanceVolume",
            category=self.CATEGORY,
            description="On-Balance Volume - Volume-momentum indicator for trend confirmation",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """OBV requires close and volume data"""
        return ["close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for OBV calculation"""
        return max(self.parameters.get("smoothing_period", 20),
                  self.parameters.get("divergence_lookback", 20))

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "smoothing_period" not in self.parameters:
            self.parameters["smoothing_period"] = 20
        if "signal_period" not in self.parameters:
            self.parameters["signal_period"] = 10
        if "divergence_lookback" not in self.parameters:
            self.parameters["divergence_lookback"] = 20
        if "volume_threshold" not in self.parameters:
            self.parameters["volume_threshold"] = 1.5

    # Property accessors for backward compatibility
    @property
    def smoothing_period(self) -> int:
        """Smoothing period for backward compatibility"""
        return self.parameters.get("smoothing_period", 20)

    @property
    def signal_period(self) -> int:
        """Signal period for backward compatibility"""
        return self.parameters.get("signal_period", 10)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "OnBalanceVolume",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return OnBalanceVolume


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLCV data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic price and volume data
    data_list = []
    current_price = base_price
    
    for i in range(n_points):
        # Add trend and volatility
        trend = np.sin(i / 30) * 0.3
        change = np.random.randn() * 0.5 + trend
        current_price = max(10, current_price + change)
        
        # Volume tends to be higher during bigger price moves
        volume_base = 1000
        volume_multiplier = 1 + abs(change) * 2
        volume = abs(np.random.randn() * volume_base * volume_multiplier) + volume_base
        
        data_list.append({
            "close": current_price,
            "volume": volume
        })
    
    data = pd.DataFrame(data_list)

    # Calculate OBV
    obv_indicator = OnBalanceVolume(smoothing_period=20)
    result = obv_indicator.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # OBV chart
    ax2.plot(result['obv'].values, label="OBV", color="purple", linewidth=2)
    ax2.plot(result['obv_sma'].values, label="OBV SMA", color="orange", linewidth=1)
    ax2.set_title("On-Balance Volume")
    ax2.legend()
    ax2.grid(True)

    # Volume and signals chart
    ax3.bar(range(len(data)), data["volume"].values, alpha=0.3, label="Volume", color="gray")
    ax3_twin = ax3.twinx()
    
    signal_colors = {
        'strong_buy': 'darkgreen',
        'buy': 'green',
        'neutral': 'gray', 
        'sell': 'red',
        'strong_sell': 'darkred',
        'bullish_divergence': 'cyan',
        'bearish_divergence': 'magenta'
    }
    
    for signal, color in signal_colors.items():
        mask = result['signals'] == signal
        if mask.any():
            indices = result.index[mask]
            ax3_twin.scatter(indices, result.loc[mask, 'signal_quality'],
                           color=color, label=signal, alpha=0.8)
    
    ax3.set_title("Volume and OBV Signals")
    ax3.set_ylabel("Volume")
    ax3_twin.set_ylabel("Signal Quality")
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("OBV calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Parameters: {obv_indicator.parameters}")
    
    # Show latest values
    latest_idx = result.dropna().index[-1]
    print(f"\nLatest OBV values:")
    print(f"OBV: {result.loc[latest_idx, 'obv']:.2f}")
    print(f"OBV Rate of Change: {result.loc[latest_idx, 'obv_roc']:.4f}")
    print(f"Signal: {result.loc[latest_idx, 'signals']}")
    print(f"Signal Quality: {result.loc[latest_idx, 'signal_quality']:.3f}")
    
    # Count signals
    signal_counts = result['signals'].value_counts()
    print(f"\nSignal distribution:")
    for signal, count in signal_counts.items():
        print(f"{signal}: {count}")