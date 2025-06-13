"""
Accumulation/Distribution Line (A/D Line)

The Accumulation/Distribution Line is a volume-based indicator designed to measure
cumulative money flow to help determine if a stock is being accumulated or distributed.

Formula:
A/D Line = Previous A/D + Current Period's Money Flow Volume
Money Flow Volume = Volume × ((Close - Low) - (High - Close)) / (High - Low)

Interpretation:
- Rising A/D Line: Accumulation phase (buying pressure)
- Falling A/D Line: Distribution phase (selling pressure)
- Divergence between price and A/D: Potential reversal signal
- Strong correlation with price: Trend confirmation

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


class AccumulationDistributionSignal(StandardIndicatorInterface):
    """
    Accumulation/Distribution Line for money flow analysis
    
    Combines price and volume to show cumulative money flow and 
    identify accumulation/distribution patterns in the market.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "volume"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        smoothing_period: int = 14,
        signal_threshold: float = 0.02,
        divergence_lookback: int = 20,
        **kwargs,
    ):
        """
        Initialize Accumulation/Distribution indicator

        Args:
            smoothing_period: Period for smoothing the A/D line (default: 14)
            signal_threshold: Threshold for significant changes (default: 0.02)
            divergence_lookback: Lookback period for divergence detection (default: 20)
        """
        super().__init__(
            smoothing_period=smoothing_period,
            signal_threshold=signal_threshold,
            divergence_lookback=divergence_lookback,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Accumulation/Distribution Line

        Args:
            data: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            pd.DataFrame: DataFrame with A/D line, signals, and analysis
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "A/D Line requires DataFrame with HLCV data"
            )

        smoothing_period = self.parameters.get("smoothing_period", 14)
        signal_threshold = self.parameters.get("signal_threshold", 0.02)
        divergence_lookback = self.parameters.get("divergence_lookback", 20)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        
        # Calculate Money Flow Multiplier
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        
        # Handle division by zero when high == low
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = volume * money_flow_multiplier
        
        # Calculate cumulative A/D Line
        ad_line = money_flow_volume.cumsum()
        
        # Calculate smoothed A/D Line
        ad_smoothed = ad_line.rolling(window=smoothing_period, min_periods=1).mean()
        
        # Calculate A/D Line rate of change
        ad_roc = ad_line.pct_change(periods=smoothing_period)
        
        # Calculate volume-weighted signals
        volume_ma = volume.rolling(window=smoothing_period).mean()
        relative_volume = volume / volume_ma
        
        # Detect accumulation/distribution phases
        accumulation_signal = (money_flow_multiplier > signal_threshold) & (relative_volume > 1.2)
        distribution_signal = (money_flow_multiplier < -signal_threshold) & (relative_volume > 1.2)
        
        # Detect divergences between price and A/D line
        price_trend = close.rolling(window=divergence_lookback).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
        )
        ad_trend = ad_line.rolling(window=divergence_lookback).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
        )
        
        # Bullish divergence: price down, A/D up
        bullish_divergence = (price_trend < 0) & (ad_trend > 0)
        # Bearish divergence: price up, A/D down
        bearish_divergence = (price_trend > 0) & (ad_trend < 0)
        
        # Calculate institutional flow detection
        institutional_threshold = volume.quantile(0.95)  # Top 5% volume
        institutional_flow = (volume > institutional_threshold) & (abs(money_flow_multiplier) > 0.5)
        
        # Smart money detection (large volume with minimal price impact)
        price_efficiency = abs(close.pct_change()) / (volume / volume_ma)
        smart_money = (relative_volume > 2.0) & (price_efficiency < price_efficiency.quantile(0.3))
        
        # Create comprehensive signals
        signal_strength = abs(money_flow_multiplier) * relative_volume
        signal_quality = signal_strength.rolling(window=5).mean()
        
        # Generate trading signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "neutral"
        
        # Strong accumulation signals
        strong_accumulation = (accumulation_signal & (signal_quality > signal_quality.quantile(0.8)))
        signals[strong_accumulation] = "strong_buy"
        
        # Strong distribution signals  
        strong_distribution = (distribution_signal & (signal_quality > signal_quality.quantile(0.8)))
        signals[strong_distribution] = "strong_sell"
        
        # Regular signals
        signals[accumulation_signal & ~strong_accumulation] = "buy"
        signals[distribution_signal & ~strong_distribution] = "sell"
        
        # Divergence signals override
        signals[bullish_divergence] = "bullish_divergence"
        signals[bearish_divergence] = "bearish_divergence"
        
        # Create result DataFrame
        result = pd.DataFrame({
            'ad_line': ad_line,
            'ad_smoothed': ad_smoothed,
            'money_flow_multiplier': money_flow_multiplier,
            'money_flow_volume': money_flow_volume,
            'ad_roc': ad_roc,
            'relative_volume': relative_volume,
            'signal_strength': signal_strength,
            'signal_quality': signal_quality,
            'accumulation': accumulation_signal.astype(int),
            'distribution': distribution_signal.astype(int),
            'bullish_divergence': bullish_divergence.astype(int),
            'bearish_divergence': bearish_divergence.astype(int),
            'institutional_flow': institutional_flow.astype(int),
            'smart_money': smart_money.astype(int),
            'signals': signals
        }, index=data.index)
        
        # Store calculation details for analysis
        self._last_calculation = {
            "ad_line": ad_line,
            "money_flow_volume": money_flow_volume,
            "signals": signals,
            "smoothing_period": smoothing_period,
            "accumulation_count": accumulation_signal.sum(),
            "distribution_count": distribution_signal.sum(),
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate A/D Line parameters"""
        smoothing_period = self.parameters.get("smoothing_period", 14)
        signal_threshold = self.parameters.get("signal_threshold", 0.02)
        divergence_lookback = self.parameters.get("divergence_lookback", 20)

        if not isinstance(smoothing_period, int) or smoothing_period < 1:
            raise IndicatorValidationError(
                f"smoothing_period must be integer >= 1, got {smoothing_period}"
            )

        if smoothing_period > 200:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"smoothing_period too large, maximum 200, got {smoothing_period}"
            )

        if not isinstance(signal_threshold, (int, float)) or signal_threshold <= 0:
            raise IndicatorValidationError(
                f"signal_threshold must be positive number, got {signal_threshold}"
            )

        if not isinstance(divergence_lookback, int) or divergence_lookback < 5:
            raise IndicatorValidationError(
                f"divergence_lookback must be integer >= 5, got {divergence_lookback}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return A/D Line metadata"""
        return IndicatorMetadata(
            name="AccumulationDistributionLine",
            category=self.CATEGORY,
            description="Accumulation/Distribution Line - Volume-based money flow indicator",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """A/D Line requires HLCV data"""
        return ["high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for A/D calculation"""
        return max(self.parameters.get("smoothing_period", 14), 
                  self.parameters.get("divergence_lookback", 20))

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "smoothing_period" not in self.parameters:
            self.parameters["smoothing_period"] = 14
        if "signal_threshold" not in self.parameters:
            self.parameters["signal_threshold"] = 0.02
        if "divergence_lookback" not in self.parameters:
            self.parameters["divergence_lookback"] = 20

    # Property accessors for backward compatibility
    @property
    def smoothing_period(self) -> int:
        """Smoothing period for backward compatibility"""
        return self.parameters.get("smoothing_period", 14)

    @property
    def signal_threshold(self) -> float:
        """Signal threshold for backward compatibility"""
        return self.parameters.get("signal_threshold", 0.02)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "AccumulationDistributionLine",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return AccumulationDistributionSignal


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLCV data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic OHLCV data
    data_list = []
    current_price = base_price
    
    for i in range(n_points):
        # Add trend and volatility
        trend = np.sin(i / 30) * 0.3
        change = np.random.randn() * 0.5 + trend
        current_price = max(10, current_price + change)
        
        high = current_price + abs(np.random.randn() * 0.3)
        low = current_price - abs(np.random.randn() * 0.3)
        close = current_price
        volume = abs(np.random.randn() * 1000) + 1000
        
        data_list.append({
            "high": high,
            "low": low, 
            "close": close,
            "volume": volume
        })
    
    data = pd.DataFrame(data_list)

    # Calculate A/D Line
    ad_indicator = AccumulationDistributionSignal(smoothing_period=14)
    result = ad_indicator.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Price chart
    ax1.plot(data["close"].values, label="Close Price", color="blue")
    ax1.set_title("Sample Price Data")
    ax1.legend()
    ax1.grid(True)

    # A/D Line chart
    ax2.plot(result['ad_line'].values, label="A/D Line", color="purple", linewidth=2)
    ax2.plot(result['ad_smoothed'].values, label="A/D Smoothed", color="orange", linewidth=1)
    ax2.set_title("Accumulation/Distribution Line")
    ax2.legend()
    ax2.grid(True)

    # Signals chart
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
            ax3.scatter(indices, result.loc[mask, 'signal_strength'], 
                       color=color, label=signal, alpha=0.7)
    
    ax3.plot(result['signal_strength'].values, label="Signal Strength", 
             color="black", alpha=0.3, linewidth=1)
    ax3.set_title("A/D Signals and Strength")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print("A/D Line calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Parameters: {ad_indicator.parameters}")
    
    # Show latest values
    latest_idx = result.dropna().index[-1] 
    print(f"\nLatest A/D values:")
    print(f"A/D Line: {result.loc[latest_idx, 'ad_line']:.2f}")
    print(f"Money Flow Multiplier: {result.loc[latest_idx, 'money_flow_multiplier']:.4f}")
    print(f"Signal: {result.loc[latest_idx, 'signals']}")
    
    # Count signals
    signal_counts = result['signals'].value_counts()
    print(f"\nSignal distribution:")
    for signal, count in signal_counts.items():
        print(f"{signal}: {count}")