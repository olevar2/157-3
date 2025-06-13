"""
Average True Range (ATR)

Average True Range is a technical analysis indicator that measures volatility.
It represents the average of true ranges over a specified number of periods.

Formula:
True Range = MAX(High - Low, ABS(High - Previous Close), ABS(Low - Previous Close))
ATR = Moving Average of True Range over N periods

Interpretation:
- High ATR: High volatility period
- Low ATR: Low volatility period  
- Rising ATR: Increasing volatility
- Falling ATR: Decreasing volatility
- ATR used for position sizing and stop-loss placement

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
from indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
    IndicatorMetadata,
)


class AverageTrueRange(StandardIndicatorInterface):
    """
    Average True Range for volatility measurement and risk management
    
    Provides dynamic volatility measurement for position sizing,
    stop-loss placement, and market regime identification.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "volatility"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 14,
        smoothing_method: str = "ema",
        volatility_threshold: float = 1.5,
        regime_periods: int = 50,
        **kwargs,
    ):
        """
        Initialize ATR indicator

        Args:
            period: Period for ATR calculation (default: 14)
            smoothing_method: Smoothing method 'ema' or 'sma' (default: 'ema')
            volatility_threshold: Threshold for high/low volatility detection (default: 1.5)
            regime_periods: Periods for volatility regime analysis (default: 50)
        """
        super().__init__(
            period=period,
            smoothing_method=smoothing_method,
            volatility_threshold=volatility_threshold,
            regime_periods=regime_periods,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Average True Range

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.DataFrame: DataFrame with ATR, volatility analysis, and signals
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "ATR requires DataFrame with 'high', 'low', 'close' columns"
            )

        period = self.parameters.get("period", 14)
        smoothing_method = self.parameters.get("smoothing_method", "ema")
        volatility_threshold = self.parameters.get("volatility_threshold", 1.5)
        regime_periods = self.parameters.get("regime_periods", 50)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range components
        hl = high - low  # High - Low
        hc = abs(high - close.shift(1))  # High - Previous Close
        lc = abs(low - close.shift(1))   # Low - Previous Close
        
        # True Range is the maximum of the three
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # Calculate ATR using specified smoothing method
        if smoothing_method.lower() == "ema":
            # Exponential Moving Average
            atr = true_range.ewm(span=period, adjust=False).mean()
        else:
            # Simple Moving Average (default fallback)
            atr = true_range.rolling(window=period, min_periods=1).mean()
        
        # Calculate percentage ATR (ATR relative to price)
        atr_percentage = (atr / close) * 100
        
        # Calculate ATR-based volatility metrics
        atr_ma = atr.rolling(window=regime_periods).mean()
        atr_std = atr.rolling(window=regime_periods).std()
        
        # Volatility regimes
        high_volatility = atr > (atr_ma + atr_std * volatility_threshold)
        low_volatility = atr < (atr_ma - atr_std * volatility_threshold)
        normal_volatility = ~(high_volatility | low_volatility)
        
        # ATR momentum and trend
        atr_momentum = atr.diff(periods=5)
        atr_trend = pd.Series(index=data.index, dtype=str)
        atr_trend[atr_momentum > 0] = "increasing"
        atr_trend[atr_momentum < 0] = "decreasing"
        atr_trend[atr_momentum == 0] = "stable"
        
        # Volatility expansion/contraction detection
        atr_expansion = (atr > atr.rolling(window=20).max().shift(1)) & high_volatility
        atr_contraction = (atr < atr.rolling(window=20).min().shift(1)) & low_volatility
        
        # ATR breakout signals
        atr_breakout_threshold = atr_ma + atr_std
        volatility_breakout = (atr > atr_breakout_threshold) & (atr.shift(1) <= atr_breakout_threshold.shift(1))
        
        # Price movement efficiency (price change relative to ATR)
        price_change = abs(close.diff())
        movement_efficiency = price_change / atr
        
        # Volatility clustering detection
        volatility_cluster = (
            high_volatility & 
            (high_volatility.rolling(window=5).sum() >= 3)
        )
        
        # ATR-based support and resistance levels
        atr_support = close - (atr * 2)
        atr_resistance = close + (atr * 2)
        
        # Risk assessment metrics
        # Position sizing factor (inverse of ATR percentage)
        position_size_factor = 1 / (atr_percentage / 100 + 0.01)  # Avoid division by zero
        
        # Stop loss suggestions
        stop_loss_long = close - (atr * 2)
        stop_loss_short = close + (atr * 2)
        
        # Volatility-adjusted signals
        volatility_regime = pd.Series(index=data.index, dtype=str)
        volatility_regime[high_volatility] = "high"
        volatility_regime[low_volatility] = "low"
        volatility_regime[normal_volatility] = "normal"
        
        # Market regime changes
        regime_change = volatility_regime != volatility_regime.shift(1)
        
        # Generate comprehensive signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "neutral"
        
        # Volatility expansion signals
        signals[atr_expansion & (movement_efficiency > movement_efficiency.quantile(0.7))] = "volatility_expansion"
        
        # Volatility contraction signals
        signals[atr_contraction & (movement_efficiency < movement_efficiency.quantile(0.3))] = "volatility_contraction"
        
        # Breakout preparation signals
        signals[volatility_breakout] = "volatility_breakout"
        
        # Regime change signals
        signals[regime_change & high_volatility] = "entering_high_vol"
        signals[regime_change & low_volatility] = "entering_low_vol"
        
        # Clustering signals
        signals[volatility_cluster] = "volatility_cluster"
        
        # Create result DataFrame
        result = pd.DataFrame({
            'atr': atr,
            'true_range': true_range,
            'atr_percentage': atr_percentage,
            'atr_ma': atr_ma,
            'atr_std': atr_std,
            'atr_momentum': atr_momentum,
            'atr_trend': atr_trend,
            'movement_efficiency': movement_efficiency,
            'position_size_factor': position_size_factor,
            'atr_support': atr_support,
            'atr_resistance': atr_resistance,
            'stop_loss_long': stop_loss_long,
            'stop_loss_short': stop_loss_short,
            'volatility_regime': volatility_regime,
            'high_volatility': high_volatility.astype(int),
            'low_volatility': low_volatility.astype(int),
            'atr_expansion': atr_expansion.astype(int),
            'atr_contraction': atr_contraction.astype(int),
            'volatility_breakout': volatility_breakout.astype(int),
            'volatility_cluster': volatility_cluster.astype(int),
            'regime_change': regime_change.astype(int),
            'signals': signals
        }, index=data.index)
        
        # Store calculation details for analysis
        self._last_calculation = {
            "atr": atr,
            "final_atr": atr.iloc[-1] if len(atr) > 0 else 0,
            "final_atr_percentage": atr_percentage.iloc[-1] if len(atr_percentage) > 0 else 0,
            "current_regime": volatility_regime.iloc[-1] if len(volatility_regime) > 0 else "normal",
            "signals": signals,
            "period": period,
            "smoothing_method": smoothing_method,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate ATR parameters"""
        period = self.parameters.get("period", 14)
        smoothing_method = self.parameters.get("smoothing_method", "ema")
        volatility_threshold = self.parameters.get("volatility_threshold", 1.5)
        regime_periods = self.parameters.get("regime_periods", 50)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be integer >= 1, got {period}"
            )

        if period > 200:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 200, got {period}"
            )

        if smoothing_method.lower() not in ["ema", "sma"]:
            raise IndicatorValidationError(
                f"smoothing_method must be 'ema' or 'sma', got {smoothing_method}"
            )

        if not isinstance(volatility_threshold, (int, float)) or volatility_threshold <= 0:
            raise IndicatorValidationError(
                f"volatility_threshold must be positive number, got {volatility_threshold}"
            )

        if not isinstance(regime_periods, int) or regime_periods < 10:
            raise IndicatorValidationError(
                f"regime_periods must be integer >= 10, got {regime_periods}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return ATR metadata"""
        return IndicatorMetadata(
            name="AverageTrueRange",
            category=self.CATEGORY,
            description="Average True Range - Volatility measurement and risk management indicator",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """ATR requires HLC data"""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for ATR calculation"""
        return max(self.parameters.get("period", 14),
                  self.parameters.get("regime_periods", 50))

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 14
        if "smoothing_method" not in self.parameters:
            self.parameters["smoothing_method"] = "ema"
        if "volatility_threshold" not in self.parameters:
            self.parameters["volatility_threshold"] = 1.5
        if "regime_periods" not in self.parameters:
            self.parameters["regime_periods"] = 50

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
    def minimum_periods(self) -> int:
        """Minimum periods required"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "AverageTrueRange",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return AverageTrueRange


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    base_price = 100

    # Generate realistic OHLC data with varying volatility
    data_list = []
    current_price = base_price
    
    for i in range(n_points):
        # Add volatility regime changes
        if i < 50:
            volatility = 0.5  # Low volatility period
        elif i < 100:
            volatility = 2.0  # High volatility period
        elif i < 150:
            volatility = 0.8  # Medium volatility period
        else:
            volatility = 1.5  # Increasing volatility
        
        change = np.random.randn() * volatility
        current_price = max(10, current_price + change)
        
        # Create realistic OHLC from current price
        high = current_price + abs(np.random.randn() * volatility * 0.5)
        low = current_price - abs(np.random.randn() * volatility * 0.5)
        close = current_price
        
        data_list.append({
            "high": high,
            "low": low,
            "close": close
        })
    
    data = pd.DataFrame(data_list)

    # Calculate ATR
    atr_indicator = AverageTrueRange(period=14)
    result = atr_indicator.calculate(data)

    # Plot results
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

    # Price chart with ATR bands
    ax1.plot(data["close"].values, label="Close Price", color="blue", linewidth=2)
    ax1.fill_between(range(len(data)), 
                     result['atr_support'].values, 
                     result['atr_resistance'].values,
                     alpha=0.2, color="gray", label="ATR Bands")
    ax1.set_title("Price with ATR Support/Resistance")
    ax1.legend()
    ax1.grid(True)

    # ATR chart
    ax2.plot(result['atr'].values, label="ATR", color="purple", linewidth=2)
    ax2.plot(result['atr_ma'].values, label="ATR MA", color="orange", linewidth=1)
    
    # Highlight volatility regimes
    high_vol_mask = result['high_volatility'] == 1
    low_vol_mask = result['low_volatility'] == 1
    
    ax2.fill_between(range(len(data)), 0, result['atr'].values,
                     where=high_vol_mask, alpha=0.3, color="red", label="High Volatility")
    ax2.fill_between(range(len(data)), 0, result['atr'].values,
                     where=low_vol_mask, alpha=0.3, color="green", label="Low Volatility")
    
    ax2.set_title("Average True Range")
    ax2.legend()
    ax2.grid(True)

    # ATR Percentage chart
    ax3.plot(result['atr_percentage'].values, label="ATR %", color="brown", linewidth=2)
    ax3.axhline(y=result['atr_percentage'].mean(), color="gray", linestyle="--", alpha=0.7, label="Average ATR %")
    ax3.set_title("ATR Percentage")
    ax3.legend()
    ax3.grid(True)

    # Signals and regime changes
    signal_colors = {
        'neutral': 'gray',
        'volatility_expansion': 'red',
        'volatility_contraction': 'green', 
        'volatility_breakout': 'orange',
        'entering_high_vol': 'darkred',
        'entering_low_vol': 'darkgreen',
        'volatility_cluster': 'purple'
    }
    
    for signal, color in signal_colors.items():
        mask = result['signals'] == signal
        if mask.any():
            indices = result.index[mask]
            ax4.scatter(indices, result.loc[mask, 'movement_efficiency'],
                       color=color, label=signal, alpha=0.7)
    
    ax4.plot(result['movement_efficiency'].values, label="Movement Efficiency", 
             color="black", alpha=0.3, linewidth=1)
    ax4.set_title("ATR Signals and Movement Efficiency")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    print("ATR calculation completed successfully!")
    print(f"Data points: {len(result)}")
    print(f"Parameters: {atr_indicator.parameters}")
    
    # Show latest values
    latest_idx = result.dropna().index[-1]
    print(f"\nLatest ATR values:")
    print(f"ATR: {result.loc[latest_idx, 'atr']:.4f}")
    print(f"ATR %: {result.loc[latest_idx, 'atr_percentage']:.2f}%")
    print(f"Volatility Regime: {result.loc[latest_idx, 'volatility_regime']}")
    print(f"Signal: {result.loc[latest_idx, 'signals']}")
    
    # Count signals and regimes
    signal_counts = result['signals'].value_counts()
    print(f"\nSignal distribution:")
    for signal, count in signal_counts.items():
        print(f"{signal}: {count}")
    
    regime_counts = result['volatility_regime'].value_counts()
    print(f"\nVolatility regime distribution:")
    for regime, count in regime_counts.items():
        print(f"{regime}: {count}")