"""
Standard Deviation Channels Indicator

Standard deviation channels create dynamic price channels based on a moving average
center line with upper and lower bands positioned at specified standard deviations
from the center line. This is similar to Bollinger Bands but more flexible.

Components:
1. Center Line: Moving average (SMA, EMA, or other)
2. Upper Channel: Center + (multiplier × standard deviation)
3. Lower Channel: Center - (multiplier × standard deviation)

Channel Analysis:
- Width indicates volatility level
- Position indicates momentum and mean reversion opportunities
- Breakouts signal potential trend changes
- Squeeze patterns indicate low volatility periods

Mathematical Foundation:
- Standard Deviation: σ = sqrt(Σ(x - μ)² / n)
- Upper Band = MA + (k × σ)
- Lower Band = MA - (k × σ)
Where k is the multiplier and MA is moving average

Applications:
- Volatility analysis
- Support/resistance identification  
- Mean reversion trading
- Trend continuation signals
- Volatility breakout strategies

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
from typing import Any, Dict, List, Union, Optional

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class StandardDeviationChannels(StandardIndicatorInterface):
    """
    Standard Deviation Channels Indicator
    
    Creates dynamic price channels using a moving average center line and
    standard deviation-based upper and lower bands.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "statistical"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,  # Period for moving average and standard deviation
        std_dev_multiplier: float = 2.0,  # Standard deviation multiplier
        ma_type: str = "sma",  # Moving average type ("sma", "ema", "wma")
        min_periods: int = 10,  # Minimum periods for calculation
        **kwargs,
    ):
        """
        Initialize Standard Deviation Channels indicator

        Args:
            period: Period for calculations (default: 20)
            std_dev_multiplier: Multiplier for standard deviation bands (default: 2.0)
            ma_type: Type of moving average for center line (default: "sma")
            min_periods: Minimum periods required for calculation (default: 10)
        """
        super().__init__(
            period=period,
            std_dev_multiplier=std_dev_multiplier,
            ma_type=ma_type,
            min_periods=min_periods,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Standard Deviation Channels
        
        Args:
            data: Price data (DataFrame with 'close' or Series of prices)
        
        Returns:
            pd.DataFrame: DataFrame with center line, upper/lower channels, and metrics
        """
        # Handle input data
        if isinstance(data, pd.Series):
            prices = data
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                prices = data["close"]
                self.validate_input_data(data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        period = self.parameters.get("period", 20)
        std_dev_multiplier = self.parameters.get("std_dev_multiplier", 2.0)
        ma_type = self.parameters.get("ma_type", "sma")
        min_periods = self.parameters.get("min_periods", 10)

        # Calculate center line (moving average)
        if ma_type == "sma":
            center_line = prices.rolling(window=period, min_periods=min_periods).mean()
        elif ma_type == "ema":
            center_line = prices.ewm(span=period, min_periods=min_periods).mean()
        elif ma_type == "wma":
            # Weighted moving average
            weights = np.arange(1, period + 1)
            center_line = prices.rolling(window=period, min_periods=min_periods).apply(
                lambda x: np.average(x, weights=weights[-len(x):]), raw=True
            )
        else:
            raise IndicatorValidationError(
                f"Unknown moving average type: {ma_type}. Use 'sma', 'ema', or 'wma'"
            )

        # Calculate rolling standard deviation
        rolling_std = prices.rolling(window=period, min_periods=min_periods).std()

        # Calculate channels
        upper_channel = center_line + (std_dev_multiplier * rolling_std)
        lower_channel = center_line - (std_dev_multiplier * rolling_std)

        # Calculate additional metrics
        channel_width = upper_channel - lower_channel
        channel_position = (prices - lower_channel) / channel_width
        
        # Calculate %B (similar to Bollinger %B)
        percent_b = (prices - lower_channel) / (upper_channel - lower_channel)
        
        # Calculate bandwidth (channel width relative to center line)
        bandwidth = channel_width / center_line

        # Identify squeeze periods (low volatility)
        squeeze_threshold = bandwidth.rolling(window=50, min_periods=25).quantile(0.2)
        is_squeeze = bandwidth < squeeze_threshold

        # Initialize result DataFrame
        result = pd.DataFrame(index=prices.index)
        result["center_line"] = center_line
        result["upper_channel"] = upper_channel
        result["lower_channel"] = lower_channel
        result["channel_width"] = channel_width
        result["channel_position"] = channel_position
        result["percent_b"] = percent_b
        result["bandwidth"] = bandwidth
        result["rolling_std"] = rolling_std
        result["is_squeeze"] = is_squeeze

        # Store calculation details for analysis
        self._last_calculation = {
            "prices": prices,
            "result": result,
            "period": period,
            "std_dev_multiplier": std_dev_multiplier,
            "ma_type": ma_type,
            "min_periods": min_periods,
        }

        return result

    def get_volatility_signals(self, channels: pd.DataFrame = None, data: Union[pd.DataFrame, pd.Series] = None) -> pd.Series:
        """
        Generate volatility-based signals
        
        Args:
            channels: Pre-calculated channels (if None, will calculate)
            data: Price data (required if channels is None)
            
        Returns:
            pd.Series: Volatility signals
        """
        if channels is None:
            if data is None:
                raise IndicatorValidationError("Either channels or data must be provided")
            channels = self.calculate(data)
        
        signals = pd.Series("normal", index=channels.index, name="volatility_signals")
        
        if "bandwidth" not in channels.columns or "is_squeeze" not in channels.columns:
            return signals
        
        bandwidth = channels["bandwidth"]
        is_squeeze = channels["is_squeeze"]
        
        # Calculate volatility percentiles
        bandwidth_80th = bandwidth.rolling(window=252, min_periods=50).quantile(0.8)
        bandwidth_20th = bandwidth.rolling(window=252, min_periods=50).quantile(0.2)
        
        # Generate signals
        high_volatility = bandwidth > bandwidth_80th
        low_volatility = bandwidth < bandwidth_20th
        
        signals.loc[high_volatility] = "high_volatility"
        signals.loc[low_volatility] = "low_volatility"
        signals.loc[is_squeeze] = "squeeze"
        
        # Detect breakouts from squeeze
        squeeze_breakout = (~is_squeeze) & (is_squeeze.shift(1))
        signals.loc[squeeze_breakout] = "squeeze_breakout"
        
        return signals

    def get_mean_reversion_signals(self, channels: pd.DataFrame = None, data: Union[pd.DataFrame, pd.Series] = None) -> pd.Series:
        """
        Generate mean reversion signals based on channel position
        
        Args:
            channels: Pre-calculated channels (if None, will calculate)
            data: Price data (required if channels is None)
            
        Returns:
            pd.Series: Mean reversion signals
        """
        if channels is None:
            if data is None:
                raise IndicatorValidationError("Either channels or data must be provided")
            channels = self.calculate(data)
        
        signals = pd.Series("neutral", index=channels.index, name="mean_reversion_signals")
        
        if "percent_b" not in channels.columns:
            return signals
        
        percent_b = channels["percent_b"]
        
        # Mean reversion signals based on %B
        # %B > 1: Price above upper band (overbought)
        # %B < 0: Price below lower band (oversold)
        # %B around 0.5: Price at center line
        
        overbought = percent_b > 1.0
        oversold = percent_b < 0.0
        extreme_overbought = percent_b > 1.2
        extreme_oversold = percent_b < -0.2
        
        signals.loc[extreme_overbought] = "extreme_sell"
        signals.loc[overbought & ~extreme_overbought] = "sell"
        signals.loc[extreme_oversold] = "extreme_buy"
        signals.loc[oversold & ~extreme_oversold] = "buy"
        
        # Reversal signals when returning to bands
        returning_from_overbought = (percent_b < 1.0) & (percent_b.shift(1) > 1.0)
        returning_from_oversold = (percent_b > 0.0) & (percent_b.shift(1) < 0.0)
        
        signals.loc[returning_from_overbought] = "reversal_sell"
        signals.loc[returning_from_oversold] = "reversal_buy"
        
        return signals

    def validate_parameters(self) -> bool:
        """Validate Standard Deviation Channels parameters"""
        period = self.parameters.get("period", 20)
        std_dev_multiplier = self.parameters.get("std_dev_multiplier", 2.0)
        ma_type = self.parameters.get("ma_type", "sma")
        min_periods = self.parameters.get("min_periods", 10)

        if not isinstance(period, int) or period < 2:
            raise IndicatorValidationError(
                f"period must be integer >= 2, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(std_dev_multiplier, (int, float)) or std_dev_multiplier <= 0:
            raise IndicatorValidationError(
                f"std_dev_multiplier must be positive number, got {std_dev_multiplier}"
            )

        if ma_type not in ["sma", "ema", "wma"]:
            raise IndicatorValidationError(
                f"ma_type must be 'sma', 'ema', or 'wma', got {ma_type}"
            )

        if not isinstance(min_periods, int) or min_periods < 1:
            raise IndicatorValidationError(
                f"min_periods must be integer >= 1, got {min_periods}"
            )

        if min_periods > period:
            raise IndicatorValidationError(
                f"min_periods cannot exceed period, got min_periods={min_periods}, period={period}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Standard Deviation Channels metadata as dictionary"""
        return {
            "name": "Standard Deviation Channels",
            "category": self.CATEGORY,
            "description": "Dynamic price channels using moving average center line and standard deviation bands",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }
    
    def _get_required_columns(self) -> List[str]:
        """Standard Deviation Channels requires price data"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Standard Deviation Channels"""
        return self.parameters.get("min_periods", 10)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "std_dev_multiplier" not in self.parameters:
            self.parameters["std_dev_multiplier"] = 2.0
        if "ma_type" not in self.parameters:
            self.parameters["ma_type"] = "sma"
        if "min_periods" not in self.parameters:
            self.parameters["min_periods"] = 10

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def std_dev_multiplier(self) -> float:
        """Standard deviation multiplier for backward compatibility"""
        return self.parameters.get("std_dev_multiplier", 2.0)

    @property
    def ma_type(self) -> str:
        """Moving average type for backward compatibility"""
        return self.parameters.get("ma_type", "sma")

    @property
    def min_periods(self) -> int:
        """Minimum periods for backward compatibility"""
        return self.parameters.get("min_periods", 10)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods property for compatibility"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "StandardDeviationChannels",
            "period": self.period,
            "std_dev_multiplier": self.std_dev_multiplier,
            "ma_type": self.ma_type,
            "min_periods": self.min_periods,
            "category": self.CATEGORY,
        }

    def interpret_channels(self, channels: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Interpret channel signals and position
        
        Args:
            channels: Calculated channels DataFrame
            current_price: Current price value
            
        Returns:
            Dict containing signal interpretation
        """
        if channels.empty:
            return {
                "signal": "insufficient_data",
                "interpretation": "Not enough data for channel analysis",
                "volatility_regime": "unknown"
            }
        
        latest_data = channels.dropna().iloc[-1]
        
        center_line = latest_data["center_line"]
        upper_channel = latest_data["upper_channel"]
        lower_channel = latest_data["lower_channel"]
        percent_b = latest_data["percent_b"]
        bandwidth = latest_data["bandwidth"]
        is_squeeze = latest_data["is_squeeze"]
        
        # Volatility regime assessment
        if is_squeeze:
            volatility_regime = "low_volatility_squeeze"
        elif bandwidth > bandwidth.quantile(0.8):
            volatility_regime = "high_volatility"
        elif bandwidth < bandwidth.quantile(0.2):
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal_volatility"
        
        # Position analysis
        if percent_b > 1.2:
            position_desc = "extremely_overbought"
            signal = "strong_sell_signal"
        elif percent_b > 1.0:
            position_desc = "overbought"
            signal = "sell_signal"
        elif percent_b > 0.8:
            position_desc = "upper_channel"
            signal = "approaching_resistance"
        elif percent_b > 0.2:
            position_desc = "middle_range"
            signal = "neutral"
        elif percent_b > 0.0:
            position_desc = "lower_channel"
            signal = "approaching_support"
        elif percent_b > -0.2:
            position_desc = "oversold"
            signal = "buy_signal"
        else:
            position_desc = "extremely_oversold"
            signal = "strong_buy_signal"
        
        # Distance from center line
        distance_from_center = abs(current_price - center_line)
        distance_pct = (distance_from_center / center_line) * 100
        
        # Overall interpretation
        if volatility_regime == "low_volatility_squeeze":
            interpretation = f"Low volatility squeeze - expect breakout soon. Price is {position_desc}"
        elif volatility_regime == "high_volatility":
            interpretation = f"High volatility environment - price is {position_desc}"
        else:
            interpretation = f"Normal volatility - price is {position_desc}"

        return {
            "signal": signal,
            "interpretation": interpretation,
            "volatility_regime": volatility_regime,
            "position_description": position_desc,
            "percent_b": percent_b,
            "bandwidth": bandwidth,
            "is_squeeze": is_squeeze,
            "current_price": current_price,
            "center_line": center_line,
            "upper_channel": upper_channel,
            "lower_channel": lower_channel,
            "distance_from_center": distance_from_center,
            "distance_percentage": distance_pct,
        }

    def get_breakout_confirmation(self, channels: pd.DataFrame, volume_data: pd.Series = None) -> pd.Series:
        """
        Detect confirmed breakouts with optional volume confirmation
        
        Args:
            channels: Calculated channels DataFrame
            volume_data: Optional volume data for confirmation
            
        Returns:
            pd.Series: Breakout confirmation signals
        """
        signals = pd.Series("no_breakout", index=channels.index, name="breakout_confirmation")
        
        if channels.empty or "percent_b" not in channels.columns:
            return signals
        
        percent_b = channels["percent_b"]
        is_squeeze = channels["is_squeeze"]
        
        # Detect breakouts
        upper_breakout = percent_b > 1.0
        lower_breakout = percent_b < 0.0
        
        # Post-squeeze breakouts are more significant
        post_squeeze_upper = upper_breakout & (is_squeeze.shift(1) | is_squeeze.shift(2))
        post_squeeze_lower = lower_breakout & (is_squeeze.shift(1) | is_squeeze.shift(2))
        
        # Sustained breakouts (multiple periods)
        sustained_upper = upper_breakout & upper_breakout.shift(1)
        sustained_lower = lower_breakout & lower_breakout.shift(1)
        
        # Apply signals
        signals.loc[post_squeeze_upper] = "strong_bullish_breakout"
        signals.loc[post_squeeze_lower] = "strong_bearish_breakout"
        signals.loc[sustained_upper & ~post_squeeze_upper] = "bullish_breakout"
        signals.loc[sustained_lower & ~post_squeeze_lower] = "bearish_breakout"
        
        # Volume confirmation if available
        if volume_data is not None:
            volume_ma = volume_data.rolling(window=20, min_periods=10).mean()
            high_volume = volume_data > (1.5 * volume_ma)
            
            # Upgrade signals with volume confirmation
            volume_confirmed_bull = (signals == "bullish_breakout") & high_volume
            volume_confirmed_bear = (signals == "bearish_breakout") & high_volume
            
            signals.loc[volume_confirmed_bull] = "volume_confirmed_bullish"
            signals.loc[volume_confirmed_bear] = "volume_confirmed_bearish"
        
        return signals


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return StandardDeviationChannels


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data with varying volatility
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    np.random.seed(42)
    
    # Create price data with periods of high and low volatility
    base_price = 100
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # Vary volatility over time
        volatility = 0.02 if i % 100 < 20 else 0.005  # High vol every 100 days for 20 days
        
        # Random walk with changing volatility
        change = np.random.normal(0.001, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    data = pd.DataFrame({
        "close": prices
    }, index=dates)
    
    # Calculate standard deviation channels
    sd_channels = StandardDeviationChannels(period=20, std_dev_multiplier=2.0, ma_type="sma")
    channels = sd_channels.calculate(data)
    
    print("Standard Deviation Channels Indicator Example:")
    
    # Display latest channel values
    latest_channels = channels.dropna().iloc[-1]
    print(f"Center Line: {latest_channels['center_line']:.2f}")
    print(f"Upper Channel: {latest_channels['upper_channel']:.2f}")
    print(f"Lower Channel: {latest_channels['lower_channel']:.2f}")
    print(f"Percent B: {latest_channels['percent_b']:.3f}")
    print(f"Bandwidth: {latest_channels['bandwidth']:.4f}")
    print(f"Is Squeeze: {latest_channels['is_squeeze']}")
    
    # Get volatility signals
    vol_signals = sd_channels.get_volatility_signals(channels)
    latest_vol_signal = vol_signals.iloc[-1]
    print(f"Volatility Signal: {latest_vol_signal}")
    
    # Get mean reversion signals
    mr_signals = sd_channels.get_mean_reversion_signals(channels)
    latest_mr_signal = mr_signals.iloc[-1]
    print(f"Mean Reversion Signal: {latest_mr_signal}")
    
    # Interpret signals
    current_price = data["close"].iloc[-1]
    interpretation = sd_channels.interpret_channels(channels, current_price)
    print(f"\nInterpretation: {interpretation}")
    
    # Show squeeze periods
    squeeze_periods = channels[channels["is_squeeze"]].index
    print(f"\nSqueeze periods detected: {len(squeeze_periods)} days")
    if len(squeeze_periods) > 0:
        print(f"Latest squeeze period: {squeeze_periods[-1] if len(squeeze_periods) > 0 else 'None'}")