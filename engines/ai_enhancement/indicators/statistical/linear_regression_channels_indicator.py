"""
Linear Regression Channels Indicator

Linear regression channels provide a dynamic channel around a linear regression trend line.
The channels consist of:
1. Central regression line (trend)
2. Upper channel (regression line + standard deviation)
3. Lower channel (regression line - standard deviation)

The channels help identify:
- Price trend direction and strength
- Support and resistance levels
- Potential reversal points
- Trend continuation patterns
- Breakout signals

Mathematical Foundation:
- Linear Regression: y = mx + b (where m is slope, b is intercept)
- Standard Error of Regression
- Confidence intervals around the regression line

Applications:
- Trend following strategies
- Support/resistance identification
- Channel breakout trading
- Mean reversion signals
- Price target estimation

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import the base indicator interface
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class LinearRegressionChannels(StandardIndicatorInterface):
    """
    Linear Regression Channels Indicator
    
    Creates dynamic channels around a linear regression trend line based on
    statistical confidence intervals and standard deviation bands.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "statistical"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,  # Period for regression calculation
        std_dev_multiplier: float = 2.0,  # Standard deviation multiplier for channels
        confidence_level: float = 0.95,  # Confidence level for regression bands
        min_periods: int = 10,  # Minimum periods for calculation
        **kwargs,
    ):
        """
        Initialize Linear Regression Channels indicator

        Args:
            period: Period for linear regression calculation (default: 20)
            std_dev_multiplier: Multiplier for standard deviation channels (default: 2.0)
            confidence_level: Confidence level for regression (default: 0.95)
            min_periods: Minimum periods required for calculation (default: 10)
        """
        super().__init__(
            period=period,
            std_dev_multiplier=std_dev_multiplier,
            confidence_level=confidence_level,
            min_periods=min_periods,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Linear Regression Channels
        
        Args:
            data: Price data (DataFrame with 'close' or Series of prices)
        
        Returns:
            pd.DataFrame: DataFrame with regression line, upper/lower channels, and statistics
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
        confidence_level = self.parameters.get("confidence_level", 0.95)
        min_periods = self.parameters.get("min_periods", 10)

        # Initialize result DataFrame
        result = pd.DataFrame(index=prices.index)
        result["regression_line"] = np.nan
        result["upper_channel"] = np.nan
        result["lower_channel"] = np.nan
        result["upper_confidence"] = np.nan
        result["lower_confidence"] = np.nan
        result["slope"] = np.nan
        result["r_squared"] = np.nan
        result["std_error"] = np.nan

        # Calculate rolling linear regression channels
        for i in range(min_periods, len(prices)):
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            
            price_window = prices.iloc[start_idx:end_idx].dropna()
            
            if len(price_window) >= min_periods:
                # Create time index for regression (0, 1, 2, ...)
                x_values = np.arange(len(price_window))
                y_values = price_window.values
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                
                # Calculate regression line at current point (end of window)
                current_x = len(price_window) - 1
                regression_value = slope * current_x + intercept
                
                # Calculate residuals and standard deviation
                predicted_values = slope * x_values + intercept
                residuals = y_values - predicted_values
                residual_std = np.std(residuals, ddof=2)  # ddof=2 for unbiased estimate
                
                # Calculate confidence intervals
                # t-value for confidence level
                degrees_freedom = len(x_values) - 2
                alpha = 1 - confidence_level
                t_value = stats.t.ppf(1 - alpha/2, degrees_freedom) if degrees_freedom > 0 else 2.0
                
                # Standard error of prediction at current point
                x_mean = np.mean(x_values)
                sum_sq_x = np.sum((x_values - x_mean)**2)
                se_prediction = std_err * np.sqrt(1 + 1/len(x_values) + (current_x - x_mean)**2 / sum_sq_x)
                
                # Calculate channels
                upper_channel = regression_value + std_dev_multiplier * residual_std
                lower_channel = regression_value - std_dev_multiplier * residual_std
                
                upper_confidence = regression_value + t_value * se_prediction
                lower_confidence = regression_value - t_value * se_prediction
                
                # Store results
                result.loc[prices.index[i], "regression_line"] = regression_value
                result.loc[prices.index[i], "upper_channel"] = upper_channel
                result.loc[prices.index[i], "lower_channel"] = lower_channel
                result.loc[prices.index[i], "upper_confidence"] = upper_confidence
                result.loc[prices.index[i], "lower_confidence"] = lower_confidence
                result.loc[prices.index[i], "slope"] = slope
                result.loc[prices.index[i], "r_squared"] = r_value**2
                result.loc[prices.index[i], "std_error"] = std_err

        # Store calculation details for analysis
        self._last_calculation = {
            "prices": prices,
            "result": result,
            "period": period,
            "std_dev_multiplier": std_dev_multiplier,
            "confidence_level": confidence_level,
            "min_periods": min_periods,
        }

        return result

    def get_channel_position(self, data: Union[pd.DataFrame, pd.Series], channels: pd.DataFrame = None) -> pd.Series:
        """
        Calculate position of price within the regression channels
        
        Args:
            data: Current price data
            channels: Pre-calculated channels (if None, will calculate)
            
        Returns:
            pd.Series: Position within channels (0 = lower channel, 1 = upper channel)
        """
        if channels is None:
            channels = self.calculate(data)
        
        # Handle input data
        if isinstance(data, pd.Series):
            prices = data
        elif isinstance(data, pd.DataFrame):
            prices = data["close"] if "close" in data.columns else data.iloc[:, 0]
        
        # Calculate relative position within channels
        upper = channels["upper_channel"]
        lower = channels["lower_channel"]
        
        # Position: 0 = at lower channel, 0.5 = at regression line, 1 = at upper channel
        channel_width = upper - lower
        position = (prices - lower) / channel_width
        
        return pd.Series(position, index=prices.index, name="channel_position")

    def get_breakout_signals(self, data: Union[pd.DataFrame, pd.Series], channels: pd.DataFrame = None) -> pd.Series:
        """
        Detect channel breakout signals
        
        Args:
            data: Current price data
            channels: Pre-calculated channels (if None, will calculate)
            
        Returns:
            pd.Series: Breakout signals (1 = upper breakout, -1 = lower breakout, 0 = within channels)
        """
        if channels is None:
            channels = self.calculate(data)
        
        # Handle input data
        if isinstance(data, pd.Series):
            prices = data
        elif isinstance(data, pd.DataFrame):
            prices = data["close"] if "close" in data.columns else data.iloc[:, 0]
        
        signals = pd.Series(0, index=prices.index, name="breakout_signals")
        
        # Upper breakout: price > upper channel
        upper_breakout = prices > channels["upper_channel"]
        signals[upper_breakout] = 1
        
        # Lower breakout: price < lower channel
        lower_breakout = prices < channels["lower_channel"]
        signals[lower_breakout] = -1
        
        return signals

    def validate_parameters(self) -> bool:
        """Validate Linear Regression Channels parameters"""
        period = self.parameters.get("period", 20)
        std_dev_multiplier = self.parameters.get("std_dev_multiplier", 2.0)
        confidence_level = self.parameters.get("confidence_level", 0.95)
        min_periods = self.parameters.get("min_periods", 10)

        if not isinstance(period, int) or period < 5:
            raise IndicatorValidationError(
                f"period must be integer >= 5, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(std_dev_multiplier, (int, float)) or std_dev_multiplier <= 0:
            raise IndicatorValidationError(
                f"std_dev_multiplier must be positive number, got {std_dev_multiplier}"
            )

        if not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
            raise IndicatorValidationError(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )

        if not isinstance(min_periods, int) or min_periods < 3:
            raise IndicatorValidationError(
                f"min_periods must be integer >= 3, got {min_periods}"
            )

        if min_periods > period:
            raise IndicatorValidationError(
                f"min_periods cannot exceed period, got min_periods={min_periods}, period={period}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Linear Regression Channels metadata as dictionary"""
        return {
            "name": "Linear Regression Channels",
            "category": self.CATEGORY,
            "description": "Dynamic channels around linear regression trend line with statistical confidence bands",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Linear Regression Channels requires price data"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Linear Regression Channels"""
        return self.parameters.get("min_periods", 10)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "std_dev_multiplier" not in self.parameters:
            self.parameters["std_dev_multiplier"] = 2.0
        if "confidence_level" not in self.parameters:
            self.parameters["confidence_level"] = 0.95
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
    def confidence_level(self) -> float:
        """Confidence level for backward compatibility"""
        return self.parameters.get("confidence_level", 0.95)

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
            "indicator": "LinearRegressionChannels",
            "period": self.period,
            "std_dev_multiplier": self.std_dev_multiplier,
            "confidence_level": self.confidence_level,
            "min_periods": self.min_periods,
            "category": self.CATEGORY,
        }
    
    def interpret_channel_signals(self, channels: pd.DataFrame, current_price: float, position: float) -> Dict[str, Any]:
        """
        Interpret channel signals and position
        
        Args:
            channels: Calculated channels DataFrame
            current_price: Current price value
            position: Current position within channels (0-1)
            
        Returns:
            Dict containing signal interpretation
        """
        if channels.empty or pd.isna(position):
            return {
                "signal": "insufficient_data",
                "interpretation": "Not enough data for channel analysis",
                "trend_strength": "unknown",
                "position_description": "unknown"
            }
        
        latest_data = channels.dropna().iloc[-1]
        regression_line = latest_data["regression_line"]
        upper_channel = latest_data["upper_channel"]
        lower_channel = latest_data["lower_channel"]
        slope = latest_data["slope"]
        r_squared = latest_data["r_squared"]
        
        # Trend strength assessment
        if r_squared > 0.8:
            trend_strength = "very_strong"
        elif r_squared > 0.6:
            trend_strength = "strong"
        elif r_squared > 0.4:
            trend_strength = "moderate"
        elif r_squared > 0.2:
            trend_strength = "weak"
        else:
            trend_strength = "very_weak"
        
        # Trend direction
        if slope > 0:
            trend_direction = "bullish"
        elif slope < 0:
            trend_direction = "bearish"
        else:
            trend_direction = "sideways"
        
        # Position analysis
        if position > 1.0:
            position_desc = "above_upper_channel"
            signal = "breakout_bullish" if slope > 0 else "resistance_test"
        elif position > 0.8:
            position_desc = "near_upper_channel"
            signal = "approaching_resistance"
        elif position > 0.6:
            position_desc = "upper_half"
            signal = "bullish_bias"
        elif position > 0.4:
            position_desc = "middle_range"
            signal = "neutral"
        elif position > 0.2:
            position_desc = "lower_half"
            signal = "bearish_bias"
        elif position > 0:
            position_desc = "near_lower_channel"
            signal = "approaching_support"
        else:
            position_desc = "below_lower_channel"
            signal = "breakout_bearish" if slope < 0 else "support_test"
        
        # Overall interpretation
        if trend_strength in ["very_strong", "strong"]:
            if position > 0.8:
                interpretation = f"Strong {trend_direction} trend with price near resistance"
            elif position < 0.2:
                interpretation = f"Strong {trend_direction} trend with price near support"
            else:
                interpretation = f"Strong {trend_direction} trend continuing"
        else:
            interpretation = f"Weak trend with price in {position_desc.replace('_', ' ')}"

        return {
            "signal": signal,
            "interpretation": interpretation,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "position_description": position_desc,
            "position_value": position,
            "r_squared": r_squared,
            "slope": slope,
            "current_price": current_price,
            "regression_line": regression_line,
            "distance_from_regression": abs(current_price - regression_line),
        }

    def get_trading_signals(self, channels: pd.DataFrame, breakout_signals: pd.Series) -> pd.Series:
        """
        Generate comprehensive trading signals based on channel analysis
        
        Args:
            channels: Calculated channels DataFrame
            breakout_signals: Breakout signals series
            
        Returns:
            pd.Series: Trading signals
        """
        signals = pd.Series("neutral", index=channels.index, name="trading_signals")
        
        if channels.empty:
            return signals
        
        # Get the slope to determine trend direction
        slope = channels["slope"]
        r_squared = channels["r_squared"]
        
        for i in range(1, len(signals)):
            current_breakout = breakout_signals.iloc[i] if i < len(breakout_signals) else 0
            current_slope = slope.iloc[i]
            current_r_squared = r_squared.iloc[i]
            
            # Only generate signals for strong trends
            if pd.notna(current_r_squared) and current_r_squared > 0.3:
                if current_breakout == 1:  # Upper breakout
                    if current_slope > 0:
                        signals.iloc[i] = "strong_buy"  # Bullish breakout in uptrend
                    else:
                        signals.iloc[i] = "resistance_test"  # Breakout in downtrend (could be reversal)
                elif current_breakout == -1:  # Lower breakout
                    if current_slope < 0:
                        signals.iloc[i] = "strong_sell"  # Bearish breakout in downtrend
                    else:
                        signals.iloc[i] = "support_test"  # Breakout in uptrend (could be reversal)
                else:
                    # Within channels - check for mean reversion opportunities
                    if current_slope > 0:
                        signals.iloc[i] = "trend_following_buy"
                    elif current_slope < 0:
                        signals.iloc[i] = "trend_following_sell"
        
        return signals


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return LinearRegressionChannels


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample trending data with noise
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    np.random.seed(42)
    
    # Create trending data with linear component and noise
    trend = np.linspace(100, 150, len(dates))  # Linear trend
    noise = np.random.normal(0, 2, len(dates))  # Random noise
    prices = trend + noise + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)  # Add cyclical component
    
    data = pd.DataFrame({
        "close": prices
    }, index=dates)
    
    # Calculate linear regression channels
    lr_channels = LinearRegressionChannels(period=30, std_dev_multiplier=2.0)
    channels = lr_channels.calculate(data)
    
    print("Linear Regression Channels Indicator Example:")
    
    # Display latest channel values
    latest_channels = channels.dropna().iloc[-1]
    print(f"Latest Regression Line: {latest_channels['regression_line']:.2f}")
    print(f"Upper Channel: {latest_channels['upper_channel']:.2f}")
    print(f"Lower Channel: {latest_channels['lower_channel']:.2f}")
    print(f"Slope: {latest_channels['slope']:.4f}")
    print(f"R-squared: {latest_channels['r_squared']:.3f}")
    
    # Get channel position
    position = lr_channels.get_channel_position(data, channels)
    latest_position = position.dropna().iloc[-1]
    print(f"Channel Position: {latest_position:.2f}")
    
    # Get breakout signals
    breakout_signals = lr_channels.get_breakout_signals(data, channels)
    latest_breakout = breakout_signals.iloc[-1]
    print(f"Breakout Signal: {latest_breakout}")
    
    # Interpret signals
    current_price = data["close"].iloc[-1]
    interpretation = lr_channels.interpret_channel_signals(channels, current_price, latest_position)
    print(f"\nInterpretation: {interpretation}")