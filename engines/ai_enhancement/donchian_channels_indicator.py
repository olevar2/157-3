"""
Donchian Channels Indicator

Donchian Channels are formed by taking the highest high and lowest low over a
specified period. The channels provide support and resistance levels and help
identify breakouts and trend direction.

Formula:
- Upper Channel = Highest high over period
- Lower Channel = Lowest low over period  
- Middle Channel = (Upper Channel + Lower Channel) / 2
- Channel Width = Upper Channel - Lower Channel
- Position = (Close - Lower Channel) / (Upper Channel - Lower Channel)

Interpretation:
- Price at upper channel: Potential resistance, overbought
- Price at lower channel: Potential support, oversold
- Breakout above upper channel: Bullish signal
- Breakdown below lower channel: Bearish signal
- Narrow channels: Low volatility, potential breakout
- Wide channels: High volatility

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


class DonchianChannelsIndicator(StandardIndicatorInterface):
    """
    Donchian Channels Indicator for breakout and trend analysis
    
    Donchian Channels help identify breakout levels and trend direction
    by tracking the highest highs and lowest lows over a specified period.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "trend"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        offset: int = 0,
        **kwargs,
    ):
        """
        Initialize Donchian Channels indicator

        Args:
            period: Period for channel calculation (default: 20)
            offset: Offset for channel lines (default: 0)
        """
        super().__init__(
            period=period,
            offset=offset,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Donchian Channels

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            pd.DataFrame: DataFrame with columns 'upper_channel', 'lower_channel', 'middle_channel', 'position'
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "Donchian Channels require DataFrame with 'high', 'low' columns"
            )

        period = self.parameters.get("period", 20)
        offset = self.parameters.get("offset", 0)
        
        high = data["high"]
        low = data["low"]
        close = data["close"] if "close" in data.columns else high  # Use high if no close
        
        # Calculate highest high and lowest low over period
        upper_channel = high.rolling(window=period, min_periods=period).max()
        lower_channel = low.rolling(window=period, min_periods=period).min()
        
        # Apply offset if specified
        if offset != 0:
            upper_channel = upper_channel.shift(offset)
            lower_channel = lower_channel.shift(offset)
        
        # Calculate middle channel
        middle_channel = (upper_channel + lower_channel) / 2
        
        # Calculate position within channel (0 = at lower, 1 = at upper)
        channel_width = upper_channel - lower_channel
        position = pd.Series(index=data.index, dtype=float)
        
        # Avoid division by zero
        valid_mask = channel_width > 0
        position[valid_mask] = (close[valid_mask] - lower_channel[valid_mask]) / channel_width[valid_mask]
        
        # Create result DataFrame
        result = pd.DataFrame({
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'middle_channel': middle_channel,
            'position': position
        }, index=data.index)
        
        # Store calculation details
        self._last_calculation = {
            "upper_channel": upper_channel,
            "lower_channel": lower_channel,
            "middle_channel": middle_channel,
            "position": position,
            "channel_width": channel_width,
            "period": period,
            "offset": offset,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate Donchian Channels parameters"""
        period = self.parameters.get("period", 20)
        offset = self.parameters.get("offset", 0)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if not isinstance(offset, int):
            raise IndicatorValidationError(
                f"offset must be integer, got {offset}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return Donchian Channels metadata"""
        return IndicatorMetadata(
            name="DonchianChannels",
            category=self.CATEGORY,
            description="Donchian Channels - Breakout levels based on highest/lowest prices",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """Donchian Channels require high and low prices"""
        return ["high", "low", "close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "offset" not in self.parameters:
            self.parameters["offset"] = 0

    @property
    def period(self) -> int:
        return self.parameters.get("period", 20)

    @property
    def offset(self) -> int:
        return self.parameters.get("offset", 0)

    @property
    def minimum_periods(self) -> int:
        return self.parameters.get("period", 20)

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "DonchianChannels",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }

# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return DonchianChannelsIndicator