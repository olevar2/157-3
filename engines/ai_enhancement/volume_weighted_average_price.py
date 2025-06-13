"""
Volume Weighted Average Price (VWAP)

VWAP is a trading benchmark that represents the average price weighted by volume.
It's particularly useful for institutional traders to assess execution quality
and identify support/resistance levels.

Formula:
VWAP = Σ(Typical Price × Volume) / Σ(Volume)
Where Typical Price = (High + Low + Close) / 3

Interpretation:
- Price above VWAP: Bullish bias (buyers in control)
- Price below VWAP: Bearish bias (sellers in control)
- VWAP as dynamic support/resistance level
- Large deviations from VWAP indicate potential reversal zones

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


class VolumeWeightedAveragePrice(StandardIndicatorInterface):
    """
    Volume Weighted Average Price for institutional trading analysis
    
    Provides fair value benchmark and identifies key support/resistance levels
    based on volume-weighted pricing.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "volume"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        std_dev_multiplier: float = 2.0,
        institutional_threshold: float = 1.5,
        reset_daily: bool = False,
        **kwargs,
    ):
        """
        Initialize VWAP indicator

        Args:
            period: Rolling period for VWAP calculation (default: 20)
            std_dev_multiplier: Standard deviation multiplier for bands (default: 2.0)
            institutional_threshold: Volume threshold for institutional moves (default: 1.5)
            reset_daily: Whether to reset VWAP daily (default: False)
        """
        super().__init__(
            period=period,
            std_dev_multiplier=std_dev_multiplier,
            institutional_threshold=institutional_threshold,
            reset_daily=reset_daily,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price

        Args:
            data: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            pd.DataFrame: DataFrame with VWAP, bands, and analysis
        """
        # Validate input data
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            raise IndicatorValidationError(
                "VWAP requires DataFrame with HLCV data"
            )

        period = self.parameters.get("period", 20)
        std_dev_multiplier = self.parameters.get("std_dev_multiplier", 2.0)
        institutional_threshold = self.parameters.get("institutional_threshold", 1.5)
        reset_daily = self.parameters.get("reset_daily", False)
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate volume-weighted values
        vw_price = typical_price * volume
        
        if reset_daily and hasattr(data.index, 'date'):
            # Daily reset VWAP (requires datetime index)
            vwap = pd.Series(index=data.index, dtype=float)
            vwap_std = pd.Series(index=data.index, dtype=float)
            
            for date in data.index.date:
                daily_mask = data.index.date == date
                daily_data = data[daily_mask]
                daily_typical = typical_price[daily_mask]
                daily_volume = volume[daily_mask]
                daily_vw_price = daily_typical * daily_volume
                
                # Cumulative VWAP for the day
                cum_vw_price = daily_vw_price.cumsum()
                cum_volume = daily_volume.cumsum()
                daily_vwap = cum_vw_price / cum_volume
                
                vwap[daily_mask] = daily_vwap
                
                # Calculate standard deviation for VWAP bands
                for i in range(len(daily_data)):
                    if i >= 1:
                        price_deviations = (daily_typical.iloc[:i+1] - daily_vwap.iloc[i]) ** 2
                        weighted_variance = (price_deviations * daily_volume.iloc[:i+1]).sum() / cum_volume.iloc[i]
                        vwap_std.iloc[daily_data.index[i]] = np.sqrt(weighted_variance)
        else:
            # Rolling period VWAP
            vwap = pd.Series(index=data.index, dtype=float)
            vwap_std = pd.Series(index=data.index, dtype=float)
            
            for i in range(len(data)):
                start_idx = max(0, i - period + 1)
                period_vw_price = vw_price.iloc[start_idx:i+1]
                period_volume = volume.iloc[start_idx:i+1]
                period_typical = typical_price.iloc[start_idx:i+1]
                
                if period_volume.sum() > 0:
                    vwap.iloc[i] = period_vw_price.sum() / period_volume.sum()
                    
                    # Calculate weighted standard deviation
                    if len(period_typical) > 1:
                        price_deviations = (period_typical - vwap.iloc[i]) ** 2
                        weighted_variance = (price_deviations * period_volume).sum() / period_volume.sum()
                        vwap_std.iloc[i] = np.sqrt(weighted_variance)
        
        # Calculate VWAP bands
        vwap_upper = vwap + (vwap_std * std_dev_multiplier)
        vwap_lower = vwap - (vwap_std * std_dev_multiplier)
        
        # Calculate relative position to VWAP
        vwap_position = (close - vwap) / vwap * 100  # Percentage above/below VWAP
        
        # Volume analysis
        volume_ma = volume.rolling(window=period).mean()
        relative_volume = volume / volume_ma
        
        # Institutional activity detection
        institutional_moves = (
            (relative_volume > institutional_threshold) & 
            (abs(vwap_position) > vwap_position.rolling(window=period).std())
        )
        
        # Price vs VWAP signals
        above_vwap = close > vwap
        below_vwap = close < vwap
        
        # VWAP crossover detection
        vwap_cross_up = (close > vwap) & (close.shift(1) <= vwap.shift(1))
        vwap_cross_down = (close < vwap) & (close.shift(1) >= vwap.shift(1))
        
        # VWAP band signals
        upper_band_touch = (high >= vwap_upper) & (close < vwap_upper)
        lower_band_touch = (low <= vwap_lower) & (close > vwap_lower)
        
        # VWAP momentum analysis
        vwap_slope = vwap.diff(periods=5)
        vwap_trend = pd.Series(index=data.index, dtype=str)
        vwap_trend[vwap_slope > 0] = "uptrend"
        vwap_trend[vwap_slope < 0] = "downtrend"
        vwap_trend[vwap_slope == 0] = "sideways"
        
        # Fair value analysis
        fair_value_premium = vwap_position
        overvalued = fair_value_premium > fair_value_premium.quantile(0.8)
        undervalued = fair_value_premium < fair_value_premium.quantile(0.2)
        
        # Volume profile analysis around VWAP
        near_vwap = abs(vwap_position) < 1.0  # Within 1% of VWAP
        vwap_acceptance = near_vwap & (relative_volume > 1.0)
        vwap_rejection = (abs(vwap_position) > 2.0) & (relative_volume > institutional_threshold)
        
        # Generate comprehensive trading signals
        signals = pd.Series(index=data.index, dtype=str)
        signals[:] = "neutral"
        
        # Strong signals with institutional volume
        strong_bullish = (
            vwap_cross_up & (relative_volume > institutional_threshold) & 
            (vwap_slope > 0)
        )
        strong_bearish = (
            vwap_cross_down & (relative_volume > institutional_threshold) & 
            (vwap_slope < 0)
        )
        
        signals[strong_bullish] = "strong_buy"
        signals[strong_bearish] = "strong_sell"
        
        # Band bounce signals
        signals[lower_band_touch & (relative_volume > 1.2)] = "buy_oversold"
        signals[upper_band_touch & (relative_volume > 1.2)] = "sell_overbought"
        
        # Trend following signals
        signals[(above_vwap & (vwap_slope > 0) & (relative_volume > 1.0)) & ~strong_bullish] = "buy"
        signals[(below_vwap & (vwap_slope < 0) & (relative_volume > 1.0)) & ~strong_bearish] = "sell"
        
        # Fair value signals
        signals[undervalued & vwap_acceptance] = "accumulate"
        signals[overvalued & vwap_rejection] = "distribute"
        
        # Create result DataFrame
        result = pd.DataFrame({
            'vwap': vwap,
            'vwap_upper': vwap_upper,
            'vwap_lower': vwap_lower,
            'vwap_std': vwap_std,
            'vwap_position': vwap_position,
            'vwap_slope': vwap_slope,
            'vwap_trend': vwap_trend,
            'relative_volume': relative_volume,
            'fair_value_premium': fair_value_premium,
            'above_vwap': above_vwap.astype(int),
            'below_vwap': below_vwap.astype(int),
            'vwap_cross_up': vwap_cross_up.astype(int),
            'vwap_cross_down': vwap_cross_down.astype(int),
            'upper_band_touch': upper_band_touch.astype(int),
            'lower_band_touch': lower_band_touch.astype(int),
            'institutional_moves': institutional_moves.astype(int),
            'overvalued': overvalued.astype(int),
            'undervalued': undervalued.astype(int),
            'vwap_acceptance': vwap_acceptance.astype(int),
            'vwap_rejection': vwap_rejection.astype(int),
            'signals': signals
        }, index=data.index)
        
        # Store calculation details for analysis
        self._last_calculation = {
            "vwap": vwap,
            "final_vwap": vwap.iloc[-1] if len(vwap) > 0 else 0,
            "final_position": vwap_position.iloc[-1] if len(vwap_position) > 0 else 0,
            "signals": signals,
            "period": period,
            "trend": vwap_trend.iloc[-1] if len(vwap_trend) > 0 else "sideways",
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate VWAP parameters"""
        period = self.parameters.get("period", 20)
        std_dev_multiplier = self.parameters.get("std_dev_multiplier", 2.0)
        institutional_threshold = self.parameters.get("institutional_threshold", 1.5)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be integer >= 1, got {period}"
            )

        if period > 500:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 500, got {period}"
            )

        if not isinstance(std_dev_multiplier, (int, float)) or std_dev_multiplier <= 0:
            raise IndicatorValidationError(
                f"std_dev_multiplier must be positive number, got {std_dev_multiplier}"
            )

        if not isinstance(institutional_threshold, (int, float)) or institutional_threshold <= 0:
            raise IndicatorValidationError(
                f"institutional_threshold must be positive number, got {institutional_threshold}"
            )

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return VWAP metadata"""
        return IndicatorMetadata(
            name="VolumeWeightedAveragePrice", 
            category=self.CATEGORY,
            description="Volume Weighted Average Price - Fair value benchmark and S/R levels",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """VWAP requires HLCV data"""
        return ["high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for VWAP calculation"""
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "std_dev_multiplier" not in self.parameters:
            self.parameters["std_dev_multiplier"] = 2.0
        if "institutional_threshold" not in self.parameters:
            self.parameters["institutional_threshold"] = 1.5
        if "reset_daily" not in self.parameters:
            self.parameters["reset_daily"] = False

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
    def minimum_periods(self) -> int:
        """Minimum periods required"""
        return self.parameters.get("period", 20)

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "VolumeWeightedAveragePrice",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return VolumeWeightedAveragePrice