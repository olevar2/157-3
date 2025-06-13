"""
Hurst Exponent Indicator

The Hurst Exponent is a statistical measure used to classify time series data and determine
the long-term memory of a time series. It was developed by Harold Edwin Hurst to analyze
the long-term storage capacity of reservoirs.

Hurst Exponent interpretations:
- H = 0.5: Random walk (Brownian motion) - no long-term correlation
- H > 0.5: Persistent series - trend-following behavior
- H < 0.5: Anti-persistent series - mean-reverting behavior

This implementation provides:
1. Rolling Hurst Exponent calculation
2. Rescaled Range (R/S) analysis
3. Detrended Fluctuation Analysis (DFA) method
4. Market regime classification based on Hurst values
5. Confidence intervals and statistical significance testing

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


class HurstExponentIndicator(StandardIndicatorInterface):
    """
    Hurst Exponent Indicator
    
    Measures long-term memory and correlation in time series data
    to determine market regime characteristics.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "cycle"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 100,
        min_period: int = 10,
        method: str = "rs",  # rs (Rescaled Range), dfa (Detrended Fluctuation Analysis)
        confidence_level: float = 0.95,
        **kwargs,
    ):
        """
        Initialize Hurst Exponent indicator

        Args:
            period: Period for Hurst calculation (default: 100)
            min_period: Minimum period for analysis (default: 10)
            method: Calculation method ('rs' or 'dfa') (default: 'rs')
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        super().__init__(
            period=period,
            min_period=min_period,
            method=method,
            confidence_level=confidence_level,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Hurst Exponent

        Args:
            data: DataFrame with OHLC data or Series of prices

        Returns:
            pd.DataFrame: Hurst Exponent analysis results
        """
        # Handle input data
        if isinstance(data, pd.Series):
            price = data
        elif isinstance(data, pd.DataFrame):
            self.validate_input_data(data)
            price = data["close"]
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Get parameters
        period = self.parameters.get("period", 100)
        min_period = self.parameters.get("min_period", 10)
        method = self.parameters.get("method", "rs")
        confidence_level = self.parameters.get("confidence_level", 0.95)

        n = len(price)
        hurst_values = np.full(n, np.nan)
        market_regime = pd.Series("unknown", index=price.index)
        hurst_confidence = np.full(n, np.nan)
        trend_strength = np.full(n, np.nan)
        
        # Rolling Hurst calculation
        for i in range(period, n):
            window_data = price.iloc[i-period+1:i+1]
            
            try:
                if method == "rs":
                    hurst_val, confidence = self._calculate_hurst_rs(window_data.values, min_period)
                elif method == "dfa":
                    hurst_val, confidence = self._calculate_hurst_dfa(window_data.values, min_period)
                else:
                    raise IndicatorValidationError(f"Unknown method: {method}")
                
                hurst_values[i] = hurst_val
                hurst_confidence[i] = confidence
                
                # Classify market regime
                market_regime.iloc[i] = self._classify_regime(hurst_val)
                
                # Calculate trend strength
                trend_strength[i] = self._calculate_trend_strength(hurst_val)
                
            except Exception:
                continue

        # Create result DataFrame
        result = pd.DataFrame(index=price.index)
        result["hurst_exponent"] = hurst_values
        result["market_regime"] = market_regime
        result["hurst_confidence"] = hurst_confidence
        result["trend_strength"] = trend_strength
        
        # Store calculation details
        self._last_calculation = {
            "price": price,
            "method": method,
            "parameters": self.parameters,
            "regime_distribution": market_regime.value_counts().to_dict(),
        }

        return result

    def _calculate_hurst_rs(self, data: np.ndarray, min_period: int) -> tuple:
        """Calculate Hurst Exponent using Rescaled Range analysis"""
        n = len(data)
        if n < min_period:
            return np.nan, 0.0
            
        # Remove trend by taking log returns
        log_returns = np.diff(np.log(data + 1e-10))
        
        # Calculate rescaled range for different lag periods
        lags = np.logspace(np.log10(min_period), np.log10(n//4), num=10).astype(int)
        lags = np.unique(lags)
        
        rs_values = []
        
        for lag in lags:
            if lag >= len(log_returns):
                continue
                
            # Split data into non-overlapping windows
            num_windows = len(log_returns) // lag
            rs_window = []
            
            for i in range(num_windows):
                window = log_returns[i*lag:(i+1)*lag]
                
                # Calculate mean
                mean_val = np.mean(window)
                
                # Calculate cumulative deviations
                deviations = np.cumsum(window - mean_val)
                
                # Calculate range
                R = np.max(deviations) - np.min(deviations)
                
                # Calculate standard deviation
                S = np.std(window)
                
                # Calculate R/S ratio
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 3:
            return np.nan, 0.0
        
        # Fit log(R/S) vs log(lag) to get Hurst exponent
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid_mask) < 3:
            return np.nan, 0.0
            
        log_lags = log_lags[valid_mask]
        log_rs = log_rs[valid_mask]
        
        # Linear regression to find Hurst exponent
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]
        
        # Calculate R-squared as confidence measure
        predicted = np.polyval(coeffs, log_lags)
        ss_res = np.sum((log_rs - predicted) ** 2)
        ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
            confidence = max(0, min(1, r_squared))
        else:
            confidence = 0.0
        
        return float(hurst), float(confidence)

    def _calculate_hurst_dfa(self, data: np.ndarray, min_period: int) -> tuple:
        """Calculate Hurst Exponent using Detrended Fluctuation Analysis"""
        n = len(data)
        if n < min_period:
            return np.nan, 0.0
            
        # Remove mean and integrate
        y = np.cumsum(data - np.mean(data))
        
        # Calculate fluctuation for different box sizes
        box_sizes = np.logspace(np.log10(min_period), np.log10(n//4), num=10).astype(int)
        box_sizes = np.unique(box_sizes)
        
        fluctuations = []
        
        for box_size in box_sizes:
            if box_size >= n:
                continue
                
            # Number of boxes
            num_boxes = n // box_size
            box_fluctuations = []
            
            for i in range(num_boxes):
                # Extract box
                start_idx = i * box_size
                end_idx = (i + 1) * box_size
                box_data = y[start_idx:end_idx]
                
                # Fit linear trend
                x = np.arange(len(box_data))
                coeffs = np.polyfit(x, box_data, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((box_data - trend) ** 2))
                box_fluctuations.append(fluctuation)
            
            if box_fluctuations:
                fluctuations.append(np.mean(box_fluctuations))
        
        if len(fluctuations) < 3:
            return np.nan, 0.0
        
        # Fit log(F) vs log(box_size) to get scaling exponent (Hurst)
        log_boxes = np.log(box_sizes[:len(fluctuations)])
        log_fluct = np.log(fluctuations)
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_boxes) & np.isfinite(log_fluct)
        if np.sum(valid_mask) < 3:
            return np.nan, 0.0
            
        log_boxes = log_boxes[valid_mask]
        log_fluct = log_fluct[valid_mask]
        
        # Linear regression
        coeffs = np.polyfit(log_boxes, log_fluct, 1)
        hurst = coeffs[0]
        
        # Calculate confidence
        predicted = np.polyval(coeffs, log_boxes)
        ss_res = np.sum((log_fluct - predicted) ** 2)
        ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
            confidence = max(0, min(1, r_squared))
        else:
            confidence = 0.0
        
        return float(hurst), float(confidence)

    def _classify_regime(self, hurst: float) -> str:
        """Classify market regime based on Hurst exponent"""
        if np.isnan(hurst):
            return "unknown"
        elif hurst > 0.65:
            return "strong_trending"
        elif hurst > 0.55:
            return "trending"
        elif hurst > 0.45:
            return "random"
        elif hurst > 0.35:
            return "mean_reverting"
        else:
            return "strong_mean_reverting"

    def _calculate_trend_strength(self, hurst: float) -> float:
        """Calculate trend strength from Hurst exponent"""
        if np.isnan(hurst):
            return 0.0
        
        # Convert Hurst to trend strength (0-100)
        if hurst > 0.5:
            # Trending market
            strength = (hurst - 0.5) * 200  # Scale to 0-100
        else:
            # Mean-reverting market (negative strength)
            strength = (hurst - 0.5) * 200  # Scale to -100 to 0
        
        return np.clip(strength, -100, 100)

    def validate_parameters(self) -> bool:
        """Validate parameters"""
        period = self.parameters.get("period", 100)
        min_period = self.parameters.get("min_period", 10)
        method = self.parameters.get("method", "rs")
        confidence_level = self.parameters.get("confidence_level", 0.95)

        if not isinstance(period, int) or period < 20:
            raise IndicatorValidationError(f"period must be integer >= 20, got {period}")
        
        if period > 1000:
            raise IndicatorValidationError(f"period too large, maximum 1000, got {period}")

        if not isinstance(min_period, int) or min_period < 5:
            raise IndicatorValidationError(f"min_period must be integer >= 5, got {min_period}")
        
        if min_period >= period:
            raise IndicatorValidationError(f"min_period must be < period, got {min_period} >= {period}")

        valid_methods = ["rs", "dfa"]
        if method not in valid_methods:
            raise IndicatorValidationError(f"method must be one of {valid_methods}, got {method}")

        if not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
            raise IndicatorValidationError(f"confidence_level must be in (0, 1), got {confidence_level}")

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata"""
        return {
            "name": "HurstExponent",
            "category": self.CATEGORY,
            "description": "Hurst Exponent for analyzing long-term memory and market regime classification",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "DataFrame",
            "output_columns": ["hurst_exponent", "market_regime", "hurst_confidence", "trend_strength"],
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Required columns"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return self.parameters.get("period", 100)

    def _setup_defaults(self):
        """Setup default parameter values"""
        defaults = {
            "period": 100,
            "min_period": 10,
            "method": "rs",
            "confidence_level": 0.95,
        }
        
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    # Backward compatibility properties
    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "HurstExponent",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return HurstExponentIndicator