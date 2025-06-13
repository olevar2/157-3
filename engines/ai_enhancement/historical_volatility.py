"""
Historical Volatility

Historical Volatility measures the rate of price change over a specified period,
typically expressed as an annualized standard deviation of returns.

Formula:
HV = STDEV(LN(Close/Previous Close)) * SQRT(252) * 100

Where:
- LN = Natural logarithm
- 252 = Typical trading days per year
- Result is expressed as percentage

Interpretation:
- High HV: High price volatility period
- Low HV: Low price volatility period
- Rising HV: Increasing uncertainty/risk
- Falling HV: Decreasing uncertainty/risk

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


class HistoricalVolatility(StandardIndicatorInterface):
    """
    Historical Volatility for risk measurement and regime detection
    
    Calculates annualized volatility from price returns for risk assessment
    and volatility regime identification.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "volatility"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        trading_days: int = 252,
        return_type: str = "log",
        regime_threshold: float = 1.5,
        **kwargs,
    ):
        """
        Initialize Historical Volatility indicator

        Args:
            period: Period for volatility calculation (default: 20)
            trading_days: Trading days per year for annualization (default: 252)
            return_type: Type of returns 'log' or 'simple' (default: 'log')
            regime_threshold: Threshold for regime detection (default: 1.5)
        """
        super().__init__(
            period=period,
            trading_days=trading_days,
            return_type=return_type,
            regime_threshold=regime_threshold,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Calculate Historical Volatility

        Args:
            data: DataFrame with 'close' column or Series of prices

        Returns:
            pd.DataFrame: DataFrame with Historical Volatility and analysis
        """
        self.validate_input_data(data)
        
        if isinstance(data, pd.Series):
            close = data
            index = data.index
        else:
            if 'close' not in data.columns:
                raise IndicatorValidationError("Historical Volatility requires 'close' column")
            close = data['close']
            index = data.index

        period = self.parameters.get("period", 20)
        trading_days = self.parameters.get("trading_days", 252)
        return_type = self.parameters.get("return_type", "log")
        regime_threshold = self.parameters.get("regime_threshold", 1.5)
        
        # Calculate returns
        if return_type.lower() == "log":
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=period, min_periods=period//2).std()
        
        # Annualize volatility
        historical_volatility = volatility * np.sqrt(trading_days) * 100
        
        # Calculate volatility percentiles for regime detection
        vol_mean = historical_volatility.rolling(window=period*3, min_periods=period).mean()
        vol_std = historical_volatility.rolling(window=period*3, min_periods=period).std()
        
        # Volatility regimes
        high_volatility = historical_volatility > (vol_mean + vol_std * regime_threshold)
        low_volatility = historical_volatility < (vol_mean - vol_std * regime_threshold)
        
        # Volatility momentum
        vol_momentum = historical_volatility.diff(periods=5)
        vol_acceleration = vol_momentum.diff(periods=3)
        
        # Generate signals
        signals = pd.Series(index=index, dtype=str)
        signals[:] = "normal_volatility"
        
        signals[high_volatility] = "high_volatility"
        signals[low_volatility] = "low_volatility"
        
        # Regime change signals
        vol_spike = (historical_volatility > historical_volatility.shift(1) * 1.5) & high_volatility
        vol_crush = (historical_volatility < historical_volatility.shift(1) * 0.7) & low_volatility
        
        signals[vol_spike] = "volatility_spike"
        signals[vol_crush] = "volatility_crush"
        
        # Create result DataFrame
        result = pd.DataFrame({
            'historical_volatility': historical_volatility,
            'returns': returns,
            'vol_mean': vol_mean,
            'vol_std': vol_std,
            'vol_momentum': vol_momentum,
            'vol_acceleration': vol_acceleration,
            'high_volatility': high_volatility.astype(int),
            'low_volatility': low_volatility.astype(int),
            'signals': signals
        }, index=index)
        
        self._last_calculation = {
            "historical_volatility": historical_volatility,
            "final_value": historical_volatility.iloc[-1] if len(historical_volatility) > 0 else 0,
            "signals": signals,
            "period": period,
        }

        return result

    def validate_parameters(self) -> bool:
        """Validate Historical Volatility parameters"""
        period = self.parameters.get("period", 20)
        trading_days = self.parameters.get("trading_days", 252)
        return_type = self.parameters.get("return_type", "log")
        regime_threshold = self.parameters.get("regime_threshold", 1.5)

        if not isinstance(period, int) or period < 2:
            raise IndicatorValidationError(f"period must be integer >= 2, got {period}")
        
        if not isinstance(trading_days, int) or trading_days < 1:
            raise IndicatorValidationError(f"trading_days must be positive integer, got {trading_days}")
            
        if return_type.lower() not in ["log", "simple"]:
            raise IndicatorValidationError(f"return_type must be 'log' or 'simple', got {return_type}")
            
        if not isinstance(regime_threshold, (int, float)) or regime_threshold <= 0:
            raise IndicatorValidationError(f"regime_threshold must be positive number, got {regime_threshold}")

        return True

    def get_metadata(self) -> IndicatorMetadata:
        """Return Historical Volatility metadata"""
        return IndicatorMetadata(
            name="HistoricalVolatility",
            category=self.CATEGORY,
            description="Historical Volatility - Annualized price volatility measurement",
            parameters=self.parameters,
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points(),
        )

    def _get_required_columns(self) -> List[str]:
        """Historical Volatility requires close prices"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "trading_days" not in self.parameters:
            self.parameters["trading_days"] = 252
        if "return_type" not in self.parameters:
            self.parameters["return_type"] = "log"
        if "regime_threshold" not in self.parameters:
            self.parameters["regime_threshold"] = 1.5

    @property
    def period(self) -> int:
        return self.parameters.get("period", 20)

    @property
    def trading_days(self) -> int:
        return self.parameters.get("trading_days", 252)

    @property
    def minimum_periods(self) -> int:
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "name": "HistoricalVolatility",
            "category": self.CATEGORY,
            "parameters": self.parameters,
            "version": self.VERSION,
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return HistoricalVolatility