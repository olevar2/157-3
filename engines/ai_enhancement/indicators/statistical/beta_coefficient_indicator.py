"""
Beta Coefficient Indicator

Beta measures the volatility of a security or portfolio in relation to the market as a whole.
Beta is calculated using regression analysis and represents the slope of the line of best fit
for each Re (return of stock) vs. Rm (return of market) data point.

Formula:
Beta = Covariance(Ra, Rm) / Variance(Rm)
Where:
- Ra = Return of the asset
- Rm = Return of the market/benchmark
- Beta = 1: Asset moves with the market
- Beta > 1: Asset is more volatile than market
- Beta < 1: Asset is less volatile than market
- Beta < 0: Asset moves opposite to market

Applications:
- Portfolio risk assessment
- Capital Asset Pricing Model (CAPM)
- Risk-adjusted performance measurement
- Hedging strategy development

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Import the base indicator interface
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class BetaCoefficientIndicator(StandardIndicatorInterface):
    """
    Beta Coefficient Indicator
    
    Measures the systematic risk and volatility of an asset relative to a benchmark.
    Beta indicates how much an asset's price moves relative to market movements.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "statistical"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 252,  # 1 year of daily data
        lookback_returns: int = 21,  # Period for return calculation
        min_periods: int = 50,  # Minimum periods for calculation
        **kwargs,
    ):
        """
        Initialize Beta Coefficient indicator

        Args:
            period: Period for beta calculation (default: 252 trading days)
            lookback_returns: Period for return calculation (default: 21 days)
            min_periods: Minimum periods required for calculation (default: 50)
        """
        super().__init__(
            period=period,
            lookback_returns=lookback_returns,
            min_periods=min_periods,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series], benchmark: Union[pd.DataFrame, pd.Series] = None) -> pd.Series:
        """
        Calculate Beta Coefficient
        
        Args:
            data: Asset price data (DataFrame with 'close' or Series of prices)
            benchmark: Benchmark price data (DataFrame with 'close' or Series of prices)
                      If None, uses market proxy estimation
        
        Returns:
            pd.Series: Beta coefficient values with same index as input data
        """
        # Handle asset data
        if isinstance(data, pd.Series):
            asset_prices = data
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                asset_prices = data["close"]
                self.validate_input_data(data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Handle benchmark data
        if benchmark is None:
            # Create synthetic market benchmark based on asset data
            # Use smoothed moving average as market proxy
            market_prices = asset_prices.rolling(window=50, min_periods=25).mean()
        elif isinstance(benchmark, pd.Series):
            market_prices = benchmark
        elif isinstance(benchmark, pd.DataFrame):
            if "close" in benchmark.columns:
                market_prices = benchmark["close"]
            else:
                raise IndicatorValidationError(
                    "Benchmark DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Benchmark must be DataFrame or Series")

        period = self.parameters.get("period", 252)
        lookback_returns = self.parameters.get("lookback_returns", 21)
        min_periods = self.parameters.get("min_periods", 50)

        # Calculate returns
        asset_returns = asset_prices.pct_change(periods=lookback_returns)
        market_returns = market_prices.pct_change(periods=lookback_returns)

        # Align indices
        common_index = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns.reindex(common_index)
        market_returns = market_returns.reindex(common_index)

        # Calculate rolling beta
        beta_values = pd.Series(index=asset_returns.index, dtype=float)

        for i in range(min_periods, len(asset_returns)):
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            
            asset_window = asset_returns.iloc[start_idx:end_idx].dropna()
            market_window = market_returns.iloc[start_idx:end_idx].dropna()
            
            # Ensure we have enough data points
            if len(asset_window) >= min_periods and len(market_window) >= min_periods:
                # Align the windows
                common_dates = asset_window.index.intersection(market_window.index)
                if len(common_dates) >= min_periods:
                    asset_aligned = asset_window.reindex(common_dates)
                    market_aligned = market_window.reindex(common_dates)
                    
                    # Remove any remaining NaN values
                    valid_mask = ~(asset_aligned.isna() | market_aligned.isna())
                    asset_clean = asset_aligned[valid_mask]
                    market_clean = market_aligned[valid_mask]
                    
                    if len(asset_clean) >= min_periods:
                        # Calculate beta using covariance and variance
                        covariance = np.cov(asset_clean, market_clean)[0, 1]
                        market_variance = np.var(market_clean, ddof=1)
                        
                        if market_variance != 0:
                            beta = covariance / market_variance
                            
                            # Apply statistical confidence bounds
                            if abs(beta) > 10:  # Sanity check for extreme values
                                beta = np.sign(beta) * 10
                                
                            beta_values.iloc[i] = beta
                        else:
                            beta_values.iloc[i] = 1.0  # Default beta when market variance is zero

        # Store calculation details for analysis
        self._last_calculation = {
            "asset_returns": asset_returns,
            "market_returns": market_returns,
            "beta": beta_values,
            "period": period,
            "lookback_returns": lookback_returns,
            "min_periods": min_periods,
            "covariance": covariance if 'covariance' in locals() else None,
            "market_variance": market_variance if 'market_variance' in locals() else None,
        }

        return pd.Series(beta_values, index=asset_returns.index, name="Beta")

    def validate_parameters(self) -> bool:
        """Validate Beta parameters"""
        period = self.parameters.get("period", 252)
        lookback_returns = self.parameters.get("lookback_returns", 21)
        min_periods = self.parameters.get("min_periods", 50)

        if not isinstance(period, int) or period < 10:
            raise IndicatorValidationError(
                f"period must be integer >= 10, got {period}"
            )

        if period > 2000:
            raise IndicatorValidationError(
                f"period too large, maximum 2000, got {period}"
            )

        if not isinstance(lookback_returns, int) or lookback_returns < 1:
            raise IndicatorValidationError(
                f"lookback_returns must be positive integer, got {lookback_returns}"
            )

        if not isinstance(min_periods, int) or min_periods < 5:
            raise IndicatorValidationError(
                f"min_periods must be integer >= 5, got {min_periods}"
            )

        if min_periods > period:
            raise IndicatorValidationError(
                f"min_periods cannot exceed period, got min_periods={min_periods}, period={period}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Beta metadata as dictionary"""
        return {
            "name": "Beta Coefficient",
            "category": self.CATEGORY,
            "description": "Beta measures systematic risk and volatility relative to market benchmark",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Beta requires price data"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Beta calculation"""
        return self.parameters.get("min_periods", 50)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 252
        if "lookback_returns" not in self.parameters:
            self.parameters["lookback_returns"] = 21
        if "min_periods" not in self.parameters:
            self.parameters["min_periods"] = 50

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 252)

    @property
    def lookback_returns(self) -> int:
        """Lookback returns period for backward compatibility"""
        return self.parameters.get("lookback_returns", 21)

    @property
    def min_periods(self) -> int:
        """Minimum periods for backward compatibility"""
        return self.parameters.get("min_periods", 50)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods property for compatibility"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "BetaCoefficient",
            "period": self.period,
            "lookback_returns": self.lookback_returns,
            "min_periods": self.min_periods,
            "category": self.CATEGORY,
        }

    def interpret_beta(self, beta_value: float) -> Dict[str, Any]:
        """
        Interpret beta coefficient value
        
        Args:
            beta_value: Beta coefficient value
            
        Returns:
            Dict containing interpretation details
        """
        if pd.isna(beta_value):
            return {
                "signal": "insufficient_data",
                "interpretation": "Not enough data for beta calculation",
                "risk_level": "unknown",
                "market_correlation": "unknown"
            }
        
        if beta_value > 1.5:
            risk_level = "very_high"
            interpretation = "Asset is highly volatile relative to market"
        elif beta_value > 1.0:
            risk_level = "high"
            interpretation = "Asset is more volatile than market"
        elif beta_value > 0.5:
            risk_level = "moderate"
            interpretation = "Asset has moderate volatility relative to market"
        elif beta_value > 0:
            risk_level = "low"
            interpretation = "Asset is less volatile than market"
        elif beta_value > -0.5:
            risk_level = "low_negative"
            interpretation = "Asset moves slightly opposite to market"
        else:
            risk_level = "high_negative"
            interpretation = "Asset strongly moves opposite to market"

        # Market correlation assessment
        if abs(beta_value) > 1.5:
            correlation = "strong"
        elif abs(beta_value) > 0.7:
            correlation = "moderate"
        elif abs(beta_value) > 0.3:
            correlation = "weak"
        else:
            correlation = "very_weak"

        return {
            "signal": "normal",
            "beta_value": beta_value,
            "interpretation": interpretation,
            "risk_level": risk_level,
            "market_correlation": correlation,
            "directional": "positive" if beta_value > 0 else "negative"
        }


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return BetaCoefficientIndicator


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Create sample data
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    np.random.seed(42)
    
    # Asset with beta > 1 (more volatile than market)
    market_returns = np.random.normal(0.001, 0.02, len(dates))
    asset_returns = 0.8 * market_returns + np.random.normal(0, 0.01, len(dates))
    
    market_prices = 100 * (1 + market_returns).cumprod()
    asset_prices = 100 * (1 + asset_returns).cumprod()
    
    data = pd.DataFrame({
        "close": asset_prices
    }, index=dates)
    
    benchmark = pd.DataFrame({
        "close": market_prices
    }, index=dates)
    
    # Calculate beta
    beta_indicator = BetaCoefficientIndicator(period=100, lookback_returns=5)
    beta_values = beta_indicator.calculate(data, benchmark)
    
    print("Beta Coefficient Indicator Example:")
    print(f"Final Beta: {beta_values.dropna().iloc[-1]:.3f}")
    print(f"Mean Beta: {beta_values.dropna().mean():.3f}")
    print(f"Beta Std: {beta_values.dropna().std():.3f}")
    
    # Interpret the latest beta value
    latest_beta = beta_values.dropna().iloc[-1]
    interpretation = beta_indicator.interpret_beta(latest_beta)
    print(f"\nInterpretation: {interpretation}")