"""
Cointegration Indicator

Cointegration tests whether two or more time series have a long-run equilibrium relationship.
This is crucial for pairs trading, statistical arbitrage, and risk management.

The Engle-Granger two-step method:
1. Test if both series are integrated of order 1 (I(1)) using ADF test
2. Estimate the cointegrating regression: Y = α + βX + ε
3. Test if residuals are stationary using ADF test on residuals
4. If residuals are stationary, series are cointegrated

Johansen test for multiple series:
- Tests for multiple cointegrating relationships
- Provides trace and eigenvalue statistics
- More robust for multiple time series

Applications:
- Pairs trading strategy development
- Portfolio risk management
- Long-term equilibrium relationship detection
- Statistical arbitrage opportunities

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
from typing import Any, Dict, List, Union, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Import the base indicator interface
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class CointegrationIndicator(StandardIndicatorInterface):
    """
    Cointegration Indicator
    
    Tests for cointegration between two or more time series using various statistical tests.
    Provides rolling cointegration analysis for dynamic relationship monitoring.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "statistical"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 252,  # 1 year of daily data
        method: str = "engle_granger",  # "engle_granger" or "johansen"
        confidence_level: float = 0.05,  # 5% significance level
        min_periods: int = 100,  # Minimum periods for reliable test
        **kwargs,
    ):
        """
        Initialize Cointegration indicator

        Args:
            period: Period for cointegration analysis (default: 252 trading days)
            method: Method for cointegration test ("engle_granger" or "johansen")
            confidence_level: Significance level for statistical tests (default: 0.05)
            min_periods: Minimum periods required for calculation (default: 100)
        """
        super().__init__(
            period=period,
            method=method,
            confidence_level=confidence_level,
            min_periods=min_periods,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series], reference_series: Union[pd.DataFrame, pd.Series] = None) -> pd.Series:
        """
        Calculate Cointegration test statistics
        
        Args:
            data: Primary time series (DataFrame with 'close' or Series of prices)
            reference_series: Reference time series for cointegration test
                             If None, uses autocorrelation-based cointegration
        
        Returns:
            pd.Series: Cointegration test statistics with same index as input data
        """
        # Handle primary data
        if isinstance(data, pd.Series):
            primary_series = data
        elif isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                primary_series = data["close"]
                self.validate_input_data(data)
            else:
                raise IndicatorValidationError(
                    "DataFrame must contain 'close' column"
                )
        else:
            raise IndicatorValidationError("Data must be DataFrame or Series")

        # Handle reference series
        if reference_series is None:
            # Create synthetic reference series based on lagged values
            reference_series = primary_series.shift(1)
        elif isinstance(reference_series, pd.Series):
            pass  # Already in correct format
        elif isinstance(reference_series, pd.DataFrame):
            if "close" in reference_series.columns:
                reference_series = reference_series["close"]
            else:
                raise IndicatorValidationError(
                    "Reference DataFrame must contain 'close' column"
                )

        period = self.parameters.get("period", 252)
        method = self.parameters.get("method", "engle_granger")
        confidence_level = self.parameters.get("confidence_level", 0.05)
        min_periods = self.parameters.get("min_periods", 100)

        # Align series
        common_index = primary_series.index.intersection(reference_series.index)
        primary_aligned = primary_series.reindex(common_index)
        reference_aligned = reference_series.reindex(common_index)

        # Calculate rolling cointegration
        cointegration_stats = pd.Series(index=primary_aligned.index, dtype=float)
        p_values = pd.Series(index=primary_aligned.index, dtype=float)
        cointegration_vectors = pd.Series(index=primary_aligned.index, dtype=object)

        for i in range(min_periods, len(primary_aligned)):
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            
            primary_window = primary_aligned.iloc[start_idx:end_idx].dropna()
            reference_window = reference_aligned.iloc[start_idx:end_idx].dropna()
            
            # Ensure we have enough data points
            if len(primary_window) >= min_periods and len(reference_window) >= min_periods:
                # Align the windows
                common_dates = primary_window.index.intersection(reference_window.index)
                if len(common_dates) >= min_periods:
                    y = primary_window.reindex(common_dates)
                    x = reference_window.reindex(common_dates)
                    
                    # Remove any remaining NaN values
                    valid_mask = ~(y.isna() | x.isna())
                    y_clean = y[valid_mask]
                    x_clean = x[valid_mask]
                    
                    if len(y_clean) >= min_periods:
                        if method == "engle_granger":
                            stat, p_value, beta = self._engle_granger_test(y_clean, x_clean)
                            cointegration_stats.iloc[i] = stat
                            p_values.iloc[i] = p_value
                            cointegration_vectors.iloc[i] = [1, -beta]  # Cointegrating vector
                        elif method == "johansen":
                            stat, p_value = self._simplified_johansen_test(y_clean, x_clean)
                            cointegration_stats.iloc[i] = stat
                            p_values.iloc[i] = p_value

        # Store calculation details for analysis
        self._last_calculation = {
            "primary_series": primary_aligned,
            "reference_series": reference_aligned,
            "cointegration_stats": cointegration_stats,
            "p_values": p_values,
            "cointegration_vectors": cointegration_vectors,
            "method": method,
            "period": period,
            "confidence_level": confidence_level,
            "min_periods": min_periods,
        }

        return pd.Series(cointegration_stats, index=primary_aligned.index, name="Cointegration")

    def _engle_granger_test(self, y: pd.Series, x: pd.Series) -> Tuple[float, float, float]:
        """
        Perform Engle-Granger cointegration test
        
        Args:
            y: Dependent variable series
            x: Independent variable series
            
        Returns:
            Tuple of (test_statistic, p_value, beta_coefficient)
        """
        try:
            # Step 1: Estimate cointegrating regression Y = α + βX + ε
            y_values = y.values
            x_values = x.values
            
            # Add constant term
            X = np.column_stack([np.ones(len(x_values)), x_values])
            
            # OLS regression
            beta_hat = np.linalg.lstsq(X, y_values, rcond=None)[0]
            alpha, beta = beta_hat[0], beta_hat[1]
            
            # Calculate residuals
            residuals = y_values - (alpha + beta * x_values)
            
            # Step 2: Test stationarity of residuals using ADF test
            adf_stat, p_value = self._augmented_dickey_fuller_test(residuals)
            
            return adf_stat, p_value, beta
            
        except Exception as e:
            # Return NaN values on calculation error
            return np.nan, 1.0, np.nan

    def _simplified_johansen_test(self, y: pd.Series, x: pd.Series) -> Tuple[float, float]:
        """
        Simplified Johansen cointegration test for two series
        
        Args:
            y: First time series
            x: Second time series
            
        Returns:
            Tuple of (trace_statistic, p_value)
        """
        try:
            # Convert to numpy arrays
            y_values = y.values
            x_values = x.values
            
            # Create VAR(1) system
            data_matrix = np.column_stack([y_values, x_values])
            
            # Calculate differences for VECM
            diff_y = np.diff(y_values)
            diff_x = np.diff(x_values)
            
            # Lagged levels
            y_lag = y_values[:-1]
            x_lag = x_values[:-1]
            
            # Estimate VECM components
            # This is a simplified version - full Johansen test requires eigenvalue decomposition
            residuals_y = diff_y - np.mean(diff_y)
            residuals_x = diff_x - np.mean(diff_x)
            
            # Calculate trace statistic approximation
            cov_matrix = np.cov(residuals_y, residuals_x)
            trace_stat = -len(residuals_y) * np.log(np.linalg.det(cov_matrix))
            
            # Approximate p-value (simplified)
            # In practice, critical values depend on number of variables and trend assumptions
            critical_5pct = 15.49  # Approximate critical value for 2 variables, no trend
            p_value = 0.01 if trace_stat > critical_5pct else 0.10
            
            return trace_stat, p_value
            
        except Exception as e:
            return np.nan, 1.0

    def _augmented_dickey_fuller_test(self, series: np.ndarray, maxlag: int = None) -> Tuple[float, float]:
        """
        Simplified Augmented Dickey-Fuller test for unit root
        
        Args:
            series: Time series to test
            maxlag: Maximum number of lags to include
            
        Returns:
            Tuple of (adf_statistic, p_value)
        """
        try:
            if maxlag is None:
                maxlag = min(12, len(series) // 4)
            
            # ADF regression: Δy_t = α + γy_{t-1} + Σβ_i Δy_{t-i} + ε_t
            y = series[maxlag:]
            dy = np.diff(series)[maxlag-1:]
            y_lag = series[maxlag-1:-1]
            
            # Add lagged differences
            X = [y_lag]
            for i in range(1, maxlag):
                if maxlag - i < len(dy):
                    lag_dy = dy[maxlag-i-1:-i] if i > 0 else dy[maxlag-i-1:]
                    if len(lag_dy) == len(y):
                        X.append(lag_dy)
            
            X = np.column_stack(X)
            
            # Add constant
            X = np.column_stack([np.ones(len(X)), X])
            
            # OLS regression
            if len(X) > 0 and len(y) == len(X):
                beta = np.linalg.lstsq(X, dy[:len(X)], rcond=None)[0]
                
                # Test statistic is t-statistic for γ coefficient
                residuals = dy[:len(X)] - X @ beta
                mse = np.sum(residuals**2) / (len(residuals) - len(beta))
                
                # Standard error of γ coefficient (second coefficient after constant)
                if len(beta) > 1:
                    X_gamma = X[:, 1]  # y_lag column
                    se_gamma = np.sqrt(mse / np.sum((X_gamma - np.mean(X_gamma))**2))
                    
                    # ADF test statistic
                    adf_stat = beta[1] / se_gamma
                    
                    # Approximate p-value using normal distribution
                    # (In practice, uses MacKinnon critical values)
                    p_value = 2 * (1 - stats.norm.cdf(abs(adf_stat)))
                    
                    return adf_stat, p_value
            
            return np.nan, 1.0
            
        except Exception as e:
            return np.nan, 1.0

    def validate_parameters(self) -> bool:
        """Validate Cointegration parameters"""
        period = self.parameters.get("period", 252)
        method = self.parameters.get("method", "engle_granger")
        confidence_level = self.parameters.get("confidence_level", 0.05)
        min_periods = self.parameters.get("min_periods", 100)

        if not isinstance(period, int) or period < 50:
            raise IndicatorValidationError(
                f"period must be integer >= 50, got {period}"
            )

        if period > 2000:
            raise IndicatorValidationError(
                f"period too large, maximum 2000, got {period}"
            )

        if method not in ["engle_granger", "johansen"]:
            raise IndicatorValidationError(
                f"method must be 'engle_granger' or 'johansen', got {method}"
            )

        if not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
            raise IndicatorValidationError(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )

        if not isinstance(min_periods, int) or min_periods < 30:
            raise IndicatorValidationError(
                f"min_periods must be integer >= 30, got {min_periods}"
            )

        if min_periods > period:
            raise IndicatorValidationError(
                f"min_periods cannot exceed period, got min_periods={min_periods}, period={period}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Cointegration metadata as dictionary"""
        return {
            "name": "Cointegration",
            "category": self.CATEGORY,
            "description": "Tests for cointegration between time series using statistical methods",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Cointegration requires price data"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Cointegration calculation"""
        return self.parameters.get("min_periods", 100)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 252
        if "method" not in self.parameters:
            self.parameters["method"] = "engle_granger"
        if "confidence_level" not in self.parameters:
            self.parameters["confidence_level"] = 0.05
        if "min_periods" not in self.parameters:
            self.parameters["min_periods"] = 100

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 252)

    @property
    def method(self) -> str:
        """Method for backward compatibility"""
        return self.parameters.get("method", "engle_granger")

    @property
    def confidence_level(self) -> float:
        """Confidence level for backward compatibility"""
        return self.parameters.get("confidence_level", 0.05)

    @property
    def min_periods(self) -> int:
        """Minimum periods for backward compatibility"""
        return self.parameters.get("min_periods", 100)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods property for compatibility"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "Cointegration",
            "period": self.period,
            "method": self.method,
            "confidence_level": self.confidence_level,
            "min_periods": self.min_periods,
            "category": self.CATEGORY,
        }

    def interpret_cointegration(self, test_stat: float, p_value: float) -> Dict[str, Any]:
        """
        Interpret cointegration test results
        
        Args:
            test_stat: Cointegration test statistic
            p_value: P-value of the test
            
        Returns:
            Dict containing interpretation details
        """
        confidence_level = self.confidence_level
        
        if pd.isna(test_stat) or pd.isna(p_value):
            return {
                "signal": "insufficient_data",
                "interpretation": "Not enough data for cointegration test",
                "cointegrated": False,
                "confidence": "unknown"
            }
        
        # Determine if series are cointegrated
        cointegrated = p_value < confidence_level
        
        if cointegrated:
            if p_value < 0.01:
                confidence = "very_high"
                interpretation = "Strong evidence of cointegration relationship"
            elif p_value < 0.05:
                confidence = "high"
                interpretation = "Significant cointegration relationship detected"
            else:
                confidence = "moderate"
                interpretation = "Some evidence of cointegration relationship"
        else:
            if p_value > 0.10:
                confidence = "low"
                interpretation = "No evidence of cointegration relationship"
            else:
                confidence = "uncertain"
                interpretation = "Weak evidence against cointegration"

        return {
            "signal": "cointegrated" if cointegrated else "not_cointegrated",
            "test_statistic": test_stat,
            "p_value": p_value,
            "interpretation": interpretation,
            "cointegrated": cointegrated,
            "confidence": confidence,
            "significance_level": confidence_level
        }

    def get_trading_signal(self, cointegration_result: Dict[str, Any], current_spread: float = None) -> str:
        """
        Generate trading signal based on cointegration analysis
        
        Args:
            cointegration_result: Result from interpret_cointegration method
            current_spread: Current spread between the series
            
        Returns:
            Trading signal string
        """
        if not cointegration_result.get("cointegrated", False):
            return "no_signal"
        
        confidence = cointegration_result.get("confidence", "unknown")
        
        if confidence in ["very_high", "high"]:
            if current_spread is not None:
                if current_spread > 2:  # Spread above 2 standard deviations
                    return "short_spread"  # Short the spread (sell overperformer, buy underperformer)
                elif current_spread < -2:  # Spread below -2 standard deviations
                    return "long_spread"  # Long the spread (buy overperformer, sell underperformer)
                else:
                    return "hold"
            else:
                return "monitor"  # Monitor for trading opportunities
        elif confidence == "moderate":
            return "cautious_monitor"
        else:
            return "no_signal"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return CointegrationIndicator


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample cointegrated data
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    np.random.seed(42)
    
    # Create two cointegrated series (with common stochastic trend)
    common_trend = np.cumsum(np.random.normal(0, 0.01, len(dates)))
    
    # Series 1: follows common trend with some noise
    series1 = 100 + common_trend + np.random.normal(0, 0.5, len(dates))
    
    # Series 2: follows common trend with different coefficient and noise
    series2 = 80 + 1.2 * common_trend + np.random.normal(0, 0.3, len(dates))
    
    data = pd.DataFrame({
        "close": series1
    }, index=dates)
    
    reference = pd.DataFrame({
        "close": series2
    }, index=dates)
    
    # Calculate cointegration
    coint_indicator = CointegrationIndicator(period=150, min_periods=50)
    coint_stats = coint_indicator.calculate(data, reference)
    
    print("Cointegration Indicator Example:")
    latest_stats = coint_stats.dropna()
    if len(latest_stats) > 0:
        latest_stat = latest_stats.iloc[-1]
        latest_pvalue = coint_indicator._last_calculation.get("p_values", pd.Series()).dropna().iloc[-1]
        
        print(f"Final Test Statistic: {latest_stat:.3f}")
        print(f"Final P-value: {latest_pvalue:.3f}")
        
        # Interpret results
        interpretation = coint_indicator.interpret_cointegration(latest_stat, latest_pvalue)
        print(f"\nInterpretation: {interpretation}")
    else:
        print("No cointegration statistics calculated")