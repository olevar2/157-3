"""
Correlation Analysis Indicator

Advanced correlation analysis for multiple time series with dynamic correlation tracking,
rolling correlation windows, and various correlation methods. Provides comprehensive
correlation metrics including Pearson, Spearman, and Kendall correlations.

Correlation Methods:
1. Pearson: Linear correlation (-1 to +1)
2. Spearman: Rank-based correlation (non-parametric)
3. Kendall: Tau correlation (robust to outliers)

Rolling Correlation Analysis:
- Dynamic correlation tracking over time
- Correlation breakdown detection
- Multi-timeframe correlation analysis
- Correlation regime identification

Applications:
- Portfolio diversification analysis
- Risk management and correlation risk
- Pairs trading strategy development
- Market regime detection
- Asset allocation optimization

Author: Platform3 AI Framework
Created: 2025-06-09
"""

import os
import sys
from typing import Any, Dict, List, Union, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Import the base indicator interface
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)


class CorrelationAnalysis(StandardIndicatorInterface):
    """
    Correlation Analysis Indicator
    
    Provides comprehensive correlation analysis with multiple correlation methods,
    rolling windows, and advanced correlation metrics for financial time series.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "statistical"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 50,  # Rolling correlation window
        method: str = "pearson",  # Correlation method
        min_periods: int = 20,  # Minimum periods for calculation
        significance_level: float = 0.05,  # For correlation significance testing
        **kwargs,
    ):
        """
        Initialize Correlation Analysis indicator

        Args:
            period: Period for rolling correlation calculation (default: 50)
            method: Correlation method ("pearson", "spearman", "kendall") (default: "pearson")
            min_periods: Minimum periods required for calculation (default: 20)
            significance_level: Significance level for correlation tests (default: 0.05)
        """
        super().__init__(
            period=period,
            method=method,
            min_periods=min_periods,
            significance_level=significance_level,
            **kwargs,
        )

    def calculate(self, data: Union[pd.DataFrame, pd.Series], reference_series: Union[pd.DataFrame, pd.Series] = None) -> pd.Series:
        """
        Calculate rolling correlation analysis
        
        Args:
            data: Primary time series (DataFrame with 'close' or Series of prices)
            reference_series: Reference time series for correlation analysis
                             If None, calculates autocorrelation
        
        Returns:
            pd.Series: Rolling correlation values with same index as input data
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
            # Calculate autocorrelation (correlation with lagged version)
            lag = self.parameters.get("autocorr_lag", 1)
            reference_series = primary_series.shift(lag)
        elif isinstance(reference_series, pd.Series):
            pass  # Already in correct format
        elif isinstance(reference_series, pd.DataFrame):
            if "close" in reference_series.columns:
                reference_series = reference_series["close"]
            else:
                raise IndicatorValidationError(
                    "Reference DataFrame must contain 'close' column"
                )

        period = self.parameters.get("period", 50)
        method = self.parameters.get("method", "pearson")
        min_periods = self.parameters.get("min_periods", 20)

        # Align series
        common_index = primary_series.index.intersection(reference_series.index)
        primary_aligned = primary_series.reindex(common_index)
        reference_aligned = reference_series.reindex(common_index)

        # Calculate rolling correlation
        if method == "pearson":
            correlation_values = primary_aligned.rolling(
                window=period, min_periods=min_periods
            ).corr(reference_aligned)
        elif method == "spearman":
            correlation_values = self._rolling_spearman_correlation(
                primary_aligned, reference_aligned, period, min_periods
            )
        elif method == "kendall":
            correlation_values = self._rolling_kendall_correlation(
                primary_aligned, reference_aligned, period, min_periods
            )
        else:
            raise IndicatorValidationError(
                f"Unknown correlation method: {method}"
            )

        # Calculate additional metrics
        correlation_pvalues = self._calculate_correlation_pvalues(
            primary_aligned, reference_aligned, period, min_periods, method
        )
        
        correlation_confidence = self._calculate_correlation_confidence(
            correlation_values, primary_aligned, period, min_periods
        )

        # Store calculation details for analysis
        self._last_calculation = {
            "primary_series": primary_aligned,
            "reference_series": reference_aligned,
            "correlation_values": correlation_values,
            "correlation_pvalues": correlation_pvalues,
            "correlation_confidence": correlation_confidence,
            "method": method,
            "period": period,
            "min_periods": min_periods,
        }

        return pd.Series(correlation_values, index=primary_aligned.index, name=f"Correlation_{method}")

    def _rolling_spearman_correlation(self, x: pd.Series, y: pd.Series, period: int, min_periods: int) -> pd.Series:
        """Calculate rolling Spearman rank correlation"""
        result = pd.Series(index=x.index, dtype=float)
        
        for i in range(min_periods, len(x)):
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            
            x_window = x.iloc[start_idx:end_idx].dropna()
            y_window = y.iloc[start_idx:end_idx].dropna()
            
            # Align windows
            common_dates = x_window.index.intersection(y_window.index)
            if len(common_dates) >= min_periods:
                x_aligned = x_window.reindex(common_dates)
                y_aligned = y_window.reindex(common_dates)
                
                # Remove any remaining NaN values
                valid_mask = ~(x_aligned.isna() | y_aligned.isna())
                x_clean = x_aligned[valid_mask]
                y_clean = y_aligned[valid_mask]
                
                if len(x_clean) >= min_periods:
                    corr, _ = stats.spearmanr(x_clean, y_clean)
                    result.iloc[i] = corr
        
        return result

    def _rolling_kendall_correlation(self, x: pd.Series, y: pd.Series, period: int, min_periods: int) -> pd.Series:
        """Calculate rolling Kendall tau correlation"""
        result = pd.Series(index=x.index, dtype=float)
        
        for i in range(min_periods, len(x)):
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            
            x_window = x.iloc[start_idx:end_idx].dropna()
            y_window = y.iloc[start_idx:end_idx].dropna()
            
            # Align windows
            common_dates = x_window.index.intersection(y_window.index)
            if len(common_dates) >= min_periods:
                x_aligned = x_window.reindex(common_dates)
                y_aligned = y_window.reindex(common_dates)
                
                # Remove any remaining NaN values
                valid_mask = ~(x_aligned.isna() | y_aligned.isna())
                x_clean = x_aligned[valid_mask]
                y_clean = y_aligned[valid_mask]
                
                if len(x_clean) >= min_periods:
                    corr, _ = stats.kendalltau(x_clean, y_clean)
                    result.iloc[i] = corr
        
        return result

    def _calculate_correlation_pvalues(self, x: pd.Series, y: pd.Series, period: int, min_periods: int, method: str) -> pd.Series:
        """Calculate p-values for correlation significance testing"""
        result = pd.Series(index=x.index, dtype=float)
        
        for i in range(min_periods, len(x)):
            start_idx = max(0, i - period + 1)
            end_idx = i + 1
            
            x_window = x.iloc[start_idx:end_idx].dropna()
            y_window = y.iloc[start_idx:end_idx].dropna()
            
            # Align windows
            common_dates = x_window.index.intersection(y_window.index)
            if len(common_dates) >= min_periods:
                x_aligned = x_window.reindex(common_dates)
                y_aligned = y_window.reindex(common_dates)
                
                # Remove any remaining NaN values
                valid_mask = ~(x_aligned.isna() | y_aligned.isna())
                x_clean = x_aligned[valid_mask]
                y_clean = y_aligned[valid_mask]
                
                if len(x_clean) >= min_periods:
                    try:
                        if method == "pearson":
                            _, p_value = stats.pearsonr(x_clean, y_clean)
                        elif method == "spearman":
                            _, p_value = stats.spearmanr(x_clean, y_clean)
                        elif method == "kendall":
                            _, p_value = stats.kendalltau(x_clean, y_clean)
                        
                        result.iloc[i] = p_value
                    except:
                        result.iloc[i] = np.nan
        
        return result

    def _calculate_correlation_confidence(self, correlations: pd.Series, data: pd.Series, period: int, min_periods: int) -> pd.Series:
        """Calculate confidence intervals for correlations using Fisher transformation"""
        result = pd.Series(index=correlations.index, dtype=float)
        
        # Fisher transformation for confidence intervals
        # z = 0.5 * ln((1 + r) / (1 - r))
        # SE(z) = 1 / sqrt(n - 3)
        
        for i in range(min_periods, len(correlations)):
            corr = correlations.iloc[i]
            if not pd.isna(corr) and abs(corr) < 0.999:  # Avoid division by zero
                n = min(period, i + 1)  # Actual sample size
                
                # Fisher transformation
                z = 0.5 * np.log((1 + corr) / (1 - corr))
                se_z = 1 / np.sqrt(n - 3) if n > 3 else np.inf
                
                # 95% confidence interval in z-space
                z_lower = z - 1.96 * se_z
                z_upper = z + 1.96 * se_z
                
                # Transform back to correlation space
                r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                
                # Confidence width as metric
                confidence_width = r_upper - r_lower
                result.iloc[i] = confidence_width
        
        return result

    def validate_parameters(self) -> bool:
        """Validate Correlation Analysis parameters"""
        period = self.parameters.get("period", 50)
        method = self.parameters.get("method", "pearson")
        min_periods = self.parameters.get("min_periods", 20)
        significance_level = self.parameters.get("significance_level", 0.05)

        if not isinstance(period, int) or period < 5:
            raise IndicatorValidationError(
                f"period must be integer >= 5, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        if method not in ["pearson", "spearman", "kendall"]:
            raise IndicatorValidationError(
                f"method must be 'pearson', 'spearman', or 'kendall', got {method}"
            )

        if not isinstance(min_periods, int) or min_periods < 3:
            raise IndicatorValidationError(
                f"min_periods must be integer >= 3, got {min_periods}"
            )

        if min_periods > period:
            raise IndicatorValidationError(
                f"min_periods cannot exceed period, got min_periods={min_periods}, period={period}"
            )

        if not isinstance(significance_level, (int, float)) or not 0 < significance_level < 1:
            raise IndicatorValidationError(
                f"significance_level must be between 0 and 1, got {significance_level}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Correlation Analysis metadata as dictionary"""
        return {
            "name": "Correlation Analysis",
            "category": self.CATEGORY,
            "description": "Advanced correlation analysis with multiple methods and rolling windows",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Correlation Analysis requires price data"""
        return ["close"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for Correlation Analysis"""
        return self.parameters.get("min_periods", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 50
        if "method" not in self.parameters:
            self.parameters["method"] = "pearson"
        if "min_periods" not in self.parameters:
            self.parameters["min_periods"] = 20
        if "significance_level" not in self.parameters:
            self.parameters["significance_level"] = 0.05

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 50)

    @property
    def method(self) -> str:
        """Method for backward compatibility"""
        return self.parameters.get("method", "pearson")

    @property
    def min_periods(self) -> int:
        """Minimum periods for backward compatibility"""
        return self.parameters.get("min_periods", 20)

    @property
    def significance_level(self) -> float:
        """Significance level for backward compatibility"""
        return self.parameters.get("significance_level", 0.05)

    @property
    def minimum_periods(self) -> int:
        """Minimum periods property for compatibility"""
        return self._get_minimum_data_points()

    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            "indicator": "CorrelationAnalysis",
            "period": self.period,
            "method": self.method,
            "min_periods": self.min_periods,
            "significance_level": self.significance_level,
            "category": self.CATEGORY,
        }

    def interpret_correlation(self, correlation_value: float, p_value: float = None) -> Dict[str, Any]:
        """
        Interpret correlation value
        
        Args:
            correlation_value: Correlation coefficient
            p_value: P-value for significance testing (optional)
            
        Returns:
            Dict containing interpretation details
        """
        if pd.isna(correlation_value):
            return {
                "signal": "insufficient_data",
                "interpretation": "Not enough data for correlation calculation",
                "strength": "unknown",
                "direction": "unknown",
                "significant": False
            }
        
        # Determine correlation strength
        abs_corr = abs(correlation_value)
        if abs_corr >= 0.8:
            strength = "very_strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very_weak"
        
        # Determine direction
        if correlation_value > 0:
            direction = "positive"
            direction_desc = "move in same direction"
        elif correlation_value < 0:
            direction = "negative"
            direction_desc = "move in opposite directions"
        else:
            direction = "neutral"
            direction_desc = "no linear relationship"
        
        # Check statistical significance
        significance_level = self.significance_level
        significant = p_value < significance_level if p_value is not None else None
        
        # Create interpretation
        if strength in ["very_strong", "strong"]:
            interpretation = f"Strong {direction} correlation - assets {direction_desc}"
        elif strength == "moderate":
            interpretation = f"Moderate {direction} correlation - some relationship exists"
        else:
            interpretation = f"Weak {direction} correlation - limited relationship"

        return {
            "signal": "correlation_detected" if abs_corr > 0.2 else "no_correlation",
            "correlation_value": correlation_value,
            "interpretation": interpretation,
            "strength": strength,
            "direction": direction,
            "significant": significant,
            "p_value": p_value,
            "confidence_level": 1 - significance_level if p_value is not None else None
        }

    def detect_correlation_regimes(self, correlation_series: pd.Series, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect correlation regime changes
        
        Args:
            correlation_series: Time series of correlation values
            threshold: Threshold for regime change detection
            
        Returns:
            Dict containing regime analysis
        """
        if len(correlation_series.dropna()) < 10:
            return {"regimes": [], "current_regime": "unknown"}
        
        clean_corr = correlation_series.dropna()
        
        # Detect regime changes using rolling standard deviation
        rolling_std = clean_corr.rolling(window=min(20, len(clean_corr)//3)).std()
        regime_changes = []
        
        current_regime = "stable"
        for i in range(1, len(clean_corr)):
            prev_corr = clean_corr.iloc[i-1]
            curr_corr = clean_corr.iloc[i]
            
            # Check for significant change
            if abs(curr_corr - prev_corr) > threshold:
                regime_changes.append({
                    "date": clean_corr.index[i],
                    "from_correlation": prev_corr,
                    "to_correlation": curr_corr,
                    "change_magnitude": abs(curr_corr - prev_corr)
                })
        
        # Classify current regime
        recent_corr = clean_corr.tail(10).mean()
        recent_std = clean_corr.tail(10).std()
        
        if recent_std > 0.2:
            current_regime = "volatile"
        elif abs(recent_corr) > 0.7:
            current_regime = "high_correlation"
        elif abs(recent_corr) < 0.2:
            current_regime = "low_correlation"
        else:
            current_regime = "moderate_correlation"
        
        return {
            "regimes": regime_changes,
            "current_regime": current_regime,
            "recent_correlation": recent_corr,
            "recent_volatility": recent_std,
            "regime_changes_count": len(regime_changes)
        }

    def get_diversification_signal(self, correlation_value: float) -> str:
        """
        Generate portfolio diversification signal based on correlation
        
        Args:
            correlation_value: Current correlation value
            
        Returns:
            Diversification signal string
        """
        if pd.isna(correlation_value):
            return "no_signal"
        
        abs_corr = abs(correlation_value)
        
        if abs_corr > 0.8:
            return "poor_diversification"  # High correlation = poor diversification
        elif abs_corr > 0.5:
            return "moderate_diversification"
        elif abs_corr > 0.2:
            return "good_diversification"
        else:
            return "excellent_diversification"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return CorrelationAnalysis


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample correlated data
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
    np.random.seed(42)
    
    # Create two correlated series
    base_returns = np.random.normal(0.001, 0.02, len(dates))
    noise1 = np.random.normal(0, 0.01, len(dates))
    noise2 = np.random.normal(0, 0.01, len(dates))
    
    # Series 1
    returns1 = 0.7 * base_returns + 0.3 * noise1
    prices1 = 100 * (1 + returns1).cumprod()
    
    # Series 2 (correlated with series 1)
    returns2 = 0.6 * base_returns + 0.4 * noise2
    prices2 = 95 * (1 + returns2).cumprod()
    
    data = pd.DataFrame({
        "close": prices1
    }, index=dates)
    
    reference = pd.DataFrame({
        "close": prices2
    }, index=dates)
    
    # Calculate correlation analysis
    corr_indicator = CorrelationAnalysis(period=50, method="pearson", min_periods=20)
    correlations = corr_indicator.calculate(data, reference)
    
    print("Correlation Analysis Indicator Example:")
    recent_correlations = correlations.dropna().tail(20)
    if len(recent_correlations) > 0:
        latest_corr = recent_correlations.iloc[-1]
        mean_corr = recent_correlations.mean()
        
        print(f"Latest Correlation: {latest_corr:.3f}")
        print(f"Mean Correlation (last 20): {mean_corr:.3f}")
        print(f"Correlation Std (last 20): {recent_correlations.std():.3f}")
        
        # Interpret results
        p_values = corr_indicator._last_calculation.get("correlation_pvalues", pd.Series())
        latest_pvalue = p_values.dropna().iloc[-1] if len(p_values.dropna()) > 0 else None
        
        interpretation = corr_indicator.interpret_correlation(latest_corr, latest_pvalue)
        print(f"\nInterpretation: {interpretation}")
        
        # Diversification signal
        div_signal = corr_indicator.get_diversification_signal(latest_corr)
        print(f"Diversification Signal: {div_signal}")
    else:
        print("No correlation values calculated")