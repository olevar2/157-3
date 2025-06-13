"""
Statistical Indicators Package

This package contains individual statistical indicator implementations
following Platform3 standards for trading-grade accuracy and reliability.

Available Indicators:
- StandardDeviationIndicator: Measures price volatility and dispersion
- CorrelationCoefficientIndicator: Measures linear relationships between series
- ZScoreIndicator: Measures deviation from mean in standard deviations
- LinearRegressionIndicator: Fits regression lines for trend analysis
- RSquaredIndicator: Measures regression model goodness-of-fit
- AutocorrelationIndicator: Measures temporal correlation patterns
- SkewnessIndicator: Measures distribution asymmetry and tail risk
- BetaCoefficientIndicator: Measures systematic risk relative to market
- CointegrationIndicator: Tests for long-run equilibrium relationships
- CorrelationAnalysis: Advanced correlation analysis with multiple methods
- LinearRegressionChannels: Dynamic channels around regression trend lines
- StandardDeviationChannels: Dynamic channels using moving averages and standard deviation
- VarianceRatioIndicator: Tests for random walk behavior and serial correlation

All indicators inherit from StandardIndicatorInterface and provide:
- Trading-grade mathematical accuracy
- Comprehensive parameter validation
- Advanced statistical features
- Performance optimization
- Quality scoring for ensemble management
"""

from .autocorrelation_indicator import AutocorrelationIndicator
from .beta_coefficient_indicator import BetaCoefficientIndicator
from .cointegration_indicator import CointegrationIndicator
from .correlation_analysis_indicator import CorrelationAnalysis
from .correlation_coefficient_indicator import CorrelationCoefficientIndicator
from .linear_regression_channels_indicator import LinearRegressionChannels
from .linear_regression_indicator import LinearRegressionIndicator
from .r_squared_indicator import RSquaredIndicator
from .skewness_indicator import SkewnessIndicator
from .standard_deviation_channels_indicator import StandardDeviationChannels
from .standard_deviation_indicator import StandardDeviationIndicator
from .variance_ratio_indicator import VarianceRatioIndicator
from .z_score_indicator import ZScoreIndicator

# Export all indicator classes
__all__ = [
    "StandardDeviationIndicator",
    "CorrelationCoefficientIndicator",
    "ZScoreIndicator",
    "LinearRegressionIndicator",
    "RSquaredIndicator",
    "AutocorrelationIndicator",
    "SkewnessIndicator",
    "BetaCoefficientIndicator",
    "CointegrationIndicator",
    "CorrelationAnalysis",
    "LinearRegressionChannels",
    "StandardDeviationChannels",
    "VarianceRatioIndicator",
]

# Registry mapping for dynamic discovery
STATISTICAL_INDICATORS = {
    "standard_deviation": StandardDeviationIndicator,
    "correlation_coefficient": CorrelationCoefficientIndicator,
    "z_score": ZScoreIndicator,
    "linear_regression": LinearRegressionIndicator,
    "r_squared": RSquaredIndicator,
    "autocorrelation": AutocorrelationIndicator,
    "skewness": SkewnessIndicator,
    "beta_coefficient": BetaCoefficientIndicator,
    "cointegration": CointegrationIndicator,
    "correlation_analysis": CorrelationAnalysis,
    "linear_regression_channels": LinearRegressionChannels,
    "standard_deviation_channels": StandardDeviationChannels,
    "variance_ratio": VarianceRatioIndicator,
}


def get_indicator_by_name(name: str):
    """
    Get indicator class by name

    Args:
        name: Indicator name (snake_case)

    Returns:
        Indicator class or None if not found
    """
    return STATISTICAL_INDICATORS.get(name.lower())


def list_available_indicators():
    """
    List all available statistical indicators

    Returns:
        List of indicator names
    """
    return list(STATISTICAL_INDICATORS.keys())
