"""
Correlation Coefficient Indicator - Statistical Analysis
Measures the linear relationship between two price series or datasets.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..base_indicator import (
    StandardIndicatorInterface,
    IndicatorMetadata,
    IndicatorValidationError,
    TradingGradeValidator
)

logger = logging.getLogger(__name__)


class CorrelationCoefficientIndicator(StandardIndicatorInterface):
    """
    Correlation Coefficient Indicator for Statistical Analysis
    
    Calculates Pearson correlation coefficient between price series,
    providing insights into asset relationships and market dynamics.
    
    Mathematical Formula:
    r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)
    Where:
    - r = correlation coefficient (-1 to +1)
    - x, y = data series
    - x̄, ȳ = means of respective series
    
    Advanced Features:
    - Rolling correlation analysis
    - Statistical significance testing
    - Multi-asset correlation matrices
    - Regime-aware correlation tracking
    
    Parameters:
        period (int): Look-back period for calculation (default: 20)
        reference_column (str): Reference column for correlation (default: 'close')
        target_column (str): Target column for correlation (default: 'volume')
        correlation_type (str): Type of correlation ('pearson', 'spearman', 'kendall')
        
    Example:
        >>> indicator = CorrelationCoefficientIndicator(period=20, 
        ...                                           reference_column='close',
        ...                                           target_column='volume')
        >>> result = indicator.calculate(ohlcv_data)
        >>> print(f"Price-Volume correlation: {result.iloc[-1]:.4f}")
    """
    
    CATEGORY = "statistical"
    VERSION = "1.0.0"  
    AUTHOR = "Platform3"
    
    def __init__(self, period: int = 20, reference_column: str = 'close',
                 target_column: str = 'volume', correlation_type: str = 'pearson', **kwargs):
        super().__init__(
            period=period,
            reference_column=reference_column,
            target_column=target_column,
            correlation_type=correlation_type,
            **kwargs
        )
        
    def _setup_defaults(self):
        """Setup default parameter values"""
        if 'period' not in self.parameters:
            self.parameters['period'] = 20
        if 'reference_column' not in self.parameters:
            self.parameters['reference_column'] = 'close'
        if 'target_column' not in self.parameters:
            self.parameters['target_column'] = 'volume'
        if 'correlation_type' not in self.parameters:
            self.parameters['correlation_type'] = 'pearson'
            
    def validate_parameters(self) -> bool:
        """Validate Correlation Coefficient indicator parameters"""
        period = self.parameters.get('period')
        reference_column = self.parameters.get('reference_column')
        target_column = self.parameters.get('target_column')
        correlation_type = self.parameters.get('correlation_type')
        
        if not isinstance(period, int) or period < 3:
            raise IndicatorValidationError("Period must be an integer >= 3")
            
        if period > 1000:
            raise IndicatorValidationError("Period cannot exceed 1000")
            
        if not isinstance(reference_column, str) or not reference_column:
            raise IndicatorValidationError("Reference column must be a non-empty string")
            
        if not isinstance(target_column, str) or not target_column:
            raise IndicatorValidationError("Target column must be a non-empty string")
            
        if reference_column == target_column:
            raise IndicatorValidationError("Reference and target columns must be different")
            
        valid_types = ['pearson', 'spearman', 'kendall']
        if correlation_type not in valid_types:
            raise IndicatorValidationError(f"Correlation type must be one of: {valid_types}")
            
        return True
        
    def _get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return [
            self.parameters.get('reference_column', 'close'),
            self.parameters.get('target_column', 'volume')
        ]
        
    def _get_minimum_data_points(self) -> int:
        """Get minimum required data points"""
        return self.parameters.get('period', 20)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Correlation Coefficient with advanced statistical features
        
        Args:
            data: DataFrame with required columns
            
        Returns:
            pd.Series: Rolling correlation coefficients
        """
        # Validate input data
        self.validate_input_data(data)
        
        # Get parameters
        period = self.parameters.get('period')
        reference_column = self.parameters.get('reference_column')
        target_column = self.parameters.get('target_column')
        correlation_type = self.parameters.get('correlation_type')
        
        # Extract data series
        reference_series = data[reference_column]
        target_series = data[target_column]
        
        # Calculate rolling correlation
        if correlation_type == 'pearson':
            correlation = reference_series.rolling(
                window=period, min_periods=period
            ).corr(target_series)
        elif correlation_type == 'spearman':
            correlation = self._rolling_spearman(reference_series, target_series, period)
        elif correlation_type == 'kendall':
            correlation = self._rolling_kendall(reference_series, target_series, period)
        else:
            raise IndicatorValidationError(f"Unsupported correlation type: {correlation_type}")
            
        # Calculate statistical significance and confidence intervals
        correlation_stats = self._calculate_correlation_statistics(
            reference_series, target_series, correlation, period
        )
        
        # Store detailed calculation information
        self._last_calculation = {
            'values': correlation,
            'period': period,
            'correlation_type': correlation_type,
            'reference_column': reference_column,
            'target_column': target_column,
            'statistics': correlation_stats,
            'quality_score': self._calculate_quality_score(correlation, correlation_stats)
        }
        
        # Validate numerical precision
        valid_correlation = correlation.dropna()
        if len(valid_correlation) > 0:
            if not TradingGradeValidator.validate_numerical_precision(valid_correlation):
                raise IndicatorValidationError("Correlation calculation failed precision validation")
                
            if not TradingGradeValidator.validate_consistency(valid_correlation):
                logger.warning("Correlation values show inconsistent patterns")
                
        return correlation
        
    def _rolling_spearman(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Spearman correlation"""
        def spearman_corr(series_x, series_y):
            try:
                corr, _ = stats.spearmanr(series_x, series_y)
                return corr if not np.isnan(corr) else np.nan
            except:
                return np.nan
                
        rolling_corr = []
        for i in range(len(x)):
            if i < window - 1:
                rolling_corr.append(np.nan)
            else:
                x_window = x.iloc[i-window+1:i+1]
                y_window = y.iloc[i-window+1:i+1]
                corr = spearman_corr(x_window, y_window)
                rolling_corr.append(corr)
                
        return pd.Series(rolling_corr, index=x.index)
        
    def _rolling_kendall(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Kendall's tau correlation"""
        def kendall_corr(series_x, series_y):
            try:
                corr, _ = stats.kendalltau(series_x, series_y)
                return corr if not np.isnan(corr) else np.nan
            except:
                return np.nan
                
        rolling_corr = []
        for i in range(len(x)):
            if i < window - 1:
                rolling_corr.append(np.nan)
            else:
                x_window = x.iloc[i-window+1:i+1]
                y_window = y.iloc[i-window+1:i+1]
                corr = kendall_corr(x_window, y_window)
                rolling_corr.append(corr)
                
        return pd.Series(rolling_corr, index=x.index)
        
    def _calculate_correlation_statistics(self, x: pd.Series, y: pd.Series, 
                                        correlation: pd.Series, period: int) -> Dict[str, Any]:
        """Calculate comprehensive correlation statistics"""
        valid_corr = correlation.dropna()
        
        if len(valid_corr) == 0:
            return {'significance': 0.0, 'confidence_interval': (np.nan, np.nan)}
            
        # Current correlation value
        current_corr = valid_corr.iloc[-1] if len(valid_corr) > 0 else np.nan
        
        # Statistical significance test
        n = period
        if not np.isnan(current_corr) and n > 2:
            # T-test for correlation significance
            t_stat = current_corr * np.sqrt((n - 2) / (1 - current_corr**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            significance = 1 - p_value if not np.isnan(p_value) else 0.0
        else:
            t_stat = np.nan
            p_value = 1.0
            significance = 0.0
            
        # Confidence interval for correlation (Fisher transformation)
        if not np.isnan(current_corr) and n > 3:
            z = np.arctanh(current_corr)  # Fisher transformation
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf(0.975)  # 95% confidence
            
            z_lower = z - z_crit * se
            z_upper = z + z_crit * se
            
            # Transform back
            corr_lower = np.tanh(z_lower)
            corr_upper = np.tanh(z_upper)
        else:
            corr_lower = np.nan
            corr_upper = np.nan
            
        # Correlation stability analysis
        if len(valid_corr) >= 10:
            recent_corr = valid_corr.tail(5)
            historical_corr = valid_corr.iloc[-20:-5] if len(valid_corr) >= 20 else pd.Series()
            
            if len(historical_corr) >= 5:
                # Test for correlation regime change
                try:
                    t_stat_regime, p_value_regime = stats.ttest_ind(recent_corr, historical_corr)
                    regime_change_probability = 1 - p_value_regime if not np.isnan(p_value_regime) else 0.0
                except:
                    regime_change_probability = 0.0
            else:
                regime_change_probability = 0.0
                
            stability = 1 - valid_corr.std() if valid_corr.std() > 0 else 1.0
        else:
            regime_change_probability = 0.0
            stability = 0.5
            
        return {
            'current_correlation': float(current_corr) if not np.isnan(current_corr) else None,
            'significance': float(significance),
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'confidence_interval': (float(corr_lower) if not np.isnan(corr_lower) else None,
                                  float(corr_upper) if not np.isnan(corr_upper) else None),
            'stability': float(max(0, min(1, stability))),
            'regime_change_probability': float(regime_change_probability)
        }        
    def _calculate_quality_score(self, correlation: pd.Series, stats: Dict[str, Any]) -> float:
        """Calculate indicator quality score for ensemble management"""
        valid_corr = correlation.dropna()
        
        if len(valid_corr) < 5:
            return 0.0
            
        try:
            # Factor 1: Statistical significance
            significance = stats.get('significance', 0.0)
            
            # Factor 2: Correlation stability
            stability = stats.get('stability', 0.5)
            
            # Factor 3: Data sufficiency
            data_sufficiency = min(1.0, len(valid_corr) / (self.parameters.get('period') * 2))
            
            # Factor 4: Correlation strength (absolute value)
            current_corr = stats.get('current_correlation', 0.0)
            strength = abs(current_corr) if current_corr is not None else 0.0
            
            # Combined quality score
            quality_score = (
                significance * 0.3 +
                stability * 0.25 +
                data_sufficiency * 0.25 +
                strength * 0.2
            )
            
            return float(max(0, min(1, quality_score)))
            
        except Exception:
            return 0.5  # Default moderate quality
            
    @property
    def minimum_periods(self) -> int:
        """Minimum periods required for calculation"""
        return self.parameters.get('period', 20)
        
    def get_config(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return {
            'period': self.parameters.get('period'),
            'reference_column': self.parameters.get('reference_column'),
            'target_column': self.parameters.get('target_column'),
            'correlation_type': self.parameters.get('correlation_type'),
            'category': self.CATEGORY,
            'version': self.VERSION
        }
        
    def get_metadata(self) -> IndicatorMetadata:
        """Get comprehensive indicator metadata"""
        return IndicatorMetadata(
            name="Correlation Coefficient Indicator",
            category=self.CATEGORY,
            description="Statistical measure of linear relationship between two data series with significance testing",
            parameters={
                "period": {
                    "type": "int",
                    "default": 20,
                    "range": [3, 1000],
                    "description": "Look-back period for correlation calculation"
                },
                "reference_column": {
                    "type": "str",
                    "default": "close",
                    "description": "Reference column for correlation analysis"
                },
                "target_column": {
                    "type": "str", 
                    "default": "volume",
                    "description": "Target column for correlation analysis"
                },
                "correlation_type": {
                    "type": "str",
                    "default": "pearson",
                    "options": ["pearson", "spearman", "kendall"],
                    "description": "Type of correlation coefficient to calculate"
                }
            },
            input_requirements=["close", "volume"],  # Default requirements
            output_type="series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self.minimum_periods,
            performance_tier="standard"
        )


def get_indicator_class():
    """Export function for registry discovery"""
    return CorrelationCoefficientIndicator