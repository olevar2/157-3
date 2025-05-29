"""
Platform3 Hurst Exponent Calculator
Advanced Hurst exponent calculation for market memory and trend persistence analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class HurstExponentCalculator:
    """
    Advanced Hurst exponent calculator for measuring market memory and trend persistence.
    
    Hurst Exponent Interpretation:
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent/trending behavior (long memory)
    - H < 0.5: Anti-persistent/mean-reverting behavior
    """
    
    def __init__(self,
                 min_window: int = 50,
                 max_window: int = 200,
                 rolling_window: int = 100,
                 num_lags: int = 20,
                 confidence_level: float = 0.95,
                 detrending_method: str = 'linear',
                 use_overlapping: bool = True):
        """
        Initialize Hurst Exponent Calculator
        
        Parameters:
        -----------
        min_window : int
            Minimum window size for calculation
        max_window : int
            Maximum window size for calculation
        rolling_window : int
            Rolling window for time series analysis
        num_lags : int
            Number of lags for R/S analysis
        confidence_level : float
            Confidence level for statistical tests
        detrending_method : str
            Method for detrending ('linear', 'polynomial', 'moving_average')
        use_overlapping : bool
            Whether to use overlapping windows
        """
        self.min_window = min_window
        self.max_window = max_window
        self.rolling_window = rolling_window
        self.num_lags = num_lags
        self.confidence_level = confidence_level
        self.detrending_method = detrending_method
        self.use_overlapping = use_overlapping
        
        # Hurst interpretation thresholds
        self.persistent_threshold = 0.55
        self.anti_persistent_threshold = 0.45
        self.strong_persistence_threshold = 0.65
        self.strong_anti_persistence_threshold = 0.35
    
    def detrend_series(self, series: np.ndarray, method: str = None) -> np.ndarray:
        """
        Detrend a time series using specified method
        
        Parameters:
        -----------
        series : np.ndarray
            Input time series
        method : str
            Detrending method
            
        Returns:
        --------
        np.ndarray
            Detrended series
        """
        if method is None:
            method = self.detrending_method
        
        n = len(series)
        x = np.arange(n)
        
        if method == 'linear':
            # Linear detrending
            slope, intercept, _, _, _ = stats.linregress(x, series)
            trend = slope * x + intercept
            return series - trend
        
        elif method == 'polynomial':
            # Polynomial detrending (degree 2)
            coeffs = np.polyfit(x, series, deg=2)
            trend = np.polyval(coeffs, x)
            return series - trend
        
        elif method == 'moving_average':
            # Moving average detrending
            window = min(20, n // 4)
            trend = pd.Series(series).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            return series - trend.values
        
        else:
            # No detrending
            return series
    
    def calculate_rs_statistic(self, series: np.ndarray, lag: int) -> float:
        """
        Calculate Rescaled Range (R/S) statistic for given lag
        
        Parameters:
        -----------
        series : np.ndarray
            Input time series
        lag : int
            Lag period
            
        Returns:
        --------
        float
            R/S statistic
        """
        if len(series) <= lag:
            return np.nan
        
        # Create non-overlapping windows
        n_windows = len(series) // lag
        if n_windows == 0:
            return np.nan
        
        rs_values = []
        
        for i in range(n_windows):
            start_idx = i * lag
            end_idx = start_idx + lag
            window_data = series[start_idx:end_idx]
            
            if len(window_data) < lag:
                continue
            
            # Calculate mean
            mean_val = np.mean(window_data)
            
            # Calculate deviations from mean
            deviations = window_data - mean_val
            
            # Calculate cumulative deviations
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(window_data, ddof=1)
            
            # Avoid division by zero
            if S > 0:
                rs_values.append(R / S)
        
        return np.mean(rs_values) if rs_values else np.nan
    
    def calculate_hurst_rs(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) method
        
        Parameters:
        -----------
        series : np.ndarray
            Input time series
            
        Returns:
        --------
        Dict containing Hurst calculation results
        """
        # Detrend the series
        detrended = self.detrend_series(series)
        
        # Calculate lags (geometric progression)
        max_lag = min(len(series) // 4, self.num_lags * 2)
        lags = np.unique(np.logspace(1, np.log10(max_lag), self.num_lags).astype(int))
        lags = lags[lags >= 2]
        
        if len(lags) < 3:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'R/S',
                'error': 'Insufficient lags for calculation'
            }
        
        # Calculate R/S statistics for each lag
        rs_stats = []
        valid_lags = []
        
        for lag in lags:
            rs_stat = self.calculate_rs_statistic(detrended, lag)
            if not np.isnan(rs_stat) and rs_stat > 0:
                rs_stats.append(rs_stat)
                valid_lags.append(lag)
        
        if len(rs_stats) < 3:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'R/S',
                'error': 'Insufficient valid R/S statistics'
            }
        
        # Log-log regression to find Hurst exponent
        try:
            log_lags = np.log10(valid_lags)
            log_rs = np.log10(rs_stats)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
            
            hurst_exponent = slope
            r_squared = r_value ** 2
            
            # Calculate confidence based on R-squared and p-value
            confidence = r_squared * (1 - p_value) if not np.isnan(p_value) else r_squared
            
            return {
                'hurst_exponent': hurst_exponent,
                'confidence': confidence,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_error': std_err,
                'method': 'R/S',
                'num_lags': len(valid_lags),
                'regression_data': {
                    'lags': valid_lags,
                    'rs_stats': rs_stats,
                    'log_lags': log_lags,
                    'log_rs': log_rs
                }
            }
            
        except Exception as e:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'R/S',
                'error': f'Regression error: {str(e)}'
            }
    
    def calculate_hurst_dfa(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA)
        
        Parameters:
        -----------
        series : np.ndarray
            Input time series
            
        Returns:
        --------
        Dict containing DFA Hurst calculation results
        """
        # Calculate cumulative sum (integration)
        y = np.cumsum(series - np.mean(series))
        
        # Define scales (window sizes)
        min_scale = 4
        max_scale = min(len(series) // 4, 100)
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 15).astype(int))
        
        if len(scales) < 3:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'DFA',
                'error': 'Insufficient scales for DFA'
            }
        
        fluctuations = []
        valid_scales = []
        
        for scale in scales:
            # Divide series into non-overlapping windows
            n_windows = len(y) // scale
            if n_windows < 1:
                continue
            
            local_fluctuations = []
            
            for i in range(n_windows):
                start_idx = i * scale
                end_idx = start_idx + scale
                window_y = y[start_idx:end_idx]
                
                # Fit polynomial trend (linear for DFA1)
                x_window = np.arange(len(window_y))
                coeffs = np.polyfit(x_window, window_y, deg=1)
                trend = np.polyval(coeffs, x_window)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean((window_y - trend) ** 2))
                local_fluctuations.append(fluctuation)
            
            if local_fluctuations:
                avg_fluctuation = np.mean(local_fluctuations)
                fluctuations.append(avg_fluctuation)
                valid_scales.append(scale)
        
        if len(fluctuations) < 3:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'DFA',
                'error': 'Insufficient valid fluctuations'
            }
        
        # Log-log regression
        try:
            log_scales = np.log10(valid_scales)
            log_fluctuations = np.log10(fluctuations)
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
            log_scales = log_scales[valid_mask]
            log_fluctuations = log_fluctuations[valid_mask]
            
            if len(log_scales) < 3:
                return {
                    'hurst_exponent': np.nan,
                    'confidence': 0.0,
                    'r_squared': 0.0,
                    'method': 'DFA',
                    'error': 'Insufficient valid points after filtering'
                }
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_fluctuations)
            
            # DFA scaling exponent (alpha) relates to Hurst exponent
            hurst_exponent = slope  # For DFA1, alpha ≈ H
            r_squared = r_value ** 2
            
            # Calculate confidence
            confidence = r_squared * (1 - p_value) if not np.isnan(p_value) else r_squared
            
            return {
                'hurst_exponent': hurst_exponent,
                'confidence': confidence,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_error': std_err,
                'method': 'DFA',
                'num_scales': len(valid_scales),
                'regression_data': {
                    'scales': valid_scales,
                    'fluctuations': fluctuations,
                    'log_scales': log_scales,
                    'log_fluctuations': log_fluctuations
                }
            }
            
        except Exception as e:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'DFA',
                'error': f'DFA regression error: {str(e)}'
            }
    
    def calculate_hurst_variance(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Hurst exponent using variance method
        
        Parameters:
        -----------
        series : np.ndarray
            Input time series
            
        Returns:
        --------
        Dict containing variance-based Hurst calculation results
        """
        # Calculate log returns
        log_returns = np.diff(np.log(series))
        
        # Define aggregation levels
        max_level = min(len(log_returns) // 10, 20)
        levels = np.arange(1, max_level + 1)
        
        if len(levels) < 3:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'Variance',
                'error': 'Insufficient aggregation levels'
            }
        
        variances = []
        valid_levels = []
        
        for level in levels:
            # Aggregate returns
            n_agg = len(log_returns) // level
            if n_agg < 5:
                continue
            
            aggregated_returns = []
            for i in range(n_agg):
                start_idx = i * level
                end_idx = start_idx + level
                agg_return = np.sum(log_returns[start_idx:end_idx])
                aggregated_returns.append(agg_return)
            
            if len(aggregated_returns) >= 5:
                variance = np.var(aggregated_returns, ddof=1)
                variances.append(variance)
                valid_levels.append(level)
        
        if len(variances) < 3:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'Variance',
                'error': 'Insufficient variance calculations'
            }
        
        # Log-log regression
        try:
            log_levels = np.log10(valid_levels)
            log_variances = np.log10(variances)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_levels, log_variances)
            
            # Hurst exponent from variance scaling
            hurst_exponent = slope / 2.0  # Variance scales as τ^(2H)
            r_squared = r_value ** 2
            
            # Calculate confidence
            confidence = r_squared * (1 - p_value) if not np.isnan(p_value) else r_squared
            
            return {
                'hurst_exponent': hurst_exponent,
                'confidence': confidence,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_error': std_err / 2.0,  # Adjust for division by 2
                'method': 'Variance',
                'num_levels': len(valid_levels),
                'regression_data': {
                    'levels': valid_levels,
                    'variances': variances,
                    'log_levels': log_levels,
                    'log_variances': log_variances
                }
            }
            
        except Exception as e:
            return {
                'hurst_exponent': np.nan,
                'confidence': 0.0,
                'r_squared': 0.0,
                'method': 'Variance',
                'error': f'Variance regression error: {str(e)}'
            }
    
    def interpret_hurst_exponent(self, hurst: float, confidence: float) -> Dict[str, Any]:
        """
        Interpret Hurst exponent value
        
        Parameters:
        -----------
        hurst : float
            Hurst exponent value
        confidence : float
            Confidence level
            
        Returns:
        --------
        Dict containing interpretation
        """
        if np.isnan(hurst):
            return {
                'market_behavior': 'UNKNOWN',
                'trend_persistence': 'UNKNOWN',
                'predictability': 'UNKNOWN',
                'trading_implication': 'No reliable signal',
                'confidence_level': 'LOW'
            }
        
        # Determine market behavior
        if hurst >= self.strong_persistence_threshold:
            market_behavior = 'STRONGLY_PERSISTENT'
            trend_persistence = 'VERY_HIGH'
            predictability = 'HIGH'
            trading_implication = 'Strong trend-following strategies'
        elif hurst >= self.persistent_threshold:
            market_behavior = 'PERSISTENT'
            trend_persistence = 'HIGH'
            predictability = 'MODERATE_TO_HIGH'
            trading_implication = 'Trend-following strategies favored'
        elif hurst <= self.strong_anti_persistence_threshold:
            market_behavior = 'STRONGLY_ANTI_PERSISTENT'
            trend_persistence = 'VERY_LOW'
            predictability = 'HIGH'
            trading_implication = 'Strong mean-reversion strategies'
        elif hurst <= self.anti_persistent_threshold:
            market_behavior = 'ANTI_PERSISTENT'
            trend_persistence = 'LOW'
            predictability = 'MODERATE_TO_HIGH'
            trading_implication = 'Mean-reversion strategies favored'
        else:
            market_behavior = 'RANDOM_WALK'
            trend_persistence = 'NEUTRAL'
            predictability = 'LOW'
            trading_implication = 'Market efficiency suggests limited predictability'
        
        # Confidence level interpretation
        if confidence >= 0.8:
            confidence_level = 'VERY_HIGH'
        elif confidence >= 0.6:
            confidence_level = 'HIGH'
        elif confidence >= 0.4:
            confidence_level = 'MODERATE'
        elif confidence >= 0.2:
            confidence_level = 'LOW'
        else:
            confidence_level = 'VERY_LOW'
        
        return {
            'market_behavior': market_behavior,
            'trend_persistence': trend_persistence,
            'predictability': predictability,
            'trading_implication': trading_implication,
            'confidence_level': confidence_level,
            'hurst_value': hurst,
            'confidence_score': confidence
        }
    
    def calculate_comprehensive_hurst(self, data: pd.DataFrame, 
                                    price_column: str = 'close') -> Dict[str, Any]:
        """
        Calculate comprehensive Hurst exponent using multiple methods
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        price_column : str
            Column to analyze
            
        Returns:
        --------
        Dict containing comprehensive Hurst analysis
        """
        if len(data) < self.min_window:
            return {
                'error': 'Insufficient data',
                'required_length': self.min_window,
                'actual_length': len(data)
            }
        
        # Extract price series
        price_series = data[price_column].dropna().values
        
        if len(price_series) < self.min_window:
            return {
                'error': 'Insufficient valid price data',
                'required_length': self.min_window,
                'actual_length': len(price_series)
            }
        
        # Calculate using multiple methods
        methods = {
            'rs': self.calculate_hurst_rs,
            'dfa': self.calculate_hurst_dfa,
            'variance': self.calculate_hurst_variance
        }
        
        results = {}
        valid_results = []
        
        for method_name, method_func in methods.items():
            try:
                result = method_func(price_series)
                results[method_name] = result
                
                if ('error' not in result and 
                    not np.isnan(result['hurst_exponent']) and
                    result['confidence'] > 0.1):
                    valid_results.append(result)
                    
            except Exception as e:
                results[method_name] = {
                    'hurst_exponent': np.nan,
                    'confidence': 0.0,
                    'method': method_name.upper(),
                    'error': f'Calculation error: {str(e)}'
                }
        
        # Calculate consensus Hurst exponent
        if valid_results:
            # Weight by confidence and R-squared
            weights = []
            hurst_values = []
            
            for result in valid_results:
                weight = result['confidence'] * result['r_squared']
                weights.append(weight)
                hurst_values.append(result['hurst_exponent'])
            
            if sum(weights) > 0:
                consensus_hurst = np.average(hurst_values, weights=weights)
                consensus_confidence = np.mean([r['confidence'] for r in valid_results])
                consensus_r_squared = np.mean([r['r_squared'] for r in valid_results])
            else:
                consensus_hurst = np.mean(hurst_values)
                consensus_confidence = np.mean([r['confidence'] for r in valid_results])
                consensus_r_squared = np.mean([r['r_squared'] for r in valid_results])
        else:
            consensus_hurst = np.nan
            consensus_confidence = 0.0
            consensus_r_squared = 0.0
        
        # Interpret consensus result
        interpretation = self.interpret_hurst_exponent(consensus_hurst, consensus_confidence)
        
        return {
            'timestamp': data.index[-1],
            'consensus_hurst': consensus_hurst,
            'consensus_confidence': consensus_confidence,
            'consensus_r_squared': consensus_r_squared,
            'interpretation': interpretation,
            'method_results': results,
            'valid_methods': len(valid_results),
            'total_methods': len(methods),
            'data_points': len(price_series),
            'summary': {
                'market_memory': 'Long' if consensus_hurst > 0.5 else 'Short' if consensus_hurst < 0.5 else 'Neutral',
                'trend_strength': abs(consensus_hurst - 0.5) * 2,  # Normalized to 0-1
                'reliability': 'High' if consensus_confidence > 0.6 else 'Medium' if consensus_confidence > 0.3 else 'Low'
            }
        }
    
    def get_rolling_hurst_analysis(self, data: pd.DataFrame, 
                                 price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate rolling Hurst exponent analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        price_column : str
            Column to analyze
            
        Returns:
        --------
        pd.DataFrame with rolling Hurst analysis
        """
        results = []
        
        for i in range(self.rolling_window, len(data)):
            subset_data = data.iloc[i-self.rolling_window:i+1]
            hurst_result = self.calculate_comprehensive_hurst(subset_data, price_column)
            
            if 'error' not in hurst_result:
                results.append({
                    'timestamp': subset_data.index[-1],
                    'hurst_exponent': hurst_result['consensus_hurst'],
                    'confidence': hurst_result['consensus_confidence'],
                    'r_squared': hurst_result['consensus_r_squared'],
                    'market_behavior': hurst_result['interpretation']['market_behavior'],
                    'trend_persistence': hurst_result['interpretation']['trend_persistence'],
                    'market_memory': hurst_result['summary']['market_memory'],
                    'trend_strength': hurst_result['summary']['trend_strength'],
                    'reliability': hurst_result['summary']['reliability'],
                    'valid_methods': hurst_result['valid_methods']
                })
            else:
                results.append({
                    'timestamp': subset_data.index[-1],
                    'hurst_exponent': np.nan,
                    'confidence': 0.0,
                    'r_squared': 0.0,
                    'market_behavior': 'UNKNOWN',
                    'trend_persistence': 'UNKNOWN',
                    'market_memory': 'Unknown',
                    'trend_strength': 0.0,
                    'reliability': 'None',
                    'valid_methods': 0
                })
        
        return pd.DataFrame(results).set_index('timestamp')

# Example usage and testing
if __name__ == "__main__":
    # Create sample data with different market regimes
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    
    # Generate sample price data with mixed behavior
    # Part 1: Trending (persistent)
    trend1 = np.cumsum(np.random.randn(100) * 0.02 + 0.001)
    
    # Part 2: Mean-reverting (anti-persistent)
    mean_rev = []
    value = 0
    for i in range(100):
        if value > 0.5:
            change = np.random.randn() * 0.01 - 0.002
        elif value < -0.5:
            change = np.random.randn() * 0.01 + 0.002
        else:
            change = np.random.randn() * 0.015
        value += change
        mean_rev.append(value)
    
    # Part 3: Random walk
    random_walk = np.cumsum(np.random.randn(100) * 0.015)
    
    # Combine all parts
    price_changes = np.concatenate([trend1, mean_rev, random_walk])
    price_base = 100 + price_changes
    
    sample_data = pd.DataFrame({
        'open': price_base * 0.999,
        'high': price_base * 1.002,
        'low': price_base * 0.998,
        'close': price_base,
        'volume': np.random.randint(1000, 10000, 300)
    }, index=dates)
    
    # Test the Hurst exponent calculator
    hurst_calc = HurstExponentCalculator(rolling_window=100)
    
    print("Testing Hurst Exponent Calculator...")
    print("=" * 50)
    
    # Test comprehensive calculation
    result = hurst_calc.calculate_comprehensive_hurst(sample_data)
    
    if 'error' not in result:
        print(f"Consensus Hurst Exponent: {result['consensus_hurst']:.4f}")
        print(f"Consensus Confidence: {result['consensus_confidence']:.3f}")
        print(f"Consensus R-squared: {result['consensus_r_squared']:.3f}")
        
        interpretation = result['interpretation']
        print(f"\nInterpretation:")
        print(f"Market Behavior: {interpretation['market_behavior']}")
        print(f"Trend Persistence: {interpretation['trend_persistence']}")
        print(f"Predictability: {interpretation['predictability']}")
        print(f"Trading Implication: {interpretation['trading_implication']}")
        print(f"Confidence Level: {interpretation['confidence_level']}")
        
        print(f"\nMethod Results:")
        for method, method_result in result['method_results'].items():
            if 'error' not in method_result:
                print(f"{method.upper()}: H={method_result['hurst_exponent']:.4f}, "
                      f"Conf={method_result['confidence']:.3f}, "
                      f"R²={method_result['r_squared']:.3f}")
            else:
                print(f"{method.upper()}: {method_result['error']}")
        
        summary = result['summary']
        print(f"\nSummary:")
        print(f"Market Memory: {summary['market_memory']}")
        print(f"Trend Strength: {summary['trend_strength']:.3f}")
        print(f"Reliability: {summary['reliability']}")
        
        # Test rolling analysis
        rolling_results = hurst_calc.get_rolling_hurst_analysis(sample_data)
        print(f"\nGenerated {len(rolling_results)} rolling Hurst calculations")
        print(f"Recent values:\n{rolling_results.tail()}")
    else:
        print(f"Error: {result['error']}")
        print(f"Required: {result['required_length']}, Actual: {result['actual_length']}")
