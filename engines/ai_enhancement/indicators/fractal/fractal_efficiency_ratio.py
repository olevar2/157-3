"""
Fractal Efficiency Ratio Indicator

A fractal efficiency ratio indicator that measures the efficiency of price movement
by comparing the linear distance between two points to the actual path taken.
This indicator helps identify trending vs. ranging market conditions and provides
insights into market fractal behavior.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class FractalEfficiencyRatio(BaseIndicator):
    """
    Fractal Efficiency Ratio Indicator
    
    Measures the efficiency of price movement using fractal geometry concepts:
    - Efficiency Ratio = Linear Distance / Path Distance
    - Values near 1.0 indicate efficient trending movement
    - Values near 0.0 indicate inefficient ranging movement
    - Fractal dimension estimation
    - Trend strength measurement
    - Market regime identification
    
    The indicator provides:
    - Efficiency ratio (0-1 scale)
    - Fractal dimension estimate
    - Trend strength score
    - Market regime classification
    - Volatility-adjusted efficiency
    """
    
    def __init__(self, 
                 period: int = 20,
                 smoothing_period: int = 5,
                 volatility_adjustment: bool = True,
                 regime_threshold: float = 0.3):
        """
        Initialize Fractal Efficiency Ratio indicator
        
        Args:
            period: Lookback period for efficiency calculation
            smoothing_period: Period for smoothing the results
            volatility_adjustment: Whether to adjust for volatility
            regime_threshold: Threshold for trend/range regime classification
        """
        super().__init__()
        self.period = period
        self.smoothing_period = smoothing_period
        self.volatility_adjustment = volatility_adjustment
        self.regime_threshold = regime_threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate fractal efficiency ratio
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'efficiency_ratio': Price movement efficiency (0-1)
            - 'fractal_dimension': Estimated fractal dimension
            - 'trend_strength': Trend strength score
            - 'market_regime': Market regime classification
            - 'volatility_adjusted_efficiency': Volatility-adjusted efficiency
        """
        try:
            if len(data) < self.period:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'efficiency_ratio': empty_series,
                    'fractal_dimension': empty_series,
                    'trend_strength': empty_series,
                    'market_regime': empty_series,
                    'volatility_adjusted_efficiency': empty_series
                }
            
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calculate basic efficiency ratio
            efficiency_ratio = self._calculate_efficiency_ratio(close)
            
            # Calculate fractal dimension
            fractal_dimension = self._calculate_fractal_dimension(close)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(efficiency_ratio, close)
            
            # Classify market regime
            market_regime = self._classify_market_regime(efficiency_ratio)
            
            # Calculate volatility-adjusted efficiency
            if self.volatility_adjustment:
                volatility_adjusted_efficiency = self._calculate_volatility_adjusted_efficiency(
                    efficiency_ratio, high, low, close
                )
            else:
                volatility_adjusted_efficiency = efficiency_ratio.copy()
            
            return {
                'efficiency_ratio': efficiency_ratio,
                'fractal_dimension': fractal_dimension,
                'trend_strength': trend_strength,
                'market_regime': market_regime,
                'volatility_adjusted_efficiency': volatility_adjusted_efficiency
            }
            
        except Exception as e:
            print(f"Error in Fractal Efficiency Ratio calculation: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'efficiency_ratio': empty_series,
                'fractal_dimension': empty_series,
                'trend_strength': empty_series,
                'market_regime': empty_series,
                'volatility_adjusted_efficiency': empty_series
            }
    
    def _calculate_efficiency_ratio(self, close: pd.Series) -> pd.Series:
        """Calculate basic efficiency ratio"""
        try:
            efficiency_ratio = pd.Series(0.0, index=close.index)
            
            for i in range(self.period, len(close)):
                # Get price segment
                price_segment = close.iloc[i-self.period:i+1]
                
                # Calculate linear distance (start to end)
                linear_distance = abs(price_segment.iloc[-1] - price_segment.iloc[0])
                
                # Calculate path distance (sum of absolute changes)
                path_distance = price_segment.diff().abs().sum()
                
                # Calculate efficiency ratio
                if path_distance > 0:
                    efficiency = linear_distance / path_distance
                else:
                    efficiency = 0
                
                efficiency_ratio.iloc[i] = min(efficiency, 1.0)  # Cap at 1.0
            
            # Apply smoothing
            if self.smoothing_period > 1:
                efficiency_ratio = efficiency_ratio.rolling(
                    window=self.smoothing_period, min_periods=1
                ).mean()
            
            return efficiency_ratio
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_fractal_dimension(self, close: pd.Series) -> pd.Series:
        """Calculate fractal dimension estimate"""
        try:
            fractal_dimension = pd.Series(1.0, index=close.index)
            
            for i in range(self.period, len(close)):
                # Get price segment
                price_segment = close.iloc[i-self.period:i+1]
                
                # Calculate path length at different scales
                scales = [1, 2, 4, 8]
                path_lengths = []
                
                for scale in scales:
                    if scale < len(price_segment):
                        # Downsample the data
                        downsampled = price_segment.iloc[::scale]
                        if len(downsampled) > 1:
                            path_length = downsampled.diff().abs().sum()
                            path_lengths.append(path_length)
                
                if len(path_lengths) >= 2:
                    # Estimate fractal dimension using log-log slope
                    # D = 1 - slope of log(path_length) vs log(scale)
                    log_scales = np.log(scales[:len(path_lengths)])
                    log_lengths = np.log(np.array(path_lengths) + 1e-8)
                    
                    # Simple linear regression
                    if len(log_scales) > 1:
                        slope = np.polyfit(log_scales, log_lengths, 1)[0]
                        dimension = 1 - slope
                        # Bound between 1 and 2 (typical range for price series)
                        dimension = np.clip(dimension, 1.0, 2.0)
                        fractal_dimension.iloc[i] = dimension
            
            return fractal_dimension
        except:
            return pd.Series(1.5, index=close.index)  # Default fractal dimension
    
    def _calculate_trend_strength(self, efficiency_ratio: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate trend strength based on efficiency and price direction"""
        try:
            trend_strength = pd.Series(0.0, index=close.index)
            
            for i in range(self.period, len(close)):
                # Get efficiency for this period
                current_efficiency = efficiency_ratio.iloc[i]
                
                # Calculate price direction over the period
                start_price = close.iloc[i-self.period]
                end_price = close.iloc[i]
                price_change = (end_price - start_price) / start_price if start_price != 0 else 0
                
                # Trend strength combines efficiency with price direction magnitude
                strength = current_efficiency * abs(price_change) * 100
                trend_strength.iloc[i] = strength
            
            # Apply smoothing
            if self.smoothing_period > 1:
                trend_strength = trend_strength.rolling(
                    window=self.smoothing_period, min_periods=1
                ).mean()
            
            return trend_strength
        except:
            return pd.Series(0, index=close.index)
    
    def _classify_market_regime(self, efficiency_ratio: pd.Series) -> pd.Series:
        """Classify market regime based on efficiency ratio"""
        try:
            market_regime = pd.Series(0, index=efficiency_ratio.index)
            
            # Apply regime classification
            # 1 = Strong Trend, 0 = Range, -1 = Weak Trend
            market_regime = np.where(
                efficiency_ratio >= self.regime_threshold * 2, 1,  # Strong trend
                np.where(
                    efficiency_ratio >= self.regime_threshold, 0,   # Weak trend/transition
                    -1  # Range/choppy
                )
            )
            
            market_regime = pd.Series(market_regime, index=efficiency_ratio.index)
            
            # Apply smoothing to reduce noise
            if self.smoothing_period > 1:
                market_regime = market_regime.rolling(
                    window=self.smoothing_period, min_periods=1
                ).mean().round()
            
            return market_regime
        except:
            return pd.Series(0, index=efficiency_ratio.index)
    
    def _calculate_volatility_adjusted_efficiency(self, efficiency_ratio: pd.Series,
                                                 high: pd.Series, low: pd.Series,
                                                 close: pd.Series) -> pd.Series:
        """Calculate volatility-adjusted efficiency ratio"""
        try:
            # Calculate True Range-based volatility
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            volatility = true_range.rolling(window=self.period, min_periods=1).mean()
            
            # Normalize volatility
            vol_normalized = volatility / close
            
            # Adjust efficiency by volatility
            # Higher volatility reduces efficiency, lower volatility increases it
            vol_factor = 1 / (1 + vol_normalized * 10)  # Scaling factor
            volatility_adjusted = efficiency_ratio * vol_factor
            
            return volatility_adjusted
        except:
            return efficiency_ratio.copy()
    
    def _calculate_directional_efficiency(self, close: pd.Series) -> pd.Series:
        """Calculate directional efficiency (considering price direction)"""
        try:
            directional_efficiency = pd.Series(0.0, index=close.index)
            
            for i in range(self.period, len(close)):
                # Get price segment
                price_segment = close.iloc[i-self.period:i+1]
                
                # Calculate directional distance (considering direction)
                net_change = price_segment.iloc[-1] - price_segment.iloc[0]
                
                # Calculate total path distance
                path_distance = price_segment.diff().abs().sum()
                
                # Directional efficiency (can be negative)
                if path_distance > 0:
                    dir_efficiency = net_change / path_distance
                else:
                    dir_efficiency = 0
                
                directional_efficiency.iloc[i] = dir_efficiency
            
            return directional_efficiency
        except:
            return pd.Series(0, index=close.index)
    
    def get_efficiency_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Get efficiency ratios"""
        result = self.calculate(data)
        return result['efficiency_ratio']
    
    def get_fractal_dimension(self, data: pd.DataFrame) -> pd.Series:
        """Get fractal dimensions"""
        result = self.calculate(data)
        return result['fractal_dimension']
    
    def get_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Get market regime classifications"""
        result = self.calculate(data)
        return result['market_regime']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with different market conditions
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    
    # Generate sample data with trending and ranging periods
    t = np.arange(150)
    
    # Create different market regimes
    trending_period = 2 * t[:50]  # Strong trend
    ranging_period = 100 + 5 * np.sin(t[50:100] * 0.5)  # Range-bound
    choppy_period = 110 + np.random.randn(50) * 3  # Choppy movement
    
    close_prices = np.concatenate([trending_period, ranging_period, choppy_period])
    noise = np.random.randn(150) * 0.5
    close_prices += noise
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 2, 150),
        'low': close_prices - np.random.uniform(0, 2, 150),
        'close': close_prices,
        'volume': np.random.lognormal(10, 0.3, 150)
    }, index=dates)
    
    # Test the indicator
    print("Testing Fractal Efficiency Ratio Indicator")
    print("=" * 50)
    
    indicator = FractalEfficiencyRatio(
        period=20,
        smoothing_period=3,
        volatility_adjustment=True,
        regime_threshold=0.3
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Efficiency ratio range: {result['efficiency_ratio'].min():.3f} to {result['efficiency_ratio'].max():.3f}")
    print(f"Fractal dimension range: {result['fractal_dimension'].min():.3f} to {result['fractal_dimension'].max():.3f}")
    print(f"Trend strength range: {result['trend_strength'].min():.3f} to {result['trend_strength'].max():.3f}")
    
    # Analyze market regimes
    regime = result['market_regime']
    strong_trend = (regime == 1).sum()
    weak_trend = (regime == 0).sum()
    ranging = (regime == -1).sum()
    
    print(f"\nMarket Regime Analysis:")
    print(f"Strong trend periods: {strong_trend}")
    print(f"Weak trend/transition periods: {weak_trend}")
    print(f"Ranging/choppy periods: {ranging}")
    
    # Show efficiency statistics by period
    print(f"\nEfficiency by period:")
    print(f"Period 1 (trend): avg efficiency = {result['efficiency_ratio'].iloc[20:50].mean():.3f}")
    print(f"Period 2 (range): avg efficiency = {result['efficiency_ratio'].iloc[70:100].mean():.3f}")
    print(f"Period 3 (choppy): avg efficiency = {result['efficiency_ratio'].iloc[120:150].mean():.3f}")
    
    # Volatility adjustment comparison
    print(f"\nVolatility Adjustment:")
    print(f"Original efficiency avg: {result['efficiency_ratio'].mean():.3f}")
    print(f"Vol-adjusted efficiency avg: {result['volatility_adjusted_efficiency'].mean():.3f}")
    
    print("\nFractal Efficiency Ratio Indicator test completed successfully!")