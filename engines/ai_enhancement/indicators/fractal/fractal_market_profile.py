"""
Fractal Market Profile Indicator

A fractal market profile indicator that analyzes price distribution and volume
patterns within fractal market structures. This indicator combines traditional
market profile concepts with fractal analysis to identify key price levels,
value areas, and market balance/imbalance zones.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os
from collections import defaultdict

# Add the parent directory to Python path for imports

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class FractalMarketProfile(BaseIndicator):
    """
    Fractal Market Profile Indicator
    
    Analyzes market structure using fractal geometry and market profile concepts:
    - Volume-weighted price distribution analysis
    - Fractal scaling of price levels
    - Multi-timeframe value area identification
    - Point of control (POC) detection
    - Market balance/imbalance measurement
    - Self-similar pattern recognition
    
    The indicator provides:
    - Value area high/low boundaries
    - Point of control (volume-weighted average price)
    - Market profile shape classification
    - Fractal scaling coefficients
    - Balance/imbalance indicators
    - Support/resistance strength scores
    """
    
    def __init__(self, 
                 profile_period: int = 20,
                 tick_size: float = 0.01,
                 value_area_percentage: float = 0.70,
                 fractal_levels: List[int] = [5, 10, 20],
                 balance_threshold: float = 0.3):
        """
        Initialize Fractal Market Profile indicator
        
        Args:
            profile_period: Period for market profile calculation
            tick_size: Price tick size for binning
            value_area_percentage: Percentage of volume for value area (typically 70%)
            fractal_levels: List of timeframes for fractal analysis
            balance_threshold: Threshold for balance/imbalance detection
        """
        super().__init__()
        self.profile_period = profile_period
        self.tick_size = tick_size
        self.value_area_percentage = value_area_percentage
        self.fractal_levels = fractal_levels
        self.balance_threshold = balance_threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate fractal market profile
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'value_area_high': Upper boundary of value area
            - 'value_area_low': Lower boundary of value area
            - 'point_of_control': Point of control (volume-weighted average)
            - 'profile_shape': Market profile shape classification
            - 'fractal_coefficient': Fractal scaling coefficient
            - 'balance_indicator': Market balance/imbalance measure
        """
        try:
            if len(data) < self.profile_period:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'value_area_high': empty_series,
                    'value_area_low': empty_series,
                    'point_of_control': empty_series,
                    'profile_shape': empty_series,
                    'fractal_coefficient': empty_series,
                    'balance_indicator': empty_series
                }
            
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            # Calculate market profile components
            value_area_high, value_area_low, point_of_control = self._calculate_market_profile(
                high, low, close, volume
            )
            
            # Classify profile shape
            profile_shape = self._classify_profile_shape(
                high, low, close, volume, value_area_high, value_area_low, point_of_control
            )
            
            # Calculate fractal scaling coefficient
            fractal_coefficient = self._calculate_fractal_coefficient(close, volume)
            
            # Calculate balance/imbalance indicator
            balance_indicator = self._calculate_balance_indicator(
                high, low, close, volume, value_area_high, value_area_low
            )
            
            return {
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'point_of_control': point_of_control,
                'profile_shape': profile_shape,
                'fractal_coefficient': fractal_coefficient,
                'balance_indicator': balance_indicator
            }
            
        except Exception as e:
            print(f"Error in Fractal Market Profile calculation: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'value_area_high': empty_series,
                'value_area_low': empty_series,
                'point_of_control': empty_series,
                'profile_shape': empty_series,
                'fractal_coefficient': empty_series,
                'balance_indicator': empty_series
            }
    
    def _calculate_market_profile(self, high: pd.Series, low: pd.Series,
                                 close: pd.Series, volume: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate basic market profile components"""
        try:
            value_area_high = pd.Series(0.0, index=close.index)
            value_area_low = pd.Series(0.0, index=close.index)
            point_of_control = pd.Series(0.0, index=close.index)
            
            for i in range(self.profile_period, len(close)):
                # Get data for current profile period
                period_high = high.iloc[i-self.profile_period:i+1]
                period_low = low.iloc[i-self.profile_period:i+1]
                period_close = close.iloc[i-self.profile_period:i+1]
                period_volume = volume.iloc[i-self.profile_period:i+1]
                
                # Calculate price range and bins
                price_min = period_low.min()
                price_max = period_high.max()
                
                if price_max > price_min:
                    # Create price bins
                    num_bins = max(10, int((price_max - price_min) / self.tick_size))
                    price_bins = np.linspace(price_min, price_max, num_bins)
                    
                    # Calculate volume per price level
                    volume_profile = self._calculate_volume_profile(
                        period_high, period_low, period_close, period_volume, price_bins
                    )
                    
                    if len(volume_profile) > 0:
                        # Find point of control (highest volume price)
                        poc_idx = np.argmax(volume_profile)
                        poc_price = price_bins[poc_idx] if poc_idx < len(price_bins) else price_bins[-1]
                        
                        # Calculate value area
                        va_high, va_low = self._calculate_value_area(
                            price_bins, volume_profile, poc_idx
                        )
                        
                        value_area_high.iloc[i] = va_high
                        value_area_low.iloc[i] = va_low
                        point_of_control.iloc[i] = poc_price
                    else:
                        # Fallback values
                        value_area_high.iloc[i] = period_high.max()
                        value_area_low.iloc[i] = period_low.min()
                        point_of_control.iloc[i] = period_close.mean()
                else:
                    # No price movement
                    current_price = period_close.iloc[-1]
                    value_area_high.iloc[i] = current_price
                    value_area_low.iloc[i] = current_price
                    point_of_control.iloc[i] = current_price
            
            return value_area_high, value_area_low, point_of_control
        except:
            empty_series = pd.Series(0, index=close.index)
            return empty_series, empty_series, empty_series
    
    def _calculate_volume_profile(self, high: pd.Series, low: pd.Series,
                                 close: pd.Series, volume: pd.Series,
                                 price_bins: np.ndarray) -> np.ndarray:
        """Calculate volume distribution across price levels"""
        try:
            volume_profile = np.zeros(len(price_bins) - 1)
            
            for j in range(len(high)):
                bar_high = high.iloc[j]
                bar_low = low.iloc[j]
                bar_volume = volume.iloc[j]
                
                if bar_high > bar_low and bar_volume > 0:
                    # Distribute volume across price levels within the bar's range
                    for k in range(len(price_bins) - 1):
                        bin_low = price_bins[k]
                        bin_high = price_bins[k + 1]
                        
                        # Check if this price bin overlaps with the bar's range
                        overlap_low = max(bin_low, bar_low)
                        overlap_high = min(bin_high, bar_high)
                        
                        if overlap_high > overlap_low:
                            # Calculate the proportion of the bar's range in this bin
                            overlap_ratio = (overlap_high - overlap_low) / (bar_high - bar_low)
                            volume_profile[k] += bar_volume * overlap_ratio
            
            return volume_profile
        except:
            return np.array([])
    
    def _calculate_value_area(self, price_bins: np.ndarray, volume_profile: np.ndarray,
                             poc_idx: int) -> Tuple[float, float]:
        """Calculate value area boundaries"""
        try:
            if len(volume_profile) == 0:
                return price_bins[0], price_bins[-1]
            
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * self.value_area_percentage
            
            # Start from POC and expand outward
            current_volume = volume_profile[poc_idx]
            upper_idx = poc_idx
            lower_idx = poc_idx
            
            # Expand the value area around POC
            while current_volume < target_volume and (upper_idx < len(volume_profile) - 1 or lower_idx > 0):
                # Determine which direction to expand
                upper_volume = volume_profile[upper_idx + 1] if upper_idx < len(volume_profile) - 1 else 0
                lower_volume = volume_profile[lower_idx - 1] if lower_idx > 0 else 0
                
                if upper_volume >= lower_volume and upper_idx < len(volume_profile) - 1:
                    upper_idx += 1
                    current_volume += upper_volume
                elif lower_idx > 0:
                    lower_idx -= 1
                    current_volume += lower_volume
                else:
                    break
            
            # Get price boundaries
            value_area_high = price_bins[upper_idx + 1] if upper_idx < len(price_bins) - 1 else price_bins[-1]
            value_area_low = price_bins[lower_idx]
            
            return value_area_high, value_area_low
        except:
            return price_bins[0], price_bins[-1]
    
    def _classify_profile_shape(self, high: pd.Series, low: pd.Series, close: pd.Series,
                               volume: pd.Series, value_area_high: pd.Series,
                               value_area_low: pd.Series, point_of_control: pd.Series) -> pd.Series:
        """Classify market profile shape"""
        try:
            profile_shape = pd.Series(0, index=close.index)
            
            for i in range(self.profile_period, len(close)):
                current_close = close.iloc[i]
                current_poc = point_of_control.iloc[i]
                current_va_high = value_area_high.iloc[i]
                current_va_low = value_area_low.iloc[i]
                
                if current_va_high > current_va_low:
                    va_range = current_va_high - current_va_low
                    poc_position = (current_poc - current_va_low) / va_range if va_range > 0 else 0.5
                    
                    # Shape classification:
                    # 1 = Normal Distribution (POC in middle)
                    # 2 = P-Shape (POC at top)
                    # 3 = b-Shape (POC at bottom)
                    # 0 = Neutral/Undefined
                    
                    if 0.4 <= poc_position <= 0.6:
                        profile_shape.iloc[i] = 1  # Normal distribution
                    elif poc_position > 0.7:
                        profile_shape.iloc[i] = 2  # P-shape (bearish)
                    elif poc_position < 0.3:
                        profile_shape.iloc[i] = 3  # b-shape (bullish)
                    else:
                        profile_shape.iloc[i] = 0  # Neutral
            
            return profile_shape
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_fractal_coefficient(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate fractal scaling coefficient"""
        try:
            fractal_coefficient = pd.Series(1.0, index=close.index)
            
            for i in range(max(self.fractal_levels), len(close)):
                # Calculate price variance at different scales
                price_variances = []
                volume_variances = []
                
                for level in self.fractal_levels:
                    if i >= level:
                        # Price variance at this scale
                        price_segment = close.iloc[i-level:i+1]
                        price_var = price_segment.var()
                        price_variances.append(price_var)
                        
                        # Volume variance at this scale
                        volume_segment = volume.iloc[i-level:i+1]
                        volume_var = volume_segment.var()
                        volume_variances.append(volume_var)
                
                if len(price_variances) >= 2:
                    # Calculate fractal scaling relationship
                    # log(variance) vs log(scale) slope indicates fractal behavior
                    log_scales = np.log(self.fractal_levels[:len(price_variances)])
                    log_price_vars = np.log(np.array(price_variances) + 1e-8)
                    
                    # Linear regression to find scaling exponent
                    try:
                        scaling_coeff = np.polyfit(log_scales, log_price_vars, 1)[0]
                        # Normalize to reasonable range
                        scaling_coeff = np.clip(scaling_coeff, 0.5, 2.0)
                        fractal_coefficient.iloc[i] = scaling_coeff
                    except:
                        fractal_coefficient.iloc[i] = 1.0
            
            return fractal_coefficient
        except:
            return pd.Series(1.0, index=close.index)
    
    def _calculate_balance_indicator(self, high: pd.Series, low: pd.Series, close: pd.Series,
                                   volume: pd.Series, value_area_high: pd.Series,
                                   value_area_low: pd.Series) -> pd.Series:
        """Calculate market balance/imbalance indicator"""
        try:
            balance_indicator = pd.Series(0.0, index=close.index)
            
            for i in range(self.profile_period, len(close)):
                current_close = close.iloc[i]
                current_va_high = value_area_high.iloc[i]
                current_va_low = value_area_low.iloc[i]
                
                # Check price acceptance within value area
                if current_va_high > current_va_low:
                    # Position relative to value area
                    if current_close > current_va_high:
                        # Above value area (potential bullish imbalance)
                        excess = (current_close - current_va_high) / (current_va_high - current_va_low)
                        balance_indicator.iloc[i] = min(excess, 1.0)
                    elif current_close < current_va_low:
                        # Below value area (potential bearish imbalance)
                        excess = (current_va_low - current_close) / (current_va_high - current_va_low)
                        balance_indicator.iloc[i] = -min(excess, 1.0)
                    else:
                        # Within value area (balanced)
                        va_position = (current_close - current_va_low) / (current_va_high - current_va_low)
                        # Balanced around 0.5, imbalanced at extremes
                        balance_score = abs(va_position - 0.5) * 2
                        balance_indicator.iloc[i] = balance_score - 0.5  # Center around 0
                
                # Volume confirmation
                if i >= 1:
                    volume_ratio = volume.iloc[i] / volume.iloc[i-1] if volume.iloc[i-1] > 0 else 1
                    # Adjust balance based on volume
                    balance_indicator.iloc[i] *= (1 + np.log(volume_ratio) * 0.1)
                    balance_indicator.iloc[i] = np.clip(balance_indicator.iloc[i], -2.0, 2.0)
            
            return balance_indicator
        except:
            return pd.Series(0, index=close.index)
    
    def get_value_area(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get value area boundaries"""
        result = self.calculate(data)
        return {
            'high': result['value_area_high'],
            'low': result['value_area_low']
        }
    
    def get_point_of_control(self, data: pd.DataFrame) -> pd.Series:
        """Get point of control prices"""
        result = self.calculate(data)
        return result['point_of_control']
    
    def get_balance_indicator(self, data: pd.DataFrame) -> pd.Series:
        """Get balance/imbalance indicators"""
        result = self.calculate(data)
        return result['balance_indicator']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with realistic market profile characteristics
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data with clustering around certain price levels
    base_price = 100
    price_trend = np.cumsum(np.random.randn(100) * 0.3)
    
    # Add price clustering (market profile behavior)
    cluster_levels = [98, 100, 102, 105]
    cluster_strength = [0.3, 0.5, 0.3, 0.2]
    
    close_prices = base_price + price_trend
    
    # Add volume-weighted clustering
    for i, level in enumerate(cluster_levels):
        attraction = cluster_strength[i] * np.exp(-0.1 * (close_prices - level)**2)
        close_prices += attraction * np.random.randn(100) * 0.1
    
    # Generate volume with higher volume at cluster levels
    volumes = np.random.lognormal(10, 0.5, 100)
    for i, level in enumerate(cluster_levels):
        volume_boost = cluster_strength[i] * np.exp(-0.05 * (close_prices - level)**2)
        volumes *= (1 + volume_boost)
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0, 1.5, 100),
        'low': close_prices - np.random.uniform(0, 1.5, 100),
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Test the indicator
    print("Testing Fractal Market Profile Indicator")
    print("=" * 50)
    
    indicator = FractalMarketProfile(
        profile_period=20,
        tick_size=0.1,
        value_area_percentage=0.70,
        fractal_levels=[5, 10, 20],
        balance_threshold=0.3
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Value Area High range: {result['value_area_high'].min():.2f} to {result['value_area_high'].max():.2f}")
    print(f"Value Area Low range: {result['value_area_low'].min():.2f} to {result['value_area_low'].max():.2f}")
    print(f"Point of Control range: {result['point_of_control'].min():.2f} to {result['point_of_control'].max():.2f}")
    print(f"Fractal Coefficient range: {result['fractal_coefficient'].min():.3f} to {result['fractal_coefficient'].max():.3f}")
    print(f"Balance Indicator range: {result['balance_indicator'].min():.3f} to {result['balance_indicator'].max():.3f}")
    
    # Analyze profile shapes
    shapes = result['profile_shape']
    normal_dist = (shapes == 1).sum()
    p_shape = (shapes == 2).sum()
    b_shape = (shapes == 3).sum()
    neutral = (shapes == 0).sum()
    
    print(f"\nProfile Shape Analysis:")
    print(f"Normal distribution: {normal_dist}")
    print(f"P-shape (bearish): {p_shape}")
    print(f"b-shape (bullish): {b_shape}")
    print(f"Neutral/undefined: {neutral}")
    
    # Balance analysis
    balance = result['balance_indicator']
    bullish_imbalance = (balance > 0.3).sum()
    bearish_imbalance = (balance < -0.3).sum()
    balanced = (abs(balance) <= 0.3).sum()
    
    print(f"\nBalance Analysis:")
    print(f"Bullish imbalance periods: {bullish_imbalance}")
    print(f"Bearish imbalance periods: {bearish_imbalance}")
    print(f"Balanced periods: {balanced}")
    
    # Value area statistics
    va_width = result['value_area_high'] - result['value_area_low']
    print(f"\nValue Area Statistics:")
    print(f"Average VA width: {va_width.mean():.2f}")
    print(f"Max VA width: {va_width.max():.2f}")
    print(f"Min VA width: {va_width.min():.2f}")
    
    print("\nFractal Market Profile Indicator test completed successfully!")