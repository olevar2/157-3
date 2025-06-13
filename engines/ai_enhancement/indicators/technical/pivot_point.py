"""
Pivot Point Indicator

Pivot Points are technical analysis indicators used to determine the overall trend
of the market over different time frames. They are calculated using the high, low,
and closing prices of the previous period.

Standard Pivot Point calculations:
- Pivot Point (P) = (High + Low + Close) / 3
- Support 1 (S1) = (2 * P) - High
- Support 2 (S2) = P - (High - Low)
- Resistance 1 (R1) = (2 * P) - Low
- Resistance 2 (R2) = P + (High - Low)

This indicator follows the CCI (Commodity Channel Index) gold standard template.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add the parent directory to the path to import base_indicator
from base_indicator import StandardIndicatorInterface


class PivotPoint(StandardIndicatorInterface):
    """
    Pivot Point Indicator
    
    Calculates pivot points and support/resistance levels based on 
    previous period's high, low, and close prices.
    """
    
    def __init__(self, 
                 calculation_method: str = 'standard',
                 include_additional_levels: bool = True):
        """
        Initialize the Pivot Point indicator.
        
        Args:
            calculation_method: Method for calculation ('standard', 'fibonacci', 'woodie', 'camarilla')
            include_additional_levels: Whether to include S3/R3 levels (default: True)
        """
        valid_methods = ['standard', 'fibonacci', 'woodie', 'camarilla']
        if calculation_method not in valid_methods:
            raise ValueError(f"calculation_method must be one of {valid_methods}")
            
        self.calculation_method = calculation_method
        self.include_additional_levels = include_additional_levels
        
        # Parameters dict for base class
        self.parameters = {
            'calculation_method': calculation_method,
            'include_additional_levels': include_additional_levels
        }
        
        # Initialize result storage
        self.pivot_point = []
        self.support_1 = []
        self.support_2 = []
        self.support_3 = []
        self.resistance_1 = []
        self.resistance_2 = []
        self.resistance_3 = []
        
        super().__init__()
    
    def get_metadata(self):
        """Return indicator metadata."""
        from base_indicator import IndicatorMetadata
        return IndicatorMetadata(
            name="Pivot Point",
            category="technical",
            description="Pivot points and support/resistance levels calculation",
            parameters=self.parameters,
            input_requirements=['high', 'low', 'close'],
            output_type="Dict[str, List[float]]",
            min_data_points=1
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        valid_methods = ['standard', 'fibonacci', 'woodie', 'camarilla']
        if self.calculation_method not in valid_methods:
            raise ValueError(f"calculation_method must be one of {valid_methods}")
        return True
        
    def _calculate_standard_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate standard pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with pivot levels
        """
        # Standard pivot point
        pp = (high + low + close) / 3
        
        # Support and resistance levels
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        
        levels = {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'resistance_1': r1,
            'resistance_2': r2
        }
        
        if self.include_additional_levels:
            s3 = low - 2 * (high - pp)
            r3 = high + 2 * (pp - low)
            levels.update({
                'support_3': s3,
                'resistance_3': r3
            })
        
        return levels
    
    def _calculate_fibonacci_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Fibonacci pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with pivot levels
        """
        # Fibonacci pivot point (same as standard)
        pp = (high + low + close) / 3
        
        # Fibonacci ratios
        range_hl = high - low
        s1 = pp - 0.382 * range_hl
        s2 = pp - 0.618 * range_hl
        r1 = pp + 0.382 * range_hl
        r2 = pp + 0.618 * range_hl
        
        levels = {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'resistance_1': r1,
            'resistance_2': r2
        }
        
        if self.include_additional_levels:
            s3 = pp - range_hl
            r3 = pp + range_hl
            levels.update({
                'support_3': s3,
                'resistance_3': r3
            })
        
        return levels
    
    def _calculate_woodie_pivots(self, high: float, low: float, close: float, 
                                open_price: float) -> Dict[str, float]:
        """
        Calculate Woodie's pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            open_price: Current period open
            
        Returns:
            Dictionary with pivot levels
        """
        # Woodie's pivot point calculation
        pp = (high + low + 2 * close) / 4
        
        # Support and resistance levels
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        
        levels = {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'resistance_1': r1,
            'resistance_2': r2
        }
        
        if self.include_additional_levels:
            s3 = low - 2 * (high - pp)
            r3 = high + 2 * (pp - low)
            levels.update({
                'support_3': s3,
                'resistance_3': r3
            })
        
        return levels
    
    def _calculate_camarilla_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Camarilla pivot points.
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with pivot levels
        """
        # Camarilla pivot point (same as standard)
        pp = (high + low + close) / 3
        
        # Camarilla coefficients
        range_hl = high - low
        s1 = close - 1.1 * range_hl / 12
        s2 = close - 1.1 * range_hl / 6
        r1 = close + 1.1 * range_hl / 12
        r2 = close + 1.1 * range_hl / 6
        
        levels = {
            'pivot_point': pp,
            'support_1': s1,
            'support_2': s2,
            'resistance_1': r1,
            'resistance_2': r2
        }
        
        if self.include_additional_levels:
            s3 = close - 1.1 * range_hl / 4
            r3 = close + 1.1 * range_hl / 4
            levels.update({
                'support_3': s3,
                'resistance_3': r3
            })
        
        return levels
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate Pivot Point levels.
        
        Args:
            data: Price data with columns ['open', 'high', 'low', 'close', 'volume']
                 or numpy array with same column order
                 
        Returns:
            Dictionary containing:
            - pivot_point: List of pivot point values
            - support_1/2/3: List of support level values
            - resistance_1/2/3: List of resistance level values
        """
        # Convert input to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if data.shape[1] >= 5:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
            else:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
        else:
            df = data.copy()
        
        # Initialize result lists
        n_periods = len(df)
        self.pivot_point = [np.nan] * n_periods
        self.support_1 = [np.nan] * n_periods
        self.support_2 = [np.nan] * n_periods
        self.support_3 = [np.nan] * n_periods
        self.resistance_1 = [np.nan] * n_periods
        self.resistance_2 = [np.nan] * n_periods
        self.resistance_3 = [np.nan] * n_periods
        
        # Calculate pivot points starting from second period
        for i in range(1, len(df)):
            # Use previous period's data
            prev_high = df.iloc[i-1]['high']
            prev_low = df.iloc[i-1]['low'] 
            prev_close = df.iloc[i-1]['close']
            
            # Current period's open (for Woodie's method)
            curr_open = df.iloc[i]['open'] if 'open' in df.columns else prev_close
            
            # Calculate based on selected method
            if self.calculation_method == 'standard':
                levels = self._calculate_standard_pivots(prev_high, prev_low, prev_close)
            elif self.calculation_method == 'fibonacci':
                levels = self._calculate_fibonacci_pivots(prev_high, prev_low, prev_close)
            elif self.calculation_method == 'woodie':
                levels = self._calculate_woodie_pivots(prev_high, prev_low, prev_close, curr_open)
            elif self.calculation_method == 'camarilla':
                levels = self._calculate_camarilla_pivots(prev_high, prev_low, prev_close)
            
            # Store results
            self.pivot_point[i] = levels['pivot_point']
            self.support_1[i] = levels['support_1']
            self.support_2[i] = levels['support_2']
            self.resistance_1[i] = levels['resistance_1']
            self.resistance_2[i] = levels['resistance_2']
            
            if self.include_additional_levels:
                self.support_3[i] = levels.get('support_3', np.nan)
                self.resistance_3[i] = levels.get('resistance_3', np.nan)
        
        return self._get_results()
    
    def _get_results(self) -> Dict[str, List[float]]:
        """Get the calculation results."""
        results = {
            'pivot_point': self.pivot_point,
            'support_1': self.support_1,
            'support_2': self.support_2,
            'resistance_1': self.resistance_1,
            'resistance_2': self.resistance_2
        }
        
        if self.include_additional_levels:
            results.update({
                'support_3': self.support_3,
                'resistance_3': self.resistance_3
            })
        
        return results
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, List[float]]:
        """
        Get trading signals based on price interaction with pivot levels.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary with support/resistance touch signals
        """
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(
            data, columns=['open', 'high', 'low', 'close', 'volume'][:data.shape[1]]
        )
        
        results = self.calculate(data)
        
        support_signals = [0.0] * len(df)
        resistance_signals = [0.0] * len(df)
        
        for i in range(1, len(df)):
            if not np.isnan(results['pivot_point'][i]):
                current_low = df.iloc[i]['low']
                current_high = df.iloc[i]['high']
                
                # Check for support level touches (price near support levels)
                tolerance = 0.002  # 0.2% tolerance
                
                for level_name in ['support_1', 'support_2', 'support_3']:
                    if level_name in results and not np.isnan(results[level_name][i]):
                        support_level = results[level_name][i]
                        if abs(current_low - support_level) / support_level <= tolerance:
                            support_signals[i] = 1.0
                            break
                
                # Check for resistance level touches
                for level_name in ['resistance_1', 'resistance_2', 'resistance_3']:
                    if level_name in results and not np.isnan(results[level_name][i]):
                        resistance_level = results[level_name][i]
                        if abs(current_high - resistance_level) / resistance_level <= tolerance:
                            resistance_signals[i] = 1.0
                            break
        
        return {
            'support_signals': support_signals,
            'resistance_signals': resistance_signals,
            'pivot_levels': results
        }


def test_pivot_point():
    """Test the Pivot Point indicator with sample data."""
    print("Testing Pivot Point Indicator...")
    
    # Create sample OHLC data
    np.random.seed(42)
    n_periods = 20
    
    data = []
    base_price = 100
    
    for i in range(n_periods):
        # Create realistic OHLC data
        open_price = base_price + np.random.randn() * 0.5
        close_price = open_price + np.random.randn() * 1.0
        high = max(open_price, close_price) + abs(np.random.randn()) * 0.5
        low = min(open_price, close_price) - abs(np.random.randn()) * 0.5
        volume = 1000 + np.random.randint(0, 500)
        
        data.append([open_price, high, low, close_price, volume])
        base_price = close_price  # Trending price
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Test different calculation methods
    methods = ['standard', 'fibonacci', 'woodie', 'camarilla']
    
    for method in methods:
        print(f"\nTesting {method.title()} Pivot Points:")
        
        indicator = PivotPoint(
            calculation_method=method,
            include_additional_levels=True
        )
        
        results = indicator.calculate(df)
        
        # Display results for last few periods
        for i in range(max(0, len(df) - 3), len(df)):
            if not np.isnan(results['pivot_point'][i]):
                print(f"Period {i}: PP={results['pivot_point'][i]:.2f}, "
                      f"S1={results['support_1'][i]:.2f}, "
                      f"R1={results['resistance_1'][i]:.2f}")
        
        # Test signals
        signals = indicator.get_signals(df)
        support_touches = sum(signals['support_signals'])
        resistance_touches = sum(signals['resistance_signals'])
        
        print(f"Support touches: {support_touches}, Resistance touches: {resistance_touches}")
    
    print("\nPivot Point test completed successfully!")
    return True


if __name__ == "__main__":
    test_pivot_point()