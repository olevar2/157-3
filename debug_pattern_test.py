#!/usr/bin/env python3
"""
Test BasePatternEngine to debug DataFrame ambiguity error
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the platform root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': np.nan,
        'low': np.nan,
        'close': np.nan,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Generate proper OHLC data
    for i in range(len(data)):
        open_price = data.iloc[i]['open']
        daily_range = abs(np.random.randn() * 2)
        data.iloc[i, data.columns.get_loc('high')] = open_price + daily_range
        data.iloc[i, data.columns.get_loc('low')] = open_price - daily_range
        data.iloc[i, data.columns.get_loc('close')] = open_price + np.random.randn() * 1
    
    return data

def test_base_pattern_engine():
    """Test BasePatternEngine that's failing with DataFrame ambiguity"""
    try:
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        # Get BasePatternEngine
        pattern_class = registry.get_indicator('engines.BasePatternEngine')
        if not pattern_class:
            print("BasePatternEngine not found in registry")
            return
            
        print(f"Found BasePatternEngine class: {pattern_class}")
        
        # Create sample data
        sample_data = create_sample_data()
        print(f"Sample data shape: {sample_data.shape}")
        
        # Try to instantiate
        try:
            pattern = pattern_class()
            print("✓ BasePatternEngine instantiated successfully")
        except Exception as e:
            print(f"✗ BasePatternEngine instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            return
            
        # Try to calculate
        try:
            result = pattern.calculate(sample_data)
            print(f"✓ BasePatternEngine calculation successful, result type: {type(result)}")
        except Exception as e:
            print(f"✗ BasePatternEngine calculation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_base_pattern_engine()
