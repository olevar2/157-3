#!/usr/bin/env python3
"""
Test individual indicator to debug DataFrame ambiguity error
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

def test_obv():
    """Test OnBalanceVolume specifically"""
    try:
        from engines.indicator_registry import IndicatorRegistry
        registry = IndicatorRegistry()
        
        # Get OBV
        obv_class = registry.get_indicator('volume.OnBalanceVolume')
        if not obv_class:
            print("OnBalanceVolume not found in registry")
            return
            
        print(f"Found OBV class: {obv_class}")
        
        # Create sample data
        sample_data = create_sample_data()
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {list(sample_data.columns)}")
        
        # Try to instantiate
        try:
            obv = obv_class()
            print("✓ OBV instantiated successfully")
        except Exception as e:
            print(f"✗ OBV instantiation failed: {e}")
            return
            
        # Try to calculate
        try:
            result = obv.calculate(sample_data['close'], sample_data['volume'])
            print(f"✓ OBV calculation successful, result type: {type(result)}")
            print(f"  Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            if hasattr(result, 'head'):
                print(f"  First few values: {result.head()}")
        except Exception as e:
            print(f"✗ OBV calculation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obv()
