#!/usr/bin/env python3
"""
Simple test to debug specific Gann indicator errors
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_simple_test_data(n_bars=50):
    """Create simple test data with proper datetime index"""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')  # Use 'D' instead of 'H'
    
    # Generate simple price data
    np.random.seed(42)
    base_price = 100.0
    
    data = []
    for i, date in enumerate(dates):
        price = base_price + i * 0.5 + np.random.normal(0, 1)
        data.append({
            'timestamp': date,
            'open': price,
            'high': price * 1.02,
            'low': price * 0.98,
            'close': price,
            'volume': 1000
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_gann_time_cycles():
    """Test GannTimeCycles specifically"""
    print("Testing GannTimeCycles...")
    
    try:
        from engines.gann.gann_time_cycles import GannTimeCycles
        
        test_data = create_simple_test_data(100)
        print(f"Test data shape: {test_data.shape}")
        print(f"Test data index type: {type(test_data.index[0])}")
        print(f"Test data columns: {test_data.columns.tolist()}")
        
        gann_cycles = GannTimeCycles()
        result = gann_cycles.calculate(test_data)
        
        print("✅ GannTimeCycles: SUCCESS")
        print(f"Result keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ GannTimeCycles: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_price_time_relationships():
    """Test PriceTimeRelationships specifically"""
    print("\nTesting PriceTimeRelationships...")
    
    try:
        from engines.gann.price_time_relationships import PriceTimeRelationships
        
        test_data = create_simple_test_data(100)
        
        gann_pt = PriceTimeRelationships()
        result = gann_pt.calculate(test_data)
        
        print("✅ PriceTimeRelationships: SUCCESS")
        print(f"Result keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ PriceTimeRelationships: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🔧 DEBUGGING GANN INDICATORS")
    print("=" * 50)
    
    success1 = test_gann_time_cycles()
    success2 = test_price_time_relationships()
    
    if success1 and success2:
        print("\n🎉 Both indicators working!")
    else:
        print(f"\n⚠️ Issues found: GannTimeCycles={'✅' if success1 else '❌'}, PriceTimeRelationships={'✅' if success2 else '❌'}")
