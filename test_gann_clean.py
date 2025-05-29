#!/usr/bin/env python3
"""
Test Gann Indicators Functionality - Clean Version
Platform3 Advanced Trading System
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data(n_bars=100):
    """Create realistic OHLC test data with daily frequency"""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')
    
    # Generate realistic price data with trend
    np.random.seed(42)
    base_price = 100.0
    
    data = []
    for i, date in enumerate(dates):
        price = base_price + i * 0.3 + np.random.normal(0, 2)
        volatility = abs(np.random.normal(0, 0.01))
        
        data.append({
            'timestamp': date,
            'open': price * (1 + np.random.normal(0, 0.005)),
            'high': price * (1 + volatility),
            'low': price * (1 - volatility),
            'close': price,
            'volume': np.random.randint(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_all_gann_indicators():
    """Test all Gann indicators"""
    print("🔧 TESTING ALL GANN INDICATORS")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data(150)
    print(f"Created test data: {test_data.shape[0]} bars")
    
    results = {}
    
    # Test 1: Gann Fan Lines
    print("\n📐 Testing Gann Fan Lines...")
    try:
        from engines.gann.gann_fan_lines import GannFanLines
        gann_fan = GannFanLines()
        fan_result = gann_fan.calculate(test_data)
        
        if fan_result:
            results['GannFanLines'] = 'SUCCESS'
            print(f"   ✅ GannFanLines: SUCCESS")
            print(f"   📊 Result keys: {list(fan_result.keys())}")
        else:
            results['GannFanLines'] = 'FAILED'
            print("   ❌ GannFanLines: No data returned")
    except Exception as e:
        results['GannFanLines'] = f'ERROR: {e}'
        print(f"   ❌ GannFanLines: ERROR - {e}")
    
    # Test 2: Gann Square of 9
    print("\n🔢 Testing Gann Square of 9...")
    try:
        from engines.gann.gann_square_of_nine import GannSquareOfNine
        gann_square = GannSquareOfNine()
        square_result = gann_square.calculate(test_data)
        
        if square_result:
            results['GannSquareOfNine'] = 'SUCCESS'
            print(f"   ✅ GannSquareOfNine: SUCCESS")
            print(f"   📊 Result keys: {list(square_result.keys())}")
        else:
            results['GannSquareOfNine'] = 'FAILED'
            print("   ❌ GannSquareOfNine: No data returned")
    except Exception as e:
        results['GannSquareOfNine'] = f'ERROR: {e}'
        print(f"   ❌ GannSquareOfNine: ERROR - {e}")
    
    # Test 3: Gann Time Cycles
    print("\n⏰ Testing Gann Time Cycles...")
    try:
        from engines.gann.gann_time_cycles import GannTimeCycles
        gann_cycles = GannTimeCycles()
        cycle_result = gann_cycles.calculate(test_data)
        
        if cycle_result:
            results['GannTimeCycles'] = 'SUCCESS'
            print(f"   ✅ GannTimeCycles: SUCCESS")
            print(f"   📊 Result keys: {list(cycle_result.keys())}")
        else:
            results['GannTimeCycles'] = 'FAILED'
            print("   ❌ GannTimeCycles: No data returned")
    except Exception as e:
        results['GannTimeCycles'] = f'ERROR: {e}'
        print(f"   ❌ GannTimeCycles: ERROR - {e}")
    
    # Test 4: Price-Time Relationships
    print("\n📊 Testing Price-Time Relationships...")
    try:
        from engines.gann.price_time_relationships import PriceTimeRelationships
        gann_pt = PriceTimeRelationships()
        pt_result = gann_pt.calculate(test_data)
        
        if pt_result:
            results['PriceTimeRelationships'] = 'SUCCESS'
            print(f"   ✅ PriceTimeRelationships: SUCCESS")
            print(f"   📊 Result keys: {list(pt_result.keys())}")
        else:
            results['PriceTimeRelationships'] = 'FAILED'
            print("   ❌ PriceTimeRelationships: No data returned")
    except Exception as e:
        results['PriceTimeRelationships'] = f'ERROR: {e}'
        print(f"   ❌ PriceTimeRelationships: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 GANN INDICATORS TEST SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for result in results.values() if result == 'SUCCESS')
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for name, result in results.items():
        status_icon = "✅" if result == "SUCCESS" else "❌"
        print(f"   {status_icon} {name}: {result}")
    
    if successful == total:
        print("\n🎉 ALL GANN INDICATORS ARE FUNCTIONAL!")
        print("🎯 GANN ANALYSIS CATEGORY: 100% COMPLETE!")
        return True
    else:
        print(f"\n⚠️  {total - successful} indicators need attention")
        return False

if __name__ == '__main__':
    success = test_all_gann_indicators()
    if success:
        print("\n🚀 MISSION PROGRESS: Ready to update completion status!")
    else:
        print("\n🔧 Additional debugging may be needed for some indicators")
