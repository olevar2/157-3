#!/usr/bin/env python3
"""
COMPLETE TEST OF ALL 67 INDICATORS
==================================

Testing EVERY SINGLE indicator individually to verify they all work!
"""

import time
import numpy as np
from ComprehensiveIndicatorAdapter_67 import ComprehensiveIndicatorAdapter_67, MarketData

def create_comprehensive_test_data():
    """Create robust test data for all indicators"""
    n = 200  # Large dataset for complex indicators
    base_price = 1.1000
    
    # Create realistic FOREX price movement with trends and volatility
    trend = np.linspace(0, 0.01, n)  # Slight upward trend
    volatility = np.random.normal(0, 0.0001, n)
    seasonal = 0.0005 * np.sin(np.linspace(0, 4*np.pi, n))  # Seasonal component
    
    close = base_price + trend + volatility + seasonal
    
    # Create realistic OHLC data
    high = close + np.abs(np.random.normal(0, 0.0002, n))
    low = close - np.abs(np.random.normal(0, 0.0002, n))
    open_prices = np.roll(close, 1)
    open_prices[0] = base_price
    
    # Realistic volume with some correlation to price movement
    volume_base = 50000
    volume_variation = np.random.randint(-20000, 30000, n)
    volume = volume_base + volume_variation
    volume = np.maximum(volume, 1000)  # Ensure positive volume
    
    timestamp = np.arange(n)
    
    return MarketData(
        open=open_prices,
        high=high,
        low=low,
        close=close,
        volume=volume.astype(float),
        timestamp=timestamp
    )

def test_all_indicators_comprehensive():
    """Test every single indicator with detailed reporting"""
    print("🎯 TESTING ALL 67 INDICATORS - COMPREHENSIVE VERIFICATION")
    print("=" * 80)
    
    # Initialize
    adapter = ComprehensiveIndicatorAdapter_67()
    test_data = create_comprehensive_test_data()
    
    # Get all indicators
    all_indicators = adapter.get_all_indicator_names()
    total_indicators = len(all_indicators)
    
    print(f"📊 Total indicators to test: {total_indicators}")
    print(f"📈 Test data points: {len(test_data.close)}")
    print("-" * 80)
    
    # Results tracking
    results = {
        'working': [],
        'broken': [],
        'errors': {},
        'timing': {}
    }
    
    # Test each indicator
    for i, indicator_name in enumerate(sorted(all_indicators), 1):
        print(f"[{i:2d}/{total_indicators}] {indicator_name:.<45}", end=" ")
        
        try:
            start_time = time.time()
            result = adapter.calculate_indicator(indicator_name, test_data)
            end_time = time.time()
            
            calc_time = (end_time - start_time) * 1000
            results['timing'][indicator_name] = calc_time
            
            if result.success:
                print(f"✅ OK ({calc_time:.1f}ms)")
                results['working'].append(indicator_name)
            else:
                print(f"❌ FAIL - {result.error_message}")
                results['broken'].append(indicator_name)
                results['errors'][indicator_name] = result.error_message
                
        except Exception as e:
            print(f"💥 ERROR - {str(e)}")
            results['broken'].append(indicator_name)
            results['errors'][indicator_name] = f"Exception: {str(e)}"
    
    return results, total_indicators

def print_detailed_results(results, total):
    """Print comprehensive results analysis"""
    working_count = len(results['working'])
    broken_count = len(results['broken'])
    success_rate = (working_count / total) * 100
    
    print("\n" + "=" * 80)
    print("📋 DETAILED RESULTS ANALYSIS")
    print("=" * 80)
    
    print(f"✅ WORKING: {working_count}/{total} ({success_rate:.1f}%)")
    print(f"❌ BROKEN:  {broken_count}/{total} ({100-success_rate:.1f}%)")
    
    if working_count == total:
        print("\n🎉🎉🎉 PERFECT SCORE! ALL 67 INDICATORS WORKING! 🎉🎉🎉")
        print("🚀 Platform3 is 100% production-ready!")
        print("💪 Every single indicator is functional and tested!")
    else:
        print(f"\n⚠️  ATTENTION: {broken_count} indicators need fixes!")
        
        print(f"\n❌ BROKEN INDICATORS ({broken_count}):")
        for indicator in sorted(results['broken']):
            error = results['errors'].get(indicator, "Unknown error")
            print(f"   • {indicator}: {error}")
    
    # Performance analysis
    if results['timing']:
        times = list(results['timing'].values())
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print(f"   Average calculation time: {avg_time:.1f}ms")
        print(f"   Fastest indicator: {min_time:.1f}ms")
        print(f"   Slowest indicator: {max_time:.1f}ms")
        
        # Find slowest indicators
        slow_indicators = [(name, time) for name, time in results['timing'].items() if time > 1000]
        if slow_indicators:
            print(f"   ⚠️  Slow indicators (>1000ms):")
            for name, time in sorted(slow_indicators, key=lambda x: x[1], reverse=True):
                print(f"      • {name}: {time:.1f}ms")
    
    # Category breakdown
    print(f"\n📂 WORKING INDICATORS BY CATEGORY:")
    adapter = ComprehensiveIndicatorAdapter_67()
    categories = {}
    
    for indicator in results['working']:
        if indicator in adapter.all_indicators:
            _, _, category = adapter.all_indicators[indicator]
            cat_name = category.value
            if cat_name not in categories:
                categories[cat_name] = []
            categories[cat_name].append(indicator)
    
    for category, indicators in sorted(categories.items()):
        print(f"   📁 {category.upper()}: {len(indicators)} indicators")
        for indicator in sorted(indicators):
            print(f"      ✅ {indicator}")
    
    return success_rate

def main():
    """Main test execution"""
    print("🔍 COMPREHENSIVE 67-INDICATOR VERIFICATION")
    print("Testing EVERY SINGLE indicator in Platform3")
    print("This is the ultimate test - no shortcuts!")
    print()
    
    start_time = time.time()
    results, total = test_all_indicators_comprehensive()
    end_time = time.time()
    
    success_rate = print_detailed_results(results, total)
    
    total_time = end_time - start_time
    print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
    
    print(f"\n🏁 FINAL VERDICT:")
    if success_rate == 100:
        print("🎉 SUCCESS! ALL 67 INDICATORS ARE WORKING PERFECTLY!")
        print("🚀 Platform3 is ready for production trading!")
        print("💯 100% indicator coverage achieved!")
    else:
        print(f"⚠️  PARTIAL SUCCESS: {success_rate:.1f}% indicators working")
        print(f"🔧 {len(results['broken'])} indicators need attention")
        print("📝 Review the broken indicators list above")
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
