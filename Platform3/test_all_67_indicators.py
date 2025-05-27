#!/usr/bin/env python3
"""
COMPREHENSIVE TEST FOR ALL 67 PLATFORM3 INDICATORS
=================================================

This test validates that ALL 67 indicators can be accessed through 
the Comprehensive Indicator Adapter.

Author: Platform3 Development Team
"""

import sys
import numpy as np
from datetime import datetime, timedelta

# Add the indicators path
sys.path.append('services/analytics-service/src/engines/indicators')

from ComprehensiveIndicatorAdapter import (
    ComprehensiveIndicatorAdapter, 
    MarketData, 
    IndicatorCategory,
    create_market_data
)

def generate_realistic_test_data(periods=100):
    """Generate realistic forex test data"""
    np.random.seed(42)
    base_price = 1.1000
    
    returns = np.random.normal(0, 0.0005, periods)
    prices = [base_price]
    
    for i in range(1, periods):
        trend = 0.0001 * np.sin(i / 20)
        new_price = prices[-1] * (1 + trend + returns[i])
        prices.append(new_price)
    
    prices = np.array(prices)
    highs = prices * (1 + np.abs(np.random.normal(0, 0.0002, periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.0002, periods)))
    volumes = np.random.randint(1000, 10000, periods)
    timestamps = np.arange(periods)
    
    return create_market_data(
        timestamps=timestamps,
        open_prices=prices,
        high_prices=highs,
        low_prices=lows,
        close_prices=prices,
        volumes=volumes
    )

def test_all_67_indicators():
    """Test ALL 67 Platform3 indicators"""
    print("🎯 COMPREHENSIVE TEST: ALL 67 PLATFORM3 INDICATORS")
    print("=" * 80)
    
    # Initialize adapter
    adapter = ComprehensiveIndicatorAdapter()
    
    # Generate test data
    print("📊 Generating realistic test data...")
    market_data = generate_realistic_test_data(100)
    print(f"✅ Generated {len(market_data.close)} data points")
    print(f"   Price range: {market_data.close.min():.5f} - {market_data.close.max():.5f}")
    print(f"   Volume range: {market_data.volume.min()} - {market_data.volume.max()}")
    
    # Test all indicators
    print(f"\n🔬 TESTING ALL {len(adapter.all_indicators)} INDICATORS:")
    print("-" * 80)
    
    results = {}
    successful = 0
    failed = 0
    total_time = 0
    
    # Group by category
    by_category = {}
    for name, (_, _, category) in adapter.all_indicators.items():
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(name)
    
    for category, indicators in by_category.items():
        print(f"\n📁 {category.value.upper()} INDICATORS:")
        
        for indicator_name in indicators:
            result = adapter.calculate_indicator(indicator_name, market_data)
            results[indicator_name] = result
            
            if result.success:
                print(f"  ✅ FUNCTIONAL {indicator_name} ({result.calculation_time:.1f}ms)")
                successful += 1
            else:
                print(f"  ❌ FAILED {indicator_name} - {result.error_message}")
                failed += 1
            
            total_time += result.calculation_time
    
    # Summary
    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print("=" * 50)
    print(f"🧪 Total Indicators Tested: {len(adapter.all_indicators)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(successful/len(adapter.all_indicators)*100):.1f}%")
    print(f"⚡ Total Time: {total_time:.1f}ms")
    print(f"⚡ Average Time: {(total_time/len(adapter.all_indicators)):.1f}ms per indicator")
    
    # Performance analysis
    if successful > 0:
        successful_times = [r.calculation_time for r in results.values() if r.success]
        print(f"🚀 Fastest: {min(successful_times):.1f}ms")
        print(f"🐌 Slowest: {max(successful_times):.1f}ms")
    
    # Category breakdown
    print(f"\n📊 CATEGORY BREAKDOWN:")
    print("-" * 30)
    for category, indicators in by_category.items():
        category_success = sum(1 for name in indicators if results[name].success)
        print(f"  {category.value}: {category_success}/{len(indicators)} ({(category_success/len(indicators)*100):.1f}%)")
    
    # Final verdict
    success_rate = successful/len(adapter.all_indicators)*100
    if success_rate == 100:
        print(f"\n🎊 OUTSTANDING! ALL 67 INDICATORS FUNCTIONAL!")
    elif success_rate >= 90:
        print(f"\n🎉 EXCELLENT! {successful}/67 indicators functional!")
    elif success_rate >= 75:
        print(f"\n🟢 GOOD! {successful}/67 indicators functional!")
    elif success_rate >= 50:
        print(f"\n🟡 MODERATE. {successful}/67 indicators functional!")
    else:
        print(f"\n🔴 NEEDS WORK. Only {successful}/67 indicators functional!")
    
    return results

if __name__ == "__main__":
    test_all_67_indicators()
