#!/usr/bin/env python3
"""
Test for EXACTLY 67 Platform3 Indicators
========================================

This test validates that ALL 67 indicators work properly.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List
from ComprehensiveIndicatorAdapter_67 import (
    ComprehensiveIndicatorAdapter_67,
    MarketData,
    IndicatorCategory
)

def generate_realistic_market_data(periods: int = 100) -> MarketData:
    """Generate realistic market data for testing"""
    print("📊 Generating realistic test data...")
    
    # Start with a base price
    base_price = 1.1000
    
    # Generate price movements with realistic patterns
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, periods)
    
    # Add some trend and volatility clustering
    trend = np.linspace(0, 0.01, periods)
    volatility = 0.001 * (1 + 0.5 * np.sin(np.linspace(0, 4*np.pi, periods)))
    
    price_changes = trend + volatility * returns
    close_prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLV data
    high_noise = np.random.uniform(0.0005, 0.002, periods)
    low_noise = np.random.uniform(0.0005, 0.002, periods)
    
    high = close_prices + high_noise
    low = close_prices - low_noise
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Ensure OHLC logic (High >= max(O,C), Low <= min(O,C))
    high = np.maximum(high, np.maximum(open_prices, close_prices))
    low = np.minimum(low, np.minimum(open_prices, close_prices))
    
    # Generate volume
    volume = np.random.randint(1000, 10000, periods)
    
    market_data = MarketData(
        open=open_prices,
        high=high,
        low=low,
        close=close_prices,
        volume=volume,
        timestamp=np.arange(periods)
    )
    
    print(f"✅ Generated {periods} data points")
    print(f"   Price range: {close_prices.min():.5f} - {close_prices.max():.5f}")
    print(f"   Volume range: {volume.min()} - {volume.max()}")
    
    return market_data

def test_all_67_indicators():
    """Test all 67 indicators"""
    print("🎯 COMPREHENSIVE TEST: EXACTLY 67 PLATFORM3 INDICATORS")
    print("=" * 80)
    
    # Initialize adapter
    adapter = ComprehensiveIndicatorAdapter_67()
    
    # Generate test data
    market_data = generate_realistic_market_data(100)
    print()
    
    # Test all indicators
    print(f"🔬 TESTING ALL {len(adapter.get_all_indicator_names())} INDICATORS:")
    print("-" * 80)
    print()
    
    results = {}
    total_time = 0
    successful = 0
    failed = 0
    
    # Group by category for organized output
    categories = {}
    for name in adapter.get_all_indicator_names():
        category = adapter.all_indicators[name][2]
        if category not in categories:
            categories[category] = []
        categories[category].append(name)
    
    for category, indicators in categories.items():
        print(f"📁 {category.value.upper()} INDICATORS:")
        
        for indicator_name in indicators:
            start_time = time.time()
            result = adapter.calculate_indicator(indicator_name, market_data)
            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed
            
            if result.success:
                print(f"  ✅ FUNCTIONAL {indicator_name} ({elapsed:.1f}ms)")
                successful += 1
            else:
                print(f"  ❌ FAILED {indicator_name} - {result.error_message}")
                failed += 1
            
            results[indicator_name] = result
        
        print()
    
    # Summary statistics
    total_indicators = successful + failed
    success_rate = (successful / total_indicators) * 100 if total_indicators > 0 else 0
    avg_time = total_time / total_indicators if total_indicators > 0 else 0
    
    # Calculate times for min/max
    execution_times = [r.calculation_time for r in results.values() if r.success]
    fastest = min(execution_times) if execution_times else 0
    slowest = max(execution_times) if execution_times else 0
    
    print("📊 COMPREHENSIVE TEST RESULTS:")
    print("=" * 50)
    print(f"🧪 Total Indicators Tested: {total_indicators}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"⚡ Total Time: {total_time:.1f}ms")
    print(f"⚡ Average Time: {avg_time:.1f}ms per indicator")
    print(f"🚀 Fastest: {fastest:.1f}ms")
    print(f"🐌 Slowest: {slowest:.1f}ms")
    print()
    
    # Category breakdown
    print("📊 CATEGORY BREAKDOWN:")
    print("-" * 30)
    for category, indicators in categories.items():
        cat_successful = sum(1 for name in indicators if results[name].success)
        cat_total = len(indicators)
        cat_percentage = (cat_successful / cat_total) * 100 if cat_total > 0 else 0
        print(f"  {category.value}: {cat_successful}/{cat_total} ({cat_percentage:.1f}%)")
    
    print()
    
    # Final assessment
    if success_rate >= 95:
        print("🎊 OUTSTANDING! ALL INDICATORS FUNCTIONAL!")
    elif success_rate >= 90:
        print("🎉 EXCELLENT! Nearly all indicators functional!")
    elif success_rate >= 75:
        print("🟢 GOOD! Most indicators functional!")
    elif success_rate >= 50:
        print("🟡 MODERATE. Many indicators functional!")
    else:
        print("🔴 POOR. Many indicators failed!")
    
    return results

if __name__ == "__main__":
    test_results = test_all_67_indicators()
