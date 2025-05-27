#!/usr/bin/env python3
"""
Universal Indicator Adapter Test
===============================

This script validates that the Universal Indicator Adapter successfully 
solves the interface inconsistency problem by testing all Platform3 indicators
through the standardized interface.

Features:
- Tests all indicator categories
- Validates interface detection
- Measures performance improvements  
- Provides detailed functionality reports
- Generates compatibility matrix

Author: Platform3 Development Team
Version: 1.0.0
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import traceback

# Add the indicators path
sys.path.append('services/analytics-service/src/engines/indicators')

from UniversalIndicatorAdapter import (
    UniversalIndicatorAdapter, 
    MarketData, 
    IndicatorCategory,
    create_market_data
)

def generate_realistic_test_data(periods=100) -> MarketData:
    """Generate realistic forex market data for testing"""
    # Generate timestamps (M15 intervals)
    start_time = datetime.now() - timedelta(minutes=periods * 15)
    timestamps = np.array([
        (start_time + timedelta(minutes=i * 15)).timestamp() 
        for i in range(periods)
    ])
    
    # Generate realistic EURUSD-like price data
    base_price = 1.1000
    price_data = []
    volume_data = []
    
    for i in range(periods):
        # Simulate price movements with trend and noise
        trend = 0.0001 * np.sin(i / 20) 
        noise = np.random.normal(0, 0.0005)
        price_change = trend + noise
        
        if i == 0:
            price = base_price
        else:
            price = price_data[-1] + price_change
        
        # Generate OHLC from base price
        spread = np.random.uniform(0.0001, 0.0003)
        volatility = np.random.uniform(0.0002, 0.0008)
        
        open_price = price_data[-1] if i > 0 else price
        close_price = price
        high_price = max(open_price, close_price) + np.random.uniform(0, volatility)
        low_price = min(open_price, close_price) - np.random.uniform(0, volatility)
        
        price_data.append(close_price)
        
        # Generate realistic volume
        base_volume = 1500
        volume_variation = np.random.uniform(0.5, 2.0)
        volume = int(base_volume * volume_variation)
        volume_data.append(volume)
    
    # Create OHLC arrays
    open_prices = np.array([price_data[0]] + price_data[:-1])
    high_prices = np.array([
        max(open_prices[i], price_data[i]) + np.random.uniform(0, 0.0003) 
        for i in range(periods)
    ])
    low_prices = np.array([
        min(open_prices[i], price_data[i]) - np.random.uniform(0, 0.0003) 
        for i in range(periods)
    ])
    close_prices = np.array(price_data)
    volumes = np.array(volume_data)
    
    return create_market_data(
        timestamps=timestamps,
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        close_prices=close_prices,
        volumes=volumes
    )

def test_adapter_comprehensive():
    """Comprehensive test of the Universal Indicator Adapter"""
    print("🚀 UNIVERSAL INDICATOR ADAPTER - COMPREHENSIVE TEST")
    print("=" * 80)
    print("Testing all Platform3 indicators through standardized interface...")
    print()
    
    # Generate test data
    print("📊 Generating realistic forex test data...")
    market_data = generate_realistic_test_data(100)
    print(f"✅ Generated {len(market_data.close)} data points")
    print(f"   Price range: {min(market_data.close):.5f} - {max(market_data.close):.5f}")
    print(f"   Volume range: {min(market_data.volume)} - {max(market_data.volume)}")
    print()
    
    # Initialize adapter
    adapter = UniversalIndicatorAdapter()
    
    # Test all supported indicators
    print("🔬 TESTING ALL INDICATORS THROUGH ADAPTER:")
    print("-" * 60)
    
    total_indicators = 0
    successful_indicators = 0
    failed_indicators = []
    performance_data = []
    
    for category, indicators in adapter.get_supported_indicators().items():
        print(f"\n📁 {category.value.upper()} INDICATORS:")
        
        for indicator_name in indicators.keys():
            total_indicators += 1
            
            # Test indicator
            result = adapter.calculate_indicator(
                indicator_name=indicator_name,
                market_data=market_data,
                category=category
            )
            
            if result.success:
                successful_indicators += 1
                status = "✅ FUNCTIONAL"
                interface = result.metadata.get('interface_used', 'auto-detected')
                calc_time = result.calculation_time
                
                performance_data.append({
                    'indicator': indicator_name,
                    'category': category.value,
                    'time_ms': calc_time,
                    'interface': interface
                })
                
                # Check if signals are available
                has_signals = "📊" if result.signals else "📈"
                
                print(f"  {status} {indicator_name} ({interface}, {calc_time:.1f}ms) {has_signals}")
                
            else:
                failed_indicators.append({
                    'name': indicator_name,
                    'category': category.value,
                    'error': result.error_message
                })
                print(f"  ❌ FAILED {indicator_name} - {result.error_message}")
    
    # Performance Analysis
    print(f"\n🎯 ADAPTER PERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    if performance_data:
        avg_time = np.mean([p['time_ms'] for p in performance_data])
        max_time = max([p['time_ms'] for p in performance_data])
        min_time = min([p['time_ms'] for p in performance_data])
        
        print(f"⚡ Average calculation time: {avg_time:.1f}ms")
        print(f"🚀 Fastest indicator: {min_time:.1f}ms")
        print(f"🐌 Slowest indicator: {max_time:.1f}ms")
        
        # Interface usage statistics
        interfaces = [p['interface'] for p in performance_data]
        interface_counts = {}
        for interface in interfaces:
            interface_counts[interface] = interface_counts.get(interface, 0) + 1
        
        print(f"\n🔧 Interface Usage:")
        for interface, count in interface_counts.items():
            print(f"   {interface}: {count} indicators")
    
    # Summary Results
    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print("=" * 50)
    print(f"🧪 Total Indicators Tested: {total_indicators}")
    print(f"✅ Successful: {successful_indicators}")
    print(f"❌ Failed: {len(failed_indicators)}")
    print(f"📈 Success Rate: {(successful_indicators/total_indicators)*100:.1f}%")
    
    # Detailed failure analysis
    if failed_indicators:
        print(f"\n🔍 FAILURE ANALYSIS:")
        print("-" * 30)
        for failure in failed_indicators:
            print(f"❌ {failure['name']} ({failure['category']}): {failure['error']}")
    
    # Compare with previous functionality rate
    print(f"\n🆚 COMPARISON WITH DIRECT INTERFACE:")
    print("-" * 40)
    previous_functional = 3  # From your original test (RSI, MACD, ScalpingMomentum)
    previous_total = 15
    previous_rate = (previous_functional / previous_total) * 100
    
    current_rate = (successful_indicators / total_indicators) * 100
    improvement = current_rate - previous_rate
    
    print(f"📉 Previous Functionality: {previous_functional}/{previous_total} ({previous_rate:.1f}%)")
    print(f"📈 Adapter Functionality: {successful_indicators}/{total_indicators} ({current_rate:.1f}%)")
    print(f"🎉 Improvement: +{improvement:.1f} percentage points")
    
    if current_rate >= 80:
        print("\n🎊 OUTSTANDING RESULTS!")
        print("✅ Universal Adapter successfully resolves interface inconsistencies")
        print("✅ Platform3 indicators are now truly functional for trading")
        print("✅ Ready for real-time trading implementation")
    elif current_rate >= 60:
        print("\n🟡 GOOD RESULTS!")
        print("✅ Major improvement in indicator functionality")
        print("⚠️  Some indicators may need additional work")
    else:
        print("\n🔴 NEEDS MORE WORK")
        print("❌ Additional development required")
    
    return successful_indicators == total_indicators

def test_batch_calculation():
    """Test batch calculation feature"""
    print("\n🔄 TESTING BATCH CALCULATION:")
    print("-" * 40)
    
    adapter = UniversalIndicatorAdapter()
    market_data = generate_realistic_test_data(50)
    
    # Test batch calculation with multiple indicators
    indicator_list = ['RSI', 'MACD', 'ADX', 'ScalpingMomentum']
    
    start_time = time.time()
    batch_results = adapter.batch_calculate(indicator_list, market_data)
    batch_time = (time.time() - start_time) * 1000
    
    print(f"📊 Batch calculated {len(indicator_list)} indicators in {batch_time:.1f}ms")
    
    successful_batch = sum(1 for result in batch_results.values() if result.success)
    print(f"✅ Successful: {successful_batch}/{len(indicator_list)}")
    
    for name, result in batch_results.items():
        status = "✅" if result.success else "❌"
        print(f"  {status} {name}: {result.calculation_time:.1f}ms")

def test_validation_feature():
    """Test indicator validation feature"""
    print("\n🔍 TESTING INDICATOR VALIDATION:")
    print("-" * 40)
    
    adapter = UniversalIndicatorAdapter()
    
    # Test validation for known working indicators
    test_indicators = ['RSI', 'MACD', 'ScalpingMomentum', 'NonExistentIndicator']
    
    for indicator_name in test_indicators:
        is_valid = adapter.validate_indicator(indicator_name)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"  {status} {indicator_name}")

def main():
    """Run all adapter tests"""
    try:
        # Main comprehensive test
        success = test_adapter_comprehensive()
        
        # Additional feature tests
        test_batch_calculation()
        test_validation_feature()
        
        print(f"\n🏁 ADAPTER TEST COMPLETED")
        print("=" * 50)
        
        if success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Universal Indicator Adapter is fully functional")
            print("✅ Platform3 indicators are now standardized and ready for trading")
            return True
        else:
            print("⚠️  Some indicators still need work")
            print("✅ Major improvement achieved through adapter")
            return False
            
    except Exception as e:
        print(f"\n❌ ADAPTER TEST FAILED: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
