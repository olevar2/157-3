#!/usr/bin/env python3
"""
Test script to verify Phase 2 robustness features work with invalid data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from engines.ai_enhancement.adaptive_indicators import AdaptiveIndicators
    print("✅ Successfully imported AdaptiveIndicators")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def create_invalid_data_scenarios():
    """Create various invalid data scenarios to test robustness."""
    
    scenarios = {}
    
    # Create base data with enough points (60 points for 50 period + 10 extra)
    base_prices = np.linspace(100, 120, 60) + np.random.normal(0, 1, 60)
    
    # Scenario 1: Data with NaN values
    print("\n🧪 Testing Scenario 1: Data with NaN values")
    data1_prices = base_prices.copy()
    data1_prices[10:15] = np.nan  # Insert some NaN values
    data1_prices[30:32] = np.nan
    data1 = pd.DataFrame({
        'close': data1_prices,
        'timestamp': pd.date_range('2024-01-01', periods=60)
    })
    scenarios['nan_data'] = data1
    
    # Scenario 2: Data with infinite values
    print("🧪 Testing Scenario 2: Data with infinite values")
    data2_prices = base_prices.copy()
    data2_prices[20] = np.inf
    data2_prices[25] = -np.inf
    data2 = pd.DataFrame({
        'close': data2_prices,
        'timestamp': pd.date_range('2024-01-01', periods=60)
    })
    scenarios['inf_data'] = data2
    
    # Scenario 3: Data with zero/negative prices
    print("🧪 Testing Scenario 3: Data with zero/negative prices")
    data3_prices = base_prices.copy()
    data3_prices[15] = 0
    data3_prices[35] = -5
    data3 = pd.DataFrame({
        'close': data3_prices,
        'timestamp': pd.date_range('2024-01-01', periods=60)
    })
    scenarios['negative_data'] = data3
    
    # Scenario 4: Empty data
    print("🧪 Testing Scenario 4: Empty data")
    data4 = pd.DataFrame({'close': [], 'timestamp': []})
    scenarios['empty_data'] = data4
      # Scenario 5: All NaN data
    print("🧪 Testing Scenario 5: All NaN data")
    data5 = pd.DataFrame({
        'close': [np.nan] * 60,
        'timestamp': pd.date_range('2024-01-01', periods=60)
    })
    scenarios['all_nan_data'] = data5
    
    # Scenario 6: Extreme price jumps (>1000% change)
    print("🧪 Testing Scenario 6: Extreme price jumps")
    data6_prices = base_prices.copy()
    data6_prices[30] = data6_prices[29] * 10  # 1000% jump
    data6_prices[31] = data6_prices[30] * 0.1  # 90% drop
    data6 = pd.DataFrame({
        'close': data6_prices,
        'timestamp': pd.date_range('2024-01-01', periods=60)
    })
    scenarios['extreme_jumps'] = data6
    
    return scenarios

def test_robustness():
    """Test the robustness of adaptive indicators with invalid data."""
    
    print("🛡️ TESTING PHASE 2 ROBUSTNESS FEATURES")
    print("=" * 60)
    
    # Initialize adaptive indicators
    try:
        indicator = AdaptiveIndicators(
            base_indicator='RSI',
            adaptation_period=14,
            optimization_window=20
        )
        print("✅ AdaptiveIndicators initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AdaptiveIndicators: {e}")
        return False
    
    # Create test scenarios
    scenarios = create_invalid_data_scenarios()
    
    results = {}
    
    # Test each scenario
    for scenario_name, data in scenarios.items():
        print(f"\n🔍 Testing {scenario_name}...")
        try:
            # This should not crash thanks to our robustness features
            result = indicator.calculate(data)
            
            if isinstance(result, pd.DataFrame):
                print(f"✅ {scenario_name}: Handled gracefully, returned DataFrame with {len(result)} rows")
                results[scenario_name] = "SUCCESS"
            else:
                print(f"⚠️ {scenario_name}: Unexpected return type: {type(result)}")
                results[scenario_name] = "WARNING"
                
        except Exception as e:
            print(f"❌ {scenario_name}: Failed with error: {e}")
            results[scenario_name] = "FAILED"
    
    # Summary
    print("\n📊 ROBUSTNESS TEST SUMMARY")
    print("=" * 40)
    
    success_count = sum(1 for result in results.values() if result == "SUCCESS")
    total_count = len(results)
    
    for scenario, result in results.items():
        status_icon = "✅" if result == "SUCCESS" else "⚠️" if result == "WARNING" else "❌"
        print(f"{status_icon} {scenario}: {result}")
    
    success_rate = (success_count / total_count) * 100
    print(f"\n🎯 ROBUSTNESS SCORE: {success_rate:.1f}% ({success_count}/{total_count} scenarios passed)")
    
    if success_rate >= 90:
        print("🎉 ROBUSTNESS TEST PASSED: System handles edge cases well!")
        return True
    else:
        print("⚠️ ROBUSTNESS TEST NEEDS IMPROVEMENT: Some edge cases not handled")
        return False

if __name__ == "__main__":
    success = test_robustness()
    sys.exit(0 if success else 1)
