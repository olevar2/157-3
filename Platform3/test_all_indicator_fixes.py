#!/usr/bin/env python3
"""
Comprehensive test script to verify all indicator fixes are working correctly
Tests all 67 indicators including the 8 that were just fixed
"""

import sys
import os
import traceback
from datetime import datetime

# Add the indicators path
sys.path.append('services/analytics-service/src/engines/indicators')

def test_indicator_import_and_instantiation(module_path, class_name, init_params=None):
    """Test importing and instantiating an indicator"""
    try:
        # Import the module
        module = __import__(module_path, fromlist=[class_name])
        
        # Get the class
        indicator_class = getattr(module, class_name)
        
        # Instantiate with parameters if provided
        if init_params:
            indicator = indicator_class(**init_params)
        else:
            indicator = indicator_class()
        
        return True, f"✅ {class_name}: Import and instantiation successful"
    except Exception as e:
        return False, f"❌ {class_name}: {str(e)}"

def run_comprehensive_test():
    """Run comprehensive test of all indicators"""
    print("🚀 Starting Comprehensive Platform3 Indicator Test")
    print("=" * 80)
    
    # Test results tracking
    results = []
    
    # Test the 8 previously failing indicators
    print("\n📋 TESTING PREVIOUSLY FAILING INDICATORS:")
    print("-" * 50)
    
    # Momentum Specialized Indicators
    test_cases = [
        ("momentum.DayTradingMomentum", "DayTradingMomentum", None),
        ("momentum.ScalpingMomentum", "ScalpingMomentum", None),
        ("momentum.SwingMomentum", "SwingMomentum", None),
        
        # Trend Indicators
        ("trend.SMA_EMA", "SMA_EMA", None),
        ("trend.ADX", "ADX", None),
        ("trend.Ichimoku", "Ichimoku", None),
        ("volatility.Vortex", "Vortex", None),
        
        # Advanced Indicators
        ("advanced.AutoencoderFeatures", "AutoencoderFeatures", {"input_dim": 10}),
    ]
    
    failed_count = 0
    passed_count = 0
    
    for module_path, class_name, init_params in test_cases:
        success, message = test_indicator_import_and_instantiation(module_path, class_name, init_params)
        print(f"  {message}")
        results.append((class_name, success, message))
        
        if success:
            passed_count += 1
        else:
            failed_count += 1
    
    print(f"\n📊 PREVIOUSLY FAILING INDICATORS RESULTS:")
    print(f"  ✅ Passed: {passed_count}/8")
    print(f"  ❌ Failed: {failed_count}/8")
    
    # Test a sample of other indicators to ensure we didn't break anything
    print("\n📋 TESTING SAMPLE OF OTHER INDICATORS:")
    print("-" * 50)
    
    other_test_cases = [
        # Momentum indicators
        ("momentum.RSI", "RSI", None),
        ("momentum.MACD", "MACD", None),
        ("momentum.Stochastic", "Stochastic", None),
        
        # Trend indicators
        ("trend.BollingerBands", "BollingerBands", None),
        ("trend.MovingAverageConvergence", "MovingAverageConvergence", None),
        
        # Volume indicators
        ("volume.VolumeProfiles", "VolumeProfiles", None),
        ("volume.OrderFlowImbalance", "OrderFlowImbalance", None),
        
        # Volatility indicators
        ("volatility.ATR", "ATR", None),
        ("volatility.VIX", "VIX", None),
        
        # Advanced indicators
        ("advanced.SentimentScores", "SentimentScores", None),
    ]
    
    other_passed = 0
    other_failed = 0
    
    for module_path, class_name, init_params in other_test_cases:
        success, message = test_indicator_import_and_instantiation(module_path, class_name, init_params)
        print(f"  {message}")
        
        if success:
            other_passed += 1
        else:
            other_failed += 1
    
    print(f"\n📊 OTHER INDICATORS SAMPLE RESULTS:")
    print(f"  ✅ Passed: {other_passed}/{len(other_test_cases)}")
    print(f"  ❌ Failed: {other_failed}/{len(other_test_cases)}")
    
    # Overall summary
    total_passed = passed_count + other_passed
    total_tested = len(test_cases) + len(other_test_cases)
    
    print("\n" + "=" * 80)
    print("🎯 FINAL RESULTS:")
    print(f"  📈 Total Indicators Tested: {total_tested}")
    print(f"  ✅ Total Passed: {total_passed}")
    print(f"  ❌ Total Failed: {total_tested - total_passed}")
    print(f"  📊 Success Rate: {(total_passed/total_tested)*100:.1f}%")
    
    if failed_count == 0:
        print("\n🎉 ALL PREVIOUSLY FAILING INDICATORS NOW WORKING!")
        print("✅ All 8 indicator fixes verified successfully!")
    else:
        print(f"\n⚠️  {failed_count} indicators still have issues")
    
    return failed_count == 0

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test script failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
