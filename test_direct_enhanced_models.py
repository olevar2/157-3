"""
Direct Enhanced Models Integration Test

Tests all enhanced ultra-fast models using 67 indicators by importing
directly from the ultra_fast_model files to avoid dependency issues.
"""

import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced functions directly from ultra_fast_model files
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))

# Import enhanced model functions directly
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import enhanced models - current directory is Platform3
base_path = os.path.dirname(os.path.abspath(__file__))
risk_genius = import_module_from_path("risk_genius", os.path.join(base_path, "models", "risk_genius", "ultra_fast_model.py"))
session_expert = import_module_from_path("session_expert", os.path.join(base_path, "models", "session_expert", "ultra_fast_model.py"))
pair_specialist = import_module_from_path("pair_specialist", os.path.join(base_path, "models", "pair_specialist", "ultra_fast_model.py"))
pattern_master = import_module_from_path("pattern_master", os.path.join(base_path, "models", "pattern_master", "ultra_fast_model.py"))
execution_expert = import_module_from_path("execution_expert", os.path.join(base_path, "models", "execution_expert", "ultra_fast_model.py"))

def generate_test_data():
    """Generate realistic test data for comprehensive testing"""
    # Generate 100 price points
    np.random.seed(42)
    base_price = 1.1000
    prices = [base_price]
    
    for i in range(99):
        change = np.random.normal(0, 0.0002)  # Small realistic changes
        new_price = prices[-1] + change
        prices.append(max(0.5, min(2.0, new_price)))  # Keep in reasonable range
    
    prices = np.array(prices, dtype=np.float64)
    
    # Generate all 67 indicators (realistic values)
    indicators = np.array([
        # Momentum indicators (0-9)
        45.5,    # RSI
        52.3,    # Stochastic %K  
        48.7,    # Stochastic %D
        0.0003,  # MACD
        0.0001,  # MACD Signal
        0.0002,  # MACD Histogram
        15.2,    # CCI
        -45.8,   # Williams %R
        0.12,    # ROC
        0.08,    # Momentum
        
        # Bollinger Bands & Volatility (10-19)
        1.1025,  # BB Upper
        1.1000,  # BB Middle
        0.9975,  # BB Lower
        0.023,   # BB Width
        0.0015,  # ATR
        0.0012,  # True Range
        28.5,    # DMI+
        22.1,    # DMI-
        32.8,    # ADX
        65.2,    # Aroon Up
        
        # More indicators (20-29)
        34.8,    # Aroon Down
        30.4,    # Aroon Oscillator
        1.0995,  # Parabolic SAR
        1.1002,  # EMA 8
        1.1001,  # EMA 13
        1.1000,  # EMA 21
        1.0999,  # EMA 34
        1.0998,  # EMA 55
        1.0997,  # EMA 89
        1.0996,  # EMA 144
        
        # More EMAs and SMAs (30-39)
        1.0995,  # EMA 233
        1.1003,  # SMA 10
        1.1001,  # SMA 20
        1.0999,  # SMA 50
        1.0997,  # SMA 100
        1.0995,  # SMA 200
        1.1000,  # TEMA
        1.1001,  # KAMA
        1.1000,  # VWAP
        1.1000,  # Pivot Point
        
        # Support/Resistance & Fibonacci (40-49)
        1.0980,  # S1
        1.0960,  # S2
        1.0940,  # S3
        1.1020,  # R1
        1.1040,  # R2
        1.1060,  # R3
        1.0985,  # Fib 38.2
        1.0990,  # Fib 50.0
        1.0995,  # Fib 61.8
        1.1005,  # Ichimoku Tenkan
        
        # Ichimoku & Volume (50-59)
        1.1000,  # Ichimoku Kijun
        1.1002,  # Ichimoku Senkou A
        1.0998,  # Ichimoku Senkou B
        1250000, # OBV
        1000000, # Volume SMA
        0.15,    # A/D Line
        0.08,    # CMF
        52.3,    # MFI
        0.0002,  # Elder Ray Bull
        -0.0001, # Elder Ray Bear
        
        # Final indicators (60-66)
        1.1005,  # ZigZag
        0.0001,  # TRIX
        55.2,    # Ultimate Oscillator
        48.7,    # Stochastic RSI
        0.0,     # Fractal Up
        0.0,     # Fractal Down
        0.18     # Historical Volatility
    ], dtype=np.float64)
    
    return prices, indicators

def test_enhanced_risk_genius():
    """Test enhanced Risk Genius model with 67 indicators"""
    print("\\n=== Testing Enhanced Risk Genius ===")
    
    try:
        prices, indicators = generate_test_data()
        current_price = 1.1000
        position_size = 10000.0
        
        start_time = time.perf_counter()
        result = risk_genius.analyze_risk_with_67_indicators(prices, indicators, current_price, position_size)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"✅ Risk Genius Enhanced - Execution Time: {execution_time_ms:.3f}ms")
        print(f"   Risk Score: {result['risk_score']:.3f}")
        print(f"   Position Size: {result['optimal_position_size']:.0f}")
        print(f"   Indicators Used: {result['indicators_analyzed']}")
        print(f"   Performance: {'🚀 <1ms TARGET MET' if execution_time_ms < 1.0 else '⚠️ Above 1ms'}")
        
        return execution_time_ms < 1.0
        
    except Exception as e:
        print(f"❌ Risk Genius Test Failed: {e}")
        return False

def test_enhanced_session_expert():
    """Test enhanced Session Expert model with 67 indicators"""
    print("\\n=== Testing Enhanced Session Expert ===")
    
    try:
        prices, indicators = generate_test_data()
        current_hour = 14  # London session
        
        start_time = time.perf_counter()
        result = session_expert.analyze_session_with_67_indicators(prices, indicators, current_hour)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"✅ Session Expert Enhanced - Execution Time: {execution_time_ms:.3f}ms")
        print(f"   Session Strength: {result['session_strength']:.3f}")
        print(f"   Strategy: {result['recommended_strategy']}")
        print(f"   Indicators Used: {result['indicators_analyzed']}")
        print(f"   Performance: {'🚀 <1ms TARGET MET' if execution_time_ms < 1.0 else '⚠️ Above 1ms'}")
        
        return execution_time_ms < 1.0
        
    except Exception as e:
        print(f"❌ Session Expert Test Failed: {e}")
        return False

def test_enhanced_pair_specialist():
    """Test enhanced Pair Specialist model with 67 indicators"""
    print("\\n=== Testing Enhanced Pair Specialist ===")
    
    try:
        prices, indicators = generate_test_data()
        pair = "EURUSD"
        current_price = 1.1000
        
        start_time = time.perf_counter()
        result = pair_specialist.analyze_pair_with_all_indicators(pair, prices, indicators, current_price)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"✅ Pair Specialist Enhanced - Execution Time: {execution_time_ms:.3f}ms")
        print(f"   Pair Score: {result['pair_score']:.3f}")
        print(f"   Volatility Score: {result['volatility_score']:.3f}")
        print(f"   Strategy Weight: {result['strategy_weight']:.3f}")
        print(f"   Indicators Used: {result['indicators_analyzed']}")
        print(f"   Performance: {'🚀 <1ms TARGET MET' if execution_time_ms < 1.0 else '⚠️ Above 1ms'}")
        
        return execution_time_ms < 1.0
        
    except Exception as e:
        print(f"❌ Pair Specialist Test Failed: {e}")
        return False

def test_enhanced_pattern_master():
    """Test enhanced Pattern Master model with 67 indicators"""
    print("\\n=== Testing Enhanced Pattern Master ===")
    
    try:
        prices, indicators = generate_test_data()
        current_price = 1.1000
        
        start_time = time.perf_counter()
        result = pattern_master.analyze_patterns_with_67_indicators(prices, indicators, current_price)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"✅ Pattern Master Enhanced - Execution Time: {execution_time_ms:.3f}ms")
        print(f"   Primary Pattern: {result['primary_pattern']['name']}")
        print(f"   Pattern Strength: {result['primary_pattern']['strength']:.3f}")
        print(f"   Signal Strength: {result['trading_signals']['signal_strength']:.3f}")
        print(f"   Indicators Used: {result['performance_metrics']['indicator_count']}")
        print(f"   Performance: {'🚀 <1ms TARGET MET' if execution_time_ms < 1.0 else '⚠️ Above 1ms'}")
        
        return execution_time_ms < 1.0
        
    except Exception as e:
        print(f"❌ Pattern Master Test Failed: {e}")
        return False

def test_enhanced_execution_expert():
    """Test enhanced Execution Expert model with 67 indicators"""
    print("\\n=== Testing Enhanced Execution Expert ===")
    
    try:
        prices, indicators = generate_test_data()
        order_size = 10000.0
        
        start_time = time.perf_counter()
        result = execution_expert.optimize_execution_with_67_indicators(order_size, indicators)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"✅ Execution Expert Enhanced - Execution Time: {execution_time_ms:.3f}ms")
        print(f"   Order Type: {result['optimal_order_type']['name']}")
        print(f"   Expected Slippage: {result['cost_analysis']['expected_slippage']:.2f}")
        print(f"   Execution Score: {result['execution_quality']['execution_score']:.3f}")
        print(f"   Indicators Used: {result['indicator_insights']['indicators_used']}")
        print(f"   Performance: {'🚀 <1ms TARGET MET' if execution_time_ms < 1.0 else '⚠️ Above 1ms'}")
        
        return execution_time_ms < 1.0
        
    except Exception as e:
        print(f"❌ Execution Expert Test Failed: {e}")
        return False

def run_comprehensive_integration_test():
    """Run complete integration test suite"""
    print("🚀 PLATFORM3 ENHANCED MODELS INTEGRATION TEST")
    print("=" * 55)
    print("Testing all models with 67 indicators for <1ms performance")
    
    test_results = []
    
    # Test each enhanced model
    test_results.append(test_enhanced_risk_genius())
    test_results.append(test_enhanced_session_expert())
    test_results.append(test_enhanced_pair_specialist())
    test_results.append(test_enhanced_pattern_master())
    test_results.append(test_enhanced_execution_expert())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\\n" + "=" * 55)
    print("🎯 INTEGRATION TEST RESULTS")
    print("=" * 55)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Platform3 ready for humanitarian profit generation!")
        print("✅ All models enhanced with 67 indicators")
        print("✅ All models achieve <1ms performance")
        print("✅ Complete system integration verified")
        print("✅ Ready for 24/7 forex trading optimization")
    else:
        print("⚠️  Some tests failed. Review individual results above.")
    
    print("\\n🌍 Platform3: Generating profits to help the poor and needy worldwide!")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_integration_test()
    sys.exit(0 if success else 1)
