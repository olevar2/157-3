"""
Quick Performance Test - Verify <1ms Execution
=============================================

Test the optimized models to ensure sub-millisecond performance.
"""

import time
import numpy as np
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def test_optimized_risk_genius():
    """Test the optimized Risk Genius model"""
    print("🚀 Testing Optimized Risk Genius Model")
    print("=" * 50)
    
    try:
        from models.risk_genius import RiskGenius
        
        # Initialize model
        start_time = time.perf_counter()
        model = RiskGenius()
        init_time = (time.perf_counter() - start_time) * 1000
        print(f"Model Initialization: {init_time:.3f}ms")
        
        # Test data
        price_data = np.random.random(100) * 1.1
        market_conditions = {
            'session': 'london',
            'account_balance': 100000,
            'entry_price': 1.1000,
            'stop_loss': 1.0950
        }
        
        # Single execution test
        start_time = time.perf_counter()
        result = model.analyze_pair_risk('EURUSD', price_data, market_conditions)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Single Execution: {execution_time:.3f}ms {'✅ PASSED' if execution_time < 1.0 else '❌ FAILED'}")
        
        # Multiple executions test
        execution_times = []
        for i in range(100):
            start_time = time.perf_counter()
            result = model.analyze_pair_risk('EURUSD', price_data, market_conditions)
            execution_times.append((time.perf_counter() - start_time) * 1000)
        
        avg_time = np.mean(execution_times)
        p95_time = np.percentile(execution_times, 95)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        
        print(f"100 Executions Statistics:")
        print(f"  Average: {avg_time:.3f}ms {'✅ PASSED' if avg_time < 1.0 else '❌ FAILED'}")
        print(f"  P95: {p95_time:.3f}ms {'✅ PASSED' if p95_time < 1.0 else '❌ FAILED'}")
        print(f"  Min: {min_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")
        
        # Performance grade
        if avg_time < 0.1:
            grade = "A++"
        elif avg_time < 0.5:
            grade = "A+"
        elif avg_time < 1.0:
            grade = "A"
        elif avg_time < 5.0:
            grade = "B"
        else:
            grade = "C"
        
        print(f"Performance Grade: {grade}")
        
        # Display sample result
        print(f"\nSample Result:")
        print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"  Position Size: {result.get('position_size', 0):.2f}")
        print(f"  Execution Time: {result.get('execution_time_ms', 0):.3f}ms")
        
        return avg_time < 1.0
        
    except Exception as e:
        print(f"❌ Error testing Risk Genius: {e}")
        return False

def test_performance_optimizer():
    """Test the performance optimizer functions directly"""
    print("\n🚀 Testing Performance Optimizer Functions")
    print("=" * 50)
    
    try:
        from models.performance_optimizer import performance_optimizer
        
        # Test data
        prices = np.random.random(100) * 1.1
        high = prices + 0.001
        low = prices - 0.001
        
        functions_to_test = [
            ('Moving Average', lambda: performance_optimizer.fast_moving_average(prices, 20)),
            ('RSI', lambda: performance_optimizer.fast_rsi(prices)),
            ('Bollinger Bands', lambda: performance_optimizer.fast_bollinger_bands(prices)),
            ('MACD', lambda: performance_optimizer.fast_macd(prices)),
            ('Stochastic', lambda: performance_optimizer.fast_stochastic(high, low, prices)),
            ('Position Sizing', lambda: performance_optimizer.fast_position_size(100000, 2.0, 1.1000, 1.0950)),
            ('Kelly Criterion', lambda: performance_optimizer.fast_kelly_criterion(0.6, 100.0, 50.0)),
            ('VaR Calculation', lambda: performance_optimizer.fast_var_calculation(np.diff(prices))),
            ('Volatility', lambda: performance_optimizer.fast_volatility(prices))
        ]
        
        all_passed = True
        
        for name, func in functions_to_test:
            # Warm up
            func()
            
            # Test multiple executions
            times = []
            for _ in range(100):
                start = time.perf_counter()
                func()
                times.append((time.perf_counter() - start) * 1000)
            
            avg_time = np.mean(times)
            passed = avg_time < 1.0
            all_passed = all_passed and passed
            
            print(f"{name:<20} {avg_time:.3f}ms  {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing Performance Optimizer: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Platform3 Performance Verification")
    print("💰 Ensuring <1ms execution for humanitarian profits")
    print("=" * 60)
    
    # Test optimizer functions
    optimizer_passed = test_performance_optimizer()
    
    # Test optimized model
    model_passed = test_optimized_risk_genius()
    
    print("\n" + "=" * 60)
    print("🎯 PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    print(f"Optimizer Functions: {'✅ PASSED' if optimizer_passed else '❌ FAILED'}")
    print(f"Optimized Model: {'✅ PASSED' if model_passed else '❌ FAILED'}")
    
    if optimizer_passed and model_passed:
        print("🚀 Platform3 ready for <1ms humanitarian profit generation!")
    else:
        print("⚠️ Further optimization needed")
    
    print("=" * 60)
