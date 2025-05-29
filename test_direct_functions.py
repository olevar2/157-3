"""
Direct function test focusing only on working models
"""
import numpy as np
import time
import sys
import os

# Generate test data
indicators_array = np.random.random((67, 100)).astype(np.float32)
prices = np.random.uniform(1.08, 1.12, 100).astype(np.float32)

print("🚀 Testing Working Enhanced Functions")
print("=" * 50)

# Test Risk Genius (we know this works)
print("\n🎯 Testing Risk Genius...")
try:
    sys.path.insert(0, os.path.join("models", "risk_genius"))
    import ultra_fast_model as risk_model
    
    start_time = time.time()
    result = risk_model.analyze_risk_with_67_indicators_simple(indicators_array)
    execution_time = (time.time() - start_time) * 1000
    
    print(f"✅ Risk Genius: {execution_time:.3f}ms")
    print(f"   Result: {result['risk_level']}, Score: {result['risk_score']:.1f}")
    
    sys.path.remove(os.path.join("models", "risk_genius"))
except Exception as e:
    print(f"❌ Risk Genius failed: {e}")

# Test Pair Specialist (we know this works from simple test)
print("\n💱 Testing Pair Specialist...")
try:
    sys.path.insert(0, os.path.join("models", "pair_specialist"))
    import ultra_fast_model as pair_model
    
    start_time = time.time()
    result = pair_model.analyze_pair_with_all_indicators(indicators_array)
    execution_time = (time.time() - start_time) * 1000
    
    print(f"✅ Pair Specialist: {execution_time:.3f}ms")
    print(f"   Result: {result['signal']}, Entry: {result['optimal_entry']:.4f}")
    
    sys.path.remove(os.path.join("models", "pair_specialist"))
except Exception as e:
    print(f"❌ Pair Specialist failed: {e}")

# Test Pattern Master with direct import
print("\n📈 Testing Pattern Master (direct)...")
try:
    sys.path.insert(0, os.path.join("models", "pattern_master"))
    
    # Check if there's an initialization issue
    print("   Importing module...")
    import ultra_fast_model as pattern_model
    print("   Module imported successfully")
    
    # List available functions
    funcs = [name for name in dir(pattern_model) if 'analyze' in name.lower()]
    print(f"   Available functions: {funcs}")
    
    # Try to call the function if it exists
    if hasattr(pattern_model, 'analyze_patterns_with_67_indicators'):
        start_time = time.time()
        result = pattern_model.analyze_patterns_with_67_indicators(
            prices=prices,
            indicators=indicators_array,
            current_price=float(prices[-1])
        )
        execution_time = (time.time() - start_time) * 1000
        print(f"✅ Pattern Master: {execution_time:.3f}ms")
        print(f"   Result: {type(result)}")
    else:
        print("❌ Function not found in module")
    
    sys.path.remove(os.path.join("models", "pattern_master"))
except Exception as e:
    print(f"❌ Pattern Master failed: {e}")

# Test Execution Expert with direct import
print("\n⚡ Testing Execution Expert (direct)...")
try:
    sys.path.insert(0, os.path.join("models", "execution_expert"))
    
    print("   Importing module...")
    import ultra_fast_model as execution_model
    print("   Module imported successfully")
    
    # List available functions
    funcs = [name for name in dir(execution_model) if 'optimize' in name.lower()]
    print(f"   Available functions: {funcs}")
    
    # Try to call the function if it exists
    if hasattr(execution_model, 'optimize_execution_with_67_indicators'):
        start_time = time.time()
        result = execution_model.optimize_execution_with_67_indicators(
            order_size=10000.0,
            indicators=indicators_array
        )
        execution_time = (time.time() - start_time) * 1000
        print(f"✅ Execution Expert: {execution_time:.3f}ms")
        print(f"   Result: {type(result)}")
    else:
        print("❌ Function not found in module")
    
    sys.path.remove(os.path.join("models", "execution_expert"))
except Exception as e:
    print(f"❌ Execution Expert failed: {e}")

print("\n✅ Direct function test completed!")
