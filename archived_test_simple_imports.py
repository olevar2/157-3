import sys
import os
import numpy as np

# Test Session Expert import
print("Testing Session Expert...")
try:
    sys.path.insert(0, os.path.join("models", "session_expert"))
    import ultra_fast_model as session_model
    print(f"✅ Session Expert imported successfully")
    print(f"Has analyze_session_with_67_indicators: {hasattr(session_model, 'analyze_session_with_67_indicators')}")
    if hasattr(session_model, 'analyze_session_with_67_indicators'):
        test_array = np.random.random((67, 100)).astype(np.float32)
        result = session_model.analyze_session_with_67_indicators(test_array)
        print(f"✅ Function call successful: {type(result)}")
    sys.path.remove(os.path.join("models", "session_expert"))
except Exception as e:
    print(f"❌ Session Expert failed: {e}")

# Test Pair Specialist import
print("\nTesting Pair Specialist...")
try:
    sys.path.insert(0, os.path.join("models", "pair_specialist"))
    import ultra_fast_model as pair_model
    print(f"✅ Pair Specialist imported successfully")
    print(f"Has analyze_pair_with_all_indicators: {hasattr(pair_model, 'analyze_pair_with_all_indicators')}")
    if hasattr(pair_model, 'analyze_pair_with_all_indicators'):
        test_array = np.random.random((67, 100)).astype(np.float32)
        result = pair_model.analyze_pair_with_all_indicators(test_array)
        print(f"✅ Function call successful: {type(result)}")
    sys.path.remove(os.path.join("models", "pair_specialist"))
except Exception as e:
    print(f"❌ Pair Specialist failed: {e}")

# Test Pattern Master import
print("\nTesting Pattern Master...")
try:
    sys.path.insert(0, os.path.join("models", "pattern_master"))
    import ultra_fast_model as pattern_model
    print(f"✅ Pattern Master imported successfully")
    print(f"Has analyze_patterns_with_67_indicators: {hasattr(pattern_model, 'analyze_patterns_with_67_indicators')}")
    if hasattr(pattern_model, 'analyze_patterns_with_67_indicators'):
        test_array = np.random.random((67, 100)).astype(np.float32)
        prices = np.random.uniform(1.08, 1.12, 100).astype(np.float32)
        result = pattern_model.analyze_patterns_with_67_indicators(
            prices=prices,
            indicators=test_array, 
            current_price=float(prices[-1])
        )
        print(f"✅ Function call successful: {type(result)}")
    sys.path.remove(os.path.join("models", "pattern_master"))
except Exception as e:
    print(f"❌ Pattern Master failed: {e}")

# Test Execution Expert import
print("\nTesting Execution Expert...")
try:
    sys.path.insert(0, os.path.join("models", "execution_expert"))
    import ultra_fast_model as execution_model
    print(f"✅ Execution Expert imported successfully")
    print(f"Has optimize_execution_with_67_indicators: {hasattr(execution_model, 'optimize_execution_with_67_indicators')}")
    if hasattr(execution_model, 'optimize_execution_with_67_indicators'):
        test_array = np.random.random((67, 100)).astype(np.float32)
        result = execution_model.optimize_execution_with_67_indicators(
            order_size=10000.0,
            indicators=test_array
        )
        print(f"✅ Function call successful: {type(result)}")
    sys.path.remove(os.path.join("models", "execution_expert"))
except Exception as e:
    print(f"❌ Execution Expert failed: {e}")

print("\n✅ Import test completed!")
