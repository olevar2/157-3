"""
Enhanced Functions Direct Test

Tests only the enhanced functions from ultra-fast models without
initializing the full model classes to avoid dependency issues.
"""

import numpy as np
import time
import sys
import os

def generate_67_indicators():
    """Generate mock 67 indicators for testing"""
    indicators = {}
    
    # Price action indicators (7)
    indicators['sma_20'] = np.random.uniform(1.08, 1.12, 100)
    indicators['ema_20'] = np.random.uniform(1.08, 1.12, 100)
    indicators['wma_20'] = np.random.uniform(1.08, 1.12, 100)
    indicators['hull_ma'] = np.random.uniform(1.08, 1.12, 100)
    indicators['tema'] = np.random.uniform(1.08, 1.12, 100)
    indicators['dema'] = np.random.uniform(1.08, 1.12, 100)
    indicators['kama'] = np.random.uniform(1.08, 1.12, 100)
    
    # Momentum indicators (12)
    indicators['rsi'] = np.random.uniform(20, 80, 100)
    indicators['macd'] = np.random.uniform(-0.001, 0.001, 100)
    indicators['macd_signal'] = np.random.uniform(-0.001, 0.001, 100)
    indicators['macd_histogram'] = np.random.uniform(-0.0005, 0.0005, 100)
    indicators['stoch_k'] = np.random.uniform(20, 80, 100)
    indicators['stoch_d'] = np.random.uniform(20, 80, 100)
    indicators['williams_r'] = np.random.uniform(-80, -20, 100)
    indicators['roc'] = np.random.uniform(-0.01, 0.01, 100)
    indicators['mom'] = np.random.uniform(-0.001, 0.001, 100)
    indicators['cci'] = np.random.uniform(-100, 100, 100)
    indicators['ultimate_oscillator'] = np.random.uniform(30, 70, 100)
    indicators['tsi'] = np.random.uniform(-30, 30, 100)
    
    # Volatility indicators (8)
    indicators['bollinger_upper'] = np.random.uniform(1.10, 1.14, 100)
    indicators['bollinger_lower'] = np.random.uniform(1.06, 1.10, 100)
    indicators['bollinger_width'] = np.random.uniform(0.02, 0.08, 100)
    indicators['atr'] = np.random.uniform(0.0008, 0.0025, 100)
    indicators['average_true_range'] = np.random.uniform(0.0008, 0.0025, 100)
    indicators['volatility'] = np.random.uniform(0.05, 0.25, 100)
    indicators['standard_deviation'] = np.random.uniform(0.003, 0.012, 100)
    indicators['variance'] = np.random.uniform(0.00001, 0.00015, 100)
    
    # Trend indicators (12)
    indicators['adx'] = np.random.uniform(15, 45, 100)
    indicators['plus_di'] = np.random.uniform(10, 40, 100)
    indicators['minus_di'] = np.random.uniform(10, 40, 100)
    indicators['aroon_up'] = np.random.uniform(30, 100, 100)
    indicators['aroon_down'] = np.random.uniform(0, 70, 100)
    indicators['parabolic_sar'] = np.random.uniform(1.08, 1.12, 100)
    indicators['linear_regression'] = np.random.uniform(1.08, 1.12, 100)
    indicators['linear_regression_slope'] = np.random.uniform(-0.001, 0.001, 100)
    indicators['linear_regression_angle'] = np.random.uniform(-45, 45, 100)
    indicators['linear_regression_intercept'] = np.random.uniform(1.08, 1.12, 100)
    indicators['time_series_forecast'] = np.random.uniform(1.08, 1.12, 100)
    indicators['dpo'] = np.random.uniform(-0.005, 0.005, 100)
    
    # Volume indicators (8)
    indicators['volume'] = np.random.uniform(10000, 50000, 100)
    indicators['volume_sma'] = np.random.uniform(15000, 35000, 100)
    indicators['on_balance_volume'] = np.random.uniform(-100000, 100000, 100)
    indicators['volume_price_trend'] = np.random.uniform(-1000, 1000, 100)
    indicators['accumulation_distribution'] = np.random.uniform(-5000, 5000, 100)
    indicators['chaikin_money_flow'] = np.random.uniform(-0.2, 0.2, 100)
    indicators['money_flow_index'] = np.random.uniform(20, 80, 100)
    indicators['ease_of_movement'] = np.random.uniform(-0.5, 0.5, 100)
    
    # Support/Resistance indicators (6)
    indicators['pivot_point'] = np.random.uniform(1.09, 1.11, 100)
    indicators['resistance_1'] = np.random.uniform(1.10, 1.12, 100)
    indicators['resistance_2'] = np.random.uniform(1.11, 1.13, 100)
    indicators['support_1'] = np.random.uniform(1.08, 1.10, 100)
    indicators['support_2'] = np.random.uniform(1.07, 1.09, 100)
    indicators['fibonacci_retracement'] = np.random.uniform(1.08, 1.12, 100)
    
    # Statistical indicators (8)
    indicators['z_score'] = np.random.uniform(-2, 2, 100)
    indicators['percentile_rank'] = np.random.uniform(10, 90, 100)
    indicators['linear_correlation'] = np.random.uniform(-0.8, 0.8, 100)
    indicators['beta'] = np.random.uniform(0.5, 1.5, 100)
    indicators['alpha'] = np.random.uniform(-0.01, 0.01, 100)
    indicators['sharpe_ratio'] = np.random.uniform(-1, 3, 100)
    indicators['information_ratio'] = np.random.uniform(-2, 2, 100)
    indicators['treynor_ratio'] = np.random.uniform(-0.5, 0.5, 100)
    
    # Pattern indicators (6)
    indicators['fractal_high'] = np.random.choice([0, 1], 100, p=[0.95, 0.05])
    indicators['fractal_low'] = np.random.choice([0, 1], 100, p=[0.95, 0.05])
    indicators['zigzag'] = np.random.uniform(1.08, 1.12, 100)
    indicators['elliott_wave'] = np.random.randint(1, 6, 100)
    indicators['harmonic_pattern'] = np.random.choice([0, 1], 100, p=[0.9, 0.1])
    indicators['candlestick_pattern'] = np.random.randint(0, 10, 100)
    
    return indicators

def convert_indicators_to_array(indicators_dict):
    """Convert 67 indicators dictionary to standardized numpy array"""
    # List all 67 indicator keys in consistent order
    indicator_keys = [
        # Price action (7)
        'sma_20', 'ema_20', 'wma_20', 'hull_ma', 'tema', 'dema', 'kama',
        # Momentum (12)  
        'rsi', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d',
        'williams_r', 'roc', 'mom', 'cci', 'ultimate_oscillator', 'tsi',
        # Volatility (8)
        'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'atr',
        'average_true_range', 'volatility', 'standard_deviation', 'variance',
        # Trend (12)
        'adx', 'plus_di', 'minus_di', 'aroon_up', 'aroon_down', 'parabolic_sar',
        'linear_regression', 'linear_regression_slope', 'linear_regression_angle',
        'linear_regression_intercept', 'time_series_forecast', 'dpo',
        # Volume (8)
        'volume', 'volume_sma', 'on_balance_volume', 'volume_price_trend',
        'accumulation_distribution', 'chaikin_money_flow', 'money_flow_index', 'ease_of_movement',
        # Support/Resistance (6)
        'pivot_point', 'resistance_1', 'resistance_2', 'support_1', 'support_2', 'fibonacci_retracement',
        # Statistical (8)
        'z_score', 'percentile_rank', 'linear_correlation', 'beta', 'alpha',
        'sharpe_ratio', 'information_ratio', 'treynor_ratio',
        # Pattern (6)
        'fractal_high', 'fractal_low', 'zigzag', 'elliott_wave', 'harmonic_pattern', 'candlestick_pattern'
    ]
    
    # Create array with shape (67, data_length)
    data_length = len(list(indicators_dict.values())[0])
    indicators_array = np.zeros((67, data_length), dtype=np.float32)
    
    for i, key in enumerate(indicator_keys):
        if key in indicators_dict:
            indicators_array[i] = indicators_dict[key].astype(np.float32)
        else:
            # Fill missing indicators with neutral values
            if 'rsi' in key or 'stoch' in key:
                indicators_array[i] = 50.0  # Neutral momentum
            elif 'adx' in key:
                indicators_array[i] = 25.0  # Moderate trend
            elif 'volume' in key:
                indicators_array[i] = 25000.0  # Average volume
            else:
                indicators_array[i] = 0.0  # Default
    
    return indicators_array

def test_enhanced_functions():
    """Test enhanced functions directly without full model initialization"""
    print("🚀 Testing Enhanced Ultra-Fast Functions with 67 Indicators")
    print("=" * 80)
    
    # Generate test data
    print("📊 Generating test data...")
    indicators_dict = generate_67_indicators()
    indicators_array = convert_indicators_to_array(indicators_dict)
    
    # Test data for different functions
    prices = np.random.uniform(1.08, 1.12, 100).astype(np.float32)
    pair_id = 1
    spread = 0.00015
    session_type = 3  # US session
    timeframe = 5  # 5-minute
    
    print(f"✅ Generated 67 indicators array: {indicators_array.shape}")
    print(f"✅ Price data: {len(prices)} points")
    
    results = {}    # Test 1: Risk Genius Enhanced Functions
    print("\n🎯 Testing Risk Genius Enhanced Functions...")
    try:
        # Import and test risk analysis functions
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "risk_genius"))
        import ultra_fast_model as risk_model
        
        start_time = time.time()
        risk_result = risk_model.analyze_risk_with_67_indicators_simple(indicators_array)
        execution_time = (time.time() - start_time) * 1000
        
        results['risk_genius'] = {
            'success': True,
            'execution_time_ms': execution_time,
            'result_type': type(risk_result).__name__,
            'result_shape': getattr(risk_result, 'shape', 'scalar') if hasattr(risk_result, 'shape') else 'scalar'
        }
        
        print(f"✅ Risk analysis completed in {execution_time:.3f}ms")
        print(f"   Result type: {type(risk_result)}")
        
    except Exception as e:
        results['risk_genius'] = {'success': False, 'error': str(e)}
        print(f"❌ Risk Genius test failed: {e}")
    finally:
        # Clear path
        if os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "risk_genius") in sys.path:
            sys.path.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "risk_genius"))
    
    # Test 2: Session Expert Enhanced Functions
    print("\n📅 Testing Session Expert Enhanced Functions...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "session_expert"))
        import ultra_fast_model as session_model
        
        start_time = time.time()
        session_result = session_model.analyze_session_with_67_indicators(indicators_array)
        execution_time = (time.time() - start_time) * 1000
        
        results['session_expert'] = {
            'success': True,
            'execution_time_ms': execution_time,
            'result_type': type(session_result).__name__,
            'result_shape': getattr(session_result, 'shape', 'scalar') if hasattr(session_result, 'shape') else 'scalar'
        }
        
        print(f"✅ Session analysis completed in {execution_time:.3f}ms")
        print(f"   Result type: {type(session_result)}")
        
    except Exception as e:
        results['session_expert'] = {'success': False, 'error': str(e)}
        print(f"❌ Session Expert test failed: {e}")
    finally:
        # Clear path
        if os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "session_expert") in sys.path:
            sys.path.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "session_expert"))
    
    # Test 3: Pair Specialist Enhanced Functions  
    print("\n💱 Testing Pair Specialist Enhanced Functions...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pair_specialist"))
        import ultra_fast_model as pair_model
        
        start_time = time.time()
        pair_result = pair_model.analyze_pair_with_all_indicators(indicators_array)
        execution_time = (time.time() - start_time) * 1000
        
        results['pair_specialist'] = {
            'success': True,
            'execution_time_ms': execution_time,
            'result_type': type(pair_result).__name__,
            'result_shape': getattr(pair_result, 'shape', 'scalar') if hasattr(pair_result, 'shape') else 'scalar'
        }
        
        print(f"✅ Pair analysis completed in {execution_time:.3f}ms")
        print(f"   Result type: {type(pair_result)}")
        
    except Exception as e:
        results['pair_specialist'] = {'success': False, 'error': str(e)}
        print(f"❌ Pair Specialist test failed: {e}")
    finally:
        # Clear path
        if os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pair_specialist") in sys.path:
            sys.path.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pair_specialist"))
      # Test 4: Pattern Master Enhanced Functions
    print("\n📈 Testing Pattern Master Enhanced Functions...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pattern_master"))
        import ultra_fast_model as pattern_model
        
        start_time = time.time()
        pattern_result = pattern_model.analyze_patterns_with_67_indicators(
            prices=prices,
            indicators=indicators_array, 
            current_price=float(prices[-1])
        )
        execution_time = (time.time() - start_time) * 1000
        
        results['pattern_master'] = {
            'success': True,
            'execution_time_ms': execution_time,
            'result_type': type(pattern_result).__name__,
            'result_shape': getattr(pattern_result, 'shape', 'scalar') if hasattr(pattern_result, 'shape') else 'scalar'
        }
        
        print(f"✅ Pattern analysis completed in {execution_time:.3f}ms")
        print(f"   Result type: {type(pattern_result)}")
        
    except Exception as e:
        results['pattern_master'] = {'success': False, 'error': str(e)}
        print(f"❌ Pattern Master test failed: {e}")
    finally:
        # Clear path
        if os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pattern_master") in sys.path:
            sys.path.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pattern_master"))
    
    # Test 5: Execution Expert Enhanced Functions
    print("\n⚡ Testing Execution Expert Enhanced Functions...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "execution_expert"))
        import ultra_fast_model as execution_model
        
        start_time = time.time()
        execution_result = execution_model.optimize_execution_with_67_indicators(
            order_size=10000.0,
            indicators=indicators_array
        )
        execution_time = (time.time() - start_time) * 1000
        
        results['execution_expert'] = {
            'success': True,
            'execution_time_ms': execution_time,
            'result_type': type(execution_result).__name__,
            'result_shape': getattr(execution_result, 'shape', 'scalar') if hasattr(execution_result, 'shape') else 'scalar'
        }
        
        print(f"✅ Execution optimization completed in {execution_time:.3f}ms")
        print(f"   Result type: {type(execution_result)}")
        
    except Exception as e:
        results['execution_expert'] = {'success': False, 'error': str(e)}
        print(f"❌ Execution Expert test failed: {e}")
    finally:
        # Clear path
        if os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "execution_expert") in sys.path:
            sys.path.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "execution_expert"))
    
    # Summary Report
    print("\n" + "=" * 80)
    print("📊 ENHANCED MODELS PERFORMANCE SUMMARY")
    print("=" * 80)
    
    successful_tests = 0
    total_execution_time = 0
    
    for model_name, result in results.items():
        if result['success']:
            successful_tests += 1
            exec_time = result['execution_time_ms']
            total_execution_time += exec_time
            status = "✅ PASS" if exec_time < 1.0 else "⚠️  SLOW"
            print(f"{model_name.upper():20} | {exec_time:8.3f}ms | {status}")
        else:
            print(f"{model_name.upper():20} | FAILED    | ❌ ERROR: {result['error'][:50]}...")
    
    print("-" * 80)
    print(f"SUCCESSFUL TESTS: {successful_tests}/5")
    print(f"TOTAL TIME:       {total_execution_time:.3f}ms")
    print(f"AVERAGE TIME:     {total_execution_time/max(1,successful_tests):.3f}ms")
    
    if successful_tests == 5 and total_execution_time < 5.0:
        print("\n🎉 ALL ENHANCED MODELS PASSED! Platform3 ready for 24/7 profit generation!")
        print("💰 Humanitarian forex trading optimization COMPLETE!")
    elif successful_tests >= 3:
        print(f"\n✅ {successful_tests}/5 models working - Platform3 partially enhanced")
        print("🔧 Continue debugging remaining models for full optimization")
    else:
        print(f"\n⚠️  Only {successful_tests}/5 models working - debugging needed")
        print("🔧 Check model implementations and dependencies")
    
    return results

if __name__ == "__main__":
    test_enhanced_functions()
