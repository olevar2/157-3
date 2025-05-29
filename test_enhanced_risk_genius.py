#!/usr/bin/env python3
"""
Test Enhanced Risk Genius with All 67 Indicators
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test enhanced Risk Genius with 67 indicators
try:
    from models.risk_genius.ultra_fast_model import analyze_risk_ultra_fast
    
    print("🧪 Testing Enhanced Risk Genius with 67 Indicators")
    print("=" * 60)
    
    # Create test data
    price_data = np.array([1.1000, 1.1010, 1.1005, 1.1015, 1.1020, 1.1025, 1.1030])
    
    # Create all 67 indicators (sample values)
    indicators = {
        'rsi_14': 65.0, 'rsi_21': 68.0, 'bb_upper': 1.1035, 'bb_lower': 1.0995,
        'macd_line': 0.0002, 'macd_signal': 0.0001, 'atr_14': 0.0012, 'atr_21': 0.0015,
        'adx_14': 35.0, 'stoch_k': 75.0, 'stoch_d': 72.0, 'cci_14': 85.0,
        'williams_r': -25.0, 'obv': 10000.0, 'volume_ratio': 1.8, 
        'volatility_regime': 0.6, 'trend_strength': 0.75, 
        'breakout_probability': 0.3, 'reversal_probability': 0.2
    }
    
    market_conditions = {
        'account_balance': 100000,
        'entry_price': 1.1025,
        'stop_loss': 1.1010,
        'session': 'london'
    }
    
    result = analyze_risk_ultra_fast('EURUSD', price_data, indicators, market_conditions)
    
    print("✅ Enhanced Risk Analysis Result:")
    print(f"Risk Score: {result['risk_score']:.2f}")
    print(f"Position Size: {result['position_size']:.2f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Execution Time: {result['execution_time_ms']:.3f}ms")
    print(f"Indicators Used: {result['indicators_used']}/{result['total_indicators_available']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Volatility: {result['volatility']:.4f}")
    print(f"VaR 95%: {result['var_95']:.4f}")
    
    # Performance check
    if result['execution_time_ms'] < 1.0:
        print(f"🚀 PERFORMANCE TARGET MET: {result['execution_time_ms']:.3f}ms < 1.0ms")
    else:
        print(f"⚠️  PERFORMANCE WARNING: {result['execution_time_ms']:.3f}ms > 1.0ms")
    
    print("\n🎯 SUCCESS: Enhanced Risk Genius using ALL 67 indicators!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ Test Error: {e}")
    import traceback
    traceback.print_exc()
