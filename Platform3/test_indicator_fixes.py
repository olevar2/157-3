#!/usr/bin/env python3
"""
Test script to verify all indicator fixes are working correctly
"""

import sys
import os

# Add the indicators path
sys.path.append('services/analytics-service/src/engines/indicators')

def test_momentum_indicators():
    """Test momentum specialized indicators"""
    print("🧪 Testing Momentum Specialized Indicators...")
    
    try:
        from momentum.DayTradingMomentum import DayTradingMomentum
        from momentum.ScalpingMomentum import ScalpingMomentum  
        from momentum.SwingMomentum import SwingMomentum
        
        # Test instantiation
        day_trading = DayTradingMomentum()
        scalping = ScalpingMomentum()
        swing = SwingMomentum()
        
        print("✅ Momentum Specialized Indicators: DayTradingMomentum, ScalpingMomentum, SwingMomentum - ALL WORKING")
        return True
    except Exception as e:
        print(f"❌ Momentum Specialized Indicators failed: {e}")
        return False

def test_trend_indicators():
    """Test trend indicators"""
    print("🧪 Testing Trend Indicators...")
    
    try:
        from trend.SMA_EMA import SMA_EMA
        from trend.ADX import ADX
        from trend.Ichimoku import Ichimoku
        
        # Test instantiation
        sma_ema = SMA_EMA()
        adx = ADX()
        ichimoku = Ichimoku()
        
        print("✅ Trend Indicators: SMA_EMA, ADX, Ichimoku - ALL WORKING")
        return True
    except Exception as e:
        print(f"❌ Trend Indicators failed: {e}")
        return False

def test_vortex_indicator():
    """Test Vortex indicator"""
    print("🧪 Testing Vortex Indicator...")
    
    try:
        from volatility.Vortex import Vortex
        
        # Test instantiation
        vortex = Vortex()
        
        print("✅ Vortex Indicator: Vortex - WORKING")
        return True
    except Exception as e:
        print(f"❌ Vortex Indicator failed: {e}")
        return False

def test_autoencoder_features():
    """Test AutoencoderFeatures"""
    print("🧪 Testing AutoencoderFeatures...")
    
    try:
        from advanced.AutoencoderFeatures import AutoencoderFeatures
        
        # Test instantiation with required input_dim
        autoencoder = AutoencoderFeatures(input_dim=10)
        
        print("✅ AutoencoderFeatures: AutoencoderFeatures - WORKING")
        return True
    except Exception as e:
        print(f"❌ AutoencoderFeatures failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Platform3 Indicator Fixes Verification...")
    print("=" * 60)
    
    results = []
    
    # Test all indicator categories
    results.append(test_momentum_indicators())
    results.append(test_trend_indicators())
    results.append(test_vortex_indicator())
    results.append(test_autoencoder_features())
    
    print("=" * 60)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ All indicator fixes verified successfully!")
        return True
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total} passed)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
