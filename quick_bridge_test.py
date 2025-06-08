#!/usr/bin/env python3
"""
Quick Test of Adaptive Indicator Bridge Functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import numpy as np
from datetime import datetime

def test_imports():
    """Test if we can import the necessary components"""
    try:
        from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge, GeniusAgentType
        print("✅ Imports successful")
        return AdaptiveIndicatorBridge, GeniusAgentType
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return None, None

async def test_basic_bridge_operations():
    """Test basic bridge operations"""
    
    # Test imports
    AdaptiveIndicatorBridge, GeniusAgentType = test_imports()
    if not AdaptiveIndicatorBridge:
        return False
    
    try:
        # Initialize bridge
        bridge = AdaptiveIndicatorBridge()
        print(f"✅ Bridge initialized with {len(bridge.indicator_registry)} indicators")
        print(f"✅ Agent mappings: {len(bridge.agent_indicator_mapping)} agents")
        
        # Create test data
        test_data = {
            'timestamp': datetime.now(),
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'open': [1.05] * 50,
            'high': [1.06] * 50,
            'low': [1.04] * 50,
            'close': [1.05] * 50,
            'volume': [1000] * 50,
        }
        
        # Test market regime detection
        print("\n🧪 Testing market regime detection...")
        regime = await bridge._detect_market_regime(test_data)
        print(f"✅ Market regime: {regime}")
        
        # Test single indicator calculation
        print("\n🧪 Testing single indicator calculation...")
        if 'correlation_analysis' in bridge.indicator_registry:
            result = await bridge._calculate_single_indicator('correlation_analysis', test_data, regime)
            print(f"✅ correlation_analysis result: {result}")
        
        # Test comprehensive package for one agent
        print("\n🧪 Testing comprehensive indicator package...")
        agent_type = GeniusAgentType.RISK_GENIUS
        
        start_time = asyncio.get_event_loop().time()
        package = await bridge.get_comprehensive_indicator_package(test_data, agent_type, max_indicators=3)
        end_time = asyncio.get_event_loop().time()
        
        print(f"✅ Package generated in {(end_time - start_time) * 1000:.1f}ms")
        print(f"   Agent: {package.agent_type.value}")
        print(f"   Indicators: {len(package.indicators)}")
        print(f"   Optimization score: {package.optimization_score:.2f}")
        print(f"   Fallback mode: {package.metadata.get('fallback_mode', False)}")
        
        if package.indicators:
            print("   Sample indicators:")
            for name, value in list(package.indicators.items())[:3]:
                print(f"     - {name}: {value}")
        
        # Test multiple agents quickly
        print(f"\n🧪 Testing all {len(GeniusAgentType)} agents...")
        success_count = 0
        
        for i, agent_type in enumerate(GeniusAgentType):
            try:
                package = await bridge.get_comprehensive_indicator_package(test_data, agent_type, max_indicators=2)
                if package and len(package.indicators) > 0:
                    success_count += 1
                    print(f"   ✅ {agent_type.value}: {len(package.indicators)} indicators")
                else:
                    print(f"   ⚠️  {agent_type.value}: No indicators generated")
            except Exception as e:
                print(f"   ❌ {agent_type.value}: Error - {str(e)}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Total agents tested: {len(GeniusAgentType)}")
        print(f"   Successful agents: {success_count}")
        print(f"   Success rate: {(success_count/len(GeniusAgentType))*100:.1f}%")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 QUICK ADAPTIVE INDICATOR BRIDGE TEST")
    print("=" * 50)
    
    try:
        success = asyncio.run(test_basic_bridge_operations())
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 TESTS PASSED - Bridge is functional!")
        else:
            print("💥 TESTS FAILED - Bridge needs fixes")
        
        return success
        
    except Exception as e:
        print(f"💥 Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'SUCCESS' if success else 'FAILURE'}")
