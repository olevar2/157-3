#!/usr/bin/env python3
"""
Minimal test for debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting minimal test...")

try:
    from engines.ai_enhancement.adaptive_indicator_bridge import GeniusAgentType
    print("✅ GeniusAgentType imported")
    
    from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
    print("✅ AdaptiveIndicatorBridge imported")
    
    bridge = AdaptiveIndicatorBridge()
    print("✅ Bridge instantiated")
    
    print(f"✅ Found {len(bridge.indicator_registry)} indicators in registry")
    print(f"✅ Found {len(bridge.agent_indicator_mapping)} agent mappings")
    
    # Test one simple method
    import asyncio
    import numpy as np
    from datetime import datetime
    
    async def simple_test():
        test_data = {
            'close': [1.05, 1.06, 1.055, 1.057, 1.052] * 10,
            'volume': [1000] * 50
        }
        
        regime = await bridge._detect_market_regime(test_data)
        print(f"✅ Market regime detection works: {regime}")
        
        return True
    
    result = asyncio.run(simple_test())
    print("✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
