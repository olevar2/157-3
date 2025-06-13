#!/usr/bin/env python3
"""
Validation Script: Bridge Indicator Mapping Verification

Validates that the AdaptiveIndicatorBridge correctly assigns indicators 
to each agent according to GENIUS_AGENT_INDICATOR_MAPPING.md
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
from engines.ai_enhancement.genius_agent_integration import GeniusAgentType
from engines.ai_enhancement.indicator_mappings import AGENT_INDICATOR_MAPPINGS

# Expected indicator counts from GENIUS_AGENT_INDICATOR_MAPPING.md
EXPECTED_INDICATOR_COUNTS = {
    "risk_genius": 24,
    "session_expert": 15,
    "pattern_master": 40,
    "execution_expert": 19,
    "pair_specialist": 14,
    "decision_master": 10,
    "ai_model_coordinator": 11,  # AI Model Coordinator
    "market_microstructure_genius": 17,  # Market Microstructure Genius
    "sentiment_integration_genius": 12   # Sentiment Integration Genius
}

def validate_indicator_mappings():
    """Validate that each agent gets the correct number of indicators"""
    print("=" * 80)
    print("VALIDATING INDICATOR BRIDGE MAPPING")
    print("=" * 80)
    
    all_passed = True
    total_indicators_assigned = 0
    
    for agent_key, expected_count in EXPECTED_INDICATOR_COUNTS.items():
        # Get indicators from the mapping
        agent_indicators = AGENT_INDICATOR_MAPPINGS.get(agent_key, {})
        
        # Count total indicators for this agent
        actual_count = sum(len(indicators) for indicators in agent_indicators.values())
        total_indicators_assigned += actual_count
        
        # Check if count matches
        status = "PASS" if actual_count == expected_count else "FAIL"
        if status == "FAIL":
            all_passed = False
        
        print(f"{agent_key:<30} Expected: {expected_count:>3} | Actual: {actual_count:>3} | {status}")
        
        # Show detailed breakdown
        if actual_count != expected_count:
            print(f"  Categories breakdown:")
            for category, indicators in agent_indicators.items():
                print(f"    {category}: {len(indicators)} indicators")
            print()
    
    print("-" * 80)
    print(f"Total indicators assigned: {total_indicators_assigned}")
    print(f"Expected total: {sum(EXPECTED_INDICATOR_COUNTS.values())}")
    
    if all_passed:
        print("✅ ALL MAPPING VALIDATIONS PASSED!")
    else:
        print("❌ SOME MAPPING VALIDATIONS FAILED!")
    
    return all_passed

async def test_bridge_indicator_access():
    """Test that the bridge can actually access indicators for each agent"""
    print("\n" + "=" * 80)
    print("TESTING BRIDGE INDICATOR ACCESS")
    print("=" * 80)
    
    bridge = AdaptiveIndicatorBridge()
    
    # Sample market data for testing
    test_market_data = {
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    }
    
    all_bridge_tests_passed = True
    
    for agent_enum in GeniusAgentType:
        agent_name = agent_enum.value
        expected_count = EXPECTED_INDICATOR_COUNTS.get(agent_name, 0)
        
        try:
            # Test bridge access
            indicators = await bridge.get_agent_indicators_async(agent_enum, test_market_data)
            actual_count = len(indicators)
            
            status = "PASS" if actual_count > 0 else "FAIL"
            if status == "FAIL":
                all_bridge_tests_passed = False
            
            print(f"{agent_name:<30} Requested: {expected_count:>3} | Received: {actual_count:>3} | {status}")
            
        except Exception as e:
            print(f"{agent_name:<30} ERROR: {str(e)}")
            all_bridge_tests_passed = False
    
    if all_bridge_tests_passed:
        print("\n✅ ALL BRIDGE ACCESS TESTS PASSED!")
    else:
        print("\n❌ SOME BRIDGE ACCESS TESTS FAILED!")
    
    return all_bridge_tests_passed

def main():
    """Run all validation tests"""
    print("Platform3 Bridge Indicator Mapping Validation")
    print("Checking alignment with GENIUS_AGENT_INDICATOR_MAPPING.md")
    print()
    
    # Test 1: Validate static mappings
    mapping_valid = validate_indicator_mappings()
    
    # Test 2: Test bridge access
    bridge_valid = asyncio.run(test_bridge_indicator_access())
    
    print("\n" + "=" * 80)
    print("FINAL VALIDATION RESULTS")
    print("=" * 80)
    
    if mapping_valid and bridge_valid:
        print("🎯 ALL VALIDATIONS PASSED!")
        print("✅ Bridge correctly maps indicators according to GENIUS_AGENT_INDICATOR_MAPPING.md")
        print("✅ All agents can access their assigned indicators through the bridge")
        return True
    else:
        print("❌ VALIDATION FAILED!")
        if not mapping_valid:
            print("❌ Indicator mapping counts don't match expected values")
        if not bridge_valid:
            print("❌ Bridge access tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
