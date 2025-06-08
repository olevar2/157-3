#!/usr/bin/env python3
"""
Diagnostic test to examine the agent mapping structure
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
from engines.ai_enhancement.registry import GeniusAgentType

def test_mapping_structure():
    print("=== DIAGNOSTIC: Agent Mapping Structure ===")
    
    try:
        # Initialize bridge
        bridge = AdaptiveIndicatorBridge()
        
        # Check if agent_indicator_mapping exists and has content
        print(f"Agent mapping exists: {hasattr(bridge, 'agent_indicator_mapping')}")
        
        if hasattr(bridge, 'agent_indicator_mapping'):
            mapping = bridge.agent_indicator_mapping
            print(f"Mapping type: {type(mapping)}")
            print(f"Mapping keys: {list(mapping.keys()) if mapping else 'None'}")
            
            # Check each agent
            for agent_type in GeniusAgentType:
                print(f"\n--- {agent_type.value} ---")
                if agent_type in mapping:
                    agent_config = mapping[agent_type]
                    print(f"Config type: {type(agent_config)}")
                    print(f"Config keys: {list(agent_config.keys()) if isinstance(agent_config, dict) else 'Not a dict'}")
                    
                    if isinstance(agent_config, dict):
                        primary = agent_config.get('primary_indicators', [])
                        secondary = agent_config.get('secondary_indicators', [])
                        print(f"Primary indicators count: {len(primary) if isinstance(primary, list) else 'Not a list'}")
                        print(f"Secondary indicators count: {len(secondary) if isinstance(secondary, list) else 'Not a list'}")
                        
                        if isinstance(primary, list) and len(primary) > 0:
                            print(f"First 3 primary: {primary[:3]}")
                        if isinstance(secondary, list) and len(secondary) > 0:
                            print(f"First 3 secondary: {secondary[:3]}")
                else:
                    print("Agent not found in mapping")
        
        # Test the get_indicators_for_agent method directly
        print(f"\n=== TESTING get_indicators_for_agent METHOD ===")
        for agent_type in GeniusAgentType:
            indicators = bridge.get_indicators_for_agent(agent_type)
            print(f"{agent_type.value}: {len(indicators)} indicators")
            if len(indicators) > 0:
                print(f"  First 3: {indicators[:3]}")
                
    except Exception as e:
        print(f"Error in diagnostic test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mapping_structure()
