#!/usr/bin/env python3
"""
Bridge Integration Validation Script
Validates that all 9 Genius Agents have proper AdaptiveIndicatorBridge integration
"""

import os
import sys
from pathlib import Path

def validate_agent_integration():
    """Validate bridge integration across all 9 agents"""
    
    print("Validating AdaptiveIndicatorBridge Integration")
    print("=" * 60)
    
    agents = [
        ("Risk Genius", "ai-platform/ai-models/intelligent-agents/risk-genius/ultra_fast_model.py"),
        ("Session Expert", "ai-platform/ai-models/intelligent-agents/session-expert/ultra_fast_model.py"),
        ("Execution Expert", "ai-platform/ai-models/intelligent-agents/execution-expert/ultra_fast_model.py"),
        ("Pair Specialist", "ai-platform/ai-models/intelligent-agents/pair-specialist/ultra_fast_model.py"),
        ("Strategy Expert", "ai-platform/ai-models/intelligent-agents/strategy-expert/ultra_fast_model.py"),
        ("Decision Master", "ai-platform/ai-models/intelligent-agents/decision-master/model.py"),
        ("Indicator Expert", "ai-platform/ai-models/intelligent-agents/indicator-expert/model.py"),
        ("Simulation Expert", "ai-platform/ai-models/intelligent-agents/simulation-expert/model.py"),
        ("Currency Pair Intelligence", "ai-platform/ai-models/intelligent-agents/currency-pair-intelligence/model.py")
    ]
    
    integration_results = {}
    
    for agent_name, file_path in agents:
        print(f"\nChecking {agent_name}...")
        
        full_path = Path(file_path)
        if not full_path.exists():
            print(f"  ERROR: File not found: {file_path}")
            integration_results[agent_name] = "MISSING_FILE"
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required imports
            has_bridge_import = "from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge" in content
            has_registry_import = "from engines.ai_enhancement.registry import GeniusAgentType" in content
            has_base_import = "from engines.ai_enhancement.genius_agent_integration import BaseAgentInterface" in content
            
            # Check for bridge usage
            has_bridge_init = "AdaptiveIndicatorBridge()" in content
            has_base_inheritance = "BaseAgentInterface" in content
            has_async_method = "get_agent_indicators_async" in content
            
            checks = {
                "Bridge Import": has_bridge_import,
                "Registry Import": has_registry_import,
                "Base Import": has_base_import,
                "Bridge Init": has_bridge_init,
                "Base Inheritance": has_base_inheritance,
                "Async Method": has_async_method
            }
            
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            
            print(f"  Integration Status: {passed_checks}/{total_checks} checks passed")
            
            for check_name, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                print(f"    {status}: {check_name}")
            
            if passed_checks == total_checks:
                integration_results[agent_name] = "FULLY_INTEGRATED"
                print(f"  SUCCESS: {agent_name} is FULLY INTEGRATED!")
            elif passed_checks >= 4:
                integration_results[agent_name] = "MOSTLY_INTEGRATED"
                print(f"  WARNING: {agent_name} is mostly integrated")
            else:
                integration_results[agent_name] = "NEEDS_WORK"
                print(f"  ERROR: {agent_name} needs more integration work")
                
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            integration_results[agent_name] = "ERROR"
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION SUMMARY")
    print("=" * 60)
    
    fully_integrated = sum(1 for status in integration_results.values() if status == "FULLY_INTEGRATED")
    total_agents = len(agents)
    
    print(f"Fully Integrated: {fully_integrated}/{total_agents} agents")
    print(f"Success Rate: {(fully_integrated/total_agents)*100:.1f}%")
    
    if fully_integrated == total_agents:
        print("\nSUCCESS: All agents have complete bridge integration!")
        print("Ready for humanitarian mission: maximizing profits for sick babies and poor families!")
    else:
        print(f"\nWARNING: {total_agents - fully_integrated} agents still need work")
    
    return integration_results

if __name__ == "__main__":
    validate_agent_integration()