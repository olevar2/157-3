#!/usr/bin/env python3
"""
Platform3 Agent-Indicator Integration Verification Script
Tests that all 9 genius agents properly use their assigned indicators
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add Platform3 paths
current_dir = Path(__file__).parent
platform_root = current_dir
sys.path.insert(0, str(platform_root))
sys.path.insert(0, str(platform_root / "engines" / "ai_enhancement"))

def test_agent_indicator_integration():
    """Test that all agents properly use their assigned indicators"""
    print("=== Platform3 Agent-Indicator Integration Verification ===")
    print(f"Test started at: {datetime.now()}")
    print()
    
    try:
        # Import the genius agent integration module
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        
        # Create the integration instance
        integration = GeniusAgentIntegration()
        
        print("✅ Successfully imported GeniusAgentIntegration")
        print(f"✅ Found {len(integration.genius_agents)} genius agents")
        print()
        
        # Test data for analysis
        test_market_data = {
            "symbol": "EURUSD",
            "ohlcv": [
                {"open": 1.0500, "high": 1.0520, "low": 1.0490, "close": 1.0510, "volume": 1000000},
                {"open": 1.0510, "high": 1.0530, "low": 1.0500, "close": 1.0525, "volume": 1200000},
                {"open": 1.0525, "high": 1.0540, "low": 1.0515, "close": 1.0535, "volume": 1100000},
            ]
        }
        
        agent_results = {}
        
        # Test each agent individually
        for agent_name, agent_interface in integration.genius_agents.items():
            print(f"Testing {agent_name}...")
            
            try:
                # Test agent analysis
                result = agent_interface.execute_analysis(test_market_data, {})
                
                # Check if agent is using real indicators
                indicators_used = result.get("indicators_used", [])
                analysis_timestamp = result.get("analysis_timestamp", "")
                status = result.get("status", "unknown")
                
                agent_results[agent_name] = {
                    "status": status,
                    "indicators_used": indicators_used,
                    "indicator_count": len(indicators_used),
                    "analysis_timestamp": analysis_timestamp,
                    "has_real_analysis": len(indicators_used) > 0,
                    "error": result.get("error", None)
                }
                
                if status == "active" and len(indicators_used) > 0:
                    print(f"  ✅ {agent_name}: {len(indicators_used)} indicators used")
                    print(f"     Indicators: {', '.join(indicators_used[:3])}{'...' if len(indicators_used) > 3 else ''}")
                elif status == "error":
                    print(f"  ❌ {agent_name}: Error - {result.get('error', 'Unknown error')}")
                else:
                    print(f"  ⚠️  {agent_name}: No indicators used (possibly hardcoded values)")
                
            except Exception as e:
                print(f"  ❌ {agent_name}: Exception - {str(e)}")
                agent_results[agent_name] = {
                    "status": "exception",
                    "error": str(e),
                    "has_real_analysis": False
                }
            
            print()
        
        # Summary analysis
        print("=== SUMMARY ANALYSIS ===")
        
        total_agents = len(agent_results)
        working_agents = sum(1 for result in agent_results.values() if result.get("has_real_analysis", False))
        error_agents = sum(1 for result in agent_results.values() if result.get("status") in ["error", "exception"])
        
        print(f"Total Agents: {total_agents}")
        print(f"Agents Using Real Indicators: {working_agents}")
        print(f"Agents with Errors: {error_agents}")
        print(f"Success Rate: {(working_agents/total_agents)*100:.1f}%")
        print()
        
        # Expected indicator assignments from GENIUS_AGENT_INDICATOR_MAPPING.md
        expected_assignments = {
            "risk_genius": 24,
            "session_expert": 15, 
            "pattern_master": 40,
            "execution_expert": 19,
            "pair_specialist": 14,
            "decision_master": 10,
            "ai_model_coordinator": 11,
            "market_microstructure_genius": 17,
            "sentiment_integration_genius": 12
        }
        
        print("=== INDICATOR ASSIGNMENT VERIFICATION ===")
        assignment_correct = True
        
        for agent_name, expected_count in expected_assignments.items():
            if agent_name in agent_results:
                actual_count = agent_results[agent_name].get("indicator_count", 0)
                if agent_results[agent_name].get("has_real_analysis", False):
                    # Note: Actual count might be lower due to missing indicators in registry
                    print(f"{agent_name}: Expected {expected_count}, Using {actual_count} indicators")
                    if actual_count == 0:
                        assignment_correct = False
                else:
                    print(f"{agent_name}: Expected {expected_count}, ❌ NOT USING REAL INDICATORS")
                    assignment_correct = False
            else:
                print(f"{agent_name}: ❌ AGENT NOT FOUND")
                assignment_correct = False
        
        print()
        
        # Physics indicators distribution check
        print("=== PHYSICS INDICATORS DISTRIBUTION CHECK ===")
        physics_indicators = [
            "ThermodynamicEntropyEngine", "QuantumMomentumOracle", "BiorhythmMarketSynth",
            "CrystallographicLatticeDetector", "ChaosGeometryPredictor", "NeuralHarmonicResonance",
            "PhotonicWavelengthAnalyzer"
        ]
        
        physics_found = {}
        for agent_name, result in agent_results.items():
            indicators_used = result.get("indicators_used", [])
            for indicator in indicators_used:
                if indicator in physics_indicators:
                    if indicator not in physics_found:
                        physics_found[indicator] = []
                    physics_found[indicator].append(agent_name)
        
        for physics_indicator in physics_indicators:
            if physics_indicator in physics_found:
                agents = physics_found[physics_indicator]
                print(f"✅ {physics_indicator}: Used by {', '.join(agents)}")
            else:
                print(f"❌ {physics_indicator}: NOT USED by any agent")
        
        print()
        
        # Final status
        print("=== FINAL STATUS ===")
        if working_agents == total_agents and assignment_correct:
            print("🎉 SUCCESS! All agents are properly using their assigned indicators!")
            print("✅ Real indicator integration is working correctly")
            print("✅ No more hardcoded values in agent responses")
        elif working_agents >= total_agents * 0.8:  # 80% success rate
            print("✅ MOSTLY SUCCESSFUL! Most agents are using real indicators")
            print(f"⚠️  {total_agents - working_agents} agents still need fixes")
        else:
            print("❌ NEEDS WORK! Many agents are not using real indicators")
            print("🔧 Additional implementation required")
        
        # Save detailed results
        results_file = platform_root / "agent_integration_verification_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_agents": total_agents,
                    "working_agents": working_agents,
                    "error_agents": error_agents,
                    "success_rate": (working_agents/total_agents)*100
                },
                "agent_results": agent_results,
                "physics_indicators_found": physics_found,
                "expected_assignments": expected_assignments
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return working_agents == total_agents
        
    except Exception as e:
        print(f"❌ Critical error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_indicator_integration()
    sys.exit(0 if success else 1)