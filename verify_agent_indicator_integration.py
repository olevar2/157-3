"""
Platform3 Agent-Indicator Integration Verification Script
Tests that all genius agents actually use their assigned indicators from GENIUS_AGENT_INDICATOR_MAPPING.md
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add Platform3 to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_agent_indicator_integration():
    """Test that each agent actually uses its assigned indicators"""
    print("TESTING AGENT-INDICATOR INTEGRATION")
    print("=" * 50)
    
    try:
        # Import the genius agent integration
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        
        # Initialize the integration system
        agent_integration = GeniusAgentIntegration()
        
        # Create test market data
        test_market_data = create_test_market_data()
        
        # Test each agent
        agents_to_test = [
            "risk_genius",
            "session_expert", 
            "pattern_master",
            "execution_expert",
            "pair_specialist",
            "decision_master",
            "ai_model_coordinator",
            "market_microstructure_genius",
            "sentiment_integration_genius"
        ]
        
        results = {}
        
        for agent_name in agents_to_test:
            print(f"\n🤖 Testing {agent_name}...")
            
            if agent_name in agent_integration.genius_agents:
                agent = agent_integration.genius_agents[agent_name]
                
                # Test if agent has assigned indicators
                if hasattr(agent, 'assigned_indicators'):
                    print(f"  ✅ Agent has assigned_indicators attribute")
                    print(f"  📊 Indicator categories: {list(agent.assigned_indicators.keys())}")
                    
                    # Count total indicators
                    total_indicators = sum(len(indicators) for indicators in agent.assigned_indicators.values())
                    print(f"  🔢 Total assigned indicators: {total_indicators}")
                    
                    # Test if agent loaded indicators
                    if hasattr(agent, 'indicators'):
                        loaded_count = sum(len(indicators) for indicators in agent.indicators.values())
                        print(f"  📦 Loaded indicators: {loaded_count}")
                    else:
                        print(f"  ❌ Agent has no 'indicators' attribute")
                else:
                    print(f"  ❌ Agent has no 'assigned_indicators' attribute")
                
                # Test analysis execution
                try:
                    analysis_result = agent.execute_analysis(test_market_data, {})
                    
                    # Check if result contains real analysis vs hardcoded values
                    if "indicators_used" in analysis_result:
                        indicators_used = analysis_result["indicators_used"]
                        print(f"  ✅ Analysis used {len(indicators_used)} indicators: {indicators_used[:3]}...")
                        
                        if len(indicators_used) > 0:
                            print(f"  🎯 REAL INDICATOR INTEGRATION CONFIRMED")
                        else:
                            print(f"  ⚠️  No indicators actually used in analysis")
                    else:
                        print(f"  ❌ No 'indicators_used' tracking in analysis result")
                    
                    # Check for analysis quality
                    analysis_keys = [k for k in analysis_result.keys() if k not in ['agent', 'status', 'analysis_timestamp']]
                    print(f"  📈 Analysis components: {len(analysis_keys)}")
                    
                    results[agent_name] = {
                        "has_assigned_indicators": hasattr(agent, 'assigned_indicators'),
                        "total_assigned": total_indicators if hasattr(agent, 'assigned_indicators') else 0,
                        "indicators_used": len(analysis_result.get("indicators_used", [])),
                        "analysis_components": len(analysis_keys),
                        "status": "REAL" if analysis_result.get("indicators_used") else "FAKE"
                    }
                    
                except Exception as e:
                    print(f"  ❌ Analysis execution failed: {e}")
                    results[agent_name] = {"status": "ERROR", "error": str(e)}
            else:
                print(f"  ❌ Agent not found in genius_agents")
                results[agent_name] = {"status": "NOT_FOUND"}
        
        print("\n" + "=" * 50)
        print("📊 FINAL INTEGRATION REPORT")
        print("=" * 50)
        
        real_agents = 0
        fake_agents = 0
        
        for agent_name, result in results.items():
            status = result.get("status", "UNKNOWN")
            if status == "REAL":
                print(f"✅ {agent_name}: REAL INTEGRATION ({result.get('indicators_used', 0)} indicators used)")
                real_agents += 1
            elif status == "FAKE":
                print(f"❌ {agent_name}: FAKE INTEGRATION (no indicators used)")
                fake_agents += 1
            else:
                print(f"⚠️  {agent_name}: {status}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Real Integrations: {real_agents}/9")
        print(f"   Fake Integrations: {fake_agents}/9")
        print(f"   Success Rate: {(real_agents/9)*100:.1f}%")
        
        if real_agents == 9:
            print("\n🎉 SUCCESS: All agents use their assigned indicators!")
        else:
            print(f"\n🚨 FAILURE: {9-real_agents} agents still use fake hardcoded values!")
        
        return results
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        return {}

def create_test_market_data():
    """Create test market data for agent testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = []
    
    for i, date in enumerate(dates):
        data.append({
            'timestamp': date,
            'open': 100 + np.random.randn() * 2,
            'high': 102 + np.random.randn() * 2,
            'low': 98 + np.random.randn() * 2,
            'close': 100 + np.random.randn() * 2,
            'volume': np.random.randint(1000, 10000)
        })
    
    return data

def test_mapping_consistency():
    """Test that the mapping file and implementation are consistent"""
    print("\nTESTING MAPPING CONSISTENCY")
    print("=" * 50)
    
    # Expected indicator counts from GENIUS_AGENT_INDICATOR_MAPPING.md
    expected_counts = {
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
    
    try:
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        agent_integration = GeniusAgentIntegration()
        
        for agent_name, expected_count in expected_counts.items():
            if agent_name in agent_integration.genius_agents:
                agent = agent_integration.genius_agents[agent_name]
                
                if hasattr(agent, 'assigned_indicators'):
                    actual_count = sum(len(indicators) for indicators in agent.assigned_indicators.values())
                    status = "✅" if actual_count == expected_count else "❌"
                    print(f"{status} {agent_name}: {actual_count}/{expected_count} indicators")
                else:
                    print(f"❌ {agent_name}: No assigned_indicators attribute")
            else:
                print(f"❌ {agent_name}: Agent not found")
                
    except Exception as e:
        print(f"Mapping consistency test failed: {e}")

if __name__ == "__main__":
    print("PLATFORM3 AGENT-INDICATOR INTEGRATION VERIFICATION")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    # Test agent-indicator integration
    results = test_agent_indicator_integration()
    
    # Test mapping consistency
    test_mapping_consistency()
    
    print(f"\nTest completed at: {datetime.now()}")
    print("=" * 60)
