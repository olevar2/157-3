"""
Enhanced Agent-Indicator Verification
Validates that all genius agents are correctly using their assigned indicators from GENIUS_AGENT_INDICATOR_MAPPING.md
Uses Knowledge Graph MCP server for validation and visualization
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Add Platform3 to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the indicator mapping integrator
from indicator_mapping_integrator import parse_indicator_mapping

def verify_agent_indicator_integration():
    """
    Comprehensive verification of agent-indicator integration
    Uses Knowledge Graph for verification and reporting
    """
    print("🔍 ENHANCED AGENT-INDICATOR INTEGRATION VERIFICATION")
    print("=" * 60)
    
    try:
        # Import the genius agent integration
        from engines.ai_enhancement.genius_agent_integration import GeniusAgentIntegration
        
        # Initialize the integration system
        agent_integration = GeniusAgentIntegration()
        
        # Load expected indicator mappings directly from the markdown file
        markdown_path = os.path.join(os.path.dirname(__file__), "GENIUS_AGENT_INDICATOR_MAPPING.md")
        expected_mappings = parse_indicator_mapping(markdown_path)
        
        # Create test market data
        test_market_data = create_test_market_data()
        
        # Define agents to test
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
        
        # Verification results
        results = {
            "verification_time": datetime.now().isoformat(),
            "agents_verified": 0,
            "total_indicators_verified": 0,
            "correctly_implemented": 0,
            "missing_indicators": 0,
            "agent_results": {}
        }
        
        # Test each agent
        for agent_name in agents_to_test:
            print(f"\n🤖 Verifying {agent_name}...")
            
            # Get expected indicators for this agent
            expected_indicators = {}
            if agent_name in expected_mappings:
                expected_indicators = expected_mappings[agent_name]
                expected_count = sum(len(indicators) for indicators in expected_indicators.values())
                print(f"  📋 Expected: {expected_count} indicators across {len(expected_indicators)} categories")
            else:
                print(f"  ⚠️ No mapping found for {agent_name} in GENIUS_AGENT_INDICATOR_MAPPING.md")
                expected_count = 0
            
            # Initialize result data
            agent_result = {
                "expected_indicators": expected_count,
                "assigned_indicators": 0,
                "loaded_indicators": 0,
                "used_indicators": 0,
                "missing_indicators": [],
                "error": None,
                "status": "NOT_IMPLEMENTED"
            }
            
            # Check if agent exists in integration system
            if agent_name in agent_integration.genius_agents:
                agent = agent_integration.genius_agents[agent_name]
                
                # Test if agent has assigned indicators
                if hasattr(agent, 'assigned_indicators'):
                    assigned_count = sum(len(indicators) for indicators in agent.assigned_indicators.values())
                    agent_result["assigned_indicators"] = assigned_count
                    print(f"  ✅ Agent has {assigned_count} assigned indicators")
                    
                    # Compare with expected indicators
                    if expected_count > 0:
                        missing = []
                        for category, indicators in expected_indicators.items():
                            if category not in agent.assigned_indicators:
                                missing.extend([f"{category}:{ind}" for ind in indicators])
                            else:
                                for ind in indicators:
                                    if ind not in agent.assigned_indicators[category]:
                                        missing.append(f"{category}:{ind}")
                        
                        if missing:
                            agent_result["missing_indicators"] = missing
                            agent_result["status"] = "PARTIALLY_IMPLEMENTED"
                            print(f"  ⚠️ Missing {len(missing)} indicators")
                        else:
                            agent_result["status"] = "FULLY_ASSIGNED"
                            print(f"  ✅ All expected indicators assigned correctly")
                else:
                    print(f"  ❌ Agent has no 'assigned_indicators' attribute")
                    agent_result["error"] = "No assigned_indicators attribute"
                
                # Test if indicators are loaded
                if hasattr(agent, 'indicators'):
                    loaded_count = sum(len(indicators) for indicators in agent.indicators.values())
                    agent_result["loaded_indicators"] = loaded_count
                    print(f"  📦 {loaded_count} indicators loaded")
                    
                    if loaded_count > 0:
                        agent_result["status"] = "INDICATORS_LOADED"
                else:
                    print(f"  ❌ Agent has no 'indicators' attribute")
                
                # Test indicator usage in analysis
                try:
                    analysis_result = agent.execute_analysis(test_market_data, {})
                    
                    if "indicators_used" in analysis_result:
                        indicators_used = analysis_result["indicators_used"]
                        agent_result["used_indicators"] = len(indicators_used)
                        print(f"  🔍 Analysis used {len(indicators_used)} indicators")
                        
                        if len(indicators_used) > 0:
                            print(f"  🎯 REAL INDICATOR INTEGRATION CONFIRMED")
                            agent_result["status"] = "FULLY_IMPLEMENTED"
                        else:
                            print(f"  ⚠️ No indicators actually used in analysis")
                    else:
                        print(f"  ⚠️ No 'indicators_used' tracking in analysis result")
                        
                except Exception as e:
                    print(f"  ❌ Analysis execution failed: {e}")
                    agent_result["error"] = str(e)
            else:
                print(f"  ❌ Agent not found in integration system")
                agent_result["error"] = "Agent not implemented in GeniusAgentIntegration"
            
            # Add agent result to overall results
            results["agents_verified"] += 1
            if agent_result["status"] == "FULLY_IMPLEMENTED":
                results["correctly_implemented"] += 1
            results["total_indicators_verified"] += expected_count
            results["missing_indicators"] += len(agent_result["missing_indicators"])
            results["agent_results"][agent_name] = agent_result
        
        # Update Knowledge Graph with verification results
        update_knowledge_graph_with_verification(results)
        
        # Save results to file
        results_path = os.path.join(os.path.dirname(__file__), 
                                    f"agent_indicator_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ VERIFICATION COMPLETE")
        print(f"  📊 Agents verified: {results['agents_verified']}")
        print(f"  📈 Indicators verified: {results['total_indicators_verified']}")
        print(f"  ✅ Correctly implemented: {results['correctly_implemented']}/{results['agents_verified']}")
        print(f"  ⚠️ Missing indicators: {results['missing_indicators']}")
        print(f"  💾 Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return {"error": str(e)}

def create_test_market_data():
    """Create synthetic market data for testing"""
    # Generate random OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Convert to list of dictionaries for API compatibility
    return df.to_dict('records')

def update_knowledge_graph_with_verification(results: Dict[str, Any]):
    """
    Update Knowledge Graph with verification results
    
    Args:
        results: Verification results dictionary
    """
    try:
        # Try to import MCP functions
        try:
            from mcp_knowledge_gra_create_entities import mcp_knowledge_gra_create_entities
            from mcp_knowledge_gra_add_observations import mcp_knowledge_gra_add_observations
            
            # Create verification result entity
            verification_entity = {
                "name": f"Verification:AgentIndicator:{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "entityType": "VerificationResult",
                "observations": [
                    f"Agent-Indicator Verification on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"Agents verified: {results['agents_verified']}",
                    f"Correctly implemented: {results['correctly_implemented']}/{results['agents_verified']}",
                    f"Missing indicators: {results['missing_indicators']}"
                ]
            }
            
            # Create entity in Knowledge Graph
            mcp_knowledge_gra_create_entities(entities=[verification_entity])
            
            # Add observations to agent entities
            for agent_name, agent_result in results["agent_results"].items():
                display_name = ' '.join(word.capitalize() for word in agent_name.split('_'))
                
                status_message = f"Status: {agent_result['status']}"
                indicators_message = f"Assigned: {agent_result['assigned_indicators']}, " \
                                    f"Loaded: {agent_result['loaded_indicators']}, " \
                                    f"Used: {agent_result['used_indicators']}"
                
                missing_message = ""
                if agent_result["missing_indicators"]:
                    missing_message = f"Missing indicators: {', '.join(agent_result['missing_indicators'][:5])}"
                    if len(agent_result["missing_indicators"]) > 5:
                        missing_message += f" and {len(agent_result['missing_indicators']) - 5} more"
                
                # Add observations to agent entity
                mcp_knowledge_gra_add_observations(observations=[
                    {
                        "entityName": f"Agent:{display_name}",
                        "contents": [
                            f"Verification {datetime.now().strftime('%Y-%m-%d')}: {status_message}",
                            indicators_message,
                            missing_message
                        ]
                    }
                ])
            
            print("✅ Verification results added to Knowledge Graph")
        except ImportError:
            print("⚠️ Could not import MCP Knowledge Graph functions")
    except Exception as e:
        print(f"⚠️ Failed to update Knowledge Graph: {e}")

def main():
    """Run verification as standalone script"""
    verify_agent_indicator_integration()

if __name__ == "__main__":
    main()
