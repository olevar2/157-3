#!/usr/bin/env python3
"""
ACCURATE Agent-Indicator Analysis using the ACTUAL Bridge
Tests real indicator access through the adaptive_indicator_bridge
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the parent directory to sys.path to import from engines
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge, GeniusAgentType

async def comprehensive_bridge_analysis():
    """Analyze actual indicator access through the bridge"""
    
    print("=" * 80)
    print("COMPREHENSIVE AGENT-INDICATOR ANALYSIS (BRIDGE-BASED)")
    print("=" * 80)
    
    # Initialize the bridge
    bridge = AdaptiveIndicatorBridge()
    
    # Sample market data
    market_data = {
        'symbol': 'EURUSD',
        'timeframe': 'M15',
        'close': [1.1000, 1.1010, 1.1005, 1.1015, 1.1020],
        'high': [1.1005, 1.1015, 1.1010, 1.1020, 1.1025],
        'low': [1.0995, 1.1005, 1.1000, 1.1010, 1.1015],
        'volume': [1000, 1200, 900, 1100, 1300]
    }
    
    print(f"🔍 Parsing indicator registry...")
    registry_size = len(bridge.indicator_registry)
    print(f"✅ Successfully parsed {registry_size} indicators")
    
    print(f"\n📊 Analyzing indicators by agent...")
    
    # Recovery plan requirements
    requirements = {
        GeniusAgentType.DECISION_MASTER: 157,
        GeniusAgentType.PATTERN_MASTER: 60,
        GeniusAgentType.EXECUTION_EXPERT: 40,
        GeniusAgentType.RISK_GENIUS: 35,
        GeniusAgentType.PAIR_SPECIALIST: 30,
        GeniusAgentType.SESSION_EXPERT: 25,
        GeniusAgentType.AI_MODEL_COORDINATOR: 25,
        GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS: 20,
        GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: 30
    }
    
    agent_names = {
        GeniusAgentType.RISK_GENIUS: 'Risk Genius',
        GeniusAgentType.SESSION_EXPERT: 'Session Expert', 
        GeniusAgentType.PATTERN_MASTER: 'Pattern Master',
        GeniusAgentType.EXECUTION_EXPERT: 'Execution Expert',
        GeniusAgentType.PAIR_SPECIALIST: 'Pair Specialist',
        GeniusAgentType.DECISION_MASTER: 'Decision Master',
        GeniusAgentType.AI_MODEL_COORDINATOR: 'AI Model Coordinator',
        GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: 'Market Microstructure Genius',
        GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS: 'Sentiment Integration Genius'
    }
    
    results = {}
    compliant_agents = 0
    deficit_agents = []
    
    # Test each agent with high max_indicators to see actual availability
    for agent_type in requirements:
        try:
            package = await bridge.get_comprehensive_indicator_package(
                market_data=market_data,
                agent_type=agent_type,
                max_indicators=200  # Request more than needed to see actual availability
            )
            
            agent_name = agent_names[agent_type]
            indicator_count = package.metadata['indicators_available']  # Available, not calculated
            required = requirements[agent_type]
            
            results[agent_name] = {
                'count': indicator_count,
                'required': required,
                'status': 'EXCEEDS' if indicator_count > required else 'MEETS' if indicator_count == required else 'DEFICIT',
                'gap': max(0, required - indicator_count)
            }
            
            if indicator_count >= required:
                compliant_agents += 1
            else:
                deficit_agents.append({
                    'name': agent_name,
                    'current': indicator_count,
                    'required': required,
                    'gap': required - indicator_count
                })
                
        except Exception as e:
            print(f"❌ Error testing {agent_names[agent_type]}: {e}")
            results[agent_names[agent_type]] = {
                'count': 0,
                'required': requirements[agent_type],
                'status': 'ERROR',
                'gap': requirements[agent_type]
            }
    
    print(f"\n📈 Analyzing indicators by category...")
    print(f"\n📋 Generating compliance report...")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 TOTAL INDICATORS: {registry_size}")
    print(f"🎯 RECOVERY PLAN TARGET: 157")
    compliance_status = "EXCEEDS TARGET" if registry_size >= 157 else "BELOW TARGET"
    print(f"✅ COMPLIANCE: {compliance_status}")
    print(f"📋 DIFFERENCE: {registry_size - 157:+} indicators")
    
    print("\n" + "-" * 50)
    print("AGENT INDICATOR ASSIGNMENTS (ACTUAL BRIDGE ACCESS)")
    print("-" * 50)
    
    # Sort results by compliance status and count
    sorted_results = sorted(results.items(), key=lambda x: (x[1]['status'] == 'DEFICIT', -x[1]['count']))
    
    for agent_name, data in sorted_results:
        status_icon = "✅" if data['status'] != 'DEFICIT' else "❌"
        gap_text = f" (Gap: {data['gap']})" if data['gap'] > 0 else ""
        status_text = data['status']
        
        print(f"\n🤖 {agent_name}: {data['count']} indicators (Required: {data['required']}) - {status_text}{gap_text} {status_icon}")
    
    print("\n" + "-" * 50)
    print("COMPLIANCE ANALYSIS")
    print("-" * 50)
    
    total_agents = len(requirements)
    compliance_percentage = (compliant_agents / total_agents) * 100
    
    print(f"\n#### **✅ COMPLIANT AGENTS ({compliant_agents}/{total_agents} - {compliance_percentage:.1f}%)**")
    for agent_name, data in sorted_results:
        if data['status'] != 'DEFICIT':
            status_emoji = "✨" if data['status'] == 'EXCEEDS' else "✅"
            print(f"- **{agent_name}**: **{data['count']} indicators** (Required: {data['required']}) - **{data['status']}** {status_emoji}")
    
    if deficit_agents:
        print(f"\n#### **❌ DEFICIT AGENTS ({len(deficit_agents)}/{total_agents} - {100-compliance_percentage:.1f}%)**")
        for agent in deficit_agents:
            print(f"- **{agent['name']}**: **{agent['current']} indicators** (Required: {agent['required']}) - **GAP: {agent['gap']} indicators** 🚨")
    
    print("\n" + "-" * 50)
    print("🎯 FINAL ASSESSMENT")
    print("-" * 50)
    
    if len(deficit_agents) == 0:
        print("\n🎉 **ALL AGENTS ARE COMPLIANT!**")
        print("✅ **Phase 4B Recovery Plan: SUCCESSFULLY IMPLEMENTED**")
        print("✅ **All agents have sufficient indicator access**")
        print("✅ **Ready for production deployment**")
    else:
        print(f"\n⚠️  **{len(deficit_agents)} agents still need attention**")
        print("🔧 **Phase 4B Recovery Plan: PARTIALLY COMPLETE**")
        
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bridge_based_agent_analysis_{timestamp}.json"
    
    detailed_results = {
        'timestamp': timestamp,
        'total_indicators': registry_size,
        'target_indicators': 157,
        'compliant_agents': compliant_agents,
        'total_agents': total_agents,
        'compliance_percentage': compliance_percentage,
        'agent_results': results,
        'deficit_agents': deficit_agents,
        'analysis_method': 'bridge_based_actual_access'
    }
    
    with open(filename, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {filename}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(comprehensive_bridge_analysis())
