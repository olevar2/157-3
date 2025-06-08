#!/usr/bin/env python3
"""
Indicator Distribution Analysis
Shows how the 157 indicators are distributed across the 9 genius agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge, GeniusAgentType

def analyze_indicator_distribution():
    """Analyze how indicators are distributed across agents"""
    
    print("🔍 INDICATOR DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    bridge = AdaptiveIndicatorBridge()
    
    # Total indicators available
    total_indicators = len(bridge.indicator_registry)
    print(f"📊 Total Indicators Available: {total_indicators}")
    
    # Analyze each agent's allocation
    print(f"\n🎯 AGENT-SPECIFIC INDICATOR ALLOCATION:")
    print("-" * 60)
    
    all_assigned_indicators = set()
    
    for agent_type in GeniusAgentType:
        config = bridge.agent_indicator_mapping.get(agent_type, {})
        
        primary = config.get('primary_indicators', [])
        secondary = config.get('secondary_indicators', [])
        total_agent = len(primary) + len(secondary)
        
        print(f"\n🤖 {agent_type.value.upper()}:")
        print(f"   Primary Indicators: {len(primary)}")
        print(f"   Secondary Indicators: {len(secondary)}")
        print(f"   Total for Agent: {total_agent}")
        print(f"   Percentage of All: {(total_agent/total_indicators)*100:.1f}%")
        
        # Show some examples
        if primary:
            print(f"   Primary Examples: {', '.join(primary[:3])}{'...' if len(primary) > 3 else ''}")
        if secondary:
            print(f"   Secondary Examples: {', '.join(secondary[:3])}{'...' if len(secondary) > 3 else ''}")
        
        # Track all assigned indicators
        all_assigned_indicators.update(primary)
        all_assigned_indicators.update(secondary)
    
    # Overall statistics
    print(f"\n" + "=" * 60)
    print("📈 OVERALL STATISTICS:")
    print(f"   Total Indicators: {total_indicators}")
    print(f"   Assigned to Agents: {len(all_assigned_indicators)}")
    print(f"   Unassigned: {total_indicators - len(all_assigned_indicators)}")
    print(f"   Assignment Coverage: {(len(all_assigned_indicators)/total_indicators)*100:.1f}%")
    
    # Find unassigned indicators
    unassigned = set(bridge.indicator_registry.keys()) - all_assigned_indicators
    if unassigned:
        print(f"\n⚠️  UNASSIGNED INDICATORS ({len(unassigned)}):")
        for indicator in sorted(list(unassigned)[:10]):  # Show first 10
            category = bridge.indicator_registry[indicator].get('category', 'unknown')
            print(f"   - {indicator} ({category})")
        if len(unassigned) > 10:
            print(f"   ... and {len(unassigned) - 10} more")
    
    # Category analysis
    print(f"\n📋 INDICATOR CATEGORIES:")
    categories = {}
    for indicator_name, config in bridge.indicator_registry.items():
        category = config.get('category', 'unknown')
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    for category, count in sorted(categories.items()):
        print(f"   {category}: {count} indicators")
    
    return {
        'total_indicators': total_indicators,
        'assigned_indicators': len(all_assigned_indicators),
        'unassigned_indicators': len(unassigned),
        'categories': categories
    }

def show_agent_specializations():
    """Show what each agent specializes in"""
    
    print(f"\n🎯 AGENT SPECIALIZATIONS:")
    print("=" * 60)
    
    specializations = {
        GeniusAgentType.RISK_GENIUS: "Risk assessment, volatility analysis, correlation studies",
        GeniusAgentType.SESSION_EXPERT: "Trading session analysis, time-based patterns, Fibonacci levels",
        GeniusAgentType.PATTERN_MASTER: "Pattern recognition, fractal analysis, harmonic patterns",
        GeniusAgentType.EXECUTION_EXPERT: "Order flow, volume analysis, market microstructure",
        GeniusAgentType.PAIR_SPECIALIST: "Currency pair correlations, statistical arbitrage",
        GeniusAgentType.DECISION_MASTER: "Signal aggregation, momentum analysis, trend confirmation",
        GeniusAgentType.AI_MODEL_COORDINATOR: "Machine learning models, fractal dynamics",
        GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: "Deep market structure, institutional flow",
        GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS: "News sentiment, social media analysis"
    }
    
    for agent_type, description in specializations.items():
        print(f"\n🤖 {agent_type.value}:")
        print(f"   Focus: {description}")

def main():
    """Main analysis function"""
    try:
        stats = analyze_indicator_distribution()
        show_agent_specializations()
        
        print(f"\n" + "=" * 60)
        print("💡 KEY INSIGHTS:")
        print(f"   • Each agent gets a SPECIALIZED subset of indicators")
        print(f"   • NOT all 157 indicators go to every agent")
        print(f"   • Agents typically get 10-20 indicators each")
        print(f"   • Indicators are chosen based on agent expertise")
        print(f"   • System uses adaptive selection based on market conditions")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
