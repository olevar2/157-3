#!/usr/bin/env python3
"""
Show specific indicators for each agent
Displays the exact indicator names and categories for each genius agent
"""

import asyncio
import sys
import os
from collections import defaultdict

# Add the parent directory to sys.path to import from engines
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge, GeniusAgentType

async def show_agent_specific_indicators():
    """Show specific indicators for each agent"""
    
    print("=" * 100)
    print("SPECIFIC INDICATORS FOR EACH GENIUS AGENT")
    print("=" * 100)
    
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
    
    agent_names = {
        GeniusAgentType.DECISION_MASTER: 'Decision Master',
        GeniusAgentType.PATTERN_MASTER: 'Pattern Master',
        GeniusAgentType.EXECUTION_EXPERT: 'Execution Expert',
        GeniusAgentType.RISK_GENIUS: 'Risk Genius',
        GeniusAgentType.PAIR_SPECIALIST: 'Pair Specialist',
        GeniusAgentType.SESSION_EXPERT: 'Session Expert',
        GeniusAgentType.AI_MODEL_COORDINATOR: 'AI Model Coordinator',
        GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: 'Market Microstructure Genius',
        GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS: 'Sentiment Integration Genius'
    }
    
    # Get indicator categories from registry
    def get_indicator_category(indicator_name):
        indicator_info = bridge.indicator_registry.get(indicator_name, {})
        return indicator_info.get('category', 'unknown')
    
    # Test each agent and get their specific indicators
    for agent_type in agent_names:
        agent_name = agent_names[agent_type]
        
        print(f"\n{'='*20} {agent_name.upper()} {'='*20}")
        
        try:
            # Get the agent's indicator configuration
            agent_config = bridge.agent_indicator_mapping.get(agent_type, {})
            
            if agent_type == GeniusAgentType.DECISION_MASTER:
                # Decision Master gets ALL indicators
                available_indicators = list(bridge.indicator_registry.keys())
                print(f"🎯 SPECIAL ACCESS: ALL {len(available_indicators)} INDICATORS")
            else:
                # Other agents get their mapped indicators
                primary = agent_config.get('primary_indicators', [])
                secondary = agent_config.get('secondary_indicators', [])
                fallback = agent_config.get('fallback_indicators', [])
                
                # Handle special cases and remove duplicates
                all_indicators = []
                for indicator_list in [primary, secondary, fallback]:
                    if isinstance(indicator_list, list):
                        all_indicators.extend(indicator_list)
                
                # Remove duplicates while preserving order
                seen = set()
                available_indicators = []
                for indicator in all_indicators:
                    if indicator not in seen and indicator in bridge.indicator_registry:
                        seen.add(indicator)
                        available_indicators.append(indicator)
            
            # Categorize indicators
            categories = defaultdict(list)
            for indicator in available_indicators:
                category = get_indicator_category(indicator)
                categories[category].append(indicator)
            
            # Display by category
            print(f"📊 TOTAL INDICATORS: {len(available_indicators)}")
            print(f"📋 CATEGORIES: {len(categories)}")
            
            for category, indicators in sorted(categories.items()):
                print(f"\n🔹 {category.upper()} ({len(indicators)} indicators):")
                
                # Show first 10 indicators, then indicate if there are more
                displayed_indicators = indicators[:10]
                for indicator in displayed_indicators:
                    print(f"   • {indicator}")
                
                if len(indicators) > 10:
                    print(f"   ... and {len(indicators) - 10} more {category} indicators")
            
            # Show adaptive features if available
            adaptive_features = agent_config.get('adaptive_features', [])
            if adaptive_features:
                print(f"\n🧠 ADAPTIVE FEATURES ({len(adaptive_features)}):")
                for feature in adaptive_features:
                    print(f"   • {feature}")
                    
        except Exception as e:
            print(f"❌ Error analyzing {agent_name}: {e}")
    
    print(f"\n{'='*100}")
    print("SUMMARY COMPARISON")
    print("=" * 100)
    
    # Create summary table
    summary_data = []
    
    for agent_type in agent_names:
        agent_name = agent_names[agent_type]
        
        try:
            agent_config = bridge.agent_indicator_mapping.get(agent_type, {})
            
            if agent_type == GeniusAgentType.DECISION_MASTER:
                indicator_count = len(bridge.indicator_registry)
                top_categories = "ALL CATEGORIES"
            else:
                # Count indicators for other agents
                primary = agent_config.get('primary_indicators', [])
                secondary = agent_config.get('secondary_indicators', [])
                fallback = agent_config.get('fallback_indicators', [])
                
                all_indicators = []
                for indicator_list in [primary, secondary, fallback]:
                    if isinstance(indicator_list, list):
                        all_indicators.extend(indicator_list)
                
                # Remove duplicates and filter existing indicators
                available_indicators = []
                seen = set()
                for indicator in all_indicators:
                    if indicator not in seen and indicator in bridge.indicator_registry:
                        seen.add(indicator)
                        available_indicators.append(indicator)
                
                indicator_count = len(available_indicators)
                
                # Get top 3 categories
                categories = defaultdict(int)
                for indicator in available_indicators:
                    category = get_indicator_category(indicator)
                    categories[category] += 1
                
                top_3 = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                top_categories = ", ".join([f"{cat}({count})" for cat, count in top_3])
            
            summary_data.append({
                'agent': agent_name,
                'count': indicator_count,
                'categories': top_categories
            })
            
        except Exception as e:
            summary_data.append({
                'agent': agent_name,
                'count': 0,
                'categories': 'ERROR'
            })
    
    # Sort by indicator count
    summary_data.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\n{'Agent':<30} {'Count':<8} {'Top Categories'}")
    print("-" * 80)
    
    for data in summary_data:
        print(f"{data['agent']:<30} {data['count']:<8} {data['categories']}")
    
    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    asyncio.run(show_agent_specific_indicators())
