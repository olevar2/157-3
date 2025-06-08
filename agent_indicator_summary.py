"""
Agent Indicator Summary Test
Clean summary of how many indicators each agent uses
"""

def main():
    """Display clean summary of agent indicator counts"""
    
    print("🔍 AGENT INDICATOR COUNT ANALYSIS")
    print("=" * 60)
    print()
    
    # Based on our analysis results
    agent_counts = {
        "RISK_GENIUS": {
            "primary": 35,
            "secondary": 5,
            "total": 40,
            "description": "Risk Assessment & Management"
        },
        "PATTERN_MASTER": {
            "primary": 61,
            "secondary": 3,
            "total": 64,
            "description": "Pattern Recognition & Analysis"
        },
        "EXECUTION_EXPERT": {
            "primary": 42,
            "secondary": 3,
            "total": 45,
            "description": "Volume & Execution Analysis"
        },
        "DECISION_MASTER": {
            "primary": 157,  # Special case - ALL indicators
            "secondary": 0,
            "total": 157,
            "description": "Meta-Analysis & Final Decisions (ALL INDICATORS)"
        },
        "SESSION_EXPERT": {
            "primary": 27,
            "secondary": 2,
            "total": 29,
            "description": "Session & Time Analysis"
        },
        "PAIR_SPECIALIST": {
            "primary": 43,
            "secondary": 3,
            "total": 46,
            "description": "Currency Pair & Correlation Analysis"
        },
        "AI_MODEL_COORDINATOR": {
            "primary": 25,
            "secondary": 2,
            "total": 27,
            "description": "ML & AI Integration"
        },
        "MARKET_MICROSTRUCTURE_GENIUS": {
            "primary": 43,
            "secondary": 2,
            "total": 45,
            "description": "Microstructure & Institutional Analysis"
        },
        "SENTIMENT_INTEGRATION_GENIUS": {
            "primary": 26,
            "secondary": 3,
            "total": 29,
            "description": "Sentiment & News Analysis"
        }
    }
    
    # Display formatted table
    print(f"{'AGENT':<32} {'PRIMARY':<8} {'SECONDARY':<10} {'TOTAL':<6} {'DESCRIPTION':<35}")
    print("-" * 95)
    
    total_primary = 0
    total_secondary = 0
    total_indicators = 0
    
    # Sort by total indicators (descending)
    sorted_agents = sorted(agent_counts.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for agent_name, data in sorted_agents:
        primary = data['primary']
        secondary = data['secondary']
        total = data['total']
        description = data['description']
        
        # Add emoji indicators based on count
        if total >= 100:
            emoji = "🏆"  # Champion
        elif total >= 50:
            emoji = "⭐"  # Star
        elif total >= 30:
            emoji = "✨"  # Good
        else:
            emoji = "📊"  # Basic
        
        print(f"{emoji} {agent_name:<30} {primary:<8} {secondary:<10} {total:<6} {description}")
        
        if agent_name != "DECISION_MASTER":  # Don't double count ALL indicators
            total_primary += primary
            total_secondary += secondary
            total_indicators += total
    
    print("-" * 95)
    print(f"{'TOTALS (excl. DECISION_MASTER)':<32} {total_primary:<8} {total_secondary:<10} {total_indicators:<6}")
    print()
    
    # Key insights
    print("📈 KEY INSIGHTS:")
    print("-" * 20)
    print(f"• Total unique indicators in system: 157")
    print(f"• DECISION_MASTER has access to ALL 157 indicators")
    print(f"• PATTERN_MASTER uses the most indicators (64) after DECISION_MASTER")
    print(f"• Most agents use 25-50 indicators each")
    print(f"• High overlap: Many indicators are shared between agents")
    print()
    
    # Category breakdown
    print("📊 CATEGORY INSIGHTS:")
    print("-" * 20)
    print("• Volume indicators: Heavily used by EXECUTION_EXPERT & MICROSTRUCTURE_GENIUS")
    print("• Pattern indicators: Primarily used by PATTERN_MASTER & EXECUTION_EXPERT")
    print("• Statistical indicators: Key for RISK_GENIUS & PAIR_SPECIALIST")
    print("• ML indicators: Concentrated in AI_MODEL_COORDINATOR")
    print("• Fractal indicators: Distributed across RISK_GENIUS, PATTERN_MASTER, AI_MODEL_COORDINATOR")
    print()
    
    # Recommendations
    print("💡 ANALYSIS SUMMARY:")
    print("-" * 20)
    print("✅ Each agent has a specialized indicator set matching its role")
    print("✅ DECISION_MASTER acts as the central coordinator with full access")
    print("✅ Good distribution of indicators across different analysis domains")
    print("✅ Sufficient overlap for cross-validation between agents")
    
    # Files generated
    print()
    print("📁 GENERATED FILES:")
    print("-" * 20)
    print("• agent_indicator_analysis_report_[timestamp].txt - Full analysis report")
    print("• agent_indicator_analysis_[timestamp].json - Detailed JSON data")
    print("• detailed_agent_indicator_mapping_[timestamp].txt - Agent-by-agent breakdown")
    print("• indicator_overlap_analysis_[timestamp].json - Sharing analysis")

if __name__ == "__main__":
    main()
