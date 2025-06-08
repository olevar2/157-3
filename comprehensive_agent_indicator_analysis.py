"""
Comprehensive Agent-Indicator Analysis Tool
Analyzes the adaptive_indicator_bridge.py to determine:
1. Total number of indicators in registry
2. Indicators assigned to each agent (with names)
3. Category distribution
4. Compliance with 157-indicator recovery plan
"""

import re
import json
from typing import Dict, Set, List, Any
from collections import defaultdict, Counter

def parse_indicator_registry(file_path: str) -> Dict[str, Any]:
    """Parse the indicator registry from adaptive_indicator_bridge.py"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the registry method boundaries
    start_line = -1
    end_line = -1
    
    for i, line in enumerate(lines):
        if 'def _build_comprehensive_157_indicator_registry' in line:
            start_line = i
        elif start_line != -1 and 'def _build_comprehensive_agent_mapping' in line:
            end_line = i
            break
    
    if start_line == -1 or end_line == -1:
        print("ERROR: Could not find registry method boundaries")
        return {}
    
    # Extract registry content
    registry_lines = lines[start_line:end_line]
    registry_content = ''.join(registry_lines)
    
    # Parse indicators using regex
    indicators = {}
    
    # Match indicator definitions
    indicator_pattern = r"'([^']+)':\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}"
    
    matches = re.findall(indicator_pattern, registry_content, re.DOTALL)
    
    for indicator_name, indicator_content in matches:
        try:
            # Parse the indicator properties
            indicator_data = {}
            
            # Extract module
            module_match = re.search(r"'module':\s*'([^']+)'", indicator_content)
            if module_match:
                indicator_data['module'] = module_match.group(1)
            
            # Extract category
            category_match = re.search(r"'category':\s*'([^']+)'", indicator_content)
            if category_match:
                indicator_data['category'] = category_match.group(1)
            
            # Extract agents - this is more complex due to GeniusAgentType enum
            agents_match = re.search(r"'agents':\s*\[(.*?)\]", indicator_content, re.DOTALL)
            if agents_match:
                agents_content = agents_match.group(1)
                # Extract agent names from GeniusAgentType.AGENT_NAME
                agent_names = re.findall(r'GeniusAgentType\.([A-Z_]+)', agents_content)
                indicator_data['agents'] = agent_names
            else:
                indicator_data['agents'] = []
            
            # Extract priority
            priority_match = re.search(r"'priority':\s*(\d+)", indicator_content)
            if priority_match:
                indicator_data['priority'] = int(priority_match.group(1))
            
            # Extract class_name
            class_match = re.search(r"'class_name':\s*'([^']+)'", indicator_content)
            if class_match:
                indicator_data['class_name'] = class_match.group(1)
            
            indicators[indicator_name] = indicator_data
            
        except Exception as e:
            print(f"Error parsing indicator {indicator_name}: {e}")
            continue
    
    return indicators

def analyze_agent_indicators(indicators: Dict[str, Any]) -> Dict[str, Dict]:
    """Analyze indicators by agent"""
    
    agent_analysis = defaultdict(lambda: {
        'indicators': [],
        'count': 0,
        'categories': defaultdict(int),
        'priorities': defaultdict(int)
    })
    
    # Map agent type names to readable names
    agent_names = {
        'RISK_GENIUS': 'Risk Genius',
        'SESSION_EXPERT': 'Session Expert', 
        'PATTERN_MASTER': 'Pattern Master',
        'EXECUTION_EXPERT': 'Execution Expert',
        'PAIR_SPECIALIST': 'Pair Specialist',
        'DECISION_MASTER': 'Decision Master',
        'AI_MODEL_COORDINATOR': 'AI Model Coordinator',
        'MARKET_MICROSTRUCTURE_GENIUS': 'Market Microstructure Genius',
        'SENTIMENT_INTEGRATION_GENIUS': 'Sentiment Integration Genius'
    }
    
    for indicator_name, indicator_data in indicators.items():
        category = indicator_data.get('category', 'unknown')
        priority = indicator_data.get('priority', 0)
        agents = indicator_data.get('agents', [])
        
        for agent in agents:
            agent_display_name = agent_names.get(agent, agent)
            agent_analysis[agent_display_name]['indicators'].append(indicator_name)
            agent_analysis[agent_display_name]['count'] += 1
            agent_analysis[agent_display_name]['categories'][category] += 1
            agent_analysis[agent_display_name]['priorities'][priority] += 1
    
    return dict(agent_analysis)

def analyze_categories(indicators: Dict[str, Any]) -> Dict[str, int]:
    """Analyze indicator distribution by category"""
    
    category_counts = defaultdict(int)
    
    for indicator_name, indicator_data in indicators.items():
        category = indicator_data.get('category', 'unknown')
        category_counts[category] += 1
    
    return dict(category_counts)

def generate_compliance_report(indicators: Dict[str, Any], agent_analysis: Dict, category_counts: Dict) -> Dict:
    """Generate compliance report against recovery plan requirements"""
    
    # Recovery plan requirements
    plan_requirements = {
        'total_indicators': 157,
        'categories': {
            'momentum': 22,
            'pattern': 30, 
            'volume': 22,
            'fractal': 19,
            'fibonacci': 6,
            'statistical': 13,
            'trend': 8,
            'volatility': 7,
            'ml_advanced': 2,
            'elliott_wave': 3,
            'gann': 6
        },
        'agent_minimums': {
            'Decision Master': 157,  # Must have ALL indicators
            'Pair Specialist': 30,
            'Sentiment Integration Genius': 20
        }
    }
    
    compliance_report = {
        'total_indicators_found': len(indicators),
        'total_indicators_required': plan_requirements['total_indicators'],
        'total_compliance': len(indicators) == plan_requirements['total_indicators'],
        'category_compliance': {},
        'agent_compliance': {},
        'issues': []
    }
    
    # Check category compliance
    for category, required_count in plan_requirements['categories'].items():
        actual_count = category_counts.get(category, 0)
        compliance_report['category_compliance'][category] = {
            'required': required_count,
            'actual': actual_count,
            'compliant': actual_count == required_count,
            'difference': actual_count - required_count
        }
        
        if actual_count != required_count:
            compliance_report['issues'].append(
                f"Category '{category}': Required {required_count}, Found {actual_count} (Diff: {actual_count - required_count})"
            )
    
    # Check agent compliance
    for agent_name, required_count in plan_requirements['agent_minimums'].items():
        actual_count = agent_analysis.get(agent_name, {}).get('count', 0)
        compliance_report['agent_compliance'][agent_name] = {
            'required': required_count,
            'actual': actual_count,
            'compliant': actual_count >= required_count,
            'difference': actual_count - required_count
        }
        
        if actual_count < required_count:
            compliance_report['issues'].append(
                f"Agent '{agent_name}': Required {required_count}, Found {actual_count} (Deficit: {required_count - actual_count})"
            )
    
    return compliance_report

def main():
    """Main analysis function"""
    
    print("=" * 80)
    print("COMPREHENSIVE AGENT-INDICATOR ANALYSIS")
    print("=" * 80)
    
    file_path = "engines/ai_enhancement/adaptive_indicator_bridge.py"
    
    try:
        # Parse the registry
        print("\n🔍 Parsing indicator registry...")
        indicators = parse_indicator_registry(file_path)
        
        if not indicators:
            print("❌ Failed to parse indicators from registry")
            return
        
        print(f"✅ Successfully parsed {len(indicators)} indicators")
        
        # Analyze by agent
        print("\n📊 Analyzing indicators by agent...")
        agent_analysis = analyze_agent_indicators(indicators)
        
        # Analyze by category
        print("\n📈 Analyzing indicators by category...")
        category_counts = analyze_categories(indicators)
        
        # Generate compliance report
        print("\n📋 Generating compliance report...")
        compliance_report = generate_compliance_report(indicators, agent_analysis, category_counts)
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 TOTAL INDICATORS: {len(indicators)}")
        print(f"🎯 RECOVERY PLAN TARGET: 157")
        print(f"✅ COMPLIANCE: {'PASS' if compliance_report['total_compliance'] else 'FAIL'}")
        
        if not compliance_report['total_compliance']:
            diff = len(indicators) - 157
            print(f"📋 DIFFERENCE: {diff:+d} indicators")
        
        print("\n" + "-" * 50)
        print("AGENT INDICATOR ASSIGNMENTS")
        print("-" * 50)
        
        for agent_name, data in sorted(agent_analysis.items()):
            print(f"\n🤖 {agent_name}: {data['count']} indicators")
            
            # Show categories
            if data['categories']:
                print("   Categories:")
                for category, count in sorted(data['categories'].items()):
                    print(f"     - {category}: {count}")
            
            # Show first 10 indicators as sample
            if data['indicators']:
                print("   Sample indicators:")
                for i, indicator in enumerate(sorted(data['indicators'])):
                    if i < 10:
                        print(f"     - {indicator}")
                    elif i == 10:
                        print(f"     ... and {len(data['indicators']) - 10} more")
                        break
        
        print("\n" + "-" * 50)
        print("CATEGORY DISTRIBUTION")
        print("-" * 50)
        
        for category, count in sorted(category_counts.items()):
            required = compliance_report['category_compliance'].get(category, {}).get('required', 'N/A')
            status = "✅" if compliance_report['category_compliance'].get(category, {}).get('compliant', False) else "❌"
            print(f"{status} {category.upper()}: {count} (Required: {required})")
        
        print("\n" + "-" * 50)
        print("COMPLIANCE ISSUES")
        print("-" * 50)
        
        if compliance_report['issues']:
            for issue in compliance_report['issues']:
                print(f"❌ {issue}")
        else:
            print("✅ No compliance issues found!")
        
        # Save detailed report
        report_data = {
            'total_indicators': len(indicators),
            'indicators': indicators,
            'agent_analysis': agent_analysis,
            'category_counts': category_counts,
            'compliance_report': compliance_report,
            'timestamp': '2025-06-07'
        }
        
        with open('comprehensive_agent_indicator_analysis_results.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: comprehensive_agent_indicator_analysis_results.json")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
