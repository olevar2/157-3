"""
Detailed Agent Indicator Mapping Test
Shows exact indicators used by each agent with category breakdown
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetailedAgentIndicatorMapper:
    """Create detailed mapping of agents to their specific indicators"""
    
    def __init__(self):
        self.agent_types = [
            "RISK_GENIUS",
            "PATTERN_MASTER", 
            "EXECUTION_EXPERT",
            "DECISION_MASTER",
            "SESSION_EXPERT",
            "PAIR_SPECIALIST",
            "AI_MODEL_COORDINATOR",
            "MARKET_MICROSTRUCTURE_GENIUS",
            "SENTIMENT_INTEGRATION_GENIUS"
        ]
    
    def parse_agent_indicators_detailed(self, file_path: str) -> Dict[str, Any]:
        """Parse the file to extract detailed agent-indicator mappings"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {}
        
        # Find the comprehensive agent mapping section
        mapping_start = content.find("def _build_comprehensive_agent_mapping")
        if mapping_start == -1:
            logger.error("Could not find agent mapping section")
            return {}
        
        mapping_section = content[mapping_start:]
        mapping_end = mapping_section.find("\n    def ") + mapping_start
        if mapping_end > mapping_start:
            mapping_section = content[mapping_start:mapping_end]
        
        # Parse each agent's indicators
        agents_data = {}
        
        for agent_type in self.agent_types:
            agent_data = self._extract_agent_indicators(mapping_section, agent_type)
            if agent_data:
                agents_data[agent_type] = agent_data
        
        return agents_data
    
    def _extract_agent_indicators(self, content: str, agent_type: str) -> Dict[str, Any]:
        """Extract indicators for a specific agent type"""
        
        # Find the agent section
        pattern = f"GeniusAgentType\\.{agent_type}.*?'adaptive_features'"
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            logger.warning(f"Could not find section for {agent_type}")
            return {}
        
        agent_section = match.group(0)
        
        # Extract primary indicators
        primary_indicators = self._extract_indicator_list(agent_section, "primary_indicators")
        
        # Extract secondary indicators  
        secondary_indicators = self._extract_indicator_list(agent_section, "secondary_indicators")
        
        # Extract adaptive features
        adaptive_features = self._extract_indicator_list(agent_section, "adaptive_features")
        
        return {
            'primary_indicators': primary_indicators,
            'secondary_indicators': secondary_indicators,
            'adaptive_features': adaptive_features,
            'total_indicators': len(primary_indicators) + len(secondary_indicators),
            'primary_count': len(primary_indicators),
            'secondary_count': len(secondary_indicators)
        }
    
    def _extract_indicator_list(self, section: str, list_name: str) -> List[str]:
        """Extract a list of indicators from a section"""
        
        # Find the list
        pattern = f"'{list_name}':\\s*\\[([^\\]]*)"
        match = re.search(pattern, section, re.DOTALL)
        
        if not match:
            return []
        
        list_content = match.group(1)
        
        # Handle special cases
        if 'ALL_INDICATORS' in list_content:
            return ['ALL_INDICATORS']
        
        # Extract quoted strings
        indicators = re.findall(r"'([^']*)'", list_content)
        
        # Filter out comments and empty strings
        indicators = [ind for ind in indicators if ind and not ind.startswith('#')]
        
        return indicators
    
    def generate_detailed_report(self, agents_data: Dict[str, Any]) -> str:
        """Generate a detailed report showing each agent's indicators"""
        
        report_lines = [
            "=" * 100,
            "DETAILED AGENT INDICATOR MAPPING REPORT",
            "=" * 100,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Summary table
        report_lines.extend([
            "SUMMARY TABLE",
            "-" * 50,
            f"{'Agent':<35} {'Primary':<10} {'Secondary':<12} {'Total':<8}",
            "-" * 65
        ])
        
        total_indicators = 0
        for agent, data in agents_data.items():
            primary_count = data['primary_count']
            secondary_count = data['secondary_count']
            total_count = data['total_indicators']
            total_indicators += total_count
            
            report_lines.append(
                f"{agent:<35} {primary_count:<10} {secondary_count:<12} {total_count:<8}"
            )
        
        report_lines.extend([
            "-" * 65,
            f"{'TOTAL':<35} {'':<10} {'':<12} {total_indicators:<8}",
            "",
            ""
        ])
        
        # Detailed breakdown for each agent
        for agent, data in agents_data.items():
            report_lines.extend([
                f"{agent}",
                "=" * len(agent),
                f"Total Indicators: {data['total_indicators']}",
                f"Primary: {data['primary_count']}, Secondary: {data['secondary_count']}",
                ""
            ])
            
            if data['primary_indicators']:
                report_lines.extend([
                    "PRIMARY INDICATORS:",
                    "-" * 20
                ])
                
                # Group by first part of name for better organization
                grouped_indicators = defaultdict(list)
                for indicator in data['primary_indicators']:
                    if indicator == 'ALL_INDICATORS':
                        grouped_indicators['SPECIAL'].append(indicator)
                    else:
                        prefix = indicator.split('_')[0] if '_' in indicator else 'other'
                        grouped_indicators[prefix].append(indicator)
                
                for group, indicators in sorted(grouped_indicators.items()):
                    if len(grouped_indicators) > 1:
                        report_lines.append(f"  {group.upper()}:")
                    for indicator in sorted(indicators):
                        report_lines.append(f"    - {indicator}")
                    report_lines.append("")
            
            if data['secondary_indicators']:
                report_lines.extend([
                    "SECONDARY INDICATORS:",
                    "-" * 22
                ])
                for indicator in sorted(data['secondary_indicators']):
                    report_lines.append(f"  - {indicator}")
                report_lines.append("")
            
            if data['adaptive_features']:
                report_lines.extend([
                    "ADAPTIVE FEATURES:",
                    "-" * 18
                ])
                for feature in sorted(data['adaptive_features']):
                    report_lines.append(f"  - {feature}")
                report_lines.append("")
            
            report_lines.extend(["-" * 80, ""])
        
        return "\n".join(report_lines)
    
    def analyze_indicator_overlap(self, agents_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which indicators are shared between agents"""
        
        # Collect all primary indicators by agent
        agent_indicators = {}
        all_indicators = set()
        
        for agent, data in agents_data.items():
            indicators = set(data['primary_indicators'])
            if 'ALL_INDICATORS' in indicators:
                # Handle special case - we'll assume this means all available indicators
                continue
            agent_indicators[agent] = indicators
            all_indicators.update(indicators)
        
        # Find overlaps
        overlap_analysis = {}
        
        for indicator in all_indicators:
            using_agents = [agent for agent, indicators in agent_indicators.items() 
                          if indicator in indicators]
            if len(using_agents) > 1:
                overlap_analysis[indicator] = using_agents
        
        # Sort by number of agents using each indicator
        sorted_overlaps = sorted(overlap_analysis.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        
        return {
            'total_shared_indicators': len(overlap_analysis),
            'shared_indicators': dict(sorted_overlaps),
            'most_shared': sorted_overlaps[0] if sorted_overlaps else None,
            'sharing_stats': {
                '2_agents': sum(1 for agents in overlap_analysis.values() if len(agents) == 2),
                '3_agents': sum(1 for agents in overlap_analysis.values() if len(agents) == 3),
                '4_plus_agents': sum(1 for agents in overlap_analysis.values() if len(agents) >= 4)
            }
        }

def main():
    """Main function to run the detailed analysis"""
    
    print("Detailed Agent Indicator Mapping Analysis")
    print("=" * 50)
    
    file_path = "d:/MD/Platform3/engines/ai_enhancement/adaptive_indicator_bridge.py"
    
    mapper = DetailedAgentIndicatorMapper()
    
    # Parse the file
    print("Parsing agent indicator mappings...")
    agents_data = mapper.parse_agent_indicators_detailed(file_path)
    
    if not agents_data:
        print("Failed to parse agent data")
        return
    
    print(f"Successfully parsed data for {len(agents_data)} agents")
    
    # Generate detailed report
    report = mapper.generate_detailed_report(agents_data)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"detailed_agent_indicator_mapping_{timestamp}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Detailed report saved to: {report_filename}")
    
    # Show summary
    print("\nSUMMARY:")
    print("-" * 20)
    
    for agent, data in agents_data.items():
        print(f"{agent}: {data['total_indicators']} total "
              f"({data['primary_count']} primary + {data['secondary_count']} secondary)")
    
    # Analyze overlaps
    overlap_analysis = mapper.analyze_indicator_overlap(agents_data)
    
    print(f"\nINDICATOR SHARING:")
    print("-" * 20)
    print(f"Shared indicators: {overlap_analysis['total_shared_indicators']}")
    print(f"Used by 2 agents: {overlap_analysis['sharing_stats']['2_agents']}")
    print(f"Used by 3 agents: {overlap_analysis['sharing_stats']['3_agents']}")
    print(f"Used by 4+ agents: {overlap_analysis['sharing_stats']['4_plus_agents']}")
    
    if overlap_analysis['most_shared']:
        most_shared = overlap_analysis['most_shared']
        print(f"Most shared: {most_shared[0]} (used by {len(most_shared[1])} agents)")
    
    # Save overlap analysis
    overlap_filename = f"indicator_overlap_analysis_{timestamp}.json"
    with open(overlap_filename, 'w', encoding='utf-8') as f:
        json.dump(overlap_analysis, f, indent=2)
    
    print(f"Overlap analysis saved to: {overlap_filename}")

if __name__ == "__main__":
    main()
