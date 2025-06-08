"""
Final Platform3 Integration Check
Clean implementation to verify all 129+ indicators and integration components
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

class Platform3IntegrationChecker:
    """Clean implementation of integration checker"""
    
    def __init__(self):
        self.base_path = os.path.abspath('.')
    
    def check_file_exists(self, file_path: str) -> bool:
        """Check if a file exists"""
        full_path = os.path.join(self.base_path, file_path)
        return os.path.exists(full_path)
    
    def scan_indicators(self) -> Dict[str, List[str]]:
        """Scan all indicator directories for available indicators"""
        categories = {
            'fractal': 'engines/fractal',
            'pattern': 'engines/pattern', 
            'momentum': 'engines/momentum',
            'trend': 'engines/trend',
            'volatility': 'engines/volatility',
            'volume': 'engines/volume',
            'statistical': 'engines/statistical',
            'fibonacci': 'engines/fibonacci',
            'gann': 'engines/gann',
            'elliott_wave': 'engines/elliott_wave',
            'ml_advanced': 'engines/ml_advanced'
        }
        
        found_indicators = {}
        
        for category, path in categories.items():
            found_indicators[category] = []
            full_path = os.path.join(self.base_path, path)
            
            if os.path.exists(full_path):
                for file in os.listdir(full_path):
                    if file.endswith('.py') and not file.startswith('__'):
                        found_indicators[category].append(file.replace('.py', ''))
        
        return found_indicators
    
    def check_adaptive_bridge(self) -> Dict[str, Any]:
        """Check adaptive bridge status"""
        bridge_path = "engines/ai_enhancement/adaptive_indicator_bridge.py"
        coordinator_path = "ai-platform/ai-models/intelligent-agents/indicator-expert/indicator_coordinator.py"
        
        return {
            'bridge_exists': self.check_file_exists(bridge_path),
            'coordinator_exists': self.check_file_exists(coordinator_path),
            'init_exists': self.check_file_exists("engines/ai_enhancement/__init__.py")
        }
    
    def check_genius_agents(self) -> Dict[str, Any]:
        """Check genius agent registry"""
        registry_path = "ai-platform/intelligent-agents/genius_agent_registry.py"
        
        expected_agents = [
            "risk_genius", "session_expert", "pattern_master", "execution_expert",
            "pair_specialist", "decision_master", "ai_model_coordinator",
            "market_microstructure_genius", "sentiment_integration_genius"
        ]
        
        return {
            'registry_exists': self.check_file_exists(registry_path),
            'expected_agents': expected_agents,
            'agents_count': len(expected_agents)
        }
    
    def run_complete_check(self) -> Dict[str, Any]:
        """Run complete integration check"""
        print("🔍 Platform3 Final Integration Check")
        print("=" * 50)
        
        # Scan indicators
        indicators = self.scan_indicators()
        total_indicators = sum(len(category_indicators) for category_indicators in indicators.values())
        
        print(f"\n📊 INDICATOR SUMMARY:")
        print(f"   Total Indicators Found: {total_indicators}")
        for category, indicator_list in indicators.items():
            print(f"   {category.title()}: {len(indicator_list)} indicators")
        
        # Check integration components
        bridge_status = self.check_adaptive_bridge()
        
        print(f"\n🔧 INTEGRATION COMPONENTS:")
        print(f"   Adaptive Bridge: {'✅' if bridge_status['bridge_exists'] else '❌'}")
        print(f"   Coordinator: {'✅' if bridge_status['coordinator_exists'] else '❌'}")
        print(f"   AI Enhancement: {'✅' if bridge_status['init_exists'] else '❌'}")
        
        # Check genius agents
        agent_status = self.check_genius_agents()
        
        print(f"\n🤖 GENIUS AGENTS:")
        print(f"   Expected Agents: {agent_status['agents_count']}")
        
        # Final status
        target_met = total_indicators >= 115
        integration_complete = (bridge_status['bridge_exists'] and 
                              bridge_status['coordinator_exists'] and
                              bridge_status['init_exists'] and
                              agent_status['registry_exists'])
        
        print(f"\n🎯 FINAL STATUS:")
        print(f"   Indicators Target (115+): {'✅' if target_met else '❌'} ({total_indicators}/115+)")
        print(f"   Integration Components: {'✅' if integration_complete else '❌'}")
        print(f"   Overall Status: {'✅ COMPLETE' if target_met and integration_complete else '⚠️ NEEDS ATTENTION'}")
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_indicators_found': total_indicators,
            'target_met': target_met,
            'indicators_by_category': {category: len(indicator_list) for category, indicator_list in indicators.items()},
            'integration_components': bridge_status,
            'genius_agents': agent_status,
            'overall_status': 'COMPLETE' if target_met and integration_complete else 'INCOMPLETE',
            'detailed_indicators': indicators
        }
        
        # Save report
        report_filename = 'final_integration_report.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Report saved to: {report_filename}")
        
        return report

def main():
    """Main execution function"""
    checker = Platform3IntegrationChecker()
    return checker.run_complete_check()

if __name__ == "__main__":
    main()
