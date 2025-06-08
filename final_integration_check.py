"""
Final Platform3 Integration Verification
Simple, clean verification of all 115+ indicators and genius agents
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

class FinalIntegrationCheck:
    def __init__(self):
        self.base_path = "d:/MD/Platform3"
        self.indicators_found = {}
        self.agents_found = {}
        
    def check_file_exists(self, file_path: str) -> bool:
        """Check if a file exists"""
        full_path = os.path.join(self.base_path, file_path)
        return os.path.exists(full_path)
    
    def scan_indicator_files(self) -> Dict[str, List[str]]:
        """Scan for all indicator files"""
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
        
        agents_expected = [
            "risk_genius", "session_expert", "pattern_master",
            "execution_expert", "pair_specialist", "decision_master",
            "ai_model_coordinator", "market_microstructure_genius",
            "sentiment_integration_genius"
        ]
        
        return {
            'registry_exists': self.check_file_exists(registry_path),
            'expected_agents': agents_expected,
            'agents_count': len(agents_expected)
        }
    
    def count_total_indicators(self, indicators: Dict[str, List[str]]) -> int:
        """Count total indicators found"""
        total = 0
        for category, indicator_list in indicators.items():
            total += len(indicator_list)
        return total
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        print("🔍 Platform3 Final Integration Check")
        print("=" * 50)
        
        # Scan for indicators
        indicators = self.scan_indicator_files()
        total_indicators = self.count_total_indicators(indicators)
        
        # Check components
        bridge_status = self.check_adaptive_bridge()
        agents_status = self.check_genius_agents()
        
        # Print results
        print(f"\n📊 INDICATOR SUMMARY:")
        print(f"   Total Indicators Found: {total_indicators}")
        
        for category, indicator_list in indicators.items():
            if indicator_list:  # Only show categories with indicators
                print(f"   {category.title()}: {len(indicator_list)} indicators")
        
        print(f"\n🔧 INTEGRATION COMPONENTS:")
        print(f"   Adaptive Bridge: {'✅' if bridge_status['bridge_exists'] else '❌'}")
        print(f"   Coordinator: {'✅' if bridge_status['coordinator_exists'] else '❌'}")
        print(f"   AI Enhancement: {'✅' if bridge_status['init_exists'] else '❌'}")
        print(f"   Agent Registry: {'✅' if agents_status['registry_exists'] else '❌'}")
        
        print(f"\n🤖 GENIUS AGENTS:")
        print(f"   Expected Agents: {agents_status['agents_count']}")
        
        # Overall status
        all_components_ok = (
            bridge_status['bridge_exists'] and 
            bridge_status['coordinator_exists'] and
            bridge_status['init_exists'] and
            agents_status['registry_exists']
        )
        
        # Target check (115+ indicators)
        target_met = total_indicators >= 115
        
        print(f"\n🎯 FINAL STATUS:")
        print(f"   Indicators Target (115+): {'✅' if target_met else '❌'} ({total_indicators}/115+)")
        print(f"   Integration Components: {'✅' if all_components_ok else '❌'}")
        print(f"   Overall Status: {'✅ COMPLETE' if target_met and all_components_ok else '⚠️ NEEDS ATTENTION'}")
        
        # Create report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_indicators_found': total_indicators,
            'target_met': target_met,
            'indicators_by_category': {k: len(v) for k, v in indicators.items()},
            'integration_components': bridge_status,
            'genius_agents': agents_status,
            'overall_status': 'COMPLETE' if target_met and all_components_ok else 'INCOMPLETE',
            'detailed_indicators': indicators
        }
        
        return report

def main():
    """Main function"""
    checker = FinalIntegrationCheck()
    report = checker.generate_report()
    
    # Save report
    with open('final_integration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Report saved to: final_integration_report.json")
    
    return report

if __name__ == "__main__":
    main()
