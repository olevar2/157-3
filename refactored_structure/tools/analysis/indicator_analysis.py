"""
Comprehensive Indicator Analysis Tool
Consolidates multiple analysis scripts for indicator validation and testing
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorAnalyzer:
    """Comprehensive indicator analysis and validation"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.analysis_results = {}
        self.logger = logging.getLogger(__name__)
        
    def analyze_indicator_count(self) -> Dict[str, Any]:
        """Analyze total indicator count"""
        try:
            # Look for indicator files
            indicator_files = []
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if 'indicator' in file.lower() and file.endswith('.py'):
                        indicator_files.append(os.path.join(root, file))
            
            return {
                "indicator_files_found": len(indicator_files),
                "files": indicator_files,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Count analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_registry_status(self) -> Dict[str, Any]:
        """Analyze registry status and completeness"""
        try:
            registry_files = []
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if 'registry' in file.lower():
                        registry_files.append(os.path.join(root, file))
            
            return {
                "registry_files_found": len(registry_files),
                "files": registry_files,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Registry analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_agent_integration(self) -> Dict[str, Any]:
        """Analyze agent integration status"""
        try:
            agent_files = []
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if 'agent' in file.lower() and file.endswith('.py'):
                        agent_files.append(os.path.join(root, file))
            
            return {
                "agent_files_found": len(agent_files),
                "files": agent_files,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Agent analysis failed: {e}")
            return {"error": str(e)}
    
    def find_duplicate_indicators(self) -> Dict[str, Any]:
        """Find duplicate indicator implementations"""
        try:
            duplicates = []
            indicator_names = set()
            
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Look for class definitions
                                lines = content.split('\n')
                                for line in lines:
                                    if line.strip().startswith('class ') and 'Indicator' in line:
                                        class_name = line.split('class ')[1].split('(')[0].strip()
                                        if class_name in indicator_names:
                                            duplicates.append({
                                                "class_name": class_name,
                                                "file": file_path
                                            })
                                        else:
                                            indicator_names.add(class_name)
                        except Exception:
                            continue
            
            return {
                "duplicates_found": len(duplicates),
                "duplicates": duplicates,
                "unique_indicators": len(indicator_names),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Duplicate analysis failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analysis methods"""
        self.logger.info("Starting comprehensive indicator analysis")
        
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "base_path": self.base_path,
            "indicator_count": self.analyze_indicator_count(),
            "registry_status": self.analyze_registry_status(),
            "agent_integration": self.analyze_agent_integration(),
            "duplicate_analysis": self.find_duplicate_indicators()
        }
        
        self.analysis_results = results
        return results
    
    def save_analysis_report(self, output_file: str = None) -> str:
        """Save analysis results to file"""
        if not self.analysis_results:
            self.run_comprehensive_analysis()
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_analysis_report_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            
            self.logger.info(f"Analysis report saved to: {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return ""


def main():
    """Main execution function"""
    analyzer = IndicatorAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print("=== COMPREHENSIVE INDICATOR ANALYSIS ===")
    print(f"Analysis completed: {results['analysis_timestamp']}")
    print(f"Base path: {results['base_path']}")
    print()
    
    # Print summary
    if 'indicator_count' in results:
        print(f"Indicator files found: {results['indicator_count'].get('indicator_files_found', 0)}")
    
    if 'registry_status' in results:
        print(f"Registry files found: {results['registry_status'].get('registry_files_found', 0)}")
    
    if 'agent_integration' in results:
        print(f"Agent files found: {results['agent_integration'].get('agent_files_found', 0)}")
    
    if 'duplicate_analysis' in results:
        print(f"Duplicate indicators: {results['duplicate_analysis'].get('duplicates_found', 0)}")
        print(f"Unique indicators: {results['duplicate_analysis'].get('unique_indicators', 0)}")
    
    # Save report
    report_file = analyzer.save_analysis_report()
    if report_file:
        print(f"\\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()