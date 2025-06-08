"""
Platform3 Indicator Implementation Truth Analysis
Cross-validates file counts vs functional class counts vs validation results
"""

import os
import json
import sys
import importlib.util
import traceback
from typing import Dict, List, Any, Set
from datetime import datetime

class TruthAnalyzer:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.engines_path = os.path.join(self.base_path, 'engines')
        
        # Results storage
        self.file_count_results = {}
        self.class_count_results = {}
        self.validation_results = {}
        self.adaptive_registry_analysis = {}
        
    def analyze_file_vs_class_discrepancy(self) -> Dict[str, Any]:
        """Analyze the discrepancy between file counts and functional class counts"""
        print("🔍 ANALYZING INDICATOR IMPLEMENTATION TRUTH")
        print("=" * 60)
        
        # 1. File-based count (like simple_integration_check.py)
        file_counts = self._count_indicator_files()
        
        # 2. Class-based count (like comprehensive_indicator_count.py) 
        class_counts = self._count_indicator_classes()
        
        # 3. Cross-validate with validation results
        validation_data = self._load_validation_results()
        
        # 4. Analyze adaptive layer expectations
        adaptive_analysis = self._analyze_adaptive_layer()
        
        # Generate comprehensive report
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_found': sum(file_counts.values()),
                'total_classes_found': sum(class_counts.values()),
                'implementation_gap': sum(file_counts.values()) - sum(class_counts.values()),
                'implementation_success_rate': (sum(class_counts.values()) / sum(file_counts.values())) * 100 if sum(file_counts.values()) > 0 else 0
            },
            'detailed_analysis': {
                'file_counts_by_category': file_counts,
                'class_counts_by_category': class_counts,
                'category_wise_gaps': self._calculate_category_gaps(file_counts, class_counts),
                'validation_results': validation_data,
                'adaptive_layer_analysis': adaptive_analysis
            },
            'root_cause_analysis': self._perform_root_cause_analysis(file_counts, class_counts),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _count_indicator_files(self) -> Dict[str, int]:
        """Count indicator files (like simple_integration_check.py does)"""
        print("📁 Counting indicator files...")
        
        categories = [
            'fractal', 'pattern', 'momentum', 'trend', 'volatility', 
            'volume', 'statistical', 'fibonacci', 'gann', 'elliott_wave', 'ml_advanced'
        ]
        
        file_counts = {}
        
        for category in categories:
            category_path = os.path.join(self.engines_path, category)
            count = 0
            
            if os.path.exists(category_path):
                for file in os.listdir(category_path):
                    if file.endswith('.py') and not file.startswith('__'):
                        count += 1
            
            file_counts[category] = count
            print(f"   {category}: {count} files")
        
        return file_counts
    
    def _count_indicator_classes(self) -> Dict[str, int]:
        """Count functional indicator classes (like comprehensive_indicator_count.py does)"""
        print("🔧 Counting functional indicator classes...")
        
        categories = [
            'fractal', 'pattern', 'momentum', 'trend', 'volatility', 
            'volume', 'statistical', 'fibonacci', 'gann', 'elliott_wave', 'ml_advanced'
        ]
        
        class_counts = {}
        
        for category in categories:
            category_path = os.path.join(self.engines_path, category)
            count = 0
            
            if os.path.exists(category_path):
                # Add category path to sys.path for imports
                if category_path not in sys.path:
                    sys.path.insert(0, category_path)
                    sys.path.insert(0, os.path.join(category_path, '..'))
                
                for file in os.listdir(category_path):
                    if file.endswith('.py') and not file.startswith('__'):
                        module_name = file[:-3]
                        try:
                            spec = importlib.util.spec_from_file_location(
                                module_name, 
                                os.path.join(category_path, file)
                            )
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                
                                # Look for indicator classes
                                for attr_name in dir(module):
                                    attr = getattr(module, attr_name)
                                    if (isinstance(attr, type) and 
                                        hasattr(attr, 'calculate') and
                                        attr_name not in ['BaseIndicator', 'object']):
                                        count += 1
                                        break
                        except Exception:
                            # File exists but class is not functional
                            pass
            
            class_counts[category] = count
            print(f"   {category}: {count} working classes")
        
        return class_counts
    
    def _calculate_category_gaps(self, file_counts: Dict[str, int], class_counts: Dict[str, int]) -> Dict[str, Dict]:
        """Calculate gaps between files and classes by category"""
        gaps = {}
        
        for category in file_counts:
            files = file_counts[category]
            classes = class_counts[category]
            gap = files - classes
            success_rate = (classes / files * 100) if files > 0 else 0
            
            gaps[category] = {
                'files': files,
                'working_classes': classes,
                'broken_files': gap,
                'success_rate_percent': success_rate,
                'status': 'GOOD' if success_rate >= 80 else 'ISSUES' if success_rate >= 50 else 'CRITICAL'
            }
        
        return gaps
    
    def _load_validation_results(self) -> Dict[str, Any]:
        """Load validation results for cross-reference"""
        validation_file = os.path.join(self.base_path, 'comprehensive_validation_results.json')
        
        try:
            with open(validation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'loading_success_rate': data.get('summary', {}).get('loading_success_rate', 0),
                    'execution_success_rate': data.get('summary', {}).get('execution_success_rate', 0),
                    'total_indicators_tested': data.get('summary', {}).get('total_indicators', 0),
                    'successful_loads': data.get('summary', {}).get('successful_loads', 0),
                    'successful_executions': data.get('summary', {}).get('successful_executions', 0)
                }
        except Exception as e:
            return {'error': f"Could not load validation results: {e}"}
    
    def _analyze_adaptive_layer(self) -> Dict[str, Any]:
        """Analyze what the adaptive layer expects vs reality"""
        analysis = {
            'adaptive_registry_entries': 8,  # From adaptive_indicators.py
            'coordinator_mapping_claims': '115+ indicators referenced in comments',
            'bridge_registry_status': 'Partially implemented with placeholders',
            'genius_agent_mappings': 9,
            'actual_functional_indicators': 67,  # From our comprehensive count
            'gap_analysis': 'Adaptive layer expects more indicators than are functionally available'
        }
        
        return analysis
    
    def _perform_root_cause_analysis(self, file_counts: Dict, class_counts: Dict) -> Dict[str, Any]:
        """Identify root causes of implementation gaps"""
        
        total_files = sum(file_counts.values())
        total_classes = sum(class_counts.values())
        
        return {
            'primary_issue': 'File presence != functional implementation',
            'gap_percentage': ((total_files - total_classes) / total_files * 100) if total_files > 0 else 0,
            'likely_causes': [
                'Import errors in indicator files',
                'Missing or incorrectly named classes',
                'Incomplete inheritance from BaseIndicator',
                'Syntax errors preventing module loading',
                'Missing calculate() method implementations',
                'Placeholder files without actual code'
            ],
            'impact_on_system': [
                'Integration tests pass based on file counts but functional tests fail',
                'Adaptive layer cannot access promised indicators',
                'Genius agents receive limited indicator data',
                'System appears complete but performs poorly',
                'User expectations (115+ indicators) not met in practice'
            ]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        return [
            'IMMEDIATE: Switch all integration tests from file-counting to class-counting',
            'URGENT: Audit and fix broken indicator files - prioritize high-usage categories',
            'MEDIUM: Update documentation to reflect actual (67) vs claimed (115+) indicator count',
            'MEDIUM: Enhance adaptive layer to gracefully handle missing indicators',
            'LONG-TERM: Implement continuous integration tests that validate functional classes',
            'LONG-TERM: Add automated indicator health monitoring to catch regressions'
        ]
    
    def generate_report(self) -> None:
        """Generate and save comprehensive truth analysis report"""
        analysis = self.analyze_file_vs_class_discrepancy()
        
        # Print summary to console
        print("\n📊 TRUTH ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Total Files Found: {analysis['summary']['total_files_found']}")
        print(f"Functional Classes: {analysis['summary']['total_classes_found']}")
        print(f"Implementation Gap: {analysis['summary']['implementation_gap']} files")
        print(f"Success Rate: {analysis['summary']['implementation_success_rate']:.1f}%")
        
        print("\n🔍 CATEGORY BREAKDOWN")
        print("=" * 40)
        for category, gap_info in analysis['detailed_analysis']['category_wise_gaps'].items():
            status_icon = "✅" if gap_info['status'] == 'GOOD' else "⚠️" if gap_info['status'] == 'ISSUES' else "❌"
            print(f"{status_icon} {category}: {gap_info['working_classes']}/{gap_info['files']} ({gap_info['success_rate_percent']:.1f}%)")
        
        print(f"\n💡 ROOT CAUSE")
        print("=" * 40)
        print(f"Primary Issue: {analysis['root_cause_analysis']['primary_issue']}")
        print(f"Gap: {analysis['root_cause_analysis']['gap_percentage']:.1f}% of files are non-functional")
        
        # Save detailed report
        report_filename = f"platform3_indicator_truth_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved: {report_filename}")
        
        return analysis

if __name__ == "__main__":
    analyzer = TruthAnalyzer()
    analyzer.generate_report()
