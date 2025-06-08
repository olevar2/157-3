#!/usr/bin/env python3
"""
Diagnostic Gap Analysis Script
Identifies exactly which indicators are missing from validation pipeline
and categorizes the root causes preventing their inclusion in testing
"""

import sys
import os
import json
import importlib
import inspect
import traceback
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from datetime import datetime

# Add the platform root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class DiagnosticGapAnalyzer:
    def __init__(self):
        self.discovered_indicators = {}
        self.tested_indicators = set()
        self.missing_indicators = {}
        self.failure_categories = {
            'import_failures': [],
            'config_missing': [],
            'dependency_issues': [],
            'invalid_signature': [],
            'module_not_found': [],
            'class_not_indicator': [],
            'other_issues': []
        }
        
    def load_discovered_indicators(self) -> Dict[str, Any]:
        """Load indicators using dynamic indicator loader"""
        print("Loading discovered indicators...")
        try:
            from dynamic_indicator_loader import load_all_indicators
            indicators, categories = load_all_indicators()
            self.discovered_indicators = indicators
            print(f"✅ Loaded {len(indicators)} discovered indicators")
            return indicators
        except Exception as e:
            print(f"❌ Failed to load discovered indicators: {e}")
            return {}
    
    def load_discovery_json(self) -> Dict[str, Any]:
        """Load indicators from the complete discovery JSON file"""
        print("Loading indicators from discovery JSON...")
        try:
            json_file = "complete_157_indicator_discovery_20250607_034303.json"
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            json_indicators = {}
            for category, indicators in data.get('indicators_by_category', {}).items():
                for indicator in indicators:
                    name = indicator.get('name', '')
                    module_path = indicator.get('module_path', '')
                    if name and module_path:
                        key = f"{category}.{name.lower()}"
                        json_indicators[key] = {
                            'name': name,
                            'module_path': module_path,
                            'category': category,
                            'priority': indicator.get('priority', 0),
                            'agents': indicator.get('agents', [])
                        }
            
            print(f"✅ Loaded {len(json_indicators)} indicators from JSON")
            return json_indicators
            
        except Exception as e:
            print(f"❌ Failed to load discovery JSON: {e}")
            return {}
    
    def extract_tested_indicators(self) -> Set[str]:
        """Extract indicators that are actually tested by the validation system"""
        print("Analyzing tested indicators...")
        tested = set()
        
        try:
            # Run the validation test to see what indicators it attempts to load
            from comprehensive_validation_test_ultimate import smart_indicator_call
            
            # Try to import the indicator registry used by validation test
            try:
                from engines.indicator_registry import IndicatorRegistry
                registry = IndicatorRegistry()
                registry_indicators = registry.get_all_indicators()
                
                for indicator_name in registry_indicators:
                    # Normalize the name format to match discovered indicators
                    normalized_name = indicator_name.lower()
                    tested.add(normalized_name)
                    
                print(f"✅ Found {len(tested)} indicators in validation registry")
                
            except Exception as e:
                print(f"⚠️ Could not load indicator registry: {e}")
                # Fallback: try to extract from validation test code
                tested = self._extract_from_validation_code()
                
        except Exception as e:
            print(f"❌ Failed to extract tested indicators: {e}")
            
        self.tested_indicators = tested
        return tested
    
    def _extract_from_validation_code(self) -> Set[str]:
        """Fallback method to extract indicators from validation test code"""
        tested = set()
        try:
            with open('comprehensive_validation_test_ultimate.py', 'r') as f:
                content = f.read()
                
            # Look for indicator imports and references
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if 'import' in line and 'engines' in line:
                    # Extract potential indicator references
                    pass  # This would require more sophisticated parsing
                    
        except Exception as e:
            print(f"⚠️ Fallback extraction failed: {e}")
            
        return tested
    
    def categorize_missing_indicators(self) -> Dict[str, List[str]]:
        """Categorize missing indicators by failure type"""
        print("Categorizing missing indicators...")
        
        discovered_keys = set(self.discovered_indicators.keys())
        tested_keys = self.tested_indicators
        
        # Find missing indicators
        missing = discovered_keys - tested_keys
        
        print(f"Found {len(missing)} missing indicators out of {len(discovered_keys)} discovered")
        
        for indicator_key in missing:
            try:
                # Try to determine why this indicator is missing
                category, name = indicator_key.split('.', 1) if '.' in indicator_key else ('unknown', indicator_key)
                
                failure_reason = self._diagnose_indicator_failure(indicator_key, category, name)
                self.failure_categories[failure_reason].append(indicator_key)
                
            except Exception as e:
                self.failure_categories['other_issues'].append(indicator_key)
                print(f"⚠️ Could not categorize {indicator_key}: {e}")
        
        return self.failure_categories
    
    def _diagnose_indicator_failure(self, indicator_key: str, category: str, name: str) -> str:
        """Diagnose why a specific indicator is failing to load"""
        try:
            # Check if the indicator class exists in discovered indicators
            if indicator_key not in self.discovered_indicators:
                return 'module_not_found'
            
            indicator_class = self.discovered_indicators[indicator_key]
            
            # Check if it's actually an indicator class
            if not self._is_valid_indicator_class(indicator_class):
                return 'class_not_indicator'
            
            # Check if it has valid constructor signature
            if not self._has_valid_signature(indicator_class):
                return 'invalid_signature'
            
            # Check for import issues
            try:
                # Try to instantiate with dummy config
                test_config = self._create_test_config(name)
                indicator_class(test_config)
                # If we get here, it might be a registry/config issue
                return 'config_missing'
            except ImportError:
                return 'dependency_issues'
            except Exception:
                return 'config_missing'
                
        except Exception:
            return 'other_issues'
    
    def _is_valid_indicator_class(self, indicator_class) -> bool:
        """Check if class is a valid indicator"""
        try:
            if not inspect.isclass(indicator_class):
                return False
            
            # Check inheritance
            mro = indicator_class.__mro__
            for base in mro:
                if base.__name__ in ['IndicatorBase', 'TechnicalIndicator', 'BaseIndicator']:
                    return True
            
            return False
        except:
            return False
    
    def _has_valid_signature(self, indicator_class) -> bool:
        """Check if indicator has valid constructor signature"""
        try:
            sig = inspect.signature(indicator_class.__init__)
            params = list(sig.parameters.keys())
            # Should at least have 'self' and some config parameter
            return len(params) >= 2
        except:
            return False
    
    def _create_test_config(self, name: str):
        """Create a test configuration for an indicator"""
        from comprehensive_validation_test_ultimate import IndicatorConfig, IndicatorType, TimeFrame
        
        return IndicatorConfig(
            name=name,
            indicator_type=IndicatorType.TREND,
            timeframe=TimeFrame.D1,
            lookback_periods=14
        )
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        print("Generating detailed diagnostic report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_discovered': len(self.discovered_indicators),
                'total_tested': len(self.tested_indicators),
                'total_missing': len(self.discovered_indicators) - len(self.tested_indicators),
                'coverage_percentage': (len(self.tested_indicators) / len(self.discovered_indicators) * 100) if self.discovered_indicators else 0
            },
            'discovered_indicators': list(self.discovered_indicators.keys()),
            'tested_indicators': list(self.tested_indicators),
            'missing_indicators': list(set(self.discovered_indicators.keys()) - self.tested_indicators),
            'failure_categories': self.failure_categories,
            'category_breakdown': {}
        }
        
        # Breakdown by category
        for indicator_key in self.discovered_indicators:
            category = indicator_key.split('.')[0] if '.' in indicator_key else 'unknown'
            if category not in report['category_breakdown']:
                report['category_breakdown'][category] = {
                    'total': 0,
                    'tested': 0,
                    'missing': 0,
                    'missing_indicators': []
                }
            
            report['category_breakdown'][category]['total'] += 1
            
            if indicator_key in self.tested_indicators:
                report['category_breakdown'][category]['tested'] += 1
            else:
                report['category_breakdown'][category]['missing'] += 1
                report['category_breakdown'][category]['missing_indicators'].append(indicator_key)
        
        return report
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete diagnostic analysis"""
        print("=" * 60)
        print("DIAGNOSTIC GAP ANALYSIS")
        print("=" * 60)
        
        # Load discovered indicators
        self.load_discovered_indicators()
        
        # Load discovery JSON for additional context
        json_indicators = self.load_discovery_json()
        
        # Extract tested indicators
        self.extract_tested_indicators()
        
        # Categorize missing indicators
        self.categorize_missing_indicators()
        
        # Generate report
        report = self.generate_detailed_report()
        
        # Save report
        report_filename = f"diagnostic_gap_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved to: {report_filename}")
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print analysis summary"""
        summary = report['summary']
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Discovered Indicators: {summary['total_discovered']}")
        print(f"Total Tested Indicators: {summary['total_tested']}")
        print(f"Missing Indicators: {summary['total_missing']}")
        print(f"Coverage: {summary['coverage_percentage']:.1f}%")
        
        print("\n📋 FAILURE CATEGORIES:")
        for category, indicators in self.failure_categories.items():
            if indicators:
                print(f"  {category.replace('_', ' ').title()}: {len(indicators)}")
                for indicator in indicators[:3]:  # Show first 3
                    print(f"    - {indicator}")
                if len(indicators) > 3:
                    print(f"    ... and {len(indicators) - 3} more")
        
        print("\n📊 CATEGORY BREAKDOWN:")
        for category, data in report['category_breakdown'].items():
            coverage = (data['tested'] / data['total'] * 100) if data['total'] > 0 else 0
            print(f"  {category}: {data['tested']}/{data['total']} ({coverage:.1f}%)")

if __name__ == "__main__":
    analyzer = DiagnosticGapAnalyzer()
    report = analyzer.run_analysis()
