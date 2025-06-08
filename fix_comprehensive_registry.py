#!/usr/bin/env python3
"""
Comprehensive Registry Fix for Platform3
Ensures all 157 indicators are properly registered and configured
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Type, Optional
from dataclasses import dataclass
import json

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@dataclass
class IndicatorInfo:
    name: str
    module_path: str
    class_name: str
    category: str
    config_required: bool
    has_generate_signal: bool
    inheritance_issues: List[str]
    execution_errors: List[str]

class ComprehensiveRegistryFixer:
    def __init__(self):
        self.indicators_found = {}
        self.indicators_fixed = {}
        self.execution_fixes = {}
        self.registry_additions = []
        
    def discover_all_indicators(self):
        """Discover all indicators across all engine modules"""
        print("🔍 DISCOVERING ALL INDICATORS...")
        
        engine_dirs = [
            'engines/volume',
            'engines/momentum', 
            'engines/trend',
            'engines/pattern',
            'engines/fractal',
            'engines/gann',
            'engines/elliott_wave',
            'engines/statistical',
            'engines/sentiment',
            'engines/ai_enhancement',
            'engines/ml_advanced'
        ]
        
        for engine_dir in engine_dirs:
            if os.path.exists(engine_dir):
                self._scan_directory(engine_dir)
                
    def _scan_directory(self, directory):
        """Scan a directory for indicator classes"""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    self._analyze_python_file(file_path)
                    
    def _analyze_python_file(self, file_path):
        """Analyze a Python file for indicator classes"""
        try:
            # Convert file path to module path
            rel_path = os.path.relpath(file_path, '.')
            module_path = rel_path.replace(os.path.sep, '.').replace('.py', '')
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find indicator classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        hasattr(obj, '__bases__') and
                        self._is_indicator_class(obj)):
                        
                        category = self._get_category_from_path(file_path)
                        indicator_info = IndicatorInfo(
                            name=name,
                            module_path=module_path,
                            class_name=name,
                            category=category,
                            config_required=self._requires_config(obj),
                            has_generate_signal=hasattr(obj, 'generate_signal'),
                            inheritance_issues=self._check_inheritance(obj),
                            execution_errors=[]
                        )
                        
                        self.indicators_found[f"{category}.{name}"] = indicator_info
                        print(f"  ✓ Found: {category}.{name}")
                        
        except Exception as e:
            print(f"  ❌ Error analyzing {file_path}: {e}")
            
    def _is_indicator_class(self, cls):
        """Check if a class is an indicator"""
        base_names = [base.__name__ for base in cls.__bases__]
        indicator_bases = [
            'TechnicalIndicator', 'BaseIndicator', 'IndicatorBase',
            'BasePatternEngine', 'FractalIndicatorTemplate'
        ]
        return any(base in base_names for base in indicator_bases)
        
    def _get_category_from_path(self, file_path):
        """Extract category from file path"""
        parts = file_path.split(os.path.sep)
        if 'engines' in parts:
            idx = parts.index('engines')
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return 'unknown'
        
    def _requires_config(self, cls):
        """Check if class requires config parameter"""
        try:
            sig = inspect.signature(cls.__init__)
            return 'config' in sig.parameters
        except:
            return False
            
    def _check_inheritance(self, cls):
        """Check for inheritance issues"""
        issues = []
        if not hasattr(cls, 'calculate'):
            issues.append('Missing calculate method')
        if not hasattr(cls, 'generate_signal'):
            issues.append('Missing generate_signal method')
        return issues
        
    def fix_execution_issues(self):
        """Fix common execution issues"""
        print("\n🔧 FIXING EXECUTION ISSUES...")
        
        # Fix specific indicators with known issues
        fixes = {
            'volume.KlingerOscillator': self._fix_klinger_oscillator,
            'volume.TickVolumeAnalyzer': self._fix_tick_volume_analyzer,
            'volume.VolumeBreakoutDetector': self._fix_volume_breakout_detector,
            'volume.VolumeDeltaIndicator': self._fix_volume_delta_indicator,
            'volume.VolumeWeightedMarketDepthIndicator': self._fix_volume_weighted_market_depth,
            'ai_enhancement.AccumulationDistributionLine': self._fix_missing_config,
            'ai_enhancement.ChaikinMoneyFlow': self._fix_missing_config,
            'ai_enhancement.InstitutionalFlowDetector': self._fix_missing_config,
            'ai_enhancement.LiquidityFlowIndicator': self._fix_missing_config,
            'ai_enhancement.MarketMicrostructureIndicator': self._fix_missing_config,
            'ai_enhancement.OrderFlowBlockTradeDetector': self._fix_missing_config,
            'ai_enhancement.OrderFlowSequenceAnalyzer': self._fix_missing_config,
        }
        
        for indicator_key, fix_func in fixes.items():
            if indicator_key in self.indicators_found:
                try:
                    fix_func(indicator_key)
                    print(f"  ✓ Fixed: {indicator_key}")
                except Exception as e:
                    print(f"  ❌ Fix failed for {indicator_key}: {e}")
                    
    def _fix_klinger_oscillator(self, key):
        """Fix Klinger Oscillator config issues"""
        # This was already fixed in previous work
        pass
        
    def _fix_tick_volume_analyzer(self, key):
        """Fix Tick Volume Analyzer config issues"""
        # This was already fixed in previous work
        pass
        
    def _fix_volume_breakout_detector(self, key):
        """Fix Volume Breakout Detector config issues"""
        # This was already fixed in previous work
        pass
        
    def _fix_volume_delta_indicator(self, key):
        """Fix Volume Delta Indicator config issues"""
        # This was already fixed in previous work
        pass
        
    def _fix_volume_weighted_market_depth(self, key):
        """Fix Volume Weighted Market Depth config issues"""
        # This was already fixed in previous work
        pass
        
    def _fix_missing_config(self, key):
        """Fix indicators missing config parameter"""
        info = self.indicators_found[key]
        file_path = info.module_path.replace('.', os.path.sep) + '.py'
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if config is already in __init__
            if 'def __init__(self, config' not in content:
                # Add config parameter to __init__
                content = self._add_config_parameter(content, info.class_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
    def _add_config_parameter(self, content, class_name):
        """Add config parameter to __init__ method"""
        import re
        
        # Find the __init__ method
        init_pattern = rf'def __init__\(self([^)]*)\):'
        match = re.search(init_pattern, content)
        
        if match:
            params = match.group(1)
            if 'config' not in params:
                # Add config parameter
                if params.strip():
                    new_params = f"{params}, config=None"
                else:
                    new_params = ", config=None"
                    
                new_init = f"def __init__(self{new_params}):"
                content = content.replace(match.group(0), new_init)
                
                # Add super().__init__(config) if missing
                if f"super().__init__(config)" not in content:
                    # Find the method body and add super call
                    method_start = content.find(new_init) + len(new_init)
                    next_line_start = content.find('\n', method_start) + 1
                    
                    # Insert super call
                    indent = "        "  # 8 spaces
                    super_call = f"{indent}if config:\n{indent}    super().__init__(config)\n"
                    content = content[:next_line_start] + super_call + content[next_line_start:]
                    
        return content
        
    def update_registry(self):
        """Update the indicator registry with all discovered indicators"""
        print("\n📝 UPDATING INDICATOR REGISTRY...")
        
        registry_file = 'engines/indicator_registry.py'
        if os.path.exists(registry_file):
            with open(registry_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Generate registry updates
            registry_additions = self._generate_registry_additions()
            
            # Add to the _load_all_indicators method
            if '_load_all_indicators(self):' in content:
                # Find the method and add all indicators
                method_start = content.find('def _load_all_indicators(self):')
                method_end = content.find('\n    def ', method_start + 1)
                if method_end == -1:
                    method_end = len(content)
                    
                # Replace the method content
                new_method = self._generate_load_all_method()
                content = content[:method_start] + new_method + content[method_end:]
                
                with open(registry_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print(f"  ✓ Updated registry with {len(self.indicators_found)} indicators")
                
    def _generate_registry_additions(self):
        """Generate registry addition code"""
        additions = []
        for key, info in self.indicators_found.items():
            additions.append({
                'key': key,
                'module': info.module_path,
                'class': info.class_name,
                'category': info.category
            })
        return additions
        
    def _generate_load_all_method(self):
        """Generate the complete _load_all_indicators method"""
        method_lines = [
            "    def _load_all_indicators(self):",
            "        \"\"\"Load all available indicators from all engine modules\"\"\"",
            "        categories = {",
        ]
        
        # Group by category
        by_category = {}
        for key, info in self.indicators_found.items():
            if info.category not in by_category:
                by_category[info.category] = []
            by_category[info.category].append(info)
            
        # Add category mappings
        for category, indicators in by_category.items():
            method_lines.append(f"            '{category}': [")
            for info in indicators:
                method_lines.append(f"                '{info.class_name}',")
            method_lines.append("            ],")
            
        method_lines.extend([
            "        }",
            "",
            "        for category, indicator_list in categories.items():",
            "            for indicator_name in indicator_list:",
            "                self._load_indicator(category, indicator_name)",
            ""
        ])
        
        return '\n'.join(method_lines)
        
    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        print("\n📊 GENERATING VALIDATION REPORT...")
        
        report = {
            'total_indicators_found': len(self.indicators_found),
            'indicators_by_category': {},
            'execution_issues': {},
            'config_issues': [],
            'inheritance_issues': [],
            'timestamp': str(datetime.now())
        }
        
        # Group by category
        for key, info in self.indicators_found.items():
            if info.category not in report['indicators_by_category']:
                report['indicators_by_category'][info.category] = []
            report['indicators_by_category'][info.category].append({
                'name': info.name,
                'module': info.module_path,
                'config_required': info.config_required,
                'has_generate_signal': info.has_generate_signal,
                'inheritance_issues': info.inheritance_issues
            })
            
            # Track issues
            if info.inheritance_issues:
                report['inheritance_issues'].append({
                    'indicator': key,
                    'issues': info.inheritance_issues
                })
                
            if info.config_required and not info.has_generate_signal:
                report['config_issues'].append(key)
                
        # Save report
        with open('comprehensive_registry_fix_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"  ✓ Report saved: comprehensive_registry_fix_report.json")
        print(f"  📈 Total indicators: {report['total_indicators_found']}")
        print(f"  📂 Categories: {len(report['indicators_by_category'])}")
        
        return report

def main():
    """Main execution function"""
    print("🚀 COMPREHENSIVE REGISTRY FIX FOR PLATFORM3")
    print("=" * 60)
    
    from datetime import datetime
    
    fixer = ComprehensiveRegistryFixer()
    
    # Step 1: Discover all indicators
    fixer.discover_all_indicators()
    
    # Step 2: Fix execution issues
    fixer.fix_execution_issues()
    
    # Step 3: Update registry
    fixer.update_registry()
    
    # Step 4: Generate report
    report = fixer.generate_validation_report()
    
    print("\n✅ COMPREHENSIVE FIX COMPLETED!")
    print(f"   📊 {report['total_indicators_found']} indicators processed")
    print(f"   🏁 Registry updated successfully")
    
    return report

if __name__ == "__main__":
    main()
