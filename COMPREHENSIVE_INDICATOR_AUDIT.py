#!/usr/bin/env python3
"""
COMPREHENSIVE INDICATOR AUDIT
============================

This script will determine the TRUTH about Platform3's indicators:
1. What indicators actually exist as working Python code
2. What indicators are just documentation/placeholders
3. What indicators can be imported and instantiated
4. What indicators can process real market data

Author: Platform3 Truth Squad
"""

import os
import sys
import importlib
import inspect
import traceback
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd

class IndicatorAuditor:
    def __init__(self):
        self.results = {
            'audit_timestamp': datetime.now().isoformat(),
            'total_files_found': 0,
            'total_classes_found': 0,
            'working_indicators': [],
            'broken_indicators': [],
            'missing_indicators': [],
            'file_locations': {},
            'import_errors': {},
            'instantiation_errors': {},
            'calculation_errors': {}
        }
        
        # Create test data
        self.test_data = self._create_test_data()
        
    def _create_test_data(self):
        """Create realistic test data for indicator testing"""
        periods = 100
        np.random.seed(42)
        
        # Generate realistic forex price data
        base_price = 1.1000
        returns = np.random.normal(0, 0.001, periods)
        prices = base_price + np.cumsum(returns)
        
        # Create OHLCV data
        high = prices + np.random.uniform(0, 0.001, periods)
        low = prices - np.random.uniform(0, 0.001, periods)
        open_prices = np.roll(prices, 1)
        open_prices[0] = base_price
        volume = np.random.randint(1000, 10000, periods)
        
        return {
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume,
            'timestamp': pd.date_range('2024-01-01', periods=periods, freq='H')
        }
    
    def find_all_indicator_files(self):
        """Find all Python files that could be indicators"""
        print("🔍 PHASE 1: Scanning for indicator files...")
        
        search_paths = [
            'services/analytics-service/src/engines',
            'models',
            'engines',
            'src',
            '.'
        ]
        
        indicator_files = []
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__') and not file.startswith('test_'):
                            full_path = os.path.join(root, file)
                            # Check if it looks like an indicator
                            if self._looks_like_indicator(full_path):
                                indicator_files.append(full_path)
                                print(f"  📄 Found: {full_path}")
        
        self.results['total_files_found'] = len(indicator_files)
        self.results['file_locations'] = {os.path.basename(f): f for f in indicator_files}
        print(f"📊 Found {len(indicator_files)} potential indicator files")
        return indicator_files
    
    def _looks_like_indicator(self, file_path):
        """Check if a file looks like it contains an indicator"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for indicator-like patterns
            indicator_patterns = [
                'class.*Calculator',
                'class.*Indicator', 
                'class.*Analysis',
                'def calculate',
                'class RSI',
                'class MACD',
                'class.*Gann',
                'class.*Fibonacci',
                'class.*Elliott'
            ]
            
            for pattern in indicator_patterns:
                if pattern.lower() in content.lower():
                    return True
            return False
        except:
            return False
    
    def extract_classes_from_files(self, file_paths):
        """Extract all classes from indicator files"""
        print("\n🔍 PHASE 2: Extracting classes from files...")
        
        classes_found = {}
        
        for file_path in file_paths:
            try:
                # Add the directory to Python path
                file_dir = os.path.dirname(file_path)
                if file_dir not in sys.path:
                    sys.path.insert(0, file_dir)
                
                # Try to import the module
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                
                # Execute the module
                spec.loader.exec_module(module)
                
                # Find all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == module_name:  # Only classes defined in this module
                        classes_found[name] = {
                            'file_path': file_path,
                            'module': module,
                            'class_obj': obj
                        }
                        print(f"  ✅ Found class: {name} in {file_path}")
                        
            except Exception as e:
                print(f"  ❌ Failed to process {file_path}: {e}")
                self.results['import_errors'][file_path] = str(e)
        
        self.results['total_classes_found'] = len(classes_found)
        print(f"📊 Found {len(classes_found)} indicator classes")
        return classes_found
    
    def test_indicator_instantiation(self, classes_found):
        """Test if each indicator class can be instantiated"""
        print("\n🔍 PHASE 3: Testing indicator instantiation...")
        
        working_indicators = []
        broken_indicators = []
        
        for class_name, class_info in classes_found.items():
            try:
                # Try to instantiate the class
                print(f"  🧪 Testing {class_name}...", end="")
                
                # Try different instantiation approaches
                instance = None
                try:
                    instance = class_info['class_obj']()
                except TypeError:
                    # Try with some common parameters
                    try:
                        instance = class_info['class_obj'](period=14)
                    except TypeError:
                        try:
                            instance = class_info['class_obj'](logger=None)
                        except:
                            instance = class_info['class_obj'](lookback_periods=20)
                
                if instance:
                    working_indicators.append({
                        'name': class_name,
                        'file_path': class_info['file_path'],
                        'instance': instance
                    })
                    print(" ✅")
                else:
                    broken_indicators.append(class_name)
                    print(" ❌ (instantiation failed)")
                    
            except Exception as e:
                broken_indicators.append(class_name)
                self.results['instantiation_errors'][class_name] = str(e)
                print(f" ❌ ({e})")
        
        self.results['working_indicators'] = [w['name'] for w in working_indicators]
        self.results['broken_indicators'] = broken_indicators
        
        print(f"📊 Working indicators: {len(working_indicators)}")
        print(f"📊 Broken indicators: {len(broken_indicators)}")
        
        return working_indicators, broken_indicators
    
    def test_indicator_calculation(self, working_indicators):
        """Test if working indicators can actually calculate with real data"""
        print("\n🔍 PHASE 4: Testing indicator calculations...")
        
        truly_functional = []
        calculation_failures = []
        
        for indicator_info in working_indicators:
            class_name = indicator_info['name']
            instance = indicator_info['instance']
            
            try:
                print(f"  🧮 Testing calculation for {class_name}...", end="")
                
                # Try different calculation methods
                result = None
                
                # Common calculation method names
                calc_methods = ['calculate', 'analyze', 'compute', 'process', 'run']
                
                for method_name in calc_methods:
                    if hasattr(instance, method_name):
                        method = getattr(instance, method_name)
                        try:
                            # Try different parameter combinations
                            if 'calculate' in method_name.lower():
                                try:
                                    result = method(self.test_data['close'])
                                except:
                                    try:
                                        result = method(
                                            self.test_data['high'],
                                            self.test_data['low'], 
                                            self.test_data['close']
                                        )
                                    except:
                                        result = method(self.test_data)
                            else:
                                result = method(self.test_data)
                            
                            if result is not None:
                                break
                                
                        except Exception as method_error:
                            continue
                
                if result is not None:
                    truly_functional.append({
                        'name': class_name,
                        'file_path': indicator_info['file_path'],
                        'result_type': type(result).__name__,
                        'result_size': len(result) if hasattr(result, '__len__') else 'scalar'
                    })
                    print(" ✅")
                else:
                    calculation_failures.append(class_name)
                    print(" ❌ (no working calculation method)")
                    
            except Exception as e:
                calculation_failures.append(class_name)
                self.results['calculation_errors'][class_name] = str(e)
                print(f" ❌ ({e})")
        
        return truly_functional, calculation_failures
    
    def generate_report(self, truly_functional, calculation_failures):
        """Generate comprehensive audit report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE INDICATOR AUDIT REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total files scanned: {self.results['total_files_found']}")
        print(f"  Total classes found: {self.results['total_classes_found']}")
        print(f"  Working indicators: {len(self.results['working_indicators'])}")
        print(f"  Broken indicators: {len(self.results['broken_indicators'])}")
        print(f"  Truly functional: {len(truly_functional)}")
        print(f"  Calculation failures: {len(calculation_failures)}")
        
        print(f"\n✅ TRULY FUNCTIONAL INDICATORS ({len(truly_functional)}):")
        for indicator in truly_functional:
            print(f"  • {indicator['name']} - {indicator['file_path']}")
        
        if self.results['broken_indicators']:
            print(f"\n❌ BROKEN INDICATORS ({len(self.results['broken_indicators'])}):")
            for indicator in self.results['broken_indicators']:
                print(f"  • {indicator}")
        
        if calculation_failures:
            print(f"\n⚠️  INSTANTIABLE BUT NON-FUNCTIONAL ({len(calculation_failures)}):")
            for indicator in calculation_failures:
                print(f"  • {indicator}")
        
        # Calculate the truth about 67 indicators
        actual_count = len(truly_functional)
        print(f"\n🎯 THE TRUTH:")
        if actual_count >= 67:
            print(f"  ✅ YOU HAVE {actual_count} FUNCTIONAL INDICATORS! (Target: 67)")
        elif actual_count >= 50:
            print(f"  🟡 YOU HAVE {actual_count} FUNCTIONAL INDICATORS (Need {67-actual_count} more)")
        else:
            print(f"  ❌ YOU HAVE ONLY {actual_count} FUNCTIONAL INDICATORS (Need {67-actual_count} more)")
        
        # Save detailed report
        detailed_report = {
            **self.results,
            'truly_functional': truly_functional,
            'calculation_failures': calculation_failures,
            'final_count': actual_count,
            'target_count': 67,
            'achievement_percentage': (actual_count / 67) * 100
        }
        
        with open('INDICATOR_TRUTH_REPORT.json', 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: INDICATOR_TRUTH_REPORT.json")
        
        return detailed_report

def main():
    """Run comprehensive indicator audit"""
    print("🚀 STARTING COMPREHENSIVE INDICATOR AUDIT")
    print("=" * 80)
    
    auditor = IndicatorAuditor()
    
    # Phase 1: Find all indicator files
    indicator_files = auditor.find_all_indicator_files()
    
    # Phase 2: Extract classes from files
    classes_found = auditor.extract_classes_from_files(indicator_files)
    
    # Phase 3: Test instantiation
    working_indicators, broken_indicators = auditor.test_indicator_instantiation(classes_found)
    
    # Phase 4: Test calculation
    truly_functional, calculation_failures = auditor.test_indicator_calculation(working_indicators)
    
    # Phase 5: Generate report
    report = auditor.generate_report(truly_functional, calculation_failures)
    
    return len(truly_functional) >= 60  # Return success if we have at least 60 functional indicators

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
