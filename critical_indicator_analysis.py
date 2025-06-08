#!/usr/bin/env python3
"""
CRITICAL INDICATOR UTILIZATION ASSESSMENT
Platform3 Complete Indicator Analysis
"""

import os
import sys
import importlib
import inspect
from pathlib import Path

def count_indicator_files():
    """Count all indicator files in engines directory"""
    engines_path = Path("engines")
    indicator_files = []
    
    for root, dirs, files in os.walk(engines_path):
        # Skip validation and test directories
        if 'validation' in root or 'test' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if (file.endswith('.py') and 
                not file.startswith('__') and 
                not file.endswith('_backup') and
                not 'backup' in file and
                file != 'indicator_base.py' and
                file != 'indicator_registry.py'):
                
                file_path = os.path.join(root, file)
                indicator_files.append(file_path)
    
    return indicator_files

def test_indicator_imports():
    """Test which indicators can be imported successfully"""
    working_indicators = []
    failed_indicators = []
    
    # Add engines to path
    sys.path.insert(0, os.path.abspath('.'))
    
    indicator_files = count_indicator_files()
    
    for file_path in indicator_files:
        try:
            # Convert file path to module path
            module_path = file_path.replace(os.sep, '.').replace('.py', '')
            
            # Try to import the module
            module = importlib.import_module(module_path)
            
            # Look for indicator classes
            classes_found = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    name not in ['IndicatorBase', 'BaseIndicator', 'TechnicalIndicator'] and
                    not name.startswith('_')):
                    classes_found.append(name)
            
            if classes_found:
                working_indicators.append({
                    'file': file_path,
                    'classes': classes_found
                })
            else:
                failed_indicators.append({
                    'file': file_path,
                    'error': 'No indicator classes found'
                })
                
        except Exception as e:
            failed_indicators.append({
                'file': file_path,
                'error': str(e)
            })
    
    return working_indicators, failed_indicators

def analyze_indicator_utilization():
    """Main analysis function"""
    print("=" * 60)
    print("CRITICAL INDICATOR UTILIZATION ANALYSIS")
    print("=" * 60)
    
    # Count total files
    indicator_files = count_indicator_files()
    print(f"Total indicator files found: {len(indicator_files)}")
    
    # Test imports
    working, failed = test_indicator_imports()
    
    print(f"\n✅ WORKING INDICATORS: {len(working)}")
    print(f"❌ FAILED INDICATORS: {len(failed)}")
    
    print(f"\nSUCCESS RATE: {len(working)}/{len(indicator_files)} = {(len(working)/len(indicator_files)*100):.1f}%")
    
    print("\n" + "="*60)
    print("WORKING INDICATORS BY CATEGORY:")
    print("="*60)
    
    # Group by category
    categories = {}
    for indicator in working:
        category = indicator['file'].split(os.sep)[1] if len(indicator['file'].split(os.sep)) > 1 else 'other'
        if category not in categories:
            categories[category] = []
        categories[category].append(indicator)
    
    total_classes = 0
    for category, indicators in sorted(categories.items()):
        class_count = sum(len(ind['classes']) for ind in indicators)
        total_classes += class_count
        print(f"{category}: {len(indicators)} files, {class_count} classes")
        for ind in indicators:
            print(f"  ✓ {ind['file']} -> {', '.join(ind['classes'])}")
    
    print(f"\nTOTAL INDICATOR CLASSES: {total_classes}")
    
    print("\n" + "="*60)
    print("FAILED INDICATORS:")
    print("="*60)
    
    for indicator in failed:
        print(f"❌ {indicator['file']}: {indicator['error']}")
    
    print("\n" + "="*60)
    print("CRITICAL FINDINGS:")
    print("="*60)
    
    if len(working) < 50:
        print("🚨 CRITICAL: Less than 50 working indicators found!")
        print("🚨 This is FAR BELOW the claimed 115+ indicators!")
    elif len(working) < 100:
        print("⚠️  WARNING: Less than 100 working indicators found!")
        print("⚠️  This may not meet the 115+ indicator target!")
    else:
        print("✅ GOOD: 100+ indicators found, target likely met!")
    
    return working, failed

if __name__ == "__main__":
    analyze_indicator_utilization()
