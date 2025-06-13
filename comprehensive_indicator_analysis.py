#!/usr/bin/env python3
"""
Comprehensive Indicator Analysis - Real Files vs Registered
Analyze what indicator files exist vs what's actually registered
"""

import os
import sys
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Set, Any

# Add project root to path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

def extract_indicator_classes_from_file(file_path: str) -> List[str]:
    """Extract potential indicator classes from a Python file"""
    try:
        # Convert file path to module name
        relative_path = file_path.replace(str(script_dir), "").replace("\\", "/").strip("/")
        if relative_path.endswith('.py'):
            relative_path = relative_path[:-3]
        
        module_name = relative_path.replace("/", ".")
        
        # Try to import the module
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            return []
        
        indicators = []
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            obj = getattr(module, name)
            
            # Check if it's a class that could be an indicator
            if inspect.isclass(obj):
                # Look for calculate method or other indicator-like methods
                if (hasattr(obj, 'calculate') or 
                    hasattr(obj, 'compute') or 
                    hasattr(obj, 'process') or
                    'indicator' in name.lower() or
                    'signal' in name.lower() or
                    'pattern' in name.lower()):
                    indicators.append(name)
        
        return indicators
    except Exception as e:
        return []

def scan_indicator_files() -> Dict[str, List[str]]:
    """Scan all indicator files and extract potential indicators"""
    indicator_dirs = [
        'engines/ai_enhancement/indicators',
        'engines/ai_enhancement', 
        'engines/pattern',
        'engines/momentum',
        'engines/trend',
        'engines/volume',
        'engines/statistical',
        'engines/fibonacci',
        'engines/fractal',
        'engines/gann'
    ]
    
    all_indicators = {}
    
    for dir_path in indicator_dirs:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__') and not file.startswith('test_'):
                        file_path = os.path.join(root, file)
                        indicators = extract_indicator_classes_from_file(file_path)
                        if indicators:
                            all_indicators[file_path] = indicators
    
    return all_indicators

def get_registered_indicators() -> Dict[str, Any]:
    """Get all currently registered indicators"""
    try:
        from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        return dict(INDICATOR_REGISTRY)
    except Exception as e:
        print(f"Error getting registered indicators: {e}")
        return {}

def analyze_indicators():
    """Comprehensive indicator analysis"""
    print("=" * 80)
    print("COMPREHENSIVE INDICATOR ANALYSIS")
    print("=" * 80)
    
    # Get available indicators from files
    print("\n1. SCANNING INDICATOR FILES...")
    file_indicators = scan_indicator_files()
    
    all_found_indicators = set()
    for file_path, indicators in file_indicators.items():
        for indicator in indicators:
            all_found_indicators.add(indicator.lower())
    
    print(f"   Found {len(all_found_indicators)} potential indicators in files")
    
    # Get registered indicators
    print("\n2. CHECKING REGISTERED INDICATORS...")
    registered = get_registered_indicators()
    print(f"   Currently registered: {len(registered)} indicators")
    
    # Find missing indicators
    registered_names = set(registered.keys())
    missing_indicators = all_found_indicators - registered_names
    
    print(f"\n3. ANALYSIS RESULTS:")
    print(f"   Indicators in files: {len(all_found_indicators)}")
    print(f"   Indicators registered: {len(registered)}")
    print(f"   Missing from registry: {len(missing_indicators)}")
    
    if missing_indicators:
        print(f"\n4. MISSING INDICATORS (first 50):")
        for i, indicator in enumerate(sorted(missing_indicators)):
            if i >= 50:
                print(f"   ... and {len(missing_indicators) - 50} more")
                break
            print(f"   - {indicator}")
    
    # Show some examples of what's in files vs registered
    print(f"\n5. FILE SAMPLE (first 20 files with indicators):")
    count = 0
    for file_path, indicators in file_indicators.items():
        if count >= 20:
            break
        print(f"   {file_path}:")
        for indicator in indicators[:3]:  # Show first 3 indicators per file
            registered_status = "✓" if indicator.lower() in registered_names else "✗"
            print(f"     {registered_status} {indicator}")
        if len(indicators) > 3:
            print(f"     ... and {len(indicators) - 3} more")
        count += 1
    
    # Show specific categories
    print(f"\n6. CATEGORY ANALYSIS:")
    categories = {
        'momentum': 0,
        'trend': 0,
        'volume': 0,
        'pattern': 0,
        'fibonacci': 0,
        'fractal': 0,
        'gann': 0,
        'statistical': 0,
        'physics': 0
    }
    
    for indicator in all_found_indicators:
        for category in categories.keys():
            if category in indicator.lower():
                categories[category] += 1
                break
    
    for category, count in categories.items():
        print(f"   {category.capitalize()}: {count} indicators found in files")

if __name__ == "__main__":
    analyze_indicators()