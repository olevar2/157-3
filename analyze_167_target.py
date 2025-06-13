#!/usr/bin/env python3
"""
Script to precisely count and identify the 167 indicators from documentation
and compare with the registry to find exactly what's missing.
"""

import sys
import os
import re

from engines.ai_enhancement.registry import get_enhanced_registry

def extract_indicators_from_documentation():
    """Extract all indicator names from the documentation"""
    try:
        with open('COMPLETE_INDICATOR_REGISTRY.md', 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading documentation: {e}")
        return set()
    
    documented_indicators = set()
    
    # Split into sections
    lines = content.split('\n')
    
    # Look for indicator patterns:
    # 1. **IndicatorName** (file.py)
    # 2. - **IndicatorName** (file.py)
    # 3. Class names in bold
    
    for line in lines:
        line = line.strip()
        
        # Pattern 1: **IndicatorName** (file.py)
        match = re.search(r'\*\*([A-Za-z][A-Za-z0-9]+)\*\*\s*\([^)]+\.py\)', line)
        if match:
            indicator_name = match.group(1)
            documented_indicators.add(indicator_name.lower())
            continue
        
        # Pattern 2: - **IndicatorName** (file.py)  
        match = re.search(r'-\s*\*\*([A-Za-z][A-Za-z0-9]+)\*\*\s*\([^)]+\.py\)', line)
        if match:
            indicator_name = match.group(1)
            documented_indicators.add(indicator_name.lower())
            continue
            
        # Pattern 3: Look for file-based indicators like "BollingerBands"
        if '(' in line and '.py)' in line and '**' in line:
            # Extract everything between ** markers before (.py)
            parts = line.split('**')
            for i in range(1, len(parts), 2):  # Take every odd index (between **)
                name = parts[i].strip()
                if name and name.isalnum() and len(name) > 3:
                    documented_indicators.add(name.lower())
    
    return documented_indicators

def get_registry_real_indicators():
    """Get all real indicators from the registry (excluding utility types)"""
    registry = get_enhanced_registry()
    
    # Get all indicators, excluding utility types and base classes
    utility_types = {
        'dict', 'list', 'optional', 'union', 'dataclass', 'tuple', 'callable',
        'baseindicator', 'standardindicatorinterface', 'abstractindicator'
    }
    
    real_indicators = set()
    for name, indicator_class in registry._indicators.items():
        if name.lower() not in utility_types:
            real_indicators.add(name.lower())
    
    return real_indicators

def analyze_167_target():
    """Analyze the gap between documentation and registry"""
    print("=== ANALYZING 167 INDICATOR TARGET ===\n")
    
    # Get documented indicators
    documented = extract_indicators_from_documentation()
    registry_indicators = get_registry_real_indicators()
    
    print(f"Documented indicators found: {len(documented)}")
    print(f"Registry real indicators: {len(registry_indicators)}")
    print(f"Target: 167")
    print(f"Gap to reach 167: {167 - len(registry_indicators)}")
    
    # Find missing from registry
    missing_from_registry = documented - registry_indicators
    extra_in_registry = registry_indicators - documented
    
    print(f"\n=== MISSING FROM REGISTRY ({len(missing_from_registry)}) ===")
    for i, indicator in enumerate(sorted(missing_from_registry), 1):
        print(f"{i:2d}. {indicator}")
    
    print(f"\n=== EXTRA IN REGISTRY ({len(extra_in_registry)}) ===")
    extra_list = sorted(extra_in_registry)
    for i, indicator in enumerate(extra_list, 1):
        print(f"{i:2d}. {indicator}")
    
    # Look for the 7 we need to add
    print(f"\n=== ANALYSIS FOR 167 TARGET ===")
    print(f"To reach exactly 167 real indicators:")
    print(f"Current registry: {len(registry_indicators)}")
    print(f"Need to add: {167 - len(registry_indicators)} indicators")
    
    if len(missing_from_registry) >= (167 - len(registry_indicators)):
        candidates = sorted(list(missing_from_registry))[:7]
        print(f"\nTOP 7 CANDIDATES TO ADD:")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate}")
    
    # Check if any extra indicators are utility types we should remove
    print(f"\n=== CHECKING FOR UTILITY TYPES TO REMOVE ===")
    suspicious_names = []
    for name in extra_list:
        if any(keyword in name for keyword in ['config', 'base', 'interface', 'abstract', 'utility']):
            suspicious_names.append(name)
    
    if suspicious_names:
        print("Possible utility types that should be excluded:")
        for name in suspicious_names:
            print(f"  - {name}")
    else:
        print("No obvious utility types found in extras")

if __name__ == "__main__":
    analyze_167_target()