#!/usr/bin/env python3
"""
Script to identify the exact 7 indicators that are missing from the registry
to reach the target of 167 real indicators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.registry import get_enhanced_registry

def read_documented_indicators():
    """Read all indicators from the documentation"""
    documented = set()
    
    # Read from COMPLETE_INDICATOR_REGISTRY.md
    try:
        with open('COMPLETE_INDICATOR_REGISTRY.md', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract indicator names from the documentation
        # Look for numbered list items or bullet points
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('*', '-', '+')) or any(line.startswith(f"{i}.") for i in range(1, 200))):
                # Extract indicator name (remove numbering, bullets, markdown)
                parts = line.split('.')
                if len(parts) > 1:
                    name = parts[1].strip()
                elif line.startswith(('*', '-', '+')):
                    name = line[1:].strip()
                else:
                    continue
                
                # Clean up the name
                name = name.split('(')[0].strip()  # Remove descriptions in parentheses
                name = name.split(':')[0].strip()  # Remove colons and descriptions
                name = name.split('-')[0].strip()  # Remove dashes and descriptions
                name = name.replace('**', '').replace('`', '').strip()  # Remove markdown
                
                if name and len(name) > 2:  # Valid indicator name
                    documented.add(name.lower().replace(' ', '').replace('_', '').replace('-', ''))
                    
    except Exception as e:
        print(f"Error reading documentation: {e}")
    
    return documented

def get_registry_indicators():
    """Get all real indicators from the registry"""
    registry = get_enhanced_registry()
    
    # Get all indicators, excluding utility types
    utility_types = {
        'dict', 'list', 'optional', 'union', 'dataclass', 'tuple', 'callable',
        'baseindicator', 'standardindicatorinterface', 'abstractindicator'
    }
    
    real_indicators = set()
    for name, indicator_class in registry._indicators.items():
        if name.lower() not in utility_types:
            real_indicators.add(name.lower())
    
    return real_indicators

def analyze_missing_indicators():
    """Find the exact indicators that are missing"""
    print("=== FINDING MISSING 7 INDICATORS ===\n")
    
    # Get documented indicators from all sources
    documented = read_documented_indicators()
    registry_indicators = get_registry_indicators()
    
    print(f"Documented indicators found: {len(documented)}")
    print(f"Registry real indicators: {len(registry_indicators)}")
    print(f"Gap: {len(documented) - len(registry_indicators)}")
    
    # Find missing indicators
    missing = documented - registry_indicators
    extra = registry_indicators - documented
    
    print(f"\n=== MISSING FROM REGISTRY ({len(missing)}) ===")
    for i, indicator in enumerate(sorted(missing), 1):
        print(f"{i:2d}. {indicator}")
    
    print(f"\n=== EXTRA IN REGISTRY ({len(extra)}) ===")
    for i, indicator in enumerate(sorted(extra), 1):
        print(f"{i:2d}. {indicator}")
    
    # Try to find patterns or naming mismatches
    print(f"\n=== POTENTIAL NAME MATCHES ===")
    for missing_indicator in sorted(missing):
        for registry_indicator in registry_indicators:
            # Check for partial matches
            if (missing_indicator in registry_indicator or 
                registry_indicator in missing_indicator or
                abs(len(missing_indicator) - len(registry_indicator)) <= 3):
                similarity = len(set(missing_indicator) & set(registry_indicator)) / max(len(missing_indicator), len(registry_indicator))
                if similarity > 0.6:
                    print(f"  '{missing_indicator}' might match '{registry_indicator}' (similarity: {similarity:.2f})")

if __name__ == "__main__":
    analyze_missing_indicators()