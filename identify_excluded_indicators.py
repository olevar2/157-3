#!/usr/bin/env python3
"""
Identify indicators that are being loaded but not counted as 'real' indicators.
This will help us understand what to include to reach the target of 167.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.registry import EnhancedIndicatorRegistry

def identify_excluded_indicators():
    """Identify what indicators are excluded from the 'real' count."""
    
    print("=== IDENTIFYING EXCLUDED INDICATORS ===\n")
    
    # Initialize registry
    registry = EnhancedIndicatorRegistry()
    
    # Get all indicators - access the internal dictionary
    all_indicators = registry._indicators
    print(f"Total indicators loaded: {len(all_indicators)}")
    
    # Identify utility types and base classes
    utility_types = set()
    real_indicators = set()
    
    utility_keywords = [
        'standardindicatorinterface', 'baseindicator', 'dict', 'list', 
        'optional', 'union', 'dataclass', 'interface', 'base', 'abstract'
    ]
    
    for name in all_indicators:
        name_lower = name.lower()
        
        # Check if it's a utility type
        is_utility = False
        for keyword in utility_keywords:
            if keyword in name_lower:
                is_utility = True
                break
        
        # Check class name patterns
        try:
            indicator_class = all_indicators[name]
            if hasattr(indicator_class, '__name__'):
                class_name_lower = indicator_class.__name__.lower()
                for keyword in utility_keywords:
                    if keyword in class_name_lower:
                        is_utility = True
                        break
        except:
            pass
        
        if is_utility:
            utility_types.add(name)
        else:
            real_indicators.add(name)
    
    print(f"Real indicators identified: {len(real_indicators)}")
    print(f"Utility types/base classes: {len(utility_types)}")
    
    print("\n=== UTILITY TYPES EXCLUDED ===")
    for i, name in enumerate(sorted(utility_types), 1):
        print(f"{i:2d}. {name}")
    
    print(f"\n=== ANALYSIS ===")
    print(f"Total loaded: {len(all_indicators)}")
    print(f"Real indicators: {len(real_indicators)}")
    print(f"Utility types: {len(utility_types)}")
    print(f"Gap to target 167: {167 - len(real_indicators)}")
    
    # Check if we can convert some utility types to real indicators
    if len(utility_types) >= (167 - len(real_indicators)):
        print(f"\n[INFO] We can reach 167 by including some utility types as real indicators")
        needed = 167 - len(real_indicators)
        print(f"Need to include {needed} more indicators")
        
        # Suggest which utility types could be real indicators
        potential_real = []
        for name in sorted(utility_types):
            # Skip obvious utility types
            if name.lower() in ['dict', 'list', 'optional', 'union', 'dataclass']:
                continue
            if 'standardindicatorinterface' in name.lower():
                continue
            if 'baseindicator' in name.lower():
                continue
            potential_real.append(name)
        
        print(f"\nPotential real indicators currently marked as utility:")
        for i, name in enumerate(potential_real[:needed], 1):
            print(f"{i:2d}. {name}")

if __name__ == "__main__":
    identify_excluded_indicators()