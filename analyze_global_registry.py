#!/usr/bin/env python3
"""
Access the global enhanced registry and analyze the indicators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the global registry instance
from engines.ai_enhancement.registry import _enhanced_registry

def analyze_global_registry():
    """Analyze the global enhanced registry."""
    
    print("=== ANALYZING GLOBAL ENHANCED REGISTRY ===\n")
    
    # Access indicators from the global registry
    indicators = _enhanced_registry._indicators
    metadata = _enhanced_registry._metadata
    
    print(f"Total indicators in global registry: {len(indicators)}")
    print(f"Total metadata entries: {len(metadata)}")
    
    if len(indicators) == 0:
        print("No indicators found - the registry may not be loaded yet.")
        return
    
    # Classify indicators
    utility_types = set()
    real_indicators = set()
    
    utility_keywords = [
        'standardindicatorinterface', 'baseindicator', 'dict', 'list', 
        'optional', 'union', 'dataclass', 'interface', 'base', 'abstract'
    ]
    
    for name, indicator_class in indicators.items():
        name_lower = name.lower()
        
        # Check if it's a utility type
        is_utility = False
        for keyword in utility_keywords:
            if keyword in name_lower:
                is_utility = True
                break
        
        # Check class name patterns
        try:
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
    
    print(f"\nReal indicators: {len(real_indicators)}")
    print(f"Utility types/base classes: {len(utility_types)}")
    print(f"Gap to target 167: {167 - len(real_indicators)}")
    
    print(f"\n=== UTILITY TYPES ===")
    for i, name in enumerate(sorted(utility_types), 1):
        print(f"{i:2d}. {name}")
    
    # Analyze duplicates (indicators with same implementation)
    class_to_names = {}
    for name, indicator_class in indicators.items():
        class_id = str(indicator_class)
        if class_id not in class_to_names:
            class_to_names[class_id] = []
        class_to_names[class_id].append(name)
    
    duplicates = {k: v for k, v in class_to_names.items() if len(v) > 1}
    
    if duplicates:
        print(f"\n=== DUPLICATE IMPLEMENTATIONS ===")
        for i, (class_id, names) in enumerate(duplicates.items(), 1):
            print(f"{i:2d}. {names} -> same implementation")
    
    # List some real indicators
    print(f"\n=== SAMPLE REAL INDICATORS (first 20) ===")
    for i, name in enumerate(sorted(real_indicators)[:20], 1):
        print(f"{i:2d}. {name}")
    
    # Check if there are any indicators we can promote from utility to real
    promotion_candidates = []
    for name in sorted(utility_types):
        # Skip obvious utility types
        if name.lower() in ['dict', 'list', 'optional', 'union', 'dataclass']:
            continue
        if 'standardindicatorinterface' in name.lower():
            continue
        
        # These might be real indicators misclassified
        promotion_candidates.append(name)
    
    if promotion_candidates:
        print(f"\n=== POTENTIAL PROMOTION CANDIDATES ===")
        needed = 167 - len(real_indicators)
        print(f"Need {needed} more real indicators")
        for i, name in enumerate(promotion_candidates[:needed], 1):
            print(f"{i:2d}. {name}")

if __name__ == "__main__":
    analyze_global_registry()