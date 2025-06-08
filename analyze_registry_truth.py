#!/usr/bin/env python3
"""
Registry Analysis - Find the true indicator count and identify duplicates
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from engines.ai_enhancement.registry import INDICATOR_REGISTRY

def analyze_registry():
    """Analyze the registry to find duplicates and count unique indicators"""
    
    print("=" * 80)
    print("REGISTRY ANALYSIS - FINDING TRUE INDICATOR COUNT")
    print("=" * 80)
    
    # Count total entries
    total_entries = len(INDICATOR_REGISTRY)
    print(f"Total registry entries: {total_entries}")
    
    # Check for duplicate classes (same class with different names)
    class_to_names = {}
    for name, cls in INDICATOR_REGISTRY.items():
        class_name = str(cls)
        if class_name not in class_to_names:
            class_to_names[class_name] = []
        class_to_names[class_name].append(name)
    
    # Find duplicates
    duplicates = {cls_name: names for cls_name, names in class_to_names.items() if len(names) > 1}
    
    print(f"\nUnique classes: {len(class_to_names)}")
    print(f"Duplicate classes found: {len(duplicates)}")
    
    if duplicates:
        print("\nDUPLICATE CLASSES:")
        for cls_name, names in duplicates.items():
            print(f"  {cls_name}")
            for name in names:
                print(f"    - {name}")
    
    # Check for dummy indicators
    dummy_count = 0
    real_count = 0
    
    for name, cls in INDICATOR_REGISTRY.items():
        if str(cls).endswith('.dummy_indicator'):
            dummy_count += 1
        else:
            real_count += 1
    
    print(f"\nIndicator breakdown:")
    print(f"  Real indicators: {real_count}")
    print(f"  Dummy indicators: {dummy_count}")
    print(f"  Total: {real_count + dummy_count}")
    
    # List all indicator names grouped by type
    print(f"\nFirst 20 indicators in registry:")
    for i, (name, cls) in enumerate(list(INDICATOR_REGISTRY.items())[:20]):
        cls_type = "DUMMY" if str(cls).endswith('.dummy_indicator') else "REAL"
        print(f"  {i+1:2d}. {name:30s} -> {cls_type}")
    
    # Calculate true unique indicator count
    true_unique_count = len(class_to_names)
    
    print(f"\n" + "=" * 80)
    print(f"TRUE UNIQUE INDICATOR COUNT: {true_unique_count}")
    print(f"TARGET COUNT: 157")
    print(f"STATUS: {'✅ TARGET MET' if true_unique_count >= 157 else '❌ TARGET NOT MET'}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_registry()
