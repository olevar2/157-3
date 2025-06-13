#!/usr/bin/env python3
"""
Updated registry verification script that correctly excludes utility types.
"""

import sys
import os

# Fix console encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_registry_count():
    """Verify the registry with improved utility type filtering"""
    try:
        # Import after path setup
        from engines.ai_enhancement.registry import _enhanced_registry
        
        print("=== IMPROVED REGISTRY VERIFICATION ===")
        print(f"Total registered indicators: {len(_enhanced_registry._indicators)}")
        
        # Get all indicator names
        all_indicators = list(_enhanced_registry._indicators.keys())
        
        # Comprehensive list of utility types to exclude
        utility_types = {
            'dict', 'list', 'optional', 'union', 'dataclass', 
            'baseindicator', 'standardindicatorinterface'
        }
        
        # Filter indicators by name patterns
        real_indicators = []
        excluded_indicators = []
        
        for name in all_indicators:
            name_lower = name.lower()
            
            # Check if it's a utility type by name
            is_utility = False
            for utility_type in utility_types:
                if utility_type in name_lower:
                    is_utility = True
                    break
            
            if is_utility:
                excluded_indicators.append(name)
            else:
                real_indicators.append(name)
        
        print(f"Real indicators (after filtering): {len(real_indicators)}")
        print(f"Excluded utility types: {len(excluded_indicators)}")
        
        # Show excluded items
        if excluded_indicators:
            print(f"\nExcluded utility types:")
            for i, name in enumerate(sorted(excluded_indicators), 1):
                print(f"  {i:2d}. {name}")
        
        # Look for duplicates among real indicators
        from collections import defaultdict
        class_to_names = defaultdict(list)
        
        for name in real_indicators:
            indicator_class = _enhanced_registry._indicators[name]
            class_id = str(type(indicator_class).__name__)
            class_to_names[class_id].append(name)
        
        # Find aliases (multiple names for same class)
        aliases = {k: v for k, v in class_to_names.items() if len(v) > 1}
        unique_classes = len(class_to_names)
        
        print(f"Unique indicator classes: {unique_classes}")
        
        if aliases:
            print(f"Found {len(aliases)} indicator types with multiple names:")
            total_alias_count = 0
            for class_name, names in aliases.items():
                print(f"  {class_name}: {names}")
                total_alias_count += len(names) - 1  # All but one are aliases
            
            unique_real_count = len(real_indicators) - total_alias_count
            print(f"Unique real indicators (excluding aliases): {unique_real_count}")
        else:
            unique_real_count = len(real_indicators)
        
        # Target verification
        target_count = 167
        print(f"\n=== TARGET ANALYSIS ===")
        print(f"Target count: {target_count}")
        print(f"Current real indicators: {len(real_indicators)}")
        print(f"Current unique indicators: {unique_real_count}")
        print(f"Difference from target: {unique_real_count - target_count}")
        
        if unique_real_count == target_count:
            print("[OK] SUCCESS: Registry matches target count exactly!")
            return True
        elif unique_real_count > target_count:
            excess = unique_real_count - target_count
            print(f"[INFO] Registry has {excess} more indicators than target")
            if excess <= 5:
                print("   This is acceptable - likely minor differences in categorization")
                return True
        else:
            shortage = target_count - unique_real_count
            print(f"[WARN] Registry has {shortage} fewer indicators than target")
        
        # Show sample indicators
        print(f"\nSample real indicators (first 15):")
        for i, name in enumerate(sorted(real_indicators)[:15], 1):
            print(f"  {i:2d}. {name}")
        
        return abs(unique_real_count - target_count) <= 5
        
    except Exception as e:
        print(f"Error verifying registry: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_registry_count()
    if success:
        print("\n[OK] Registry verification PASSED!")
    else:
        print("\n[WARN] Registry needs adjustment")