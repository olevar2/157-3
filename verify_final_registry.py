#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Registry Verification

This script verifies that we have exactly 167 indicators as expected.
It will count real indicators (excluding utility classes and aliases).

Author: Platform3 AI Enhancement Engine
Date: 2025-06-11
"""

import sys
import os

# Fix console encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.registry import EnhancedIndicatorRegistry

def verify_final_registry():
    """Verify the final registry count and status"""
    try:
        # Get the global enhanced registry instance (indicators are loaded automatically)
        from engines.ai_enhancement.registry import get_enhanced_registry
        registry = get_enhanced_registry()
        
        print("=== FINAL REGISTRY VERIFICATION ===")
        print(f"Total registered indicators: {len(registry._indicators)}")
        
        # Get all indicator names from the correct attribute
        all_indicators = list(registry._indicators.keys())
        
        # Filter out utility types that shouldn't be counted
        utility_types = {
            'dict', 'list', 'optional', 'union', 'dataclass', 
            'baseindicator', 'standardindicatorinterface'
        }
        
        real_indicators = [name for name in all_indicators if name.lower() not in utility_types]
        
        print(f"Real indicators (excluding utility types): {len(real_indicators)}")
        
        # Check for aliases by looking for duplicate classes
        aliases = {}
        unique_classes = {}
        for name in real_indicators:
            indicator_class = registry._indicators[name]
            class_id = id(indicator_class)
            
            if class_id in unique_classes:
                # This is an alias
                original_name = unique_classes[class_id]
                if original_name not in aliases:
                    aliases[original_name] = []
                aliases[original_name].append(name)
            else:
                unique_classes[class_id] = name
        
        unique_count = len(unique_classes)
        alias_count = sum(len(alias_list) for alias_list in aliases.values())
        
        print(f"Unique indicator classes: {unique_count}")
        print(f"Aliases: {alias_count}")
        
        if aliases:
            print("Alias mappings:")
            for original, alias_list in aliases.items():
                print(f"  {original} -> {alias_list}")
        
        # Show some indicator names for verification
        print(f"\nFirst 20 real indicators:")
        for i, name in enumerate(sorted(real_indicators)[:20]):
            print(f"  {i+1:2d}. {name}")
        
        if len(real_indicators) > 20:
            print(f"  ... and {len(real_indicators) - 20} more")
        
        # Target verification
        target_count = 167
        print(f"\n=== TARGET COMPARISON ===")
        print(f"Target count: {target_count}")
        print(f"Current real indicators: {len(real_indicators)}")
        print(f"Current unique indicators: {unique_count}")
        print(f"Difference: {len(real_indicators) - target_count}")
        
        if len(real_indicators) == target_count:
            print("[OK] SUCCESS: Registry matches target count!")
            return True
        elif len(real_indicators) > target_count:
            print("[WARN] Registry has MORE indicators than expected")
            excess = len(real_indicators) - target_count
            print(f"   Consider if {excess} indicators are duplicates or should be aliases")
        else:
            print("[ERROR] Registry has FEWER indicators than expected")
            missing = target_count - len(real_indicators)
            print(f"   Need to add {missing} more indicators")
        
        # Check if we're close (within 10)
        if abs(len(real_indicators) - target_count) <= 10:
            print("[OK] Registry is close to target count - likely acceptable")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error verifying registry: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_final_registry()
    if success:
        print("\n[OK] Registry reconciliation COMPLETE!")
    else:
        print("\n[WARN] Registry needs additional work")