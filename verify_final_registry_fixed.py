"""
Fixed Final Registry Verification

This script verifies that we have exactly 167 indicators as expected.
It will count real indicators (excluding utility classes and aliases).

Author: Platform3 AI Enhancement Engine
Date: 2025-06-11
"""

import sys
import os

from engines.ai_enhancement.registry import EnhancedIndicatorRegistry

def verify_final_registry():
    """Verify the final registry count and status"""
    try:
        # Initialize registry
        registry = EnhancedIndicatorRegistry()
        
        print("=== FINAL REGISTRY VERIFICATION ===")
        print(f"Total registered indicators: {len(registry._indicators)}")
        
        # Get all indicator names
        all_indicators = list(registry._indicators.keys())
        
        # Filter out utility types that shouldn't be counted
        utility_types = {
            'dict', 'list', 'optional', 'union', 'dataclass', 
            'baseindicator', 'standardindicatorinterface',
            'try'  # This appears to be from syntax errors
        }
        
        real_indicators = [name for name in all_indicators if name not in utility_types]
        
        print(f"Real indicators (excluding utility types): {len(real_indicators)}")
        
        # Check for aliases
        aliases = getattr(registry, '_aliases', {})
        print(f"Aliases: {len(aliases)}")
        
        if aliases:
            print("Alias mappings:")
            for alias, target in aliases.items():
                print(f"  {alias} -> {target}")
        
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
        print(f"Difference: {len(real_indicators) - target_count}")
        
        if len(real_indicators) == target_count:
            print("[SUCCESS] Registry matches target count!")
            return True
        elif len(real_indicators) > target_count:
            print("[WARNING] Registry has MORE indicators than expected")
            excess = len(real_indicators) - target_count
            print(f"   Consider if {excess} indicators are duplicates or should be aliases")
        else:
            print("[ERROR] Registry has FEWER indicators than expected")
            missing = target_count - len(real_indicators)
            print(f"   Need to add {missing} more indicators")
        
        # Show all indicators sorted by category if requested
        print(f"\n=== ALL INDICATORS BY NAME ===")
        for i, name in enumerate(sorted(real_indicators), 1):
            print(f"  {i:3d}. {name}")
        
        return len(real_indicators) == target_count
        
    except Exception as e:
        print(f"Error verifying registry: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_final_registry()
    if success:
        print("\n[SUCCESS] Registry reconciliation COMPLETE!")
    else:
        print("\n[INFO] Registry needs additional work")