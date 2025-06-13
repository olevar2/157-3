#!/usr/bin/env python3
"""
Final Registry Cleanup - Fix all issues to reach exactly 167 indicators
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from engines.ai_enhancement.registry import get_enhanced_registry
    registry = get_enhanced_registry()
    
    print("=== FINAL REGISTRY CLEANUP ===")
    print(f"Starting with: {len(registry._indicators)} indicators")
    print()
    
    # 1. Remove utility types and base classes
    utility_types_to_remove = [
        'dict', 'list', 'optional', 'union', 'dataclass', 
        'baseindicator', 'standardindicatorinterface'
    ]
    
    removed_utilities = []
    for utility in utility_types_to_remove:
        if utility in registry._indicators:
            del registry._indicators[utility]
            removed_utilities.append(utility)
            print(f"[OK] Removed utility type: {utility}")
    
    print(f"Removed {len(removed_utilities)} utility types")
    print()
    
    # 2. Remove the 2 aliases
    aliases_to_remove = ['bollinger_bands', 'donchian_channels']
    removed_aliases = []
    for alias in aliases_to_remove:
        if alias in registry._indicators:
            del registry._indicators[alias]
            removed_aliases.append(alias)
            print(f"[OK] Removed alias: {alias}")
    
    print(f"Removed {len(removed_aliases)} aliases")
    print()
    
    # 3. Add any truly missing indicators by checking actual file system
    # Let's check what we have vs what we need
    current_count = len(registry._indicators)
    target_count = 167
    
    print(f"Current count after cleanup: {current_count}")
    print(f"Target count: {target_count}")
    print(f"Need to add: {target_count - current_count} indicators")
    print()
    
    # Check what's actually missing by looking at specific indicators
    # that should be present based on our documentation
    critical_indicators = [
        'chaikinoscillator', 'negativevolumeindex', 'pricevolumetrend',
        'trixindicator', 'truestrengthindexindicator', 'ultimateoscillatorindicator',
        'volumeoscillator', 'volumerateofchange', 'vortexindicator', 'williamsrindicator',
        'stochasticindicator', 'relativestrengthindexindicator', 'movingaverageconvergencedivergenceindicator',
        'selfsimilaritysignal', 'threeInsidesignal', 'threelineStrikesignal', 'threeoutsidesignal'
    ]
    
    missing_critical = []
    for indicator in critical_indicators:
        if indicator not in registry._indicators:
            missing_critical.append(indicator)
    
    print("=== MISSING CRITICAL INDICATORS ===")
    for indicator in missing_critical:
        print(f"- {indicator}")
    
    print(f"\nMissing critical indicators: {len(missing_critical)}")
    
    # Final summary
    final_count = len(registry._indicators)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"[OK] Removed {len(removed_utilities)} utility types")
    print(f"[OK] Removed {len(removed_aliases)} aliases")
    print(f"[COUNT] Final registry count: {final_count}")
    print(f"[TARGET] Target count: {target_count}")
    
    if final_count == target_count:
        print("[SUCCESS] Registry now has exactly 167 indicators!")
    elif final_count < target_count:
        shortage = target_count - final_count
        print(f"[SHORTAGE] Need {shortage} more indicators")
        print("[INFO] This suggests some real indicator files are not being loaded properly")
    else:
        excess = final_count - target_count
        print(f"[EXCESS] Have {excess} extra indicators")
        print("[INFO] This suggests there are still some duplicates or unwanted entries")
    
    print(f"\n[IMPROVEMENT] Registry improvement: {165 - final_count} indicators removed/cleaned up")

except Exception as e:
    print(f"Error during cleanup: {e}")
    import traceback
    traceback.print_exc()