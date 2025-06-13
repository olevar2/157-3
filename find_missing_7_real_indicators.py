#!/usr/bin/env python3

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from engines.ai_enhancement.registry import EnhancedIndicatorRegistry

def main():
    # Set up logging to avoid spam
    logging.basicConfig(level=logging.WARNING)
    
    print("=== FINDING MISSING 7 REAL INDICATORS ===")
    print()
    
    # Initialize registry
    registry = EnhancedIndicatorRegistry()
    
    # Get all registered indicators
    try:
        # Access the internal registry
        registered_indicators = registry._indicators
        print(f"Total registered indicators: {len(registered_indicators)}")
        
        # Count real indicators (excluding utility types)
        utility_types = {
            'standardindicatorinterface',
            'baseindicator', 
            'dict', 'list', 'optional', 'union', 'dataclass'
        }
        
        real_indicators = []
        aliases = []
        utility_excluded = []
        
        for name, indicator_class in registered_indicators.items():
            name_lower = name.lower()
            
            # Check if it's a utility type
            if name_lower in utility_types:
                utility_excluded.append(name)
                continue
                
            # Check if it's an alias (duplicate class with different name)
            class_name = indicator_class.__name__ if hasattr(indicator_class, '__name__') else str(indicator_class)
            existing_indicators = [r for r in real_indicators if r[1].__name__ == class_name]
            
            if existing_indicators:
                aliases.append((name, existing_indicators[0][0]))
            else:
                real_indicators.append((name, indicator_class))
        
        print(f"Real indicators found: {len(real_indicators)}")
        print(f"Aliases found: {len(aliases)}")
        print(f"Utility types excluded: {len(utility_excluded)}")
        print()
        
        # Show aliases
        if aliases:
            print("=== ALIASES FOUND ===")
            for alias, original in aliases:
                print(f"  {alias} -> {original}")
            print()
        
        # Show utility types excluded
        if utility_excluded:
            print("=== UTILITY TYPES EXCLUDED ===")
            for util in utility_excluded:
                print(f"  {util}")
            print()
        
        # Calculate gap
        target = 167
        current_real = len(real_indicators)
        gap = target - current_real
        
        print(f"=== GAP ANALYSIS ===")
        print(f"Target real indicators: {target}")
        print(f"Current real indicators: {current_real}")
        print(f"Gap: {gap}")
        print()
        
        if gap > 0:
            print(f"[MISSING] Need {gap} more real indicators")
            
            # Now scan for indicator files that might not be loaded
            indicators_dir = Path("engines/ai_enhancement/indicators")
            if indicators_dir.exists():
                print("\n=== SCANNING FOR UNLOADED INDICATOR FILES ===")
                
                loaded_names = {name.lower() for name, _ in real_indicators}
                
                # Scan all Python files
                unloaded_files = []
                for py_file in indicators_dir.rglob("*.py"):
                    if py_file.name == "__init__.py":
                        continue
                    
                    file_stem = py_file.stem.lower()
                    
                    # Check if this file name (or likely class name) is loaded
                    potential_names = [
                        file_stem,
                        file_stem.replace('_', ''),
                        file_stem + 'indicator',
                        file_stem.replace('_', '') + 'indicator'
                    ]
                    
                    if not any(name in loaded_names for name in potential_names):
                        unloaded_files.append(py_file)
                
                print(f"Found {len(unloaded_files)} potentially unloaded files:")
                for file in unloaded_files[:10]:  # Show first 10
                    print(f"  {file}")
                
                if len(unloaded_files) > 10:
                    print(f"  ... and {len(unloaded_files) - 10} more")
        
        elif gap == 0:
            print("[PERFECT] Registry matches target exactly!")
        else:
            print(f"[OVER] Registry has {abs(gap)} more indicators than target")
        
        print("\n=== FIRST 20 REAL INDICATORS ===")
        for i, (name, _) in enumerate(real_indicators[:20], 1):
            print(f"  {i:2d}. {name}")
        
        if len(real_indicators) > 20:
            print(f"  ... and {len(real_indicators) - 20} more")
    
    except Exception as e:
        print(f"Error accessing registry: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()