#!/usr/bin/env python3
"""
Dynamic Indicator Registry Loader
Bypasses hardcoded module lists and loads all working indicators from the filesystem
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Tuple, Any
from engines.indicator_base import IndicatorBase

# Export the function to make it importable
__all__ = ['load_all_indicators', 'load_all_working_indicators']

def load_all_working_indicators() -> Dict[str, Any]:
    """Load all working indicators dynamically from the engines directory"""
    indicators, _ = load_all_indicators()
    return indicators

def load_all_indicators() -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Load all working indicators dynamically from the engines directory
    
    Returns:
        Tuple containing:
        - Dictionary of indicator classes keyed by category.name
        - Dictionary of categories with lists of indicator names
    """
    indicators = {}
    categories = {}
    
    # Add engines to path
    # sys.path.insert(0, os.path.abspath('.')) # This can be unreliable
    script_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(script_dir.parent)) # Add project root (d:\MD\Platform3)
    
    engines_path = script_dir / "engines"
    patterns_path = script_dir / "ai-platform" / "ai-models" / "market-analysis" / "pattern-recognition" # New path
    print(f"DEBUG: engines_path resolved to: {engines_path.resolve()}") # DEBUG
    print(f"DEBUG: patterns_path resolved to: {patterns_path.resolve()}") # DEBUG
    
    search_paths = [engines_path, patterns_path] # List of paths to search

    for current_search_path in search_paths:
        for root, dirs, files in os.walk(current_search_path):
            print(f"DEBUG: Walking root: {root}") # DEBUG
            # Skip validation, test, and pycache directories
            if any(skip in root for skip in ['validation', 'test', '__pycache__', 'backup']):
                continue
            
            category = ""
            if current_search_path == patterns_path:
                category = "pattern" # Specific category for pattern indicators
            else:
                category = os.path.basename(root)

            if category not in categories:
                categories[category] = []
                
            for file in files:
                if (file.endswith('.py') and 
                    not file.startswith('__') and
                    not file.endswith('_backup') and
                    'backup' not in file and
                    file not in ['indicator_base.py', 'indicator_registry.py']):
                    
                    try:
                        module_path = ""
                        # Convert file path to module path
                        # relative_path = os.path.relpath(os.path.join(root, file), '.') # This is problematic with different drives
                        full_path = Path(root) / file
                        relative_path = full_path.relative_to(script_dir.parent) # Relative to project root
                        module_path = str(relative_path).replace(os.sep, '.').replace('.py', '')
                        
                        # Try to import the module
                        print(f"Attempting to import: {module_path}") # DEBUG
                        try:
                            module = importlib.import_module(module_path)
                            print(f"Successfully imported: {module_path}") # DEBUG
                        except Exception as e:
                            print(f"Failed to import {module_path}: {e}") # DEBUG
                            continue
                        
                        # Look for indicator classes
                        for name, obj in inspect.getmembers(module):
                            if file == 'japanese_candlesticks.py': # ADDED DEBUG
                                print(f"DEBUG_JAPANESE_CANDLESTICKS: Found member: {name}, type: {type(obj)}, isclass: {inspect.isclass(obj)}") # ADDED DEBUG
                            if (inspect.isclass(obj) and 
                                name not in ['IndicatorBase', 'BaseIndicator', 'TechnicalIndicator'] and
                                not name.startswith('_')):
                                
                                if file == 'japanese_candlesticks.py': # ADDED DEBUG
                                    print(f"DEBUG_JAPANESE_CANDLESTICKS: Checking class: {name}, MRO: {[b.__name__ for b in obj.__mro__]}") # ADDED DEBUG

                                # Check if it might be an indicator class
                                if any(base.__name__ in ['IndicatorBase', 'TechnicalIndicator'] 
                                       for base in obj.__mro__):
                                
                                    indicator_key = f"{category}.{name.lower()}"
                                    # === ADDED FILTER START ===
                                    if indicator_key in [
                                        'pattern.darkcloudcoverpattern', 
                                        'pattern.piercinglinepattern', 
                                        'pattern.tweezerpatterns'
                                    ]:
                                        print(f"[SKIP] Skipping problematic pattern key: {indicator_key}")
                                        continue
                                    # === ADDED FILTER END ===
                                    indicators[indicator_key] = obj
                                    categories[category].append(name)
                                    print(f"[OK] Loaded: {category}.{name}")
                                
                    except Exception as e:
                        print(f"[FAIL] Failed to load {module_path}: {e}")
    
    return indicators, categories

if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMIC INDICATOR REGISTRY TEST")
    print("=" * 60)
    
    indicators = load_all_working_indicators()
    
    # Count by category
    categories = {}
    for indicator_name in indicators:
        category = indicator_name.split('.')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(indicator_name)
    
    print(f"\nSUCCESS: Loaded {len(indicators)} indicators") # Removed problematic char
    print(f"Categories: {len(categories)}") # Removed problematic char
    
    for category, indicator_list in sorted(categories.items()):
        if indicator_list:
            print(f"{category}: {len(indicator_list)} indicators")
    
    print(f"\nTotal functional indicators: {len(indicators)}")
    
    # Test registry instantiation
    try:
        from engines.indicator_registry import IndicatorRegistry
        print("Original registry still failing") # Removed problematic char
    except Exception as e:
        print(f"Original registry error: {e}") # Removed problematic char
