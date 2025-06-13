#!/usr/bin/env python3
"""
Debug script to examine what indicators are actually loaded in the registry.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.ai_enhancement.registry import EnhancedIndicatorRegistry

def debug_registry_indicators():
    """Debug the registry to see what indicators are loaded."""
    
    print("=== DEBUGGING REGISTRY INDICATORS ===\n")
    
    # Initialize registry
    registry = EnhancedIndicatorRegistry()
    
    # Check if registry has indicators
    print(f"Registry type: {type(registry)}")
    print(f"Registry attributes: {[attr for attr in dir(registry) if not attr.startswith('__')]}")
    
    # Try different ways to access indicators
    print(f"\nTrying to access _indicators...")
    try:
        indicators = registry._indicators
        print(f"_indicators type: {type(indicators)}")
        print(f"_indicators length: {len(indicators)}")
        
        if len(indicators) > 0:
            print(f"\nFirst 10 indicator names:")
            for i, name in enumerate(list(indicators.keys())[:10], 1):
                print(f"{i:2d}. {name}")
        else:
            print("No indicators found in _indicators")
            
    except Exception as e:
        print(f"Error accessing _indicators: {e}")
    
    # Try get_indicators method if it exists
    print(f"\nTrying get_indicators method...")
    try:
        if hasattr(registry, 'get_indicators'):
            indicators = registry.get_indicators()
            print(f"get_indicators() returned: {type(indicators)}, length: {len(indicators)}")
        else:
            print("No get_indicators method found")
    except Exception as e:
        print(f"Error calling get_indicators: {e}")
    
    # Try other potential methods
    for method_name in ['get_all_indicators', 'list_indicators', 'indicators']:
        print(f"\nTrying {method_name}...")
        try:
            if hasattr(registry, method_name):
                method = getattr(registry, method_name)
                if callable(method):
                    result = method()
                    print(f"{method_name}() returned: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                else:
                    print(f"{method_name} is not callable: {type(method)}")
            else:
                print(f"No {method_name} method found")
        except Exception as e:
            print(f"Error with {method_name}: {e}")
    
    # Check metadata
    print(f"\nTrying to access _metadata...")
    try:
        metadata = registry._metadata
        print(f"_metadata type: {type(metadata)}")
        print(f"_metadata length: {len(metadata)}")
    except Exception as e:
        print(f"Error accessing _metadata: {e}")

if __name__ == "__main__":
    debug_registry_indicators()