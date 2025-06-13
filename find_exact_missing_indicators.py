#!/usr/bin/env python3
"""
Find exactly which 2 indicators are missing to reach 167 target
"""

import sys
from pathlib import Path
sys.path.append('.')

def load_registry():
    """Load the indicator registry and get real indicators count"""
    try:
        from engines.ai_enhancement.registry import indicator_registry
        
        # Get all indicators excluding utility types
        real_indicators = {name: cls for name, cls in indicator_registry._indicators.items() 
                          if not name.lower() in ['dict', 'list', 'optional', 'union', 'dataclass', 'baseindicator', 'standardindicatorinterface']}
        
        return real_indicators
    except Exception as e:
        print(f"Error loading registry: {e}")
        return {}

def get_documented_indicators():
    """Get the full list of documented indicators from COMPLETE_INDICATOR_REGISTRY.md"""
    documented_file = Path('COMPLETE_INDICATOR_REGISTRY.md')
    if not documented_file.exists():
        print("Documentation file not found")
        return []
    
    content = documented_file.read_text()
    lines = content.split('\n')
    
    # Look for indicators with file references
    documented_indicators = []
    for line in lines:
        if '.py)' in line and '**' in line:
            # Extract indicator name from markdown bold format
            if '**' in line:
                parts = line.split('**')
                if len(parts) >= 2:
                    indicator_name = parts[1].strip().lower()
                    documented_indicators.append(indicator_name)
    
    return documented_indicators

def main():
    print("=== FINDING EXACT MISSING INDICATORS ===")
    
    # Load registry
    registry_indicators = load_registry()
    print(f"Registry has {len(registry_indicators)} real indicators")
    
    # Get documented indicators  
    documented_indicators = get_documented_indicators()
    print(f"Documentation has {len(documented_indicators)} indicators")
    
    # Find missing indicators
    registry_names = set(name.lower() for name in registry_indicators.keys())
    documented_names = set(documented_indicators)
    
    missing_from_registry = documented_names - registry_names
    extra_in_registry = registry_names - documented_names
    
    print(f"\nMissing from registry ({len(missing_from_registry)}):")
    for name in sorted(missing_from_registry):
        print(f"  - {name}")
    
    print(f"\nExtra in registry ({len(extra_in_registry)}):")
    for name in sorted(extra_in_registry):
        print(f"  - {name}")
    
    # Target analysis
    target = 167
    current = len(registry_indicators)
    difference = target - current
    
    print(f"\nTarget: {target}")
    print(f"Current: {current}")
    print(f"Need to add: {difference}")
    
    # Check if missing indicators have files
    print(f"\n=== CHECKING FOR MISSING INDICATOR FILES ===")
    indicators_dir = Path('engines/ai_enhancement/indicators')
    
    for indicator in sorted(missing_from_registry):
        # Convert to potential file name variations
        file_variations = [
            f"{indicator}.py",
            f"{indicator}_indicator.py",
            f"{indicator}_signal.py"
        ]
        
        found_file = None
        for root in indicators_dir.rglob('*.py'):
            if root.stem.lower() in [indicator, f"{indicator}_indicator", f"{indicator}_signal"]:
                found_file = root
                break
        
        if found_file:
            print(f"  [FILE EXISTS] {indicator} -> {found_file}")
        else:
            print(f"  [NO FILE] {indicator}")

if __name__ == "__main__":
    main()