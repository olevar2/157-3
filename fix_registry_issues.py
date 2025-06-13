#!/usr/bin/env python3
"""
Registry Fix Script
Fix the 2 extra indicators and 2 aliases issue in Platform3 registry

Issues to resolve:
1. Two extra indicators beyond the documented 167
2. Two aliases causing confusion: BollingerBands/bollinger_bands, DonchianChannels/donchian_channels
3. Utility types incorrectly registered as indicators
"""

import sys
from pathlib import Path

# Add Platform3 root to Python path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

def analyze_registry_contents():
    """Analyze the current registry contents to identify issues"""
    print("=== ANALYZING REGISTRY CONTENTS ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry, INDICATOR_REGISTRY
        
        # Get all registered indicators
        all_indicators = _enhanced_registry._indicators
        aliases = _enhanced_registry._aliases
        
        print(f"Total indicators in enhanced registry: {len(all_indicators)}")
        print(f"Total aliases: {len(aliases)}")
        print(f"Total in INDICATOR_REGISTRY: {len(INDICATOR_REGISTRY)}")
        
        # Find utility types that shouldn't be indicators
        utility_types = []
        for name, obj in all_indicators.items():
            # Check for Python built-in types and typing utilities
            if hasattr(obj, '__module__'):
                module = obj.__module__
                if module in ['builtins', 'typing']:
                    utility_types.append(name)
            
            # Check for names that are clearly not indicators
            if name in ['dict', 'list', 'optional', 'union', 'dataclass', 
                       'baseindicator', 'standardindicatorinterface']:
                utility_types.append(name)
        
        print(f"\nUtility types incorrectly registered: {utility_types}")
        
        # Find exact aliases
        print(f"\nAliases found: {aliases}")
        
        # Find potential duplicates
        duplicates = {}
        for name in all_indicators.keys():
            normalized = name.replace('_', '').lower()
            if normalized not in duplicates:
                duplicates[normalized] = []
            duplicates[normalized].append(name)
        
        actual_duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
        print(f"\nPotential duplicates: {actual_duplicates}")
        
        return {
            'total_indicators': len(all_indicators),
            'utility_types': utility_types,
            'aliases': aliases,
            'duplicates': actual_duplicates
        }
        
    except Exception as e:
        print(f"Error analyzing registry: {e}")
        return None

def fix_aliases():
    """Fix the alias issues by keeping only primary names"""
    print("\n=== FIXING ALIASES ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry
        
        # Remove aliases from the main indicator registry
        aliases_to_fix = ['bollinger_bands', 'donchian_channels']
        
        for alias in aliases_to_fix:
            if alias in _enhanced_registry._indicators:
                del _enhanced_registry._indicators[alias]
                print(f"Removed alias '{alias}' from indicators")
            
            if alias in _enhanced_registry._aliases:
                primary_name = _enhanced_registry._aliases[alias]
                print(f"Kept alias mapping: '{alias}' -> '{primary_name}'")
        
        return True
        
    except Exception as e:
        print(f"Error fixing aliases: {e}")
        return False

def remove_utility_types():
    """Remove utility types that are not real indicators"""
    print("\n=== REMOVING UTILITY TYPES ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry
        
        utility_types = ['dict', 'list', 'optional', 'union', 'dataclass', 
                        'baseindicator', 'standardindicatorinterface']
        
        removed_count = 0
        for utility_type in utility_types:
            if utility_type in _enhanced_registry._indicators:
                del _enhanced_registry._indicators[utility_type]
                print(f"Removed utility type: {utility_type}")
                removed_count += 1
            
            if utility_type in _enhanced_registry._metadata:
                del _enhanced_registry._metadata[utility_type]
        
        print(f"Removed {removed_count} utility types")
        return removed_count
        
    except Exception as e:
        print(f"Error removing utility types: {e}")
        return 0

def validate_final_count():
    """Validate that we now have exactly 167 indicators"""
    print("\n=== VALIDATING FINAL COUNT ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry
        
        final_count = len(_enhanced_registry._indicators)
        aliases_count = len(_enhanced_registry._aliases)
        
        print(f"Final indicator count: {final_count}")
        print(f"Aliases count: {aliases_count}")
        print(f"Total (indicators + aliases): {final_count + aliases_count}")
        
        if final_count == 167:
            print("✅ SUCCESS: Registry now has exactly 167 indicators!")
            return True
        else:
            print(f"❌ ERROR: Expected 167 indicators, got {final_count}")
            
            # Show what we have now
            print("\nCurrent indicators:")
            for i, name in enumerate(sorted(_enhanced_registry._indicators.keys()), 1):
                print(f"{i:3d}. {name}")
            
            return False
            
    except Exception as e:
        print(f"Error validating count: {e}")
        return False

def main():
    """Main function to fix all registry issues"""
    print("PLATFORM3 REGISTRY FIX SCRIPT")
    print("=" * 50)
    
    # Step 1: Analyze current state
    analysis = analyze_registry_contents()
    if not analysis:
        print("❌ Failed to analyze registry")
        return
    
    print(f"\nCurrent state:")
    print(f"- Total indicators: {analysis['total_indicators']}")
    print(f"- Expected: 167")
    print(f"- Extra indicators: {analysis['total_indicators'] - 167}")
    
    # Step 2: Remove utility types
    removed_utilities = remove_utility_types()
    
    # Step 3: Fix aliases
    aliases_fixed = fix_aliases()
    
    # Step 4: Validate final count
    success = validate_final_count()
    
    if success:
        print("\n🎉 ALL REGISTRY ISSUES FIXED!")
        print("✅ Registry now has exactly 167 real indicators")
        print("✅ Aliases resolved")
        print("✅ Utility types removed")
    else:
        print("\n❌ Some issues remain")
        
        # Re-analyze to see current state
        print("\nRe-analyzing after fixes...")
        final_analysis = analyze_registry_contents()
        if final_analysis:
            extra = final_analysis['total_indicators'] - 167
            if extra > 0:
                print(f"Still have {extra} extra indicators")
            elif extra < 0:
                print(f"Missing {abs(extra)} indicators")

if __name__ == "__main__":
    main()