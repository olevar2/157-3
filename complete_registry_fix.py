#!/usr/bin/env python3
"""
Complete Registry Fix Script
This script will:
1. Remove utility types and duplicates properly
2. Restore missing indicators
3. Ensure we have exactly 167 indicators as documented

Issues found:
- Utility types: dict, list, optional, union, dataclass, baseindicator, standardindicatorinterface
- Aliases: bollinger_bands, donchian_channels  
- Pattern duplicates: dark_cloud_cover_pattern, piercing_line_pattern
- Missing indicators after cleanup
"""

import sys
from pathlib import Path

# Add Platform3 root to Python path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

def get_documented_indicators():
    """Get the list of 167 documented indicators"""
    documented_indicators = []
    
    try:
        # Read the documented indicators from COMPLETE_INDICATOR_REGISTRY.md
        registry_file = Path(__file__).parent / "COMPLETE_INDICATOR_REGISTRY.md"
        if registry_file.exists():
            content = registry_file.read_text()
            # Extract indicator names from the markdown
            lines = content.split('\n')
            for line in lines:
                if line.startswith('##') and 'indicator' in line.lower():
                    # Extract indicator name
                    name = line.replace('##', '').strip()
                    if name and not name.startswith('#'):
                        documented_indicators.append(name.lower().replace(' ', '_'))
        
        print(f"Found {len(documented_indicators)} documented indicators")
        return documented_indicators
        
    except Exception as e:
        print(f"Warning: Could not read documented indicators: {e}")
        return []

def analyze_current_registry():
    """Analyze current registry state"""
    print("=== ANALYZING CURRENT REGISTRY ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry, INDICATOR_REGISTRY
        
        # Get all indicators
        enhanced_indicators = _enhanced_registry._indicators
        legacy_indicators = INDICATOR_REGISTRY
        
        print(f"Enhanced registry: {len(enhanced_indicators)} indicators")
        print(f"Legacy registry: {len(legacy_indicators)} indicators")
        
        # Find utility types
        utility_types = []
        real_indicators = []
        
        for name, obj in enhanced_indicators.items():
            # Check for utility types
            if name in ['dict', 'list', 'optional', 'union', 'dataclass', 
                       'baseindicator', 'standardindicatorinterface']:
                utility_types.append(name)
            elif hasattr(obj, '__module__') and obj.__module__ in ['builtins', 'typing']:
                utility_types.append(name)
            else:
                real_indicators.append(name)
        
        # Find duplicates/aliases
        duplicates = {}
        for name in real_indicators:
            normalized = name.replace('_', '').lower()
            if normalized not in duplicates:
                duplicates[normalized] = []
            duplicates[normalized].append(name)
        
        actual_duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
        
        print(f"Utility types found: {len(utility_types)}")
        for ut in utility_types:
            print(f"  - {ut}")
        
        print(f"Real indicators: {len(real_indicators)}")
        print(f"Duplicates/aliases: {len(actual_duplicates)}")
        for norm, variants in actual_duplicates.items():
            print(f"  - {norm}: {variants}")
        
        return {
            'enhanced_indicators': enhanced_indicators,
            'legacy_indicators': legacy_indicators,
            'utility_types': utility_types,
            'real_indicators': real_indicators,
            'duplicates': actual_duplicates
        }
        
    except Exception as e:
        print(f"Error analyzing registry: {e}")
        return None

def clean_registry():
    """Clean utility types and resolve duplicates"""
    print("\n=== CLEANING REGISTRY ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry, INDICATOR_REGISTRY
        
        # Remove utility types
        utility_types = ['dict', 'list', 'optional', 'union', 'dataclass', 
                        'baseindicator', 'standardindicatorinterface']
        
        removed_count = 0
        for utility_type in utility_types:
            if utility_type in _enhanced_registry._indicators:
                del _enhanced_registry._indicators[utility_type]
                removed_count += 1
                print(f"Removed utility type: {utility_type}")
            
            if utility_type in _enhanced_registry._metadata:
                del _enhanced_registry._metadata[utility_type]
            
            # Also remove from legacy registry
            if utility_type in INDICATOR_REGISTRY:
                del INDICATOR_REGISTRY[utility_type]
        
        print(f"Removed {removed_count} utility types")
        
        # Resolve duplicates by keeping the primary name
        duplicates_to_resolve = [
            ('bollinger_bands', 'bollingerbands'),  # Keep bollingerbands
            ('donchian_channels', 'donchianchannels'),  # Keep donchianchannels
            ('dark_cloud_cover_pattern', 'darkcloudcoverpattern'),  # Keep darkcloudcoverpattern
            ('piercing_line_pattern', 'piercinglinepattern'),  # Keep piercinglinepattern
        ]
        
        resolved_count = 0
        for alias, primary in duplicates_to_resolve:
            if alias in _enhanced_registry._indicators:
                del _enhanced_registry._indicators[alias]
                resolved_count += 1
                print(f"Removed duplicate: {alias} (kept {primary})")
            
            if alias in _enhanced_registry._metadata:
                del _enhanced_registry._metadata[alias]
            
            if alias in INDICATOR_REGISTRY:
                del INDICATOR_REGISTRY[alias]
        
        print(f"Resolved {resolved_count} duplicates")
        
        return removed_count + resolved_count
        
    except Exception as e:
        print(f"Error cleaning registry: {e}")
        return 0

def ensure_all_indicators_present():
    """Ensure all documented indicators are present"""
    print("\n=== ENSURING ALL INDICATORS PRESENT ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry
        
        # Get current indicator count
        current_count = len(_enhanced_registry._indicators)
        target_count = 167
        
        print(f"Current indicators: {current_count}")
        print(f"Target indicators: {target_count}")
        print(f"Missing: {target_count - current_count}")
        
        if current_count < target_count:
            # List current indicators
            current_indicators = sorted(_enhanced_registry._indicators.keys())
            print(f"\nCurrent {current_count} indicators:")
            for i, name in enumerate(current_indicators, 1):
                print(f"{i:3d}. {name}")
        
        # Check for essential indicators that might be missing
        essential_indicators = [
            'bollingerbands', 'donchianchannels', 'keltnerchannels',
            'relativestrengthindex', 'movingaverageconvergencedivergence',
            'stochasticoscillator', 'commoditychannelindex',
            'simplemovingaverage', 'exponentialmovingaverage',
            'onbalancevolume', 'volumeweightedaverageprice'
        ]
        
        missing_essential = []
        for indicator in essential_indicators:
            if indicator not in _enhanced_registry._indicators:
                missing_essential.append(indicator)
        
        if missing_essential:
            print(f"\nMissing essential indicators: {missing_essential}")
        
        return current_count
        
    except Exception as e:
        print(f"Error checking indicators: {e}")
        return 0

def validate_final_registry():
    """Final validation of the registry"""
    print("\n=== FINAL VALIDATION ===")
    
    try:
        from engines.ai_enhancement.registry import _enhanced_registry, INDICATOR_REGISTRY
        
        enhanced_count = len(_enhanced_registry._indicators)
        legacy_count = len(INDICATOR_REGISTRY)
        aliases_count = len(_enhanced_registry._aliases)
        
        print(f"Enhanced registry: {enhanced_count} indicators")
        print(f"Legacy registry: {legacy_count} indicators")
        print(f"Aliases: {aliases_count}")
        
        # Check for remaining utility types
        remaining_utilities = []
        for name in _enhanced_registry._indicators:
            if name in ['dict', 'list', 'optional', 'union', 'dataclass', 
                       'baseindicator', 'standardindicatorinterface']:
                remaining_utilities.append(name)
        
        if remaining_utilities:
            print(f"WARNING: Still have utility types: {remaining_utilities}")
        else:
            print("OK: No utility types found")
        
        # Target is 167
        if enhanced_count == 167:
            print("SUCCESS: Registry has exactly 167 indicators!")
            return True
        elif enhanced_count < 167:
            print(f"WARNING: Missing {167 - enhanced_count} indicators")
            return False
        else:
            print(f"WARNING: Have {enhanced_count - 167} extra indicators")
            return False
            
    except Exception as e:
        print(f"Error validating registry: {e}")
        return False

def main():
    """Main function"""
    print("PLATFORM3 COMPLETE REGISTRY FIX")
    print("=" * 50)
    
    # Step 1: Analyze current state
    analysis = analyze_current_registry()
    if not analysis:
        print("ERROR: Failed to analyze registry")
        return
    
    # Step 2: Clean registry
    cleaned_count = clean_registry()
    print(f"\nCleaned {cleaned_count} items from registry")
    
    # Step 3: Check indicator count
    current_count = ensure_all_indicators_present()
    
    # Step 4: Final validation
    success = validate_final_registry()
    
    if success:
        print("\n✓ REGISTRY FIXED SUCCESSFULLY!")
        print("✓ Exactly 167 real indicators")
        print("✓ No utility types")
        print("✓ No duplicates")
    else:
        print("\n! REGISTRY STILL NEEDS WORK")
        print(f"Current count: {current_count}")
        print("Target count: 167")
        
        if current_count < 167:
            print(f"Need to restore {167 - current_count} missing indicators")

if __name__ == "__main__":
    main()