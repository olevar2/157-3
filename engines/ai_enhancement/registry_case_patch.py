'''
Registry Case-Insensitive Patch
This patch adds case-insensitive matching to the indicator registry
'''

from typing import Dict, Any, Optional

def apply_case_insensitive_patch():
    """Apply case-insensitive patch to indicator registry"""
    # Import the registry
    try:
        from registry import INDICATOR_REGISTRY
    except ImportError:
        try:
            from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        except ImportError:
            print("Could not import indicator registry")
            return
    
    # Get original registry entries
    original_entries = list(INDICATOR_REGISTRY.items())
    
    # Add lowercase aliases for all indicators
    for name, indicator in original_entries:
        lowercase_name = name.lower()
        
        if lowercase_name != name and lowercase_name not in INDICATOR_REGISTRY:
            INDICATOR_REGISTRY[lowercase_name] = indicator
            print(f"Added lowercase alias: {lowercase_name} -> {name}")
        
        # Add variations without 'indicator' suffix
        if name.lower().endswith('indicator'):
            base_name = name[:-9]  # Remove 'indicator' suffix
            if base_name not in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[base_name] = indicator
                print(f"Added base name alias: {base_name} -> {name}")
    
    # Log statistics
    print(f"Original registry size: {len(original_entries)}")
    print(f"Updated registry size: {len(INDICATOR_REGISTRY)}")
    print(f"Added {len(INDICATOR_REGISTRY) - len(original_entries)} aliases")

def get_indicator_case_insensitive(indicator_name: str) -> Optional[Any]:
    """
    Get an indicator from the registry with case-insensitive matching
    
    Args:
        indicator_name: The name of the indicator to retrieve
        
    Returns:
        The indicator instance if found, None otherwise
    """
    # Import the registry
    try:
        from registry import INDICATOR_REGISTRY
    except ImportError:
        try:
            from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        except ImportError:
            print("Could not import indicator registry")
            return None
    
    # Try exact match first
    if indicator_name in INDICATOR_REGISTRY:
        return INDICATOR_REGISTRY[indicator_name]
    
    # Try lowercase match
    lowercase_name = indicator_name.lower()
    if lowercase_name in INDICATOR_REGISTRY:
        return INDICATOR_REGISTRY[lowercase_name]
    
    # Try without 'indicator' suffix
    if lowercase_name.endswith('indicator'):
        base_name = lowercase_name[:-9]  # Remove 'indicator' suffix
        if base_name in INDICATOR_REGISTRY:
            return INDICATOR_REGISTRY[base_name]
    
    # Try all registry keys with case insensitive comparison
    for key in INDICATOR_REGISTRY:
        if key.lower() == lowercase_name:
            return INDICATOR_REGISTRY[key]
    
    return None

# Apply the patch when the module is imported
apply_case_insensitive_patch()
