'''
Platform3 Indicator Registry Aliases
This file provides aliases for indicators in the registry to match names in GENIUS_AGENT_INDICATOR_MAPPING.md
'''

from typing import Dict, Any

# Try to import the registry
try:
    from registry import INDICATOR_REGISTRY
except ImportError:
    try:
        from engines.ai_enhancement.registry import INDICATOR_REGISTRY
    except ImportError:
        INDICATOR_REGISTRY = {}

# Create aliases for indicator names
def create_registry_aliases() -> None:
    """Create aliases for indicator names in the registry"""
    # Mapping from mapping file name to registry name
    name_mapping = {
    }
    
    # Create aliases in the registry
    for alias, real_name in name_mapping.items():
        if real_name in INDICATOR_REGISTRY and alias not in INDICATOR_REGISTRY:
            INDICATOR_REGISTRY[alias] = INDICATOR_REGISTRY[real_name]
            print(f"Created alias {alias} -> {real_name}")

# Call the function to create aliases
create_registry_aliases()
