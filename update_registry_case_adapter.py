"""
Platform3 Registry Case Adapter
Ensures indicators in the registry can be accessed with the proper case from GENIUS_AGENT_INDICATOR_MAPPING.md
"""

import os
import sys
import logging
from pathlib import Path
import re
from typing import Dict, List, Any, Set, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Platform3 to path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

def update_registry_with_case_aliases():
    """Update the indicator registry with case-insensitive aliases"""
    try:
        # Import registry
        from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        
        # Create a dictionary to map lowercase names to their original registry keys
        lowercase_map = {}
        for key in list(INDICATOR_REGISTRY.keys()):
            lowercase_map[key.lower()] = key
        
        # Create an alias dictionary to map PascalCase names to their registry keys
        alias_map = {}
        
        # Process each indicator in GENIUS_AGENT_INDICATOR_MAPPING.md
        indicators = parse_indicators_from_mapping()
        
        for indicator in indicators:
            # Check if indicator exists in registry directly
            if indicator in INDICATOR_REGISTRY:
                continue
                
            # Try lowercase match
            indicator_lower = indicator.lower()
            if indicator_lower in lowercase_map:
                # Add alias from PascalCase to registry key
                registry_key = lowercase_map[indicator_lower]
                INDICATOR_REGISTRY[indicator] = INDICATOR_REGISTRY[registry_key]
                logger.info(f"Added case alias: {indicator} -> {registry_key}")
                alias_map[indicator] = registry_key
            
            # Try without suffix
            for suffix in ["Indicator", "Signal", "Type", "Pattern"]:
                if indicator.endswith(suffix):
                    base_name = indicator[:-len(suffix)]
                    base_name_lower = base_name.lower()
                    if base_name_lower in lowercase_map:
                        registry_key = lowercase_map[base_name_lower]
                        INDICATOR_REGISTRY[indicator] = INDICATOR_REGISTRY[registry_key]
                        logger.info(f"Added name variant alias: {indicator} -> {registry_key}")
                        alias_map[indicator] = registry_key
        
        # Track original registry size
        original_size = len(INDICATOR_REGISTRY)
        
        # Add all remaining case variants
        # This ensures we match all possible case variations
        for original_key in list(INDICATOR_REGISTRY.keys()):
            # Add lowercase variant
            lowercase_key = original_key.lower()
            if lowercase_key != original_key and lowercase_key not in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[lowercase_key] = INDICATOR_REGISTRY[original_key]
            
            # Add PascalCase variant (convert snake_case to PascalCase)
            if "_" in original_key:
                pascal_key = "".join(word.capitalize() for word in original_key.split("_"))
                if pascal_key not in INDICATOR_REGISTRY:
                    INDICATOR_REGISTRY[pascal_key] = INDICATOR_REGISTRY[original_key]
                    alias_map[pascal_key] = original_key
            
            # Add camelCase variant
            if "_" in original_key:
                words = original_key.split("_")
                camel_key = words[0] + "".join(word.capitalize() for word in words[1:])
                if camel_key not in INDICATOR_REGISTRY:
                    INDICATOR_REGISTRY[camel_key] = INDICATOR_REGISTRY[original_key]
                    alias_map[camel_key] = original_key
        
        # Add indicator names without suffix
        for original_key in list(INDICATOR_REGISTRY.keys()):
            for suffix in ["Indicator", "Signal", "Type", "Pattern"]:
                if original_key.endswith(suffix.lower()):
                    base_key = original_key[:-len(suffix)]
                    if base_key not in INDICATOR_REGISTRY:
                        INDICATOR_REGISTRY[base_key] = INDICATOR_REGISTRY[original_key]
                        alias_map[base_key] = original_key
                        logger.info(f"Added base name alias: {base_key} -> {original_key}")
        
        # Report results
        updated_size = len(INDICATOR_REGISTRY)
        aliases_added = updated_size - original_size
        
        print(f"Original registry size: {original_size}")
        print(f"Updated registry size: {updated_size}")
        print(f"Added {aliases_added} aliases")
        
        # Generate the registry aliases file
        create_registry_aliases_file(alias_map)
        
        return True
    except Exception as e:
        logger.error(f"Failed to update registry: {e}")
        return False

def parse_indicators_from_mapping() -> Set[str]:
    """Parse all indicator names from GENIUS_AGENT_INDICATOR_MAPPING.md"""
    indicators = set()
    
    try:
        # Read mapping file with UTF-8 encoding to handle emojis and special characters
        mapping_path = os.path.join(current_dir, "GENIUS_AGENT_INDICATOR_MAPPING.md")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to extract indicator names
        # Pattern matches indicators listed like: IndicatorName, AnotherIndicator
        pattern = r'\*\*.*\*\*:\s*(.*?)(?=\n|$)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            # Split by comma and clean up
            indicator_names = [name.strip() for name in match.split(',')]
            for name in indicator_names:
                if name:  # Skip empty names
                    indicators.add(name)
        
        logger.info(f"Found {len(indicators)} indicators in mapping file")
    except Exception as e:
        logger.error(f"Failed to parse indicators from mapping: {e}")
    
    return indicators

def create_registry_aliases_file(alias_map: Dict[str, str]):
    """Create a Python file with registry aliases for future use"""
    try:
        file_path = os.path.join(current_dir, "engines", "ai_enhancement", "registry_aliases.py")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write('"""\n')
            f.write('Platform3 Registry Aliases\n')
            f.write('Generated file with indicator name aliases to handle case sensitivity\n')
            f.write('"""\n\n')
            
            f.write('# Mapping from alias names to registry keys\n')
            f.write('REGISTRY_ALIASES = {\n')
            
            # Sort for readability
            sorted_aliases = sorted(alias_map.items())
            for alias, registry_key in sorted_aliases:
                f.write(f'    "{alias}": "{registry_key}",\n')
            
            f.write('}\n\n')
            
            # Add utility function
            f.write('def get_registry_key(indicator_name):\n')
            f.write('    """\n')
            f.write('    Get the registry key for an indicator name, handling case sensitivity\n')
            f.write('    \n')
            f.write('    Args:\n')
            f.write('        indicator_name: The indicator name to look up\n')
            f.write('        \n')
            f.write('    Returns:\n')
            f.write('        The registry key for the indicator, or the original name if not found\n')
            f.write('    """\n')
            f.write('    return REGISTRY_ALIASES.get(indicator_name, indicator_name)\n')
        
        logger.info(f"Generated registry alias file at {file_path}")
    except Exception as e:
        logger.error(f"Failed to create registry aliases file: {e}")

def main():
    """Run the registry case adapter"""
    print("Platform3 Registry Case Adapter")
    print("=" * 40)
    
    success = update_registry_with_case_aliases()
    
    if success:
        print("✓ Registry updated successfully with case-insensitive aliases")
    else:
        print("✗ Failed to update registry")

if __name__ == "__main__":
    main()
