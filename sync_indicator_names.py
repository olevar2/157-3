"""
Platform3 Indicator Name Synchronizer
This script synchronizes the indicator names between the registry and the mapping file
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Platform3 to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def get_registry_indicators() -> List[str]:
    """Get all indicators from the registry"""
    try:
        try:
            from registry import INDICATOR_REGISTRY
        except ImportError:
            from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        
        return list(INDICATOR_REGISTRY.keys())
    except Exception as e:
        logger.error(f"Failed to get registry indicators: {e}")
        return []

def get_mapping_file_indicators() -> List[str]:
    """Get all indicators mentioned in the GENIUS_AGENT_INDICATOR_MAPPING.md file"""
    from indicator_mapping_integrator import parse_indicator_mapping
    
    mapping_file = os.path.join(current_dir, "GENIUS_AGENT_INDICATOR_MAPPING.md")
    
    if not os.path.exists(mapping_file):
        logger.error(f"Mapping file not found: {mapping_file}")
        return []
    
    try:
        agent_mappings = parse_indicator_mapping(mapping_file)
        
        # Collect all indicators from all agents
        all_indicators = set()
        for agent, categories in agent_mappings.items():
            for category, indicators in categories.items():
                all_indicators.update(indicators)
        
        return list(all_indicators)
    except Exception as e:
        logger.error(f"Failed to get mapping file indicators: {e}")
        return []

def create_name_mapping(registry_indicators: List[str], mapping_indicators: List[str]) -> Dict[str, str]:
    """Create a mapping between indicator names in the mapping file and registry"""
    name_mapping = {}
    
    # Normalize registry names for comparison
    registry_normalized = {name.lower().replace('_', '').replace(' ', ''): name for name in registry_indicators}
    
    for mapping_name in mapping_indicators:
        normalized = mapping_name.lower().replace('_', '').replace(' ', '')
        
        # Check for exact match first
        if mapping_name in registry_indicators:
            name_mapping[mapping_name] = mapping_name
            continue
        
        # Check for normalized match
        if normalized in registry_normalized:
            name_mapping[mapping_name] = registry_normalized[normalized]
            continue
        
        # Check for match without 'Indicator' suffix
        if normalized.endswith('indicator'):
            base_name = normalized[:-9]
            for reg_norm, reg_name in registry_normalized.items():
                if reg_norm == base_name or reg_norm == base_name + 'indicator':
                    name_mapping[mapping_name] = reg_name
                    break
        
        # Try other variations
        variations = [
            normalized,
            normalized + 'indicator',
            normalized.replace('indicator', '')
        ]
        
        for variation in variations:
            for reg_norm, reg_name in registry_normalized.items():
                if variation == reg_norm:
                    name_mapping[mapping_name] = reg_name
                    break
    
    return name_mapping

def update_mapping_file_with_registry_names(name_mapping: Dict[str, str]) -> None:
    """Update the GENIUS_AGENT_INDICATOR_MAPPING.md file with registry indicator names"""
    mapping_file = os.path.join(current_dir, "GENIUS_AGENT_INDICATOR_MAPPING.md")
    
    if not os.path.exists(mapping_file):
        logger.error(f"Mapping file not found: {mapping_file}")
        return
    
    try:
        with open(mapping_file, 'r') as f:
            content = f.read()
        
        # For each mapping, replace the mapping name with the registry name
        for mapping_name, registry_name in name_mapping.items():
            if mapping_name != registry_name:
                # Use word boundaries to avoid partial replacements
                pattern = r'\b' + re.escape(mapping_name) + r'\b'
                content = re.sub(pattern, registry_name, content)
        
        # Write back to the file
        with open(mapping_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated mapping file with {len(name_mapping)} registry indicator names")
    except Exception as e:
        logger.error(f"Failed to update mapping file: {e}")

def generate_registry_alias_file() -> None:
    """Generate a file with aliases for indicator registry access"""
    registry_indicators = get_registry_indicators()
    mapping_indicators = get_mapping_file_indicators()
    
    # Create a mapping between indicator names
    name_mapping = create_name_mapping(registry_indicators, mapping_indicators)
    
    # Find indicators that could not be mapped
    unmapped = [name for name in mapping_indicators if name not in name_mapping]
    
    # Generate the content for the registry alias file
    content = """'''
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
    \"\"\"Create aliases for indicator names in the registry\"\"\"
    # Mapping from mapping file name to registry name
    name_mapping = {
"""
    
    # Add the name mapping
    for mapping_name, registry_name in name_mapping.items():
        if mapping_name != registry_name:
            content += f"        '{mapping_name}': '{registry_name}',\n"
    
    content += """    }
    
    # Create aliases in the registry
    for alias, real_name in name_mapping.items():
        if real_name in INDICATOR_REGISTRY and alias not in INDICATOR_REGISTRY:
            INDICATOR_REGISTRY[alias] = INDICATOR_REGISTRY[real_name]
            print(f"Created alias {alias} -> {real_name}")
"""
    
    # Add note about unmapped indicators
    if unmapped:
        content += "\n    # Note: The following indicators could not be mapped to the registry:\n"
        for name in unmapped:
            content += f"    # - {name}\n"
    
    content += """
# Call the function to create aliases
create_registry_aliases()
"""
    
    # Write the file
    output_file = os.path.join(current_dir, "engines", "ai_enhancement", "registry_aliases.py")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Generated registry alias file at {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate registry alias file: {e}")

def main():
    """Main function"""
    logger.info("Synchronizing indicator names between registry and mapping file...")
    
    # Get indicators from registry and mapping file
    registry_indicators = get_registry_indicators()
    mapping_indicators = get_mapping_file_indicators()
    
    logger.info(f"Found {len(registry_indicators)} indicators in registry")
    logger.info(f"Found {len(mapping_indicators)} indicators in mapping file")
    
    # Create a mapping between indicator names
    name_mapping = create_name_mapping(registry_indicators, mapping_indicators)
    
    # Find indicators that could not be mapped
    unmapped = [name for name in mapping_indicators if name not in name_mapping]
    
    logger.info(f"Mapped {len(name_mapping)} indicators")
    logger.info(f"Could not map {len(unmapped)} indicators: {', '.join(unmapped)}")
    
    # Update the mapping file with registry names
    # Disabled for now to preserve original mapping file
    # update_mapping_file_with_registry_names(name_mapping)
    
    # Generate a file with aliases for indicator registry access
    generate_registry_alias_file()
    
    logger.info("Synchronization complete!")

if __name__ == "__main__":
    main()
