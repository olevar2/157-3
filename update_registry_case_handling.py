"""
Platform3 Registry Case-Insensitive Update
This script updates the registry to support case-insensitive indicator name matching
"""

import os
import sys
import logging
from typing import Dict, Any, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Platform3 to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def update_registry():
    """Update the registry to support case-insensitive indicator name matching"""
    try:
        # Import the registry
        try:
            from registry import INDICATOR_REGISTRY
        except ImportError:
            from engines.ai_enhancement.registry import INDICATOR_REGISTRY
            
        # Get original registry entries
        original_entries = list(INDICATOR_REGISTRY.items())
        
        # Add lowercase aliases for all indicators
        for name, indicator in original_entries:
            lowercase_name = name.lower()
            
            if lowercase_name != name and lowercase_name not in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[lowercase_name] = indicator
                logger.info(f"Added lowercase alias: {lowercase_name} -> {name}")
            
            # Add variations without 'indicator' suffix
            if name.lower().endswith('indicator'):
                base_name = name[:-9]  # Remove 'indicator' suffix
                if base_name not in INDICATOR_REGISTRY:
                    INDICATOR_REGISTRY[base_name] = indicator
                    logger.info(f"Added base name alias: {base_name} -> {name}")
        
        # Log statistics
        logger.info(f"Original registry size: {len(original_entries)}")
        logger.info(f"Updated registry size: {len(INDICATOR_REGISTRY)}")
        logger.info(f"Added {len(INDICATOR_REGISTRY) - len(original_entries)} aliases")
        
        return True
    except Exception as e:
        logger.error(f"Failed to update registry: {e}")
        return False

def create_patch_file():
    """Create a patch file to update the registry module"""
    patch_content = """'''
Registry Case-Insensitive Patch
This patch adds case-insensitive matching to the indicator registry
'''

from typing import Dict, Any, Optional

def apply_case_insensitive_patch():
    \"\"\"Apply case-insensitive patch to indicator registry\"\"\"
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
    \"\"\"
    Get an indicator from the registry with case-insensitive matching
    
    Args:
        indicator_name: The name of the indicator to retrieve
        
    Returns:
        The indicator instance if found, None otherwise
    \"\"\"
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
"""

    # Write the patch file
    patch_file = os.path.join(current_dir, "engines", "ai_enhancement", "registry_case_patch.py")
    os.makedirs(os.path.dirname(patch_file), exist_ok=True)
    
    try:
        with open(patch_file, 'w') as f:
            f.write(patch_content)
            
        logger.info(f"Created registry case-insensitive patch at {patch_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create patch file: {e}")
        return False

def main():
    """Main function to run the registry update"""
    logger.info("Updating registry for case-insensitive matching...")
    
    # Update the registry directly
    if update_registry():
        logger.info("Registry updated successfully")
    else:
        logger.error("Failed to update registry directly")
    
    # Create a patch file for future use
    if create_patch_file():
        logger.info("Patch file created successfully")
    else:
        logger.error("Failed to create patch file")
    
    logger.info("Registry update complete")

if __name__ == "__main__":
    main()
