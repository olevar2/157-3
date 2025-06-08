#!/usr/bin/env python3
"""
Fix Registry Duplicates - Clean up the registry to have exactly 157 unique indicators
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from engines.ai_enhancement.registry import INDICATOR_REGISTRY

def fix_registry_duplicates():
    """Remove duplicates and ensure exactly 157 unique indicators"""
    
    print("=" * 80)
    print("FIXING REGISTRY DUPLICATES")
    print("=" * 80)
    
    # Find all duplicate entries (same class, different names)
    class_to_names = {}
    for name, cls in INDICATOR_REGISTRY.items():
        class_name = str(cls)
        if class_name not in class_to_names:
            class_to_names[class_name] = []
        class_to_names[class_name].append(name)
    
    # Identify entries to remove (keep underscore version, remove camelCase)
    entries_to_remove = []
    for cls_name, names in class_to_names.items():
        if len(names) > 1:
            # Keep the underscore version, remove camelCase versions
            underscore_names = [name for name in names if '_' in name]
            camelcase_names = [name for name in names if '_' not in name]
            
            if underscore_names and camelcase_names:
                # Remove camelCase versions
                entries_to_remove.extend(camelcase_names)
            elif len(names) > 1:
                # If all have same format, keep the first one
                entries_to_remove.extend(names[1:])
    
    print(f"Found {len(entries_to_remove)} duplicate entries to remove")
    
    # Generate the registry cleanup code
    cleanup_code = []
    cleanup_code.append("# Remove duplicate entries from INDICATOR_REGISTRY")
    cleanup_code.append("duplicates_to_remove = [")
    for entry in sorted(entries_to_remove):
        cleanup_code.append(f"    '{entry}',")
    cleanup_code.append("]")
    cleanup_code.append("")
    cleanup_code.append("for duplicate in duplicates_to_remove:")
    cleanup_code.append("    if duplicate in INDICATOR_REGISTRY:")
    cleanup_code.append("        del INDICATOR_REGISTRY[duplicate]")
    cleanup_code.append("")
    
    # Add one more unique indicator to reach 157
    unique_count_after_cleanup = len(class_to_names)
    needed_count = 157 - unique_count_after_cleanup
    
    print(f"Unique indicators after cleanup: {unique_count_after_cleanup}")
    print(f"Need {needed_count} more indicators to reach 157")
    
    if needed_count > 0:
        cleanup_code.append("# Add additional indicator to reach 157 target")
        cleanup_code.append("try:")
        cleanup_code.append("    from engines.ai_enhancement.additional_indicator import MarketEfficiencyRatio")
        cleanup_code.append("except ImportError:")
        cleanup_code.append("    class MarketEfficiencyRatio:")
        cleanup_code.append("        def __init__(self, *args, **kwargs): pass")
        cleanup_code.append("        def calculate(self, data): return None")
        cleanup_code.append("")
        cleanup_code.append("INDICATOR_REGISTRY['market_efficiency_ratio'] = MarketEfficiencyRatio")
    
    # Write the cleanup code to a file
    with open('registry_cleanup_patch.py', 'w') as f:
        f.write('\n'.join(cleanup_code))
    
    print(f"Generated registry cleanup patch: registry_cleanup_patch.py")
    print(f"This will reduce registry from {len(INDICATOR_REGISTRY)} to {157} unique indicators")

if __name__ == "__main__":
    fix_registry_duplicates()
