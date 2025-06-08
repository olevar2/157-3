#!/usr/bin/env python3
"""
CRITICAL FIX: Remove all dummy indicators from registry
This script identifies and fixes the dangerous dummy indicator problem
"""

import re
from pathlib import Path

def fix_registry_remove_dummy_indicators():
    """Remove all dummy indicator assignments - they cause wrong trading results"""
    
    registry_file = Path("d:/MD/Platform3/engines/ai_enhancement/registry.py")
    
    with open(registry_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove all lines that assign dummy_indicator
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Skip lines that assign dummy_indicator
        if 'dummy_indicator' in line and '=' in line:
            print(f"REMOVING DANGEROUS LINE: {line.strip()}")
            continue
        fixed_lines.append(line)
    
    # Write back the fixed content
    with open(registry_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Fixed registry file: removed {len(lines) - len(fixed_lines)} dummy indicator assignments")

if __name__ == "__main__":
    fix_registry_remove_dummy_indicators()
