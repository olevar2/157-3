#!/usr/bin/env python3
"""
Precise Indicator Counter - Only counts actual trading indicators in engines folder
Excludes: __init__.py, __pycache__, models, services, base classes, utilities
"""

import os
import glob

def count_actual_indicators():
    """Count only actual trading indicator implementations in engines folder"""
    
    engines_path = "engines"
    if not os.path.exists(engines_path):
        print(f"❌ Engines folder not found at: {engines_path}")
        return 0
    
    # Files to exclude (not actual indicators)
    exclude_files = {
        '__init__.py',
        'indicator_base.py',
        'base_indicator.py',
        'utils.py',
        'helpers.py',
        'constants.py',
        'config.py'
    }
    
    # Directories to exclude 
    exclude_dirs = {
        '__pycache__',
        '.git',
        'node_modules',
        'venv',
        'env'
    }
    
    indicators_found = []
    total_count = 0
    
    print("🔍 PRECISE INDICATOR COUNT - Engines Folder Only")
    print("=" * 60)
    
    # Walk through engines directory only
    for root, dirs, files in os.walk(engines_path):
        # Remove excluded directories from search
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Get relative path for display
        rel_path = os.path.relpath(root, engines_path)
        if rel_path == '.':
            category = "ROOT"
        else:
            category = rel_path.replace('\\', '/').upper()
        
        # Count Python files that are actual indicators
        category_indicators = []
        for file in files:
            if (file.endswith('.py') and 
                file not in exclude_files and
                not file.startswith('test_') and
                not file.startswith('debug_')):
                
                category_indicators.append(file)
                indicators_found.append(os.path.join(root, file))
        
        if category_indicators:
            print(f"\n📂 {category}:")
            for indicator in sorted(category_indicators):
                print(f"   ✅ {indicator}")
            print(f"   📊 Subtotal: {len(category_indicators)} indicators")
            total_count += len(category_indicators)
    
    print("\n" + "=" * 60)
    print(f"🎯 TOTAL ACTUAL INDICATORS: {total_count}")
    print("=" * 60)
    
    # Show some examples of what we found
    if indicators_found:
        print(f"\n📋 Sample indicators found:")
        for i, indicator in enumerate(indicators_found[:10]):  # Show first 10
            rel_path = os.path.relpath(indicator, engines_path)
            print(f"   {i+1}. {rel_path}")
        if len(indicators_found) > 10:
            print(f"   ... and {len(indicators_found) - 10} more")
    
    return total_count

if __name__ == "__main__":
    count = count_actual_indicators()
    print(f"\n🏆 FINAL COUNT: {count} trading indicators")
