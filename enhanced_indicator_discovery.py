#!/usr/bin/env python3
"""
Enhanced Indicator Discovery Script
==================================

This script uses a broader pattern to find ALL indicator classes, regardless of inheritance.
It then analyzes each class to determine if it's a legitimate technical indicator.

Purpose: Get the TRUE count of implemented indicators in Platform3
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

def is_indicator_class(class_name, file_content):
    """Determine if a class is likely an indicator based on name and content"""
    
    # Skip obvious non-indicators
    skip_patterns = [
        'Config', 'Helper', 'Result', 'Data', 'Signal', 'Enum',
        'Exception', 'Error', 'Test', 'Mock', 'Base'
    ]
    
    if any(pattern in class_name for pattern in skip_patterns):
        return False
    
    # Look for indicator-like patterns in the class
    indicator_evidence = [
        'calculate', 'compute', 'analyze', 'detect', 'scan',
        'moving_average', 'oscillator', 'momentum', 'trend',
        'volume', 'volatility', 'pattern', 'signal'
    ]
    
    content_lower = file_content.lower()
    evidence_count = sum(1 for pattern in indicator_evidence if pattern in content_lower)
    
    return evidence_count >= 2

def enhanced_indicator_count(engines_path):
    """Enhanced indicator counting with broader pattern matching"""
    
    # Initialize results
    category_counts = defaultdict(list)
    total_indicators = 0
    
    # Get all subdirectories in engines
    engines_dir = Path(engines_path)
    categories = [d.name for d in engines_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
    
    # Broader pattern to find ANY class definition
    class_pattern = re.compile(r'class\s+([A-Z][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:')
    
    print("="*80)
    print("ENHANCED PLATFORM3 INDICATOR DISCOVERY")
    print("="*80)
    print()
    
    for category in sorted(categories):
        category_path = engines_dir / category
        
        print(f"[CATEGORY] {category}")
        print("-" * 50)
        
        category_indicators = []
        
        # Get all Python files (excluding backups)
        python_files = [
            f for f in category_path.glob('*.py') 
            if not f.name.startswith('__') 
            and '.backup' not in f.name
            and not f.name.endswith(('_old.py', '_new.py'))
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find all classes in file
                matches = class_pattern.findall(content)
                
                if matches:
                    for class_name in matches:
                        # Check if this looks like an indicator
                        if is_indicator_class(class_name, content):
                            category_indicators.append({
                                'name': class_name,
                                'file': py_file.name,
                                'path': str(py_file)
                            })
                            print(f"  [FOUND] {class_name} (in {py_file.name})")
                        else:
                            print(f"  [SKIP] {class_name} (utility/config class)")
                        
            except Exception as e:
                print(f"  [ERROR] Error reading {py_file.name}: {e}")
        
        # Store results for this category
        category_counts[category] = category_indicators
        category_total = len(category_indicators)
        total_indicators += category_total
        
        print(f"  [TOTAL] {category_total} indicators in {category}")
        print()
    
    return total_indicators, category_counts

if __name__ == "__main__":
    from datetime import datetime
    
    # Define engines path
    engines_path = r"d:\MD\Platform3\engines"
    
    if not os.path.exists(engines_path):
        print(f"[ERROR] Engines directory not found: {engines_path}")
        exit(1)
    
    # Run enhanced count
    total, categories = enhanced_indicator_count(engines_path)
    
    # Summary report
    print("="*80)
    print("ENHANCED SUMMARY REPORT")
    print("="*80)
    print()
    
    for category, indicators in sorted(categories.items()):
        if indicators:  # Only show categories with indicators
            print(f"{category:25} : {len(indicators):3d} indicators")
    
    print("-" * 50)
    print(f"{'TOTAL FOUND':25} : {total:3d} indicators")
    print(f"{'CLAIMED IN DOCS':25} : 115+ indicators")
    print(f"{'DISCREPANCY':25} : {abs(total - 115):3d} indicators")
    print()
    
    # Save enhanced report
    report = {
        'timestamp': str(datetime.now()),
        'method': 'enhanced_pattern_matching',
        'total_indicators': total,
        'claimed_indicators': 115,
        'discrepancy': abs(total - 115),
        'categories': {cat: [ind['name'] for ind in indicators] for cat, indicators in categories.items()},
        'detailed_results': dict(categories)
    }
    
    report_file = Path(engines_path).parent / 'enhanced_indicator_discovery_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"[SAVED] Enhanced report saved to: {report_file}")
    
    if total >= 115:
        print(f"[VERDICT] CLAIM VERIFIED: Found {total} indicators (>= 115)")
    else:
        print(f"[VERDICT] CLAIM DISPUTED: Found only {total} indicators (< 115)")
        print(f"[SHORTFALL] Missing {115 - total} indicators from claimed count")