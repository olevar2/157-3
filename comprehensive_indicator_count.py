#!/usr/bin/env python3
"""
Comprehensive Indicator Count Script
====================================

This script systematically counts all indicator implementations in the Platform3 engines directory
to determine the true state of indicator coverage vs documented claims.

Purpose: Investigate discrepancy between claimed 115+ indicators and validation results
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

def count_indicators_in_directory(engines_path):
    """Count all indicator implementations across all categories"""
    
    # Initialize results
    category_counts = defaultdict(list)
    total_indicators = 0
    
    # Define indicator categories (based on directory structure)
    categories = [
        'ai_enhancement', 'core_momentum', 'core_trend', 'cycle', 'divergence',
        'elliott_wave', 'fibonacci', 'fractal', 'gann', 'momentum', 'pattern', 
        'performance', 'pivot', 'sentiment', 'statistical', 'trend', 
        'validation', 'volatility', 'volume'
    ]
    
    # Pattern to find indicator classes
    class_pattern = re.compile(r'class\s+(\w+)\s*\(\s*(?:.*?)?Indicator.*?\):')
    
    print("="*80)
    print("PLATFORM3 COMPREHENSIVE INDICATOR COUNT")
    print("="*80)
    print()
    
    for category in categories:
        category_path = Path(engines_path) / category
        
        if not category_path.exists():
            print(f"[WARNING] Category '{category}' not found")
            continue
            
        print(f"[INFO] Analyzing category: {category}")
        print("-" * 50)
        
        category_indicators = []
        
        # Get all Python files in category (excluding backups and __pycache__)
        python_files = [
            f for f in category_path.glob('*.py') 
            if not f.name.startswith('__') 
            and '.backup' not in f.name
            and '_old' not in f.name
            and '_new' not in f.name
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find all indicator classes in file
                matches = class_pattern.findall(content)
                
                if matches:
                    for class_name in matches:
                        # Skip config/helper classes
                        if 'Config' not in class_name and 'Helper' not in class_name:
                            category_indicators.append({
                                'name': class_name,
                                'file': py_file.name,
                                'path': str(py_file)
                            })
                            print(f"  [OK] {class_name} (in {py_file.name})")
                        
            except Exception as e:
                print(f"  [ERROR] Error reading {py_file.name}: {e}")
        
        # Store results for this category
        category_counts[category] = category_indicators
        category_total = len(category_indicators)
        total_indicators += category_total
        
        print(f"  [SUMMARY] Category total: {category_total} indicators")
        print()
    
    # Summary report
    print("="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print()
    
    for category, indicators in category_counts.items():
        print(f"{category:20} : {len(indicators):3d} indicators")
    
    print("-" * 50)
    print(f"{'TOTAL':20} : {total_indicators:3d} indicators")
    print()
    
    # Save detailed report
    report = {
        'timestamp': str(datetime.now()),
        'total_indicators': total_indicators,
        'categories': dict(category_counts),
        'summary': {category: len(indicators) for category, indicators in category_counts.items()}
    }
    
    report_file = Path(engines_path).parent / 'comprehensive_indicator_count_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"[REPORT] Detailed report saved to: {report_file}")
    
    return total_indicators, category_counts

if __name__ == "__main__":
    from datetime import datetime
    
    # Define engines path
    engines_path = r"d:\MD\Platform3\engines"
    
    if not os.path.exists(engines_path):
        print(f"[ERROR] Engines directory not found: {engines_path}")
        exit(1)
    
    # Run the count
    total, categories = count_indicators_in_directory(engines_path)
    
    print("="*80)
    print("INVESTIGATION CONCLUSION")
    print("="*80)
    print()
    print(f"[INVESTIGATION] TOTAL INDICATORS FOUND: {total}")
    print(f"[DOCUMENTATION] CLAIMED IN DOCUMENTATION: 115+")
    print(f"[ANALYSIS] DISCREPANCY: {abs(total - 115)} indicators")
    print()
    
    if total >= 115:
        print("[CONCLUSION] CLAIM VERIFIED: The project does indeed have 115+ indicators implemented")
    else:
        print("[CONCLUSION] CLAIM DISPUTED: Fewer indicators found than documented")
        print("   - Possible causes:")
        print("     * Some indicators may be in different locations")
        print("     * Some indicators may be broken and not loading")
        print("     * Documentation may be outdated")
        print("     * Validation system may have bugs")