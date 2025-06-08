#!/usr/bin/env python3
"""
Count indicators in the adaptive bridge registry
"""

import re

def count_indicators():
    try:
        with open('engines/ai_enhancement/adaptive_indicator_bridge.py', 'r') as f:
            content = f.read()        # Count all indicators in the registry by finding patterns like 'indicator_name': {  
        indicators = re.findall(r"^\s+'([^']+)':\s*{", content, re.MULTILINE)
        print(f'Total indicators found: {len(indicators)}')

        # Count by category
        categories = re.findall(r'# ====== (\w+) INDICATORS \((\d+) indicators\)', content)
        category_total = sum(int(count) for _, count in categories)
        print(f'Total from category headers: {category_total}')

        print("\nCategory breakdown:")
        for category, count in categories:
            print(f'{category}: {count} indicators')
            
        # Show some sample indicators
        print(f"\nFirst 10 indicators:")
        for i, indicator in enumerate(indicators[:10]):
            print(f"{i+1}. {indicator}")
            
        if len(indicators) > 10:
            print(f"... and {len(indicators) - 10} more")
            
        return len(indicators), category_total
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0

if __name__ == "__main__":
    count_indicators()
