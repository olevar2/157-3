#!/usr/bin/env python3
"""
Quick check of the final indicator count
"""

from engines.ai_enhancement.registry import validate_registry

def main():
    print("=== FINAL INDICATOR COUNT ===")
    result = validate_registry()
    
    print(f"Real indicators: {result['real_count']}")
    print(f"Aliases: {result['alias_count']}")
    print(f"Total entries: {result['total_count']}")
    
    # Target is 167, check difference
    target = 167
    actual = result['real_count']
    difference = actual - target
    
    print(f"\nTarget: {target}")
    print(f"Actual: {actual}")
    print(f"Difference: {difference:+d}")
    
    if difference == 0:
        print("✅ PERFECT! We have exactly 167 indicators!")
    elif difference > 0:
        print(f"⚠️  We have {difference} extra indicators")
    else:
        print(f"❌ We are missing {abs(difference)} indicators")

if __name__ == "__main__":
    main()