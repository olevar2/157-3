#!/usr/bin/env python3
"""
Dynamic Indicator Registry Loader - REDIRECTOR TO AUTHORITATIVE REGISTRY
This module is now a redirector that simply returns the authoritative registry
from engines.ai_enhancement.registry
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Type, Tuple, Any, Optional

# Make sure we can import from engines
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

# Import authoritative registry
from engines.ai_enhancement.registry import INDICATOR_REGISTRY, get_indicator_categories

# Export the function to make it importable
__all__ = ["load_all_indicators", "load_all_working_indicators"]


def load_all_working_indicators() -> Dict[str, Any]:
    """
    Load all working indicators from the authoritative registry

    Returns:
        Dict[str, Any]: Dictionary of all working indicators from the authoritative registry
    """
    return INDICATOR_REGISTRY


def load_all_indicators() -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Load all working indicators from the authoritative registry

    Returns:
        Tuple containing:
        - Dictionary of indicator classes from the authoritative registry
        - Dictionary of categories with lists of indicator names
    """
    categories = get_indicator_categories()
    return INDICATOR_REGISTRY, categories

    return indicators, categories


if __name__ == "__main__":
    print("=" * 60)
    print("PLATFORM3 INDICATOR REGISTRY TEST")
    print("=" * 60)

    indicators = load_all_working_indicators()
    indicators_count = len(indicators)

    # Get indicator categories
    categories = get_indicator_categories()
    categories_count = len(categories)

    print(
        f"\nSUCCESS: Loaded {indicators_count} indicators from authoritative registry"
    )
    print(f"Categories: {categories_count}")

    for category, indicator_list in sorted(categories.items()):
        if indicator_list:
            print(f"{category}: {len(indicator_list)} indicators")

    print(f"\nTotal functional indicators: {indicators_count}")

    print("\nNOTE: This module is now a redirector to engines.ai_enhancement.registry")
    print(
        "which is the single authoritative source for all callable indicator objects."
    )
