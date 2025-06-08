"""
Update registry.py to include ALL 157 indicators
This will create the complete registry with all stubs
"""

# Read current registry to see the structure
with open('engines/ai_enhancement/registry.py', 'r') as f:
    current_registry = f.read()

# Generate import statements for all new stub files
import_statements = []
import_statements.append("# Complete 157 Indicator Imports - Generated Stubs")

categories = [
    'momentum', 'fibonacci', 'divergence', 'statistical', 'volatility', 
    'gann', 'fractal', 'trend', 'volume', 'pattern', 'cycle', 
    'sentiment', 'ml_advanced', 'elliott_wave', 'core_trend', 
    'pivot', 'core_momentum'
]

# Generate try/except import blocks for each category
for category in categories:
    import_statements.append(f'''
try:
    from engines.ai_enhancement.{category}_indicators_complete import *
except ImportError as e:
    print(f"Warning: Could not import {category} complete indicators: {{e}}")''')

imports_section = "\\n".join(import_statements)

# Generate registry entries
# Load the complete indicator data to get the mapping
import json
with open('complete_157_indicator_discovery_20250607_034303.json', 'r') as f:
    data = json.load(f)

# Collect all indicators
all_indicators = []
for category, indicators in data['indicators_by_category'].items():
    for indicator in indicators:
        if isinstance(indicator, dict) and 'name' in indicator:
            all_indicators.append(indicator['name'])

# Remove duplicates and sort
unique_indicators = sorted(set(all_indicators))

print(f"Preparing to add {len(unique_indicators)} indicators to registry")

# Generate registry entries
registry_entries = []
registry_entries.append("    # Complete 157 Indicator Registry - All Platform3 Indicators")

for indicator in unique_indicators:
    registry_entries.append(f"    '{indicator.lower()}': {indicator},")

registry_section = "\\n".join(registry_entries)

# Create the complete registry update
complete_registry_addition = f'''

{imports_section}

# Additional registry entries for complete 157 indicator support
ADDITIONAL_INDICATORS = {{
{registry_section}
}}

# Merge with existing registry
INDICATOR_REGISTRY.update(ADDITIONAL_INDICATORS)
'''

# Save this as a patch file first
with open('registry_157_complete_patch.py', 'w') as f:
    f.write(complete_registry_addition)

print("Generated complete registry patch!")
print(f"The patch will add {len(unique_indicators)} indicators")
print("Next: Apply this patch to registry.py")
