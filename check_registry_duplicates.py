#!/usr/bin/env python3
import sys, os
import pandas as pd

# تأكد أنك في جذر المشروع:
# cd /path/to/Platform3
sys.path.insert(0, os.getcwd())

from engines.ai_enhancement.registry import INDICATOR_REGISTRY

rows = []
for name, cls in INDICATOR_REGISTRY.items():
    rows.append({
        'registry_key': name,
        'module': cls.__module__,
        'class_name': cls.__name__,
        'object_id': id(cls),
    })

df = pd.DataFrame(rows)
print("=== All Indicators ===")
print(df[['registry_key','module','class_name']].to_string(index=False))

dups = df.groupby('object_id').filter(lambda g: len(g) > 1)
if not dups.empty:
    print("\n=== Duplicates Detected ===")
    print(dups[['registry_key','module','class_name']].to_string(index=False))
else:
    print("\nNo duplicates found.")
