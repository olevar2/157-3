#!/usr/bin/env python3
"""
DEBUG VERSION: Python Bridge for ComprehensiveIndicatorAdapter_67
"""

import sys
import json

print("DEBUG: Script started", file=sys.stderr, flush=True)

try:
    print("DEBUG: Importing basic modules", file=sys.stderr, flush=True)
    import numpy as np
    from datetime import datetime
    import argparse
    
    print("DEBUG: Basic modules imported", file=sys.stderr, flush=True)
    
    # Add the indicators path
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    platform3_dir = os.path.join(current_dir, '..', '..', '..', '..')
    sys.path.append(platform3_dir)
    
    print(f"DEBUG: Added path: {platform3_dir}", file=sys.stderr, flush=True)
    
    print("DEBUG: Attempting to import ComprehensiveIndicatorAdapter_67", file=sys.stderr, flush=True)
    from ComprehensiveIndicatorAdapter_67 import (
        ComprehensiveIndicatorAdapter_67,
        MarketData,
        IndicatorCategory
    )
    print("DEBUG: ComprehensiveIndicatorAdapter_67 imported successfully", file=sys.stderr, flush=True)
    
    # Simple test
    print("DEBUG: Creating adapter instance", file=sys.stderr, flush=True)
    adapter = ComprehensiveIndicatorAdapter_67()
    print("DEBUG: Adapter created successfully", file=sys.stderr, flush=True)
    
    print("DEBUG: Getting indicator names", file=sys.stderr, flush=True)
    names = adapter.get_all_indicator_names()
    print(f"DEBUG: Got {len(names)} indicator names", file=sys.stderr, flush=True)
    
    # Read input
    print("DEBUG: Reading stdin", file=sys.stderr, flush=True)
    input_text = sys.stdin.read()
    print(f"DEBUG: Read {len(input_text)} characters", file=sys.stderr, flush=True)
    
    input_data = json.loads(input_text)
    action = input_data.get('action', 'unknown')
    print(f"DEBUG: Action: {action}", file=sys.stderr, flush=True)
    
    result = {
        'debug': True,
        'action': action,
        'adapter_initialized': True,
        'indicator_count': len(names),
        'status': 'success'
    }
    
    print(json.dumps(result, indent=2))
    
except Exception as e:
    print(f"DEBUG: Exception occurred: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    
    error_result = {
        'debug': True,
        'error': str(e),
        'success': False
    }
    print(json.dumps(error_result, indent=2))
    sys.exit(1)
