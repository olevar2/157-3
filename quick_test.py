#!/usr/bin/env python3
"""
Quick test for volatility indicators
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

# Apply logger fix
import types
import logging as std_logging

# Create a proper logging.platform3_logger module without overriding logging
platform3_logger_module = types.ModuleType('platform3_logger')

class Platform3Logger:
    def __init__(self, name='Platform3', level=std_logging.INFO):
        self.logger = std_logging.getLogger(name)
        if not self.logger.handlers:
            handler = std_logging.StreamHandler()
            formatter = std_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message, extra=None): 
        self.logger.info(message)
    def warning(self, message, extra=None): 
        self.logger.warning(message)
    def error(self, message, extra=None): 
        self.logger.error(message)
    def debug(self, message, extra=None): 
        self.logger.debug(message)

class LogMetadata:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): 
            setattr(self, k, v)

def log_performance(operation, duration, **kwargs): 
    pass

platform3_logger_module.Platform3Logger = Platform3Logger
platform3_logger_module.LogMetadata = LogMetadata
platform3_logger_module.log_performance = log_performance

# Install only the platform3_logger module, not the whole logging module
sys.modules['logging.platform3_logger'] = platform3_logger_module

# Test volatility import
print("Testing volatility indicators...")
try:
    from engines.volatility import AverageTrueRange
    print('✓ AverageTrueRange import successful!')
except Exception as e:
    print(f'✗ AverageTrueRange import failed: {e}')

try:
    from engines.volatility import BollingerBands
    print('✓ BollingerBands import successful!')
except Exception as e:
    print(f'✗ BollingerBands import failed: {e}')

# Test registry
try:
    print("\nTesting full registry...")
    from engines.registry import IndicatorRegistry
    registry = IndicatorRegistry()
    
    categories = {}
    for name, info in registry.indicators.items():
        category = info.get('category', 'unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(name)
    
    print(f"Total indicators: {len(registry.indicators)}")
    print(f"Volatility indicators: {len(categories.get('volatility', []))}")
    print(f"Fibonacci indicators: {len(categories.get('fibonacci', []))}")
    
except Exception as e:
    print(f'✗ Registry test failed: {e}')
