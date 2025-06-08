#!/usr/bin/env python3
"""
AI Services Package Initialization
"""

from typing import Dict, Any

# Export components
from .model_registry import ModelRegistry, get_model_registry

__all__ = ['ModelRegistry', 'get_model_registry']
