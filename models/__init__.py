"""
Platform3 Models Package

This package provides a unified interface to all AI/ML models in the platform.
Models are self-contained with their code, weights, and configuration.
"""

from .registry import ModelRegistry, BaseModel, ModelMetadata, get_model_registry

__all__ = [
    'ModelRegistry',
    'BaseModel', 
    'ModelMetadata',
    'get_model_registry'
]

# Version
__version__ = '1.0.0'
