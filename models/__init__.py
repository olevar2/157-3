"""
Platform3 Models Package

This package provides a unified interface to all AI/ML models in the platform.
Models are self-contained with their code, weights, and configuration.
"""

# Note: ModelRegistry temporarily disabled due to missing registry.py
# from .registry import ModelRegistry, BaseModel, ModelMetadata, get_model_registry
from .market_data import OHLCV, MarketData, PriceData, OHLCVList, create_ohlcv, generate_test_data

__all__ = [
    # Registry components temporarily disabled
    # 'ModelRegistry',
    # 'BaseModel', 
    # 'ModelMetadata',
    # 'get_model_registry',
    # Market data models
    'OHLCV',
    'MarketData',
    'PriceData', 
    'OHLCVList',
    'create_ohlcv',
    'generate_test_data'
]

# Version
__version__ = '1.0.0'
