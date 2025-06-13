"""
Platform3 - AI Trading System for Humanitarian Impact

A comprehensive algorithmic trading platform powered by artificial intelligence,
designed to generate sustainable returns for humanitarian causes worldwide.

Key Features:
- 9 Specialized AI Trading Agents
- 115+ Technical Indicators
- Real-time Market Analysis
- Risk Management & Portfolio Optimization
- Humanitarian Impact Tracking
- 24/7 Autonomous Operation

Modules:
- ai_platform: Core AI services and coordination
- shared: Common utilities and base classes
- services: Microservices architecture
- config: Configuration management
- engines: Trading execution engines
"""

__version__ = "1.0.0"
__author__ = "Platform3 Team"
__email__ = "info@platform3.ai"
__license__ = "MIT"

# Core imports for easy access
from ai_platform.coordination.engine.platform3_engine import Platform3TradingEngine
from shared.ai_model_base import EnhancedAIModelBase, AIModelPerformanceMonitor
from shared.logging.platform3_logger import Platform3Logger

# Configuration
from config.production_config_manager import ProductionConfigManager

# Version information
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version() -> str:
    """Get the current Platform3 version."""
    return __version__

def get_engine():
    """Get a configured Platform3 trading engine instance."""
    return Platform3TradingEngine()

__all__ = [
    # Core classes
    "Platform3TradingEngine",
    "EnhancedAIModelBase", 
    "AIModelPerformanceMonitor",
    "Platform3Logger",
    "ProductionConfigManager",
    
    # Utility functions
    "get_version",
    "get_engine",
    
    # Version info
    "VERSION_INFO",
    "__version__"
]