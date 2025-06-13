# --- START OF FILE __init__.py ---

"""
Platform3 AI Enhancement Engine
================================

This package is the core of the AI-powered trading analysis system. It
integrates 9 specialized "Genius Agents" with a library of 167 advanced
technical indicators.

The primary entry point for using this engine is the `GeniusAgentIntegration`
class, which orchestrates the entire analysis process.

Key Exports:
- GeniusAgentIntegration: The main class for running analyses.
- GeniusAgentType: An enum for identifying the 9 specific agents.
- INDICATOR_REGISTRY: A dictionary of all 167 available indicators.
- get_indicator: A function to retrieve a specific indicator callable.
- IndicatorBase: The foundational class for all indicators.
- BasePatternEngine: The foundational class for all pattern indicators.
"""

__version__ = "3.0.0"
__author__ = "Platform3 AI Team"
__purpose__ = "Humanitarian Profit Generation Through Mathematical Precision"

# --- Core Foundational Classes ---
from .ai_enhancement.indicator_base import (
    IndicatorBase,
    TechnicalIndicator,
    IndicatorConfig,
    IndicatorResult,
    IndicatorSignal,
    MarketData,
    SignalType,
    IndicatorType,
    TimeFrame,
    IndicatorStatus,
)
from .base_pattern import (
    BasePatternEngine,
    PatternSignal,
    PatternType,
    PatternStrength,
)

# --- Central Indicator Registry and Accessor ---
# This is the single source of truth for all 167 indicators.
from .ai_enhancement.registry import (
    INDICATOR_REGISTRY,
    get_indicator,
    validate_registry,
    GeniusAgentType,
)

# --- Core Integration and Orchestration Components ---
from .ai_enhancement.adaptive_indicator_bridge import AdaptiveIndicatorBridge
from .ai_enhancement.genius_agent_integration import GeniusAgentIntegration


# --- Public API for the AI Enhancement Engine ---
# This defines what other parts of the platform can import from this module.
__all__ = [
    # Main Orchestration
    "GeniusAgentIntegration",
    
    # Agent Identification
    "GeniusAgentType",
    
    # Indicator Access
    "INDICATOR_REGISTRY",
    "get_indicator",
    "validate_registry",
    
    # Foundational Base Classes
    "IndicatorBase",
    "BasePatternEngine",
    
    # Core Data Structures
    "IndicatorSignal",
    "PatternSignal",
    "MarketData",
    "IndicatorResult",
    "IndicatorConfig",
    
    # Core Enums
    "SignalType",
    "IndicatorType",
    "PatternType",
    "PatternStrength",
    "TimeFrame",
    "IndicatorStatus",

    # The bridge is used internally by the integration layer
    "AdaptiveIndicatorBridge",
]

# --- Module-level validation on import ---
import logging
logger = logging.getLogger(__name__)

try:
    count = validate_registry()
    if count == 121:
        logger.info(f"AI Enhancement Engine initialized successfully. All {count} indicators are available.")
    else:
        logger.warning(f"AI Enhancement Engine initialized with {count} indicators (expected 121).")
except Exception as e:
    logger.exception(f"A critical error occurred during AI Enhancement Engine initialization: {e}")

# --- END OF FILE __init__.py ---