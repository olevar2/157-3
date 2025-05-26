"""
Algorithmic Arbitrage Engine Module
Advanced arbitrage detection and execution for forex trading platform

This module provides comprehensive arbitrage capabilities including:
- Spatial arbitrage detection across multiple brokers
- Triangular arbitrage identification
- Real-time price comparison and analysis
- Risk-managed arbitrage execution

Author: Platform3 Development Team
Version: 1.0.0
"""

from .ArbitrageEngine import (
    ArbitrageEngine,
    ArbitrageOpportunity,
    ArbitrageConfig,
    ArbitrageType,
    OpportunityStatus,
    PriceQuote
)

from .PriceComparator import (
    PriceComparator,
    PriceComparison,
    PriceStatistics
)

__all__ = [
    # Main arbitrage engine
    'ArbitrageEngine',
    'ArbitrageOpportunity',
    'ArbitrageConfig',
    'ArbitrageType',
    'OpportunityStatus',
    'PriceQuote',
    
    # Price comparison
    'PriceComparator',
    'PriceComparison',
    'PriceStatistics'
]

__version__ = "1.0.0"
__author__ = "Platform3 Development Team"
