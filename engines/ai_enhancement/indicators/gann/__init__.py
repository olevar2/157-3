"""
Gann Analysis Indicators Module

This module contains comprehensive Gann analysis indicators implementing
W.D. Gann's geometric and temporal methodologies for financial markets.

Indicators included:
- GannAnglesIndicator: Implements Gann angle analysis with 45-degree lines
- GannSquareIndicator: Square of Nine and geometric price analysis
- GannFanIndicator: Fan lines from significant price points
- GannTimeCycleIndicator: Time-based cycle analysis
- GannPriceTimeIndicator: Price-time relationship analysis

All indicators follow Platform3 StandardIndicatorInterface pattern.
"""

from .gann_angles_indicator import GannAnglesIndicator
from .gann_fan_indicator import GannFanIndicator
from .gann_price_time_indicator import GannPriceTimeIndicator
from .gann_square_indicator import GannSquareIndicator
from .gann_time_cycle_indicator import GannTimeCycleIndicator

__all__ = [
    "GannAnglesIndicator",
    "GannSquareIndicator",
    "GannFanIndicator",
    "GannTimeCycleIndicator",
    "GannPriceTimeIndicator",
]

# Export dictionary for registry integration
GANN_INDICATORS = {
    "gann_angles": GannAnglesIndicator,
    "gann_square": GannSquareIndicator,
    "gann_fan": GannFanIndicator,
    "gann_time_cycle": GannTimeCycleIndicator,
    "gann_price_time": GannPriceTimeIndicator,
}
