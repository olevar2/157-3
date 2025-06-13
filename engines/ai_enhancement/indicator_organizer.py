"""
Platform3 Indicator Organization System
Provides structured access to indicators with performance optimization
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum


class IndicatorCategory(Enum):
    """Categorize indicators by their primary function"""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"
    GANN = "gann"
    FRACTAL = "fractal"
    OSCILLATOR = "oscillator"
    STATISTICAL = "statistical"
    MICROSTRUCTURE = "microstructure"


@dataclass
class IndicatorSpec:
    """Specification for an indicator"""

    name: str
    category: IndicatorCategory
    implementation_class: Any
    primary_use: str
    performance_weight: float = 1.0  # For optimization
    computational_cost: str = "medium"  # low, medium, high
    agent_affinity: List[str] = None  # Which agents benefit most


class IndicatorOrganizer:
    """
    Organizes indicators by category and optimizes access patterns
    """

    def __init__(self):
        self.categories: Dict[IndicatorCategory, List[IndicatorSpec]] = {}
        self.indicator_specs: Dict[str, IndicatorSpec] = {}
        self._build_organization()

    def _build_organization(self):
        """Build the organized structure"""
        from engines.ai_enhancement.registry import INDICATOR_REGISTRY
        from engines.ai_enhancement.real_gann_indicators import GANN_INDICATORS

        # Categorize Gann indicators
        gann_specs = [
            IndicatorSpec(
                name="gann_angles_calculator",
                category=IndicatorCategory.GANN,
                implementation_class=GANN_INDICATORS["gann_angles_calculator"],
                primary_use="Support/resistance angle calculations",
                performance_weight=1.2,
                computational_cost="medium",
                agent_affinity=["PATTERN_MASTER", "MARKET_MICROSTRUCTURE_GENIUS"],
            ),
            IndicatorSpec(
                name="gann_grid",
                category=IndicatorCategory.GANN,
                implementation_class=GANN_INDICATORS["gann_grid"],
                primary_use="Grid-based price level analysis",
                performance_weight=1.0,
                computational_cost="low",
                agent_affinity=["PATTERN_MASTER"],
            ),
            IndicatorSpec(
                name="gann_pattern_detector",
                category=IndicatorCategory.PATTERN,
                implementation_class=GANN_INDICATORS["gann_pattern_detector"],
                primary_use="Gann-specific pattern recognition",
                performance_weight=1.3,
                computational_cost="high",
                agent_affinity=["PATTERN_MASTER"],
            ),
            IndicatorSpec(
                name="gann_fan_lines",
                category=IndicatorCategory.GANN,
                implementation_class=GANN_INDICATORS["gann_fan_lines"],
                primary_use="Fan line trend analysis",
                performance_weight=1.1,
                computational_cost="medium",
                agent_affinity=["PATTERN_MASTER", "SESSION_EXPERT"],
            ),
            IndicatorSpec(
                name="gann_square_of_nine",
                category=IndicatorCategory.GANN,
                implementation_class=GANN_INDICATORS["gann_square_of_nine"],
                primary_use="Square of Nine price calculations",
                performance_weight=1.0,
                computational_cost="low",
                agent_affinity=["PATTERN_MASTER"],
            ),
            IndicatorSpec(
                name="gann_time_cycles",
                category=IndicatorCategory.GANN,
                implementation_class=GANN_INDICATORS["gann_time_cycles"],
                primary_use="Time cycle analysis",
                performance_weight=1.4,
                computational_cost="medium",
                agent_affinity=["PATTERN_MASTER", "MARKET_MICROSTRUCTURE_GENIUS"],
            ),
            IndicatorSpec(
                name="price_time_relationships",
                category=IndicatorCategory.GANN,
                implementation_class=GANN_INDICATORS["price_time_relationships"],
                primary_use="Price-time correlation analysis",
                performance_weight=1.3,
                computational_cost="medium",
                agent_affinity=["PATTERN_MASTER", "MARKET_MICROSTRUCTURE_GENIUS"],
            ),
        ]

        # Store specs
        for spec in gann_specs:
            if spec.category not in self.categories:
                self.categories[spec.category] = []
            self.categories[spec.category].append(spec)
            self.indicator_specs[spec.name] = spec

    def get_indicators_by_category(self, category: IndicatorCategory) -> List[str]:
        """Get all indicator names in a category"""
        return [spec.name for spec in self.categories.get(category, [])]

    def get_optimized_indicators_for_agent(self, agent_type: str) -> List[str]:
        """Get indicators optimized for a specific agent"""
        result = []
        for spec in self.indicator_specs.values():
            if spec.agent_affinity and agent_type in spec.agent_affinity:
                result.append(spec.name)

        # Sort by performance weight (descending)
        return sorted(
            result,
            key=lambda name: self.indicator_specs[name].performance_weight,
            reverse=True,
        )

    def get_low_cost_indicators(self) -> List[str]:
        """Get indicators with low computational cost"""
        return [
            spec.name
            for spec in self.indicator_specs.values()
            if spec.computational_cost == "low"
        ]

    def get_indicator_info(self, name: str) -> Optional[IndicatorSpec]:
        """Get detailed info about an indicator"""
        return self.indicator_specs.get(name)


# Global organizer instance
INDICATOR_ORGANIZER = IndicatorOrganizer()

# Export for use
__all__ = [
    "IndicatorOrganizer",
    "IndicatorCategory",
    "IndicatorSpec",
    "INDICATOR_ORGANIZER",
]
