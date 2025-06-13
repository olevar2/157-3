"""
Smart Agent-Indicator Mapping Service
Provides optimized indicator assignments based on agent capabilities and market conditions
"""

from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

from engines.ai_enhancement.genius_agents import GeniusAgentType
from engines.ai_enhancement.indicator_organizer import (
    INDICATOR_ORGANIZER,
    IndicatorCategory,
)


@dataclass
class AgentProfile:
    """Profile defining an agent's characteristics and optimal indicator set"""

    agent_type: GeniusAgentType
    specialty: str
    primary_categories: List[IndicatorCategory]
    secondary_categories: List[IndicatorCategory]
    max_indicators: int
    computation_preference: str  # "fast", "balanced", "comprehensive"


class SmartAgentMappingService:
    """
    Intelligent service that maps indicators to agents based on:
    1. Agent specialization
    2. Indicator performance characteristics
    3. Market conditions
    4. Computational efficiency
    """

    def __init__(self):
        self.agent_profiles = self._build_agent_profiles()
        self.base_mappings = self._build_base_mappings()

    def _build_agent_profiles(self) -> Dict[GeniusAgentType, AgentProfile]:
        """Define optimized profiles for each agent"""
        return {
            GeniusAgentType.PATTERN_MASTER: AgentProfile(
                agent_type=GeniusAgentType.PATTERN_MASTER,
                specialty="Pattern recognition and Gann analysis",
                primary_categories=[
                    IndicatorCategory.PATTERN,
                    IndicatorCategory.GANN,
                    IndicatorCategory.FRACTAL,
                ],
                secondary_categories=[
                    IndicatorCategory.TREND,
                    IndicatorCategory.OSCILLATOR,
                ],
                max_indicators=52,  # Reduced from 60 due to alias cleanup
                computation_preference="comprehensive",
            ),
            GeniusAgentType.SESSION_EXPERT: AgentProfile(
                agent_type=GeniusAgentType.SESSION_EXPERT,
                specialty="Session-based trading and timing",
                primary_categories=[
                    IndicatorCategory.TREND,
                    IndicatorCategory.MOMENTUM,
                ],
                secondary_categories=[
                    IndicatorCategory.GANN
                ],  # Limited Gann for timing
                max_indicators=25,
                computation_preference="balanced",
            ),
            GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: AgentProfile(
                agent_type=GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS,
                specialty="Microstructure and institutional flow",
                primary_categories=[
                    IndicatorCategory.MICROSTRUCTURE,
                    IndicatorCategory.VOLUME,
                ],
                secondary_categories=[IndicatorCategory.GANN],  # Time-price analysis
                max_indicators=35,
                computation_preference="fast",
            ),
            GeniusAgentType.RISK_GENIUS: AgentProfile(
                agent_type=GeniusAgentType.RISK_GENIUS,
                specialty="Risk assessment and management",
                primary_categories=[
                    IndicatorCategory.VOLATILITY,
                    IndicatorCategory.STATISTICAL,
                ],
                secondary_categories=[IndicatorCategory.TREND],
                max_indicators=35,
                computation_preference="balanced",
            ),
            GeniusAgentType.EXECUTION_EXPERT: AgentProfile(
                agent_type=GeniusAgentType.EXECUTION_EXPERT,
                specialty="Order execution and volume analysis",
                primary_categories=[
                    IndicatorCategory.VOLUME,
                    IndicatorCategory.MICROSTRUCTURE,
                ],
                secondary_categories=[IndicatorCategory.MOMENTUM],
                max_indicators=40,
                computation_preference="fast",
            ),
        }

    def _build_base_mappings(self) -> Dict[GeniusAgentType, Dict[str, List[str]]]:
        """Build optimized base mappings with cleaned Gann indicators"""
        return {
            GeniusAgentType.PATTERN_MASTER: {
                "gann_indicators": [
                    "gann_angles_calculator",  # Primary for support/resistance
                    "gann_grid",  # Grid analysis
                    "gann_pattern_detector",  # Core pattern recognition
                    "gann_fan_lines",  # Trend analysis
                    "gann_square_of_nine",  # Price levels
                    "gann_time_cycles",  # Timing analysis
                    "price_time_relationships",  # Correlation analysis
                ],
                "pattern_indicators": [
                    "abandoned_baby_signal",
                    "belt_hold_type",
                    "dark_cloud_type",
                    "doji_type",
                    "doji_type_fixed",
                    "elliott_wave_type",
                    "engulfing_type",
                    "engulfing_type_fixed",
                    "fibonacci_pattern_type",
                    "hammer_type",
                    "harami_type",
                    "harmonic_point",
                    "high_wave_candle_pattern",
                    "inverted_hammer_shooting_star_pattern",
                    "japanese_candlestick_pattern_type",
                    "kicking_signal",
                    "long_legged_doji_pattern",
                    "marubozu_pattern",
                    "matching_signal",
                    "piercing_line_type",
                    "soldiers_signal",
                    "spinning_top_pattern",
                    "star_signal",
                    "three_inside_signal",
                    "three_line_strike_signal",
                    "three_outside_signal",
                    "tweezer_type",
                    "head_shoulders_pattern",
                    "double_top_bottom_pattern",
                ],
                "fractal_indicators": [
                    "fractal_efficiency_ratio",
                    "attractor_point",
                    "fractal_channel_indicator",
                    "fractal_chaos_oscillator",
                    "fractal_correlation_dimension",
                    "fractal_dimension_calculator",
                    "fractal_energy_indicator",
                    "fractal_market_hypothesis",
                    "fractal_market_profile",
                    "fractal_wave_type",
                    "hurst_exponent_calculator",
                    "self_similarity_signal",
                ],
            },
            GeniusAgentType.SESSION_EXPERT: {
                "gann_indicators": [
                    "gann_fan_lines"  # Only fan lines for timing (was gann_angle)
                ],
                "trend_indicators": [
                    "trend_signal",
                    "alligator_trend",
                    "cycle_period_identification",
                    "dominant_cycle_analysis",
                    "fisher_signal_type",
                    "market_regime",
                    "hurst_exponent",
                    "market_regime_detection",
                    "phase_analysis",
                ],
                "timing_indicators": [
                    "wave_type",
                    "wave_structure",
                    "time_cycle",
                    "pivot_type",
                ],
            },
            GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS: {
                "gann_indicators": [
                    "gann_time_cycles",  # Timing institutional moves
                    "price_time_relationships",  # Price-time correlation in microstructure
                    "gann_angles_calculator",  # Support/resistance in order flow
                ],
                "microstructure_indicators": [
                    "market_microstructure_signal",
                    "liquidity_flow_signal",
                    "institutional_flow_signal",
                    "order_flow_analysis",
                    "bid_ask_spread_analysis",
                    "market_depth_analysis",
                ],
                "volume_indicators": [
                    "volume_profile",
                    "volume_weighted_average_price",
                    "accumulation_distribution_signal",
                    "chaikin_money_flow_signal",
                ],
            },
        }

    def get_optimized_indicators(
        self, agent_type: GeniusAgentType, market_condition: str = "normal"
    ) -> List[str]:
        """Get optimized indicator list for an agent based on current conditions"""
        profile = self.agent_profiles.get(agent_type)
        if not profile:
            return []

        base_mapping = self.base_mappings.get(agent_type, {})
        indicators = []

        # Add all indicators from base mapping
        for category_indicators in base_mapping.values():
            indicators.extend(category_indicators)

        # Remove duplicates while preserving order
        seen = set()
        unique_indicators = []
        for indicator in indicators:
            if indicator not in seen:
                seen.add(indicator)
                unique_indicators.append(indicator)

        return unique_indicators[: profile.max_indicators]

    def get_gann_indicators_for_agent(self, agent_type: GeniusAgentType) -> List[str]:
        """Get only Gann indicators for a specific agent"""
        base_mapping = self.base_mappings.get(agent_type, {})
        return base_mapping.get("gann_indicators", [])

    def validate_agent_mapping(self, agent_type: GeniusAgentType) -> Dict[str, any]:
        """Validate that an agent's mapping is optimal"""
        profile = self.agent_profiles.get(agent_type)
        if not profile:
            return {"valid": False, "error": "Unknown agent type"}

        indicators = self.get_optimized_indicators(agent_type)
        gann_indicators = self.get_gann_indicators_for_agent(agent_type)

        return {
            "valid": True,
            "agent_type": agent_type.value,
            "total_indicators": len(indicators),
            "gann_indicators": len(gann_indicators),
            "gann_list": gann_indicators,
            "within_limits": len(indicators) <= profile.max_indicators,
            "specialty": profile.specialty,
        }


# Global service instance
SMART_MAPPING_SERVICE = SmartAgentMappingService()

# Export for use
__all__ = ["SmartAgentMappingService", "AgentProfile", "SMART_MAPPING_SERVICE"]
