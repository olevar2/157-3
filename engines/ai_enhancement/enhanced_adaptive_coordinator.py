"""
Enhanced Adaptive Indicator Coordinator
Manages all 160 indicators for optimal AI agent utilization
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import indicator modules
from engines.fractal import *
from engines.pattern import *
from engines.momentum import *
from engines.trend import *
from engines.volatility import *
from engines.volume import *
from engines.statistical import *
from engines.fibonacci import *
from engines.gann import *
from engines.elliott_wave import *
from engines.ml_advanced import *


@dataclass
class IndicatorConfig:
    """Configuration for individual indicators"""

    name: str
    category: str
    weight: float
    timeframes: List[str]
    required_periods: int
    agent_affinity: List[str]  # Which agents prefer this indicator
    market_conditions: List[str]  # When this indicator works best


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"


class EnhancedAdaptiveCoordinator:
    """Advanced coordinator for all 160 indicators with AI agent optimization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicator_configs = self._initialize_indicator_configs()
        self.agent_preferences = self._initialize_agent_preferences()
        self.performance_tracker = IndicatorPerformanceTracker()
        self.market_regime_detector = MarketRegimeDetector()

    def _initialize_indicator_configs(self) -> Dict[str, IndicatorConfig]:
        """Initialize configuration for all 160 indicators"""
        configs = {}

        # Fractal Geometry Indicators (18)
        fractal_indicators = [
            "chaos_theory_indicators",
            "fractal_breakout",
            "fractal_channel",
            "fractal_chaos_oscillator",
            "fractal_correlation_dimension",
            "fractal_dimension_calculator",
            "fractal_efficiency_ratio",
            "fractal_energy_indicator",
            "fractal_market_hypothesis",
            "fractal_market_profile",
            "fractal_momentum_oscillator",
            "fractal_volume_analysis",
            "fractal_wave_counter",
            "frama",
            "mandelbrot_fractal",
            "mfdfa",
            "self_similarity_detector",
        ]

        for indicator in fractal_indicators:
            configs[indicator] = IndicatorConfig(
                name=indicator,
                category="fractal_geometry",
                weight=0.7,
                timeframes=["15m", "1h", "4h"],
                required_periods=50,
                agent_affinity=["pattern_master", "decision_master"],
                market_conditions=["volatile", "breakout"],
            )

        # Pattern Recognition Indicators (30)
        pattern_indicators = [
            "abandoned_baby_pattern",
            "belt_hold_pattern",
            "dark_cloud_cover_pattern",
            "doji_recognition",
            "elliott_wave_analysis",
            "engulfing_pattern",
            "hammer_hanging_man",
            "harami_pattern",
            "harmonic_pattern_detector",
            "high_wave_candle",
            "inverted_hammer_shooting_star",
            "japanese_candlestick_patterns",
            "kicking_pattern",
            "long_legged_doji",
            "marubozu",
            "matching_pattern",
            "piercing_line_pattern",
            "soldiers_pattern",
            "spinning_top",
            "star_pattern",
            "three_inside_pattern",
            "three_line_strike_pattern",
            "three_outside_pattern",
            "tweezer_patterns",
        ]

        for indicator in pattern_indicators:
            configs[indicator] = IndicatorConfig(
                name=indicator,
                category="pattern_recognition",
                weight=0.8,
                timeframes=["5m", "15m", "1h"],
                required_periods=20,
                agent_affinity=["pattern_master", "execution_expert"],
                market_conditions=["trending", "breakout"],
            )

        # Momentum Indicators (19)
        momentum_indicators = [
            "awesome_oscillator",
            "cci",
            "chande_momentum_oscillator",
            "commodity_channel_index",
            "detrended_price_oscillator",
            "know_sure_thing",
            "macd",
            "mfi",
            "momentum",
            "percentage_price_oscillator",
            "roc",
            "rsi",
            "stochastic",
            "trix",
            "true_strength_index",
            "ultimate_oscillator",
            "williams_r",
        ]

        for indicator in momentum_indicators:
            configs[indicator] = IndicatorConfig(
                name=indicator,
                category="momentum",
                weight=0.9,
                timeframes=["5m", "15m", "30m", "1h"],
                required_periods=14,
                agent_affinity=["execution_expert", "decision_master"],
                market_conditions=["trending", "ranging"],
            )

        # Volume Indicators (18)
        volume_indicators = [
            "accumulation_distribution",
            "chaikin_money_flow",
            "ease_of_movement",
            "force_index",
            "klinger_oscillator",
            "negative_volume_index",
            "obv",
            "OrderFlowImbalance",
            "positive_volume_index",
            "price_volume_rank",
            "SmartMoneyIndicators",
            "TickVolumeIndicators",
            "VolumeProfiles",
            "VolumeSpreadAnalysis",
            "volume_oscillator",
            "volume_price_trend",
            "volume_rate_of_change",
            "vwap",
        ]

        for indicator in volume_indicators:
            configs[indicator] = IndicatorConfig(
                name=indicator,
                category="volume",
                weight=0.85,
                timeframes=["5m", "15m", "1h"],
                required_periods=20,
                agent_affinity=["execution_expert", "market_microstructure_genius"],
                market_conditions=["trending", "breakout"],
            )

        # Continue for all other categories...
        # (Statistical, Fibonacci, Gann, etc.)

        return configs

    def _initialize_agent_preferences(self) -> Dict[str, Dict]:
        """Initialize indicator preferences for each genius agent"""
        return {
            "risk_genius": {
                "primary_categories": ["volatility", "statistical"],
                "preferred_indicators": [
                    "atr",
                    "bollinger_bands",
                    "standard_deviation",
                    "beta_coefficient",
                    "variance_ratio",
                ],
                "weight_multiplier": 1.2,
                "max_indicators": 25,
            },
            "session_expert": {
                "primary_categories": ["volume", "fibonacci"],
                "preferred_indicators": [
                    "volume_profile",
                    "vwap",
                    "fibonacci_time_zones",
                ],
                "weight_multiplier": 1.1,
                "max_indicators": 15,
            },
            "pattern_master": {
                "primary_categories": ["pattern_recognition", "fractal_geometry"],
                "preferred_indicators": [
                    "doji_recognition",
                    "fractal_breakout",
                    "harmonic_pattern_detector",
                ],
                "weight_multiplier": 1.3,
                "max_indicators": 35,
            },
            "execution_expert": {
                "primary_categories": ["momentum", "volume"],
                "preferred_indicators": ["rsi", "macd", "obv", "vwap"],
                "weight_multiplier": 1.25,
                "max_indicators": 20,
            },
            "pair_specialist": {
                "primary_categories": ["statistical", "momentum"],
                "preferred_indicators": [
                    "correlation_coefficient",
                    "cointegration",
                    "rsi",
                ],
                "weight_multiplier": 1.0,
                "max_indicators": 18,
            },
            "decision_master": {
                "primary_categories": ["all"],
                "preferred_indicators": ["custom_ai_composite_indicator"],
                "weight_multiplier": 1.4,
                "max_indicators": 25,
            },
            "ai_model_coordinator": {
                "primary_categories": ["ml_advanced", "statistical"],
                "preferred_indicators": [
                    "custom_ai_composite_indicator",
                    "linear_regression",
                ],
                "weight_multiplier": 1.3,
                "max_indicators": 15,
            },
            "market_microstructure_genius": {
                "primary_categories": ["volume", "pattern_recognition"],
                "preferred_indicators": ["OrderFlowImbalance", "SmartMoneyIndicators"],
                "weight_multiplier": 1.2,
                "max_indicators": 20,
            },
            "sentiment_integration_genius": {
                "primary_categories": ["momentum", "statistical"],
                "preferred_indicators": ["rsi", "williams_r", "correlation_analysis"],
                "weight_multiplier": 1.1,
                "max_indicators": 15,
            },
        }

    def get_optimal_indicators_for_agent(
        self, agent_name: str, market_data: Dict, market_regime: MarketRegime
    ) -> List[str]:
        """Get optimal indicator set for specific agent based on current market conditions"""

        if agent_name not in self.agent_preferences:
            self.logger.warning(f"Unknown agent: {agent_name}")
            return []

        agent_config = self.agent_preferences[agent_name]
        optimal_indicators = []

        # Get indicators by category preference
        for category in agent_config["primary_categories"]:
            if category == "all":
                category_indicators = list(self.indicator_configs.keys())
            else:
                category_indicators = [
                    name
                    for name, config in self.indicator_configs.items()
                    if config.category == category
                ]

            # Filter by market conditions
            suitable_indicators = [
                indicator
                for indicator in category_indicators
                if self._is_suitable_for_regime(indicator, market_regime)
            ]

            optimal_indicators.extend(suitable_indicators)

        # Add preferred indicators with higher priority
        preferred = agent_config["preferred_indicators"]
        for indicator in preferred:
            if indicator not in optimal_indicators:
                optimal_indicators.insert(0, indicator)  # High priority

        # Limit to max indicators and sort by performance
        optimal_indicators = self._sort_by_performance(optimal_indicators)
        max_indicators = agent_config["max_indicators"]

        return optimal_indicators[:max_indicators]

    def _is_suitable_for_regime(
        self, indicator_name: str, regime: MarketRegime
    ) -> bool:
        """Check if indicator is suitable for current market regime"""
        if indicator_name not in self.indicator_configs:
            return False

        config = self.indicator_configs[indicator_name]
        return regime.value in config.market_conditions

    def _sort_by_performance(self, indicators: List[str]) -> List[str]:
        """Sort indicators by recent performance metrics"""
        performance_scores = {}

        for indicator in indicators:
            score = self.performance_tracker.get_recent_performance(indicator)
            performance_scores[indicator] = score

        return sorted(
            indicators, key=lambda x: performance_scores.get(x, 0.5), reverse=True
        )

    def calculate_multi_agent_signals(self, market_data: Dict) -> Dict[str, Any]:
        """Calculate signals for all agents using optimal indicator sets"""

        # Detect current market regime
        current_regime = self.market_regime_detector.detect_regime(market_data)

        agent_signals = {}

        for agent_name in self.agent_preferences.keys():
            # Get optimal indicators for this agent
            optimal_indicators = self.get_optimal_indicators_for_agent(
                agent_name, market_data, current_regime
            )

            # Calculate signals using these indicators
            agent_signals[agent_name] = self._calculate_agent_signals(
                agent_name, optimal_indicators, market_data
            )

        # Generate master decision combining all agent inputs
        master_decision = self._synthesize_agent_decisions(agent_signals, market_data)

        return {
            "agent_signals": agent_signals,
            "master_decision": master_decision,
            "market_regime": current_regime.value,
            "total_indicators_used": sum(
                len(signals["indicators_used"]) for signals in agent_signals.values()
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_agent_signals(
        self, agent_name: str, indicators: List[str], market_data: Dict
    ) -> Dict[str, Any]:
        """Calculate signals for specific agent using assigned indicators"""
        signals = {
            "agent": agent_name,
            "indicators_used": indicators,
            "signals": [],
            "confidence": 0.0,
            "recommendation": "HOLD",
        }

        buy_signals = 0
        sell_signals = 0
        total_weight = 0.0

        for indicator_name in indicators:
            try:
                # Calculate indicator value
                result = self._calculate_indicator(indicator_name, market_data)
                if result is None:
                    continue

                # Get indicator configuration
                config = self.indicator_configs.get(indicator_name)
                if not config:
                    continue

                # Apply agent-specific weight multiplier
                agent_config = self.agent_preferences[agent_name]
                weight = config.weight * agent_config["weight_multiplier"]

                # Determine signal direction
                signal_direction = self._interpret_indicator_signal(
                    indicator_name, result
                )

                signals["signals"].append(
                    {
                        "indicator": indicator_name,
                        "value": result,
                        "signal": signal_direction,
                        "weight": weight,
                    }
                )

                if signal_direction == "BUY":
                    buy_signals += weight
                elif signal_direction == "SELL":
                    sell_signals += weight

                total_weight += weight

            except Exception as e:
                self.logger.error(
                    f"Error calculating {indicator_name} for {agent_name}: {e}"
                )
                continue

        # Calculate final recommendation
        if total_weight > 0:
            buy_strength = buy_signals / total_weight
            sell_strength = sell_signals / total_weight

            signals["confidence"] = abs(buy_strength - sell_strength)

            if buy_strength > sell_strength and signals["confidence"] > 0.6:
                signals["recommendation"] = "BUY"
            elif sell_strength > buy_strength and signals["confidence"] > 0.6:
                signals["recommendation"] = "SELL"
            else:
                signals["recommendation"] = "HOLD"

        return signals

    def _calculate_indicator(
        self, indicator_name: str, market_data: Dict
    ) -> Optional[float]:
        """Calculate specific indicator value"""
        try:
            # This would contain the actual indicator calculation logic
            # For now, return a placeholder
            return np.random.random()  # Replace with actual calculation
        except Exception as e:
            self.logger.error(f"Failed to calculate {indicator_name}: {e}")
            return None

    def _interpret_indicator_signal(self, indicator_name: str, value: float) -> str:
        """Interpret indicator value as BUY/SELL/HOLD signal"""
        # This would contain indicator-specific interpretation logic
        # For now, simple threshold-based logic
        if value > 0.7:
            return "BUY"
        elif value < 0.3:
            return "SELL"
        else:
            return "HOLD"

    def _synthesize_agent_decisions(
        self, agent_signals: Dict, market_data: Dict
    ) -> Dict[str, Any]:
        """Synthesize all agent decisions into master decision"""

        total_agents = len(agent_signals)
        buy_votes = 0
        sell_votes = 0
        total_confidence = 0.0

        agent_weights = {
            "decision_master": 2.0,
            "risk_genius": 1.5,
            "execution_expert": 1.8,
            "pattern_master": 1.6,
            "pair_specialist": 1.2,
            "session_expert": 1.0,
            "ai_model_coordinator": 1.7,
            "market_microstructure_genius": 1.3,
            "sentiment_integration_genius": 1.1,
        }

        for agent_name, signals in agent_signals.items():
            weight = agent_weights.get(agent_name, 1.0)
            confidence = signals["confidence"]
            recommendation = signals["recommendation"]

            if recommendation == "BUY":
                buy_votes += weight * confidence
            elif recommendation == "SELL":
                sell_votes += weight * confidence

            total_confidence += confidence * weight

        # Calculate master decision
        total_votes = buy_votes + sell_votes
        master_confidence = total_confidence / sum(agent_weights.values())

        if total_votes > 0:
            buy_strength = buy_votes / total_votes
            sell_strength = sell_votes / total_votes

            if buy_strength > 0.65 and master_confidence > 0.7:
                master_recommendation = "STRONG_BUY"
            elif buy_strength > 0.55:
                master_recommendation = "BUY"
            elif sell_strength > 0.65 and master_confidence > 0.7:
                master_recommendation = "STRONG_SELL"
            elif sell_strength > 0.55:
                master_recommendation = "SELL"
            else:
                master_recommendation = "HOLD"
        else:
            master_recommendation = "HOLD"

        return {
            "recommendation": master_recommendation,
            "confidence": master_confidence,
            "buy_strength": buy_votes,
            "sell_strength": sell_votes,
            "participating_agents": total_agents,
            "consensus_level": (
                1.0 - abs(buy_strength - sell_strength) if total_votes > 0 else 0.0
            ),
        }


class IndicatorPerformanceTracker:
    """Track indicator performance over time"""

    def __init__(self):
        self.performance_history = {}

    def get_recent_performance(self, indicator_name: str) -> float:
        """Get recent performance score for indicator (0.0 to 1.0)"""
        return self.performance_history.get(indicator_name, 0.5)

    def update_performance(self, indicator_name: str, accuracy: float):
        """Update performance tracking for indicator"""
        if indicator_name not in self.performance_history:
            self.performance_history[indicator_name] = []

        self.performance_history[indicator_name].append(accuracy)

        # Keep only recent history (last 100 signals)
        if len(self.performance_history[indicator_name]) > 100:
            self.performance_history[indicator_name] = self.performance_history[
                indicator_name
            ][-100:]


class MarketRegimeDetector:
    """Detect current market regime for adaptive indicator selection"""

    def detect_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime based on market data"""
        # Simplified regime detection - would be more sophisticated in production
        volatility = self._calculate_volatility(market_data)
        trend_strength = self._calculate_trend_strength(market_data)

        if volatility > 0.8:
            return MarketRegime.VOLATILE
        elif trend_strength > 0.7:
            return MarketRegime.TRENDING
        elif trend_strength < 0.3:
            return MarketRegime.RANGING
        else:
            return MarketRegime.QUIET

    def _calculate_volatility(self, market_data: Dict) -> float:
        """Calculate current volatility level"""
        # Simplified calculation
        return np.random.random()

    def _calculate_trend_strength(self, market_data: Dict) -> float:
        """Calculate current trend strength"""
        # Simplified calculation
        return np.random.random()


# Global coordinator instance
enhanced_coordinator = EnhancedAdaptiveCoordinator()


def get_coordinator():
    """Get the global enhanced coordinator instance"""
    return enhanced_coordinator
