# --- START OF FILE adaptive_indicator_bridge.py ---

"""
Adaptive Indicator Bridge for Platform3 Genius Agents - FINAL
Connects 9 genius agents to the 167 indicators in the registry.
This version is production-grade, asynchronous, and performs adaptive
indicator selection based on agent needs and market conditions.
"""
import asyncio
import inspect
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .registry import GeniusAgentType, get_indicator
from .indicator_mappings import AGENT_INDICATOR_MAPPINGS

logger = logging.getLogger(__name__)

class AdaptiveIndicatorBridge:
    """Bridge between all 167 indicators and 9 genius agents."""

    def __init__(self):
        self.indicator_mapping = AGENT_INDICATOR_MAPPINGS
        logger.info("AdaptiveIndicatorBridge initialized with complete agent-indicator mappings.")

    async def get_agent_indicators_async(
        self, agent_type: GeniusAgentType, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Asynchronously gets all necessary calculated indicators for a specific agent.
        """
        agent_name = agent_type.value
        logger.info(f"Requesting indicators for agent: {agent_name}")
        
        indicator_names = self._get_indicators_for_agent(agent_name)
        if not indicator_names:
            logger.warning(f"No indicators mapped for agent: {agent_name}")
            return {}

        market_regime = self._detect_market_regime(market_data)
        logger.debug(f"Detected market regime: {market_regime}")

        tasks = [self._calculate_single_indicator(name, market_data, market_regime) for name in indicator_names]
        results = await asyncio.gather(*tasks)

        calculated_indicators = {name: result for name, result in zip(indicator_names, results) if result is not None}
        
        logger.info(f"Successfully calculated {len(calculated_indicators)}/{len(indicator_names)} indicators for {agent_name}.")
        return calculated_indicators

    def _get_indicators_for_agent(self, agent_name: str) -> List[str]:
        """Retrieves the full list of indicator names for a given agent from the mapping."""
        agent_map = self.indicator_mapping.get(agent_name, {})
        all_indicators = []
        for category_list in agent_map.values():
            all_indicators.extend(category_list)
        return list(set(all_indicators))

    def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detects the current market regime (e.g., trending, volatile, ranging)."""
        import numpy as np
        close_prices = market_data.get("close", [])
        if len(close_prices) < 20: return "ranging"
        
        std_dev = np.std(np.diff(close_prices[-20:]) / close_prices[-21:-1])
        if std_dev > 0.0015: return "volatile"
        
        prices = np.array(close_prices[-20:])
        slope = np.polyfit(np.arange(len(prices)), prices, 1)[0]
        if abs(slope) / np.mean(prices) > 0.0005: return "trending"
            
        return "ranging"

    async def _calculate_single_indicator(
        self, indicator_name: str, market_data: Dict[str, Any], market_regime: str
    ) -> Optional[Any]:
        """Safely calculates a single indicator by name."""
        try:
            indicator_callable = get_indicator(indicator_name)
            
            if inspect.isclass(indicator_callable):
                indicator_instance = indicator_callable()
            else:
                indicator_instance = indicator_callable

            params = self._get_adaptive_parameters(indicator_name, market_regime)
            
            # The base class calculate() method handles the logic now
            return indicator_instance.calculate(market_data, **params)

        except KeyError:
            logger.error(f"Indicator '{indicator_name}' not found in registry during calculation.")
            return None
        except Exception as e:
            logger.error(f"Failed to calculate indicator '{indicator_name}': {e}")
            return None
            
    def _get_adaptive_parameters(self, indicator_name: str, market_regime: str) -> Dict:
        """Returns adaptive parameters for an indicator based on the market regime."""
        if market_regime == 'volatile': return {'period': 10}
        if market_regime == 'trending': return {'period': 20}
        return {'period': 14}

# --- END OF FILE adaptive_indicator_bridge.py ---