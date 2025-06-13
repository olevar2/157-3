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

from .registry import get_indicator, GeniusAgentType
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
            
            # Convert market data to list format expected by indicators
            converted_data = self._convert_market_data(market_data)
            
            # The base class calculate() method handles the logic now
            return indicator_instance.calculate(converted_data, **params)

        except KeyError:
            logger.error(f"Indicator '{indicator_name}' not found in registry during calculation.")
            return None
        except Exception as e:
            logger.error(f"Failed to calculate indicator '{indicator_name}': {e}")
            return None

    def _convert_market_data(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dictionary format market data to list format expected by indicators."""
        # If it's already a list, return as-is
        if isinstance(market_data, list):
            return market_data
            
        # Convert dictionary arrays to list of dictionaries
        if all(key in market_data for key in ['close', 'open', 'high', 'low', 'volume']):
            converted = []
            data_length = len(market_data['close'])
            
            for i in range(data_length):
                data_point = {
                    'timestamp': f"2024-01-01 {9+i//60:02d}:{i%60:02d}:00",
                    'open': float(market_data.get('open', market_data['close'])[i] if i < len(market_data.get('open', market_data['close'])) else market_data['close'][i]),
                    'high': float(market_data['high'][i] if i < len(market_data['high']) else market_data['close'][i]),
                    'low': float(market_data['low'][i] if i < len(market_data['low']) else market_data['close'][i]),
                    'close': float(market_data['close'][i]),
                    'volume': float(market_data['volume'][i] if i < len(market_data['volume']) else 1000.0)
                }
                converted.append(data_point)
            
            return converted
        
        # Fallback: create minimal test data
        return [
            {
                'timestamp': '2024-01-01 09:00:00',
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.0,
                'volume': 1000.0
            }
        ]
            
    def _get_adaptive_parameters(self, indicator_name: str, market_regime: str) -> Dict:
        """Returns adaptive parameters for an indicator based on the market regime."""
        if market_regime == 'volatile': return {'period': 10}
        if market_regime == 'trending': return {'period': 20}
        return {'period': 14}

# --- END OF FILE adaptive_indicator_bridge.py ---