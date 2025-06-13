# --- START OF FILE genius_agent_integration.py ---

"""
Genius Agent Integration Interface - FINAL
Connects all 9 genius agents to the 167 indicators through the enhanced
adaptive coordinator and bridge. This file demonstrates the full, asynchronous
workflow of agent analysis.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

# Local imports from the same engine
from .adaptive_indicator_bridge import AdaptiveIndicatorBridge
from .registry import GeniusAgentType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgentInterface:
    """Base class for all Genius Agent interfaces."""
    
    def __init__(self, agent_type: GeniusAgentType, bridge: AdaptiveIndicatorBridge):
        self.agent_type = agent_type
        self.name = agent_type.value
        self.bridge = bridge
        self.logger = logging.getLogger(self.name)
        self.logger.info(f"Initialized.")

    async def execute_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full analysis pipeline for the agent:
        1. Fetches its specific indicators from the bridge.
        2. Synthesizes the results into a trading recommendation.
        """
        self.logger.info("Requesting indicators from the bridge...")
        indicator_results = await self.bridge.get_agent_indicators_async(self.agent_type, market_data)
        
        if not indicator_results:
            self.logger.warning("No indicator results returned from bridge. Synthesizing with empty data.")
            return self._synthesize({})
            
        self.logger.info(f"Received {len(indicator_results)} indicator results. Synthesizing decision...")
        return self._synthesize(indicator_results)

    def _synthesize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesizes indicator results into a recommendation.
        This method should be overridden by each specific agent class.
        """
        if not results: return {"recommendation": "HOLD", "confidence": 0.1, "details": "No indicators available."}
        
        # Default synthesis logic
        avg_value = sum(v for v in results.values() if isinstance(v, (int, float))) / len(results)
        if avg_value > 0.6: rec = "BUY"
        elif avg_value < 0.4: rec = "SELL"
        else: rec = "HOLD"
        confidence = abs(avg_value - 0.5) * 2
        return {"recommendation": rec, "confidence": confidence, "details": f"Avg indicator value: {avg_value:.2f}"}

class RiskGeniusInterface(BaseAgentInterface):
    def _synthesize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        risk_scores = [v for k, v in results.items() if 'risk' in k.lower() or 'volatility' in k.lower() or 'variance' in k.lower()]
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        if avg_risk > 0.7: risk_level = "HIGH"
        elif avg_risk > 0.4: risk_level = "MEDIUM"
        else: risk_level = "LOW"
        return {"recommendation": "HOLD", "confidence": 0.9, "risk_level": risk_level, "details": f"Synthesized risk level from {len(risk_scores)} indicators."}

class PatternMasterInterface(BaseAgentInterface):
    def _synthesize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bullish_patterns = sum(1 for k, v in results.items() if 'bull' in k.lower() and v > 0.7)
        bearish_patterns = sum(1 for k, v in results.items() if 'bear' in k.lower() and v > 0.7)
        if bullish_patterns > 2: return {"recommendation": "STRONG_BUY", "confidence": 0.85, "details": f"Found {bullish_patterns} bullish patterns."}
        if bearish_patterns > 2: return {"recommendation": "STRONG_SELL", "confidence": 0.85, "details": f"Found {bearish_patterns} bearish patterns."}
        return {"recommendation": "HOLD", "confidence": 0.4, "details": "No strong patterns detected."}

class SessionExpertInterface(BaseAgentInterface): pass
class ExecutionExpertInterface(BaseAgentInterface): pass
class PairSpecialistInterface(BaseAgentInterface): pass
class DecisionMasterInterface(BaseAgentInterface): pass
class AIModelCoordinatorInterface(BaseAgentInterface): pass
class MarketMicrostructureInterface(BaseAgentInterface): pass
class SentimentIntegrationInterface(BaseAgentInterface): pass

class GeniusAgentIntegration:
    """The main class to orchestrate all 9 Genius Agents."""
    def __init__(self):
        self.logger = logging.getLogger("GeniusAgentIntegration")
        self.bridge = AdaptiveIndicatorBridge()
        self.agents: Dict[str, BaseAgentInterface] = {
            agent.value: globals()[f"{''.join(word.capitalize() for word in agent.value.split('_'))}Interface"](agent, self.bridge)
            for agent in GeniusAgentType
        }
        self.logger.info(f"Initialized with {len(self.agents)} agents.")

    async def analyze_market_data(self, market_data: Dict) -> Dict[str, Any]:
        self.logger.info("Starting full market analysis with all agents.")
        tasks = [agent.execute_analysis(market_data) for agent in self.agents.values()]
        agent_results = await asyncio.gather(*tasks)
        final_analyses = {agent.name: result for agent, result in zip(self.agents.values(), agent_results)}
        self.logger.info("All agent analyses complete. Synthesizing final trading decision.")
        return self._generate_final_decision(final_analyses)

    def _generate_final_decision(self, agent_analyses: Dict[str, Any]) -> Dict[str, Any]:
        recommendations = [res.get("recommendation", "HOLD") for res in agent_analyses.values()]
        confidences = [res.get("confidence", 0.0) for res in agent_analyses.values()]
        
        final_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        buy_votes = recommendations.count("BUY") + recommendations.count("STRONG_BUY")
        sell_votes = recommendations.count("SELL") + recommendations.count("STRONG_SELL")

        if buy_votes > sell_votes and final_confidence > 0.5: final_action = "BUY"
        elif sell_votes > buy_votes and final_confidence > 0.5: final_action = "SELL"
        else: final_action = "HOLD"

        return {
            "timestamp": datetime.now().isoformat(),
            "final_action": final_action,
            "confidence": round(final_confidence, 3),
            "individual_agent_analyses": agent_analyses
        }

async def main():
    import json
    import numpy as np
    
    market_data = {"close": (np.random.randn(100).cumsum() + 100).tolist()}
    
    integration = GeniusAgentIntegration()
    final_decision = await integration.analyze_market_data(market_data)
    
    print(json.dumps(final_decision, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
# --- END OF FILE genius_agent_integration.py ---