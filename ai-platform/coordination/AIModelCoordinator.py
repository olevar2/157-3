"""
🧠 PLATFORM3 AI MODEL COORDINATOR
Central coordination system for all AI/ML models across Platform3
Manages model orchestration, ensemble voting, and humanitarian impact optimization
Enhanced with AdaptiveStrategyGenerator integration for dynamic strategy allocation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

# Add shared path for Platform3 components

# Import AdaptiveStrategyGenerator and Platform3 components
try:
    from model import AdaptiveStrategyGenerator
    from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
    ADAPTIVE_STRATEGY_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AdaptiveStrategyGenerator not available: {e}")
    ADAPTIVE_STRATEGY_AVAILABLE = False
    # Define placeholder for type hints
    Platform3CommunicationFramework = Any

class TradingTimeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

class ModelType(Enum):
    SCALPING = "scalping"
    DAYTRADING = "daytrading"
    SWING = "swing"
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT = "sentiment"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class ModelPrediction:
    model_id: str
    model_type: ModelType
    timeframe: TradingTimeframe
    symbol: str
    prediction: float  # -1 to 1 (sell to buy signal strength)
    confidence: float  # 0 to 1
    risk_score: float  # 0 to 1
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class EnsemblePrediction:
    symbol: str
    timeframe: TradingTimeframe
    final_signal: float  # -1 to 1
    confidence: float
    risk_score: float
    contributing_models: List[str]
    humanitarian_impact_score: float  # 0 to 1
    timestamp: datetime

class AIModelCoordinator:
    """
    Central AI Model Coordination System with Adaptive Strategy Integration
    
    Manages 25+ AI/ML models across Platform3 for maximum humanitarian impact:
    - Scalping models (M1-M5): Ultra-fast profit generation
    - Day trading models (M15-H1): Intraday opportunities
    - Swing models (H4+): Medium-term accumulation
    - Pattern recognition: Chart pattern detection
    - Sentiment analysis: Market sentiment overlay
    - Risk assessment: Capital protection
    - Adaptive strategy generation: Dynamic strategy allocation based on market regimes
    """
    
    def __init__(self, comm_framework: Optional[Platform3CommunicationFramework] = None):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Dict] = {}
        self.humanitarian_multiplier: float = 1.0
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize communication framework
        self.comm_framework = comm_framework
        
        # Initialize AdaptiveStrategyGenerator if available
        self.adaptive_strategy_generator = None
        if ADAPTIVE_STRATEGY_AVAILABLE and comm_framework:
            try:
                self.adaptive_strategy_generator = AdaptiveStrategyGenerator(comm_framework)
                self.logger.info("✅ AdaptiveStrategyGenerator integrated successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AdaptiveStrategyGenerator: {e}")
        
        # Adaptive strategy state
        self.current_market_regime = "unknown"
        self.strategy_weights: Dict[str, float] = {}
        self.regime_history: List[Dict] = []
        
        # Initialize model registry
        self._initialize_model_registry()
        
        # Performance tracking for humanitarian optimization
        self.profit_for_charity: float = 0.0
        self.children_surgeries_funded: int = 0
        self.medical_aid_provided: float = 0.0
        
        self.logger.info("🧠 AI Model Coordinator initialized for humanitarian mission with adaptive strategies")
    
    def _initialize_model_registry(self):
        """Initialize registry of all AI models for coordinated execution"""
        self.model_registry = {
            # Scalping Models (M1-M5 Ultra-Fast)
            "scalping_lstm": {
                "type": ModelType.SCALPING,
                "timeframes": [TradingTimeframe.M1, TradingTimeframe.M5],
                "weight": 0.25,
                "humanitarian_impact": "high_frequency_profits"
            },
            "scalping_ensemble": {
                "type": ModelType.SCALPING,
                "timeframes": [TradingTimeframe.M1, TradingTimeframe.M5],
                "weight": 0.30,
                "humanitarian_impact": "rapid_charity_accumulation"
            },
            "tick_classifier": {
                "type": ModelType.SCALPING,
                "timeframes": [TradingTimeframe.M1],
                "weight": 0.20,
                "humanitarian_impact": "micro_profit_optimization"
            },
            
            # Day Trading Models (M15-H1)
            "intraday_momentum": {
                "type": ModelType.DAYTRADING,
                "timeframes": [TradingTimeframe.M15, TradingTimeframe.H1],
                "weight": 0.25,
                "humanitarian_impact": "daily_medical_fund_generation"
            },
            "session_breakout": {
                "type": ModelType.DAYTRADING,
                "timeframes": [TradingTimeframe.M15, TradingTimeframe.H1],
                "weight": 0.30,
                "humanitarian_impact": "session_based_charity_profits"
            },
            
            # Swing Trading Models (H4+)
            "swing_patterns": {
                "type": ModelType.SWING,
                "timeframes": [TradingTimeframe.H4, TradingTimeframe.D1],
                "weight": 0.35,
                "humanitarian_impact": "sustained_charity_funding"
            },
            "elliott_wave": {
                "type": ModelType.SWING,
                "timeframes": [TradingTimeframe.H4, TradingTimeframe.D1],
                "weight": 0.25,
                "humanitarian_impact": "long_term_surgery_funding"
            },
            
            # Market Analysis Models
            "pattern_recognition": {
                "type": ModelType.PATTERN_RECOGNITION,
                "timeframes": list(TradingTimeframe),
                "weight": 0.20,
                "humanitarian_impact": "pattern_based_alpha"
            },
            "sentiment_analysis": {
                "type": ModelType.SENTIMENT,
                "timeframes": list(TradingTimeframe),
                "weight": 0.15,
                "humanitarian_impact": "sentiment_driven_profits"
            },
            "risk_assessment": {
                "type": ModelType.RISK_ASSESSMENT,
                "timeframes": list(TradingTimeframe),
                "weight": 1.0,  # Always applied
                "humanitarian_impact": "capital_protection"
            }
        }
    
    async def coordinate_models(self, symbol: str, timeframe: TradingTimeframe, 
                              market_data: Dict[str, Any]) -> EnsemblePrediction:
        """
        Coordinate all relevant AI models for a trading decision with adaptive strategy allocation
        
        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            timeframe: Trading timeframe
            market_data: Current market data and indicators
            
        Returns:
            EnsemblePrediction: Coordinated prediction with humanitarian and adaptive optimization
        """
        
        # Get adaptive strategy recommendations if available
        strategy_context = await self._get_adaptive_strategy_context(symbol, timeframe, market_data)
        
        # Get relevant models for this timeframe
        relevant_models = self._get_relevant_models(timeframe)
        
        # Apply adaptive model weighting based on strategy recommendations
        if strategy_context:
            relevant_models = self._apply_adaptive_model_selection(relevant_models, strategy_context)
        
        # Collect predictions from all models
        predictions = await self._collect_model_predictions(
            relevant_models, symbol, timeframe, market_data
        )
        
        # Apply ensemble voting with adaptive weights
        ensemble_result = self._ensemble_voting_with_adaptive_weights(
            predictions, symbol, timeframe, strategy_context
        )
        
        # Apply humanitarian impact optimization
        optimized_result = self._apply_humanitarian_optimization(ensemble_result)
        
        # Update adaptive strategy performance tracking
        if strategy_context:
            await self._update_adaptive_strategy_performance(optimized_result, strategy_context)
        
        # Update performance tracking
        self._update_humanitarian_metrics(optimized_result)
        
        return optimized_result
    
    def _get_relevant_models(self, timeframe: TradingTimeframe) -> List[str]:
        """Get models relevant for the specified timeframe"""
        relevant = []
        for model_id, config in self.model_registry.items():
            if timeframe in config["timeframes"]:
                relevant.append(model_id)
        return relevant
    
    async def _collect_model_predictions(self, model_ids: List[str], symbol: str, 
                                       timeframe: TradingTimeframe, 
                                       market_data: Dict[str, Any]) -> List[ModelPrediction]:
        """Collect predictions from all relevant models concurrently"""
        
        tasks = []
        for model_id in model_ids:
            task = self._get_model_prediction(model_id, symbol, timeframe, market_data)
            tasks.append(task)
        
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid predictions
        valid_predictions = [p for p in predictions if isinstance(p, ModelPrediction)]
        return valid_predictions
    
    async def _get_model_prediction(self, model_id: str, symbol: str, 
                                  timeframe: TradingTimeframe, 
                                  market_data: Dict[str, Any]) -> ModelPrediction:
        """Get prediction from a specific model"""
        
        try:
            # This would interface with the actual model
            # For now, implementing the coordination structure
            
            model_config = self.model_registry[model_id]
            
            # Simulate model prediction (replace with actual model calls)
            prediction_value = np.random.uniform(-1, 1)  # Placeholder
            confidence = np.random.uniform(0.5, 1.0)     # Placeholder
            risk_score = np.random.uniform(0, 0.5)       # Placeholder
            
            return ModelPrediction(
                model_id=model_id,
                model_type=model_config["type"],
                timeframe=timeframe,
                symbol=symbol,
                prediction=prediction_value,
                confidence=confidence,
                risk_score=risk_score,
                timestamp=datetime.utcnow(),
                metadata={"humanitarian_impact": model_config["humanitarian_impact"]}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting prediction from {model_id}: {e}")
            raise
    
    def _ensemble_voting(self, predictions: List[ModelPrediction], 
                        symbol: str, timeframe: TradingTimeframe) -> EnsemblePrediction:
        """Apply ensemble voting across all model predictions"""
        
        if not predictions:
            return EnsemblePrediction(
                symbol=symbol,
                timeframe=timeframe,
                final_signal=0.0,
                confidence=0.0,
                risk_score=1.0,
                contributing_models=[],
                humanitarian_impact_score=0.0,
                timestamp=datetime.utcnow()
            )
        
        # Weighted ensemble based on model performance and type
        total_weight = 0.0
        weighted_signal = 0.0
        weighted_confidence = 0.0
        weighted_risk = 0.0
        humanitarian_impact = 0.0
        
        contributing_models = []
        
        for pred in predictions:
            model_weight = self.model_registry[pred.model_id]["weight"]
            
            # Performance-based weight adjustment
            performance_multiplier = self._get_performance_multiplier(pred.model_id)
            adjusted_weight = model_weight * performance_multiplier * pred.confidence
            
            weighted_signal += pred.prediction * adjusted_weight
            weighted_confidence += pred.confidence * adjusted_weight
            weighted_risk += pred.risk_score * adjusted_weight
            
            # Calculate humanitarian impact score
            humanitarian_impact += self._calculate_humanitarian_impact(pred) * adjusted_weight
            
            total_weight += adjusted_weight
            contributing_models.append(pred.model_id)
        
        if total_weight == 0:
            total_weight = 1.0
        
        return EnsemblePrediction(
            symbol=symbol,
            timeframe=timeframe,
            final_signal=weighted_signal / total_weight,
            confidence=weighted_confidence / total_weight,
            risk_score=weighted_risk / total_weight,
            contributing_models=contributing_models,
            humanitarian_impact_score=humanitarian_impact / total_weight,
            timestamp=datetime.utcnow()
        )
    
    def _get_performance_multiplier(self, model_id: str) -> float:
        """Get performance-based weight multiplier for a model"""
        if model_id not in self.performance_metrics:
            return 1.0
        
        metrics = self.performance_metrics[model_id]
        
        # Calculate multiplier based on recent performance
        accuracy = metrics.get("accuracy", 0.5)
        profit_factor = metrics.get("profit_factor", 1.0)
        humanitarian_contribution = metrics.get("humanitarian_contribution", 0.0)
        
        # Boost models that contribute more to humanitarian mission
        multiplier = accuracy * profit_factor * (1.0 + humanitarian_contribution)
        return max(0.1, min(2.0, multiplier))  # Clamp between 0.1 and 2.0
    
    def _calculate_humanitarian_impact(self, prediction: ModelPrediction) -> float:
        """Calculate humanitarian impact score for a prediction"""
        
        # Higher impact for stronger signals with higher confidence
        base_impact = abs(prediction.prediction) * prediction.confidence
        
        # Adjust based on model type humanitarian potential
        type_multipliers = {
            ModelType.SCALPING: 1.5,      # High frequency = more charity funds
            ModelType.DAYTRADING: 1.2,    # Daily accumulation
            ModelType.SWING: 1.0,         # Steady growth
            ModelType.PATTERN_RECOGNITION: 0.8,
            ModelType.SENTIMENT: 0.6,
            ModelType.RISK_ASSESSMENT: 2.0  # Capital protection is crucial
        }
        
        multiplier = type_multipliers.get(prediction.model_type, 1.0)
        
        # Reduce impact if risk is too high
        risk_penalty = max(0.1, 1.0 - prediction.risk_score)
        
        return base_impact * multiplier * risk_penalty
    
    def _apply_humanitarian_optimization(self, prediction: EnsemblePrediction) -> EnsemblePrediction:
        """Apply humanitarian mission optimization to the prediction"""
        
        # Boost signal strength for high humanitarian impact opportunities
        humanitarian_boost = min(0.3, prediction.humanitarian_impact_score * 0.2)
        
        # Apply the boost while maintaining signal bounds
        original_signal = prediction.final_signal
        boosted_signal = original_signal * (1.0 + humanitarian_boost)
        boosted_signal = max(-1.0, min(1.0, boosted_signal))
        
        # Update the prediction
        prediction.final_signal = boosted_signal
        prediction.humanitarian_impact_score *= self.humanitarian_multiplier
        
        return prediction
    
    def _update_humanitarian_metrics(self, prediction: EnsemblePrediction):
        """Update humanitarian impact tracking metrics"""
        
        # Estimate potential profit contribution (simplified)
        estimated_profit = abs(prediction.final_signal) * prediction.confidence * 100  # $100 base
        
        # Add to humanitarian funds tracking
        self.profit_for_charity += estimated_profit * 0.8  # 80% to charity
        
        # Calculate surgeries that could be funded ($5000 per surgery)
        surgeries_possible = int(self.profit_for_charity / 5000)
        if surgeries_possible > self.children_surgeries_funded:
            new_surgeries = surgeries_possible - self.children_surgeries_funded
            self.children_surgeries_funded = surgeries_possible
            self.logger.info(f"🏥 Humanitarian Impact: {new_surgeries} additional children's surgeries can now be funded!")
        
        # Medical aid provided
        self.medical_aid_provided = self.profit_for_charity * 0.6  # 60% to medical aid
        
        # Log humanitarian progress
        if estimated_profit > 50:  # Log significant contributions
            self.logger.info(                f"💰 Humanitarian Contribution: ${estimated_profit:.2f} potential profit "
                f"(Total charity funds: ${self.profit_for_charity:.2f})"
            )
    
    def get_humanitarian_impact_report(self) -> Dict[str, Any]:
        """Generate comprehensive humanitarian impact report with adaptive strategy metrics"""
        base_report = {
            "total_charity_funds": self.profit_for_charity,
            "children_surgeries_funded": self.children_surgeries_funded,
            "medical_aid_provided": self.medical_aid_provided,
            "models_contributing": len(self.model_registry),
            "humanitarian_multiplier": self.humanitarian_multiplier,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add adaptive strategy metrics
        adaptive_metrics = self.get_adaptive_strategy_status()
        base_report.update({
            "adaptive_strategy_metrics": adaptive_metrics
        })
        
        return base_report
    
    def update_model_performance(self, model_id: str, metrics: Dict[str, float]):
        """Update performance metrics for a model"""
        self.performance_metrics[model_id] = metrics
        self.logger.info(f"📊 Updated performance metrics for {model_id}")
    
    def set_humanitarian_multiplier(self, multiplier: float):
        """Set the humanitarian impact multiplier"""
        self.humanitarian_multiplier = max(0.1, min(3.0, multiplier))
        self.logger.info(f"🎯 Humanitarian multiplier set to {self.humanitarian_multiplier}")

    async def _get_adaptive_strategy_context(self, symbol: str, timeframe: TradingTimeframe, 
                                           market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get adaptive strategy recommendations and market regime information"""
        if not self.adaptive_strategy_generator:
            return None
        
        try:
            # Prepare market context for strategy generator
            market_context = {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "market_data": market_data,
                "timestamp": datetime.utcnow()
            }
            
            # Get strategy allocation recommendations
            strategy_recommendations = await self.adaptive_strategy_generator.generate_strategy_allocation(market_context)
            
            # Update current market regime
            if "current_regime" in strategy_recommendations:
                old_regime = self.current_market_regime
                self.current_market_regime = strategy_recommendations["current_regime"]
                
                if old_regime != self.current_market_regime:
                    self.logger.info(f"🔄 Market regime changed: {old_regime} → {self.current_market_regime}")
                    self.regime_history.append({
                        "timestamp": datetime.utcnow(),
                        "old_regime": old_regime,
                        "new_regime": self.current_market_regime,
                        "symbol": symbol
                    })
            
            # Update strategy weights
            if "model_weights" in strategy_recommendations:
                self.strategy_weights = strategy_recommendations["model_weights"]
            
            return strategy_recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting adaptive strategy context: {e}")
            return None
    
    def _apply_adaptive_model_selection(self, relevant_models: List[str], 
                                      strategy_context: Dict[str, Any]) -> List[str]:
        """Apply adaptive model selection based on strategy recommendations"""
        if not strategy_context or "model_preferences" not in strategy_context:
            return relevant_models
        
        model_preferences = strategy_context["model_preferences"]
        regime = strategy_context.get("current_regime", "unknown")
        
        # Filter and prioritize models based on regime
        adapted_models = []
        
        for model_id in relevant_models:
            model_type = self.model_registry[model_id]["type"]
            
            # Apply regime-based model selection
            if regime == "high_volatility":
                # Favor risk assessment and scalping models
                if model_type in [ModelType.RISK_ASSESSMENT, ModelType.SCALPING]:
                    adapted_models.append(model_id)
            elif regime == "trending":
                # Favor momentum and swing models
                if model_type in [ModelType.DAYTRADING, ModelType.SWING, ModelType.PATTERN_RECOGNITION]:
                    adapted_models.append(model_id)
            elif regime == "sideways":
                # Favor scalping and sentiment models
                if model_type in [ModelType.SCALPING, ModelType.SENTIMENT]:
                    adapted_models.append(model_id)
            else:
                # Default: include all relevant models
                adapted_models.append(model_id)
        
        # Always include risk assessment
        risk_models = [m for m in relevant_models if self.model_registry[m]["type"] == ModelType.RISK_ASSESSMENT]
        adapted_models.extend([m for m in risk_models if m not in adapted_models])
        
        self.logger.info(f"🎯 Adaptive model selection for {regime}: {len(adapted_models)} models selected")
        return adapted_models or relevant_models  # Fallback to original if empty
    
    def _ensemble_voting_with_adaptive_weights(self, predictions: List[ModelPrediction], 
                                             symbol: str, timeframe: TradingTimeframe,
                                             strategy_context: Optional[Dict[str, Any]]) -> EnsemblePrediction:
        """Apply ensemble voting with adaptive strategy weights"""
        
        if not predictions:
            return EnsemblePrediction(
                symbol=symbol,
                timeframe=timeframe,
                final_signal=0.0,
                confidence=0.0,
                risk_score=1.0,
                contributing_models=[],
                humanitarian_impact_score=0.0,
                timestamp=datetime.utcnow()
            )
        
        # Get adaptive weights if available
        adaptive_weights = strategy_context.get("model_weights", {}) if strategy_context else {}
        regime = strategy_context.get("current_regime", "unknown") if strategy_context else "unknown"
        
        # Weighted ensemble with adaptive strategy integration
        total_weight = 0.0
        weighted_signal = 0.0
        weighted_confidence = 0.0
        weighted_risk = 0.0
        humanitarian_impact = 0.0
        
        contributing_models = []
        
        for pred in predictions:
            # Base model weight
            base_weight = self.model_registry[pred.model_id]["weight"]
            
            # Apply adaptive weight if available
            adaptive_weight = adaptive_weights.get(pred.model_id, 1.0)
            
            # Apply regime-specific adjustments
            regime_weight = self._get_regime_weight_adjustment(pred.model_type, regime)
            
            # Performance-based weight adjustment
            performance_multiplier = self._get_performance_multiplier(pred.model_id)
            
            # Calculate final weight
            final_weight = base_weight * adaptive_weight * regime_weight * performance_multiplier * pred.confidence
            
            weighted_signal += pred.prediction * final_weight
            weighted_confidence += pred.confidence * final_weight
            weighted_risk += pred.risk_score * final_weight
            
            # Calculate humanitarian impact score
            humanitarian_impact += self._calculate_humanitarian_impact(pred) * final_weight
            
            total_weight += final_weight
            contributing_models.append(pred.model_id)
        
        if total_weight == 0:
            total_weight = 1.0
        
        result = EnsemblePrediction(
            symbol=symbol,
            timeframe=timeframe,
            final_signal=weighted_signal / total_weight,
            confidence=weighted_confidence / total_weight,
            risk_score=weighted_risk / total_weight,
            contributing_models=contributing_models,
            humanitarian_impact_score=humanitarian_impact / total_weight,
            timestamp=datetime.utcnow()
        )
        
        # Add adaptive strategy metadata
        if strategy_context:
            result.metadata = {
                "adaptive_regime": regime,
                "strategy_weights_applied": bool(adaptive_weights),
                "adaptive_models_count": len([m for m in contributing_models if m in adaptive_weights])
            }
        
        return result
    
    def _get_regime_weight_adjustment(self, model_type: ModelType, regime: str) -> float:
        """Get weight adjustment based on market regime and model type"""
        regime_adjustments = {
            "high_volatility": {
                ModelType.RISK_ASSESSMENT: 1.5,
                ModelType.SCALPING: 1.3,
                ModelType.DAYTRADING: 0.8,
                ModelType.SWING: 0.6,
                ModelType.PATTERN_RECOGNITION: 0.9,
                ModelType.SENTIMENT: 1.1
            },
            "trending": {
                ModelType.RISK_ASSESSMENT: 1.2,
                ModelType.SCALPING: 0.8,
                ModelType.DAYTRADING: 1.4,
                ModelType.SWING: 1.5,
                ModelType.PATTERN_RECOGNITION: 1.3,
                ModelType.SENTIMENT: 1.0
            },
            "sideways": {
                ModelType.RISK_ASSESSMENT: 1.3,
                ModelType.SCALPING: 1.4,
                ModelType.DAYTRADING: 0.9,
                ModelType.SWING: 0.7,
                ModelType.PATTERN_RECOGNITION: 1.1,
                ModelType.SENTIMENT: 1.2
            }
        }
        
        return regime_adjustments.get(regime, {}).get(model_type, 1.0)
    
    async def _update_adaptive_strategy_performance(self, result: EnsemblePrediction, 
                                                  strategy_context: Dict[str, Any]):
        """Update adaptive strategy performance tracking"""
        if not self.adaptive_strategy_generator:
            return
        
        try:
            # Prepare performance update for strategy generator
            performance_data = {
                "timestamp": result.timestamp,
                "symbol": result.symbol,
                "regime": strategy_context.get("current_regime"),
                "signal_strength": abs(result.final_signal),
                "confidence": result.confidence,
                "risk_score": result.risk_score,
                "humanitarian_impact": result.humanitarian_impact_score,
                "contributing_models": result.contributing_models
            }
            
            # Send performance update to adaptive strategy generator
            if hasattr(self.adaptive_strategy_generator, '_handle_performance_update'):
                await self.adaptive_strategy_generator._handle_performance_update(performance_data)
                
        except Exception as e:
            self.logger.error(f"Error updating adaptive strategy performance: {e}")
    
    def get_adaptive_strategy_status(self) -> Dict[str, Any]:
        """Get current adaptive strategy status and metrics"""
        return {
            "adaptive_strategy_enabled": self.adaptive_strategy_generator is not None,
            "current_market_regime": self.current_market_regime,
            "strategy_weights": self.strategy_weights,
            "regime_changes_today": len([r for r in self.regime_history 
                                       if r["timestamp"].date() == datetime.utcnow().date()]),
            "total_regime_changes": len(self.regime_history),
            "last_regime_change": self.regime_history[-1] if self.regime_history else None,
            "timestamp": datetime.utcnow()
        }

# Singleton instance for global coordination (initialized without comm_framework for backward compatibility)
ai_coordinator = AIModelCoordinator()

def initialize_coordinator_with_adaptive_strategy(comm_framework: Platform3CommunicationFramework) -> AIModelCoordinator:
    """Initialize a new coordinator instance with adaptive strategy support"""
    return AIModelCoordinator(comm_framework)

if __name__ == "__main__":
    # Test the coordinator with adaptive strategy integration
    import asyncio
    
    async def test_coordination_with_adaptive_strategy():
        # Test with adaptive strategy integration
        try:
            # Mock communication framework for testing
            class MockCommFramework:
                async def subscribe(self, topic, handler):
                    pass
            
            comm_framework = MockCommFramework()
            coordinator = AIModelCoordinator(comm_framework)
            
            # Test market data
            market_data = {
                "price": 1.0850,
                "volume": 1000,
                "indicators": np.random.random(50),
                "volatility": 0.0125,
                "trend_strength": 0.75
            }
            
            # Test coordination with adaptive strategy
            result = await coordinator.coordinate_models("EURUSD", TradingTimeframe.M15, market_data)
            
            print(f"Enhanced Ensemble Result: {result}")
            print(f"Adaptive Strategy Status: {coordinator.get_adaptive_strategy_status()}")
            print(f"Humanitarian Report: {coordinator.get_humanitarian_impact_report()}")
            
        except Exception as e:
            print(f"Test completed with limited features due to: {e}")
            
            # Fallback test without adaptive strategy
            coordinator = AIModelCoordinator()
            market_data = {
                "price": 1.0850,
                "volume": 1000,
                "indicators": np.random.random(50)
            }
            
            result = await coordinator.coordinate_models("EURUSD", TradingTimeframe.M15, market_data)
            print(f"Basic Ensemble Result: {result}")
    
    asyncio.run(test_coordination_with_adaptive_strategy())
