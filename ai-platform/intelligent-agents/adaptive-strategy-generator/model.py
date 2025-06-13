# MANDATORY imports as per shrimp-rules.md
import sys
from pathlib import Path

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, MLError
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
from shared.ai_model_base import EnhancedAIModelBase

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
import hashlib

class AdaptiveStrategyGenerator(EnhancedAIModelBase):
    """
    Adaptive Strategy Generation Agent for Platform3
    
    Provides intelligent strategy management through:
    - Automated strategy generation using advanced ML techniques
    - Market regime-driven strategy switching
    - Performance attribution and optimization
    - Dynamic strategy weight adjustment
    
    Integrates with Platform3's Analytics Framework and AIModelCoordinator.
    """
    
    def __init__(self, comm_framework: Platform3CommunicationFramework):
        super().__init__(
            model_name="AdaptiveStrategyGenerator",
            version="1.0.0",
            description="AI-powered adaptive strategy generation and management"
        )
        
        self.comm_framework = comm_framework
        self.logger = Platform3Logger("AdaptiveStrategyGenerator")
        self.error_system = Platform3ErrorSystem("AdaptiveStrategyGenerator")
        
        # Strategy generation and management
        self.strategy_registry = {}
        self.active_strategies = {}
        self.strategy_performance_history = {}
        self.market_regime_history = []
        
        # ML Models for strategy generation and regime detection
        self.regime_detector = None
        self.strategy_evaluator = None
        self.strategy_generator = None
        
        # Configuration parameters
        self.config = {
            "regime_detection_threshold": 0.1,
            "strategy_performance_window": 24,  # hours
            "min_strategy_performance_period": 2,  # hours
            "max_active_strategies": 10,
            "strategy_weight_adjustment_factor": 0.1,
            "new_strategy_generation_threshold": 0.05,
            "underperforming_strategy_threshold": -0.02,
            "adaptation_cycle_minutes": 15
        }
        
        # Performance tracking
        self.adaptation_metrics = {
            "total_adaptations": 0,
            "strategies_generated": 0,
            "strategies_deactivated": 0,
            "regime_switches_detected": 0,
            "successful_adaptations": 0,
            "average_adaptation_performance": 0.0
        }
        
        self.is_running = False
        
    async def start(self):
        """Initialize and start the Adaptive Strategy Generator service"""
        try:
            self.logger.info("AdaptiveStrategyGenerator starting...")
            self.is_running = True
            
            # Initialize ML models and strategy registry
            await self._initialize_models()
            await self._load_existing_strategies()
            
            # Subscribe to performance updates from Analytics Framework
            await self.comm_framework.subscribe(
                "analytics.strategy_performance_update",
                self._handle_performance_update
            )
            
            # Subscribe to market regime changes
            await self.comm_framework.subscribe(
                "market_data.regime_change",
                self._handle_regime_change
            )
            
            # Start periodic adaptation cycle
            asyncio.create_task(self._periodic_adaptation_cycle())
            
            # Start continuous monitoring
            asyncio.create_task(self._continuous_monitoring())
            
            self.logger.info("AdaptiveStrategyGenerator started successfully.")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to start AdaptiveStrategyGenerator: {e}"),
                "start"
            )
            raise

    async def stop(self):
        """Gracefully stop the Adaptive Strategy Generator service"""
        self.logger.info("AdaptiveStrategyGenerator stopping...")
        self.is_running = False
        await self._save_strategy_state()
        self.logger.info("AdaptiveStrategyGenerator stopped.")

    async def _initialize_models(self):
        """Initialize ML models for regime detection and strategy generation"""
        try:
            # Initialize regime detection model
            self.regime_detector = MarketRegimeDetector()
            
            # Initialize strategy evaluation model
            self.strategy_evaluator = StrategyPerformanceEvaluator()
            
            # Initialize strategy generation model (RL/Genetic Programming)
            self.strategy_generator = StrategyGenerationEngine()
            
            self.logger.info("ML models initialized successfully.")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to initialize ML models: {e}"),
                "_initialize_models"
            )

    async def _load_existing_strategies(self):
        """Load existing strategies from the Model Registry"""
        try:
            # Request strategy list from Model Registry
            strategy_list = await self.comm_framework.request(
                "model_registry.get_active_strategies", {}
            )
            
            for strategy_info in strategy_list.get('strategies', []):
                strategy_id = strategy_info['strategy_id']
                self.strategy_registry[strategy_id] = strategy_info
                
                # Initialize performance tracking
                self.strategy_performance_history[strategy_id] = []
            
            self.logger.info(f"Loaded {len(self.strategy_registry)} existing strategies.")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to load existing strategies: {e}"),
                "_load_existing_strategies"
            )

    async def _periodic_adaptation_cycle(self):
        """Main adaptation cycle that runs periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["adaptation_cycle_minutes"] * 60)
                
                self.logger.info("Starting periodic strategy adaptation cycle...")
                
                # Analyze current market regime
                current_regime = await self._identify_market_regime()
                
                # Get comprehensive performance data
                performance_data = await self._get_comprehensive_performance_data()
                
                # Generate adaptation recommendations
                recommendations = await self._generate_adaptation_recommendations(
                    performance_data, current_regime
                )
                
                # Apply recommendations
                if recommendations:
                    await self._apply_recommendations(recommendations)
                    self.adaptation_metrics["total_adaptations"] += 1
                
                # Check for new strategy generation opportunities
                if await self._should_generate_new_strategy(performance_data, current_regime):
                    await self._generate_new_strategy(current_regime)
                
                self.logger.info("Adaptation cycle completed successfully.")
                
            except Exception as e:
                self.error_system.handle_error(
                    MLError(f"Error in periodic adaptation cycle: {e}"),
                    "_periodic_adaptation_cycle"
                )

    async def _identify_market_regime(self) -> str:
        """Identify current market regime using multiple indicators"""
        try:
            # Get regime detection AI indicator
            regime_indicator = await self.comm_framework.request(
                "market_data.get_indicator_value",
                {"indicator_name": "REGIME_DETECTION_AI", "symbol": "PLATFORM_WIDE"}
            )
            
            # Get additional market indicators
            volatility = await self.comm_framework.request(
                "market_data.get_indicator_value",
                {"indicator_name": "VOLATILITY_REGIME", "symbol": "PLATFORM_WIDE"}
            )
            
            trend_strength = await self.comm_framework.request(
                "market_data.get_indicator_value",
                {"indicator_name": "TREND_STRENGTH", "symbol": "PLATFORM_WIDE"}
            )
            
            volume_regime = await self.comm_framework.request(
                "market_data.get_indicator_value",
                {"indicator_name": "VOLUME_REGIME", "symbol": "PLATFORM_WIDE"}
            )
            
            # Use regime detector model to classify
            regime_features = [regime_indicator, volatility, trend_strength, volume_regime]
            detected_regime = self.regime_detector.detect_regime(regime_features)
            
            # Update regime history
            regime_entry = {
                "regime": detected_regime,
                "timestamp": datetime.now().isoformat(),
                "confidence": self.regime_detector.get_confidence(),
                "features": regime_features
            }
            
            self.market_regime_history.append(regime_entry)
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(hours=72)
            self.market_regime_history = [
                entry for entry in self.market_regime_history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
            ]
            
            self.logger.debug(f"Current market regime detected: {detected_regime}")
            return detected_regime
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to identify market regime: {e}"),
                "_identify_market_regime"
            )
            return "UNKNOWN"

    async def _get_comprehensive_performance_data(self) -> List[Dict[str, Any]]:
        """Get comprehensive performance data from Analytics Framework"""
        try:
            # Request performance data from all analytics services
            performance_data = []
            
            # Get performance from SessionAnalytics
            session_performance = await self.comm_framework.request(
                "analytics.session_analytics.get_strategy_performance",
                {"time_frame": "hourly", "lookback_hours": self.config["strategy_performance_window"]}
            )
            performance_data.extend(session_performance.get('strategies', []))
            
            # Get performance from LiveStrategyMonitor
            live_performance = await self.comm_framework.request(
                "analytics.live_strategy_monitor.get_current_performance",
                {"include_details": True}
            )
            performance_data.extend(live_performance.get('strategies', []))
            
            # Get performance comparison data
            comparison_data = await self.comm_framework.request(
                "analytics.performance_comparator.get_strategy_rankings",
                {"time_frame": "daily"}
            )
            
            # Merge and enrich performance data
            enriched_data = await self._enrich_performance_data(performance_data, comparison_data)
            
            self.logger.debug(f"Retrieved performance data for {len(enriched_data)} strategies.")
            return enriched_data
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to get comprehensive performance data: {e}"),
                "_get_comprehensive_performance_data"
            )
            return []

    async def _enrich_performance_data(self, performance_data: List[Dict], 
                                     comparison_data: Dict) -> List[Dict[str, Any]]:
        """Enrich performance data with additional metrics and analysis"""
        try:
            enriched_strategies = []
            
            for strategy in performance_data:
                strategy_id = strategy.get('strategy_id')
                
                # Add regime-specific performance analysis
                regime_performance = await self._analyze_regime_specific_performance(strategy_id)
                strategy['regime_performance'] = regime_performance
                
                # Add risk-adjusted metrics
                risk_metrics = await self._calculate_risk_adjusted_metrics(strategy)
                strategy['risk_metrics'] = risk_metrics
                
                # Add ranking information
                ranking_info = comparison_data.get('rankings', {}).get(strategy_id, {})
                strategy['ranking'] = ranking_info
                
                # Add optimal regime classification
                strategy['optimal_regimes'] = await self._identify_optimal_regimes(strategy_id)
                
                enriched_strategies.append(strategy)
            
            return enriched_strategies
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to enrich performance data: {e}"),
                "_enrich_performance_data"
            )
            return performance_data

    async def _generate_adaptation_recommendations(self, performance_data: List[Dict[str, Any]], 
                                                 current_regime: str) -> List[Dict[str, Any]]:
        """Generate intelligent adaptation recommendations"""
        try:
            recommendations = []
            
            for strategy in performance_data:
                strategy_id = strategy.get('strategy_id')
                current_performance = strategy.get('pnl', 0)
                recent_drawdown = strategy.get('recent_drawdown', 0)
                optimal_regimes = strategy.get('optimal_regimes', [])
                current_weight = strategy.get('current_weight', 0)
                
                # Analyze strategy suitability for current regime
                is_optimal_for_regime = current_regime in optimal_regimes
                regime_performance = strategy.get('regime_performance', {}).get(current_regime, {})
                regime_sharpe = regime_performance.get('sharpe_ratio', 0)
                
                # Generate weight adjustment recommendations
                recommendation = await self._generate_weight_recommendation(
                    strategy_id, current_performance, recent_drawdown,
                    is_optimal_for_regime, regime_sharpe, current_weight, current_regime
                )
                
                if recommendation:
                    recommendations.append(recommendation)
            
            # Generate strategy activation/deactivation recommendations
            activation_recs = await self._generate_activation_recommendations(
                performance_data, current_regime
            )
            recommendations.extend(activation_recs)
            
            self.logger.info(f"Generated {len(recommendations)} adaptation recommendations.")
            return recommendations
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to generate adaptation recommendations: {e}"),
                "_generate_adaptation_recommendations"
            )
            return []

    async def _generate_weight_recommendation(self, strategy_id: str, performance: float,
                                            drawdown: float, is_optimal: bool, 
                                            regime_sharpe: float, current_weight: float,
                                            regime: str) -> Optional[Dict[str, Any]]:
        """Generate weight adjustment recommendation for a strategy"""
        try:
            # Decision logic for weight adjustments
            target_weight = current_weight
            adjustment_reason = ""
            
            # Increase weight if performing well in current regime
            if is_optimal and performance > 0 and regime_sharpe > 1.0 and drawdown < 0.01:
                target_weight = min(current_weight * 1.2, 0.3)  # Max 30% allocation
                adjustment_reason = f"Outperforming in {regime} regime (Sharpe: {regime_sharpe:.2f})"
                
            # Decrease weight if underperforming in current regime
            elif not is_optimal and (performance < 0 or drawdown > 0.03):
                target_weight = max(current_weight * 0.5, 0.01)  # Min 1% allocation
                adjustment_reason = f"Underperforming in {regime} regime"
                  # Moderate adjustment for mixed signals
            elif is_optimal and performance > 0 and regime_sharpe < 0.5:
                target_weight = current_weight * 1.1
                adjustment_reason = f"Moderate increase for {regime} regime suitability"
                
            # Only recommend if significant change
            if abs(target_weight - current_weight) > 0.05:
                return {
                    "type": "adjust_model_weight",
                    "model_id": strategy_id,
                    "current_weight": current_weight,
                    "target_weight": round(target_weight, 3),
                    "adjustment_factor": round(target_weight / current_weight, 3) if current_weight > 0 else 1.0,
                    "reason": adjustment_reason,
                    "regime": regime,
                    "priority": "medium"
                }
            
            return None
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to generate weight recommendation for {strategy_id}: {e}"),
                "_generate_weight_recommendation"
            )
            return None

    async def _generate_activation_recommendations(self, performance_data: List[Dict[str, Any]], 
                                                 current_regime: str) -> List[Dict[str, Any]]:
        """Generate strategy activation/deactivation recommendations"""
        try:
            recommendations = []
            
            # Identify severely underperforming strategies for deactivation
            for strategy in performance_data:
                strategy_id = strategy.get('strategy_id')
                performance = strategy.get('pnl', 0)
                drawdown = strategy.get('recent_drawdown', 0)
                consecutive_losses = strategy.get('consecutive_losses', 0)
                
                # Deactivation criteria
                if (performance < self.config["underperforming_strategy_threshold"] and 
                    drawdown > 0.05 and consecutive_losses > 10):
                    
                    recommendations.append({
                        "type": "deactivate_strategy",
                        "model_id": strategy_id,
                        "reason": f"Severe underperformance: PnL {performance:.3f}, Drawdown {drawdown:.3f}",
                        "regime": current_regime,
                        "priority": "high"
                    })
                    self.adaptation_metrics["strategies_deactivated"] += 1
            
            # Check for strategies to reactivate based on regime change
            inactive_strategies = await self._get_inactive_strategies()
            for strategy_id in inactive_strategies:
                if await self._should_reactivate_strategy(strategy_id, current_regime):
                    recommendations.append({
                        "type": "activate_strategy",
                        "model_id": strategy_id,
                        "reason": f"Suitable for {current_regime} regime",
                        "regime": current_regime,
                        "priority": "medium"
                    })
            
            return recommendations
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to generate activation recommendations: {e}"),
                "_generate_activation_recommendations"
            )
            return []

    async def _apply_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Apply adaptation recommendations through AIModelCoordinator"""
        try:
            for recommendation in recommendations:
                rec_type = recommendation.get('type')
                
                if rec_type == "adjust_model_weight":
                    await self._apply_weight_adjustment(recommendation)
                elif rec_type == "activate_strategy":
                    await self._apply_strategy_activation(recommendation)
                elif rec_type == "deactivate_strategy":
                    await self._apply_strategy_deactivation(recommendation)
                
                # Log successful application
                self.logger.info(f"Applied recommendation: {recommendation.get('type')} for {recommendation.get('model_id')}")
            
            # Publish adaptation summary
            await self.comm_framework.publish(
                "ai_model.adaptive_strategy_recommendations",
                {
                    "total_recommendations": len(recommendations),
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.adaptation_metrics["successful_adaptations"] += len(recommendations)
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to apply recommendations: {e}"),
                "_apply_recommendations"
            )

    async def _apply_weight_adjustment(self, recommendation: Dict[str, Any]):
        """Apply weight adjustment recommendation"""
        try:
            await self.comm_framework.request(
                "ai_model_coordinator.adjust_model_weight",
                {
                    "model_id": recommendation["model_id"],
                    "new_weight": recommendation["target_weight"],
                    "reason": recommendation["reason"]
                }
            )
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to apply weight adjustment: {e}"),
                "_apply_weight_adjustment"
            )

    async def _apply_strategy_activation(self, recommendation: Dict[str, Any]):
        """Apply strategy activation recommendation"""
        try:
            await self.comm_framework.request(
                "ai_model_coordinator.activate_model",
                {
                    "model_id": recommendation["model_id"],
                    "reason": recommendation["reason"]
                }
            )
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to apply strategy activation: {e}"),
                "_apply_strategy_activation"
            )

    async def _apply_strategy_deactivation(self, recommendation: Dict[str, Any]):
        """Apply strategy deactivation recommendation"""
        try:
            await self.comm_framework.request(
                "ai_model_coordinator.deactivate_model",
                {
                    "model_id": recommendation["model_id"],
                    "reason": recommendation["reason"]
                }
            )
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to apply strategy deactivation: {e}"),
                "_apply_strategy_deactivation"
            )

    async def _should_generate_new_strategy(self, performance_data: List[Dict[str, Any]], 
                                          current_regime: str) -> bool:
        """Determine if a new strategy should be generated"""
        try:
            # Check if there's a performance gap in the current regime
            regime_strategies = [
                s for s in performance_data 
                if current_regime in s.get('optimal_regimes', [])
            ]
            
            if not regime_strategies:
                self.logger.info(f"No strategies optimized for {current_regime} regime. Generation recommended.")
                return True
            
            # Check if existing strategies are underperforming
            avg_performance = sum(s.get('pnl', 0) for s in regime_strategies) / len(regime_strategies)
            if avg_performance < self.config["new_strategy_generation_threshold"]:
                self.logger.info(f"Existing strategies underperforming in {current_regime} regime. Generation recommended.")
                return True
            
            # Check if we have room for more strategies
            active_count = len([s for s in performance_data if s.get('is_active', True)])
            if active_count < self.config["max_active_strategies"]:
                return True
            
            return False
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to determine if new strategy should be generated: {e}"),
                "_should_generate_new_strategy"
            )
            return False

    async def _generate_new_strategy(self, current_regime: str):
        """Generate a new trading strategy using advanced ML techniques"""
        try:
            self.logger.info(f"Generating new strategy for {current_regime} regime...")
            
            # Collect training data for strategy generation
            training_data = await self._collect_strategy_training_data(current_regime)
            
            # Generate strategy using ML model
            new_strategy = await self.strategy_generator.generate_strategy(
                regime=current_regime,
                training_data=training_data,
                existing_strategies=list(self.strategy_registry.keys())
            )
            
            if new_strategy:
                # Validate and register new strategy
                strategy_id = await self._register_new_strategy(new_strategy, current_regime)
                
                # Start backtesting
                await self._initiate_strategy_backtesting(strategy_id, new_strategy)
                
                self.adaptation_metrics["strategies_generated"] += 1
                self.logger.info(f"New strategy generated: {strategy_id}")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to generate new strategy: {e}"),
                "_generate_new_strategy"
            )

    async def _collect_strategy_training_data(self, regime: str) -> Dict[str, Any]:
        """Collect training data for strategy generation"""
        try:
            # Get historical market data for the regime
            historical_data = await self.comm_framework.request(
                "market_data.get_historical_regime_data",
                {"regime": regime, "lookback_days": 30}
            )
            
            # Get performance patterns of existing strategies
            strategy_patterns = await self.comm_framework.request(
                "analytics.get_strategy_patterns",
                {"regime": regime, "include_features": True}
            )
            
            # Get indicator effectiveness in this regime
            indicator_effectiveness = await self.comm_framework.request(
                "analytics.get_indicator_effectiveness",
                {"regime": regime}
            )
            
            return {
                "historical_data": historical_data,
                "strategy_patterns": strategy_patterns,
                "indicator_effectiveness": indicator_effectiveness,
                "regime": regime
            }
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to collect training data: {e}"),
                "_collect_strategy_training_data"
            )
            return {}

    async def _register_new_strategy(self, strategy: Dict[str, Any], regime: str) -> str:
        """Register a new strategy in the system"""
        try:
            # Generate unique strategy ID
            strategy_id = f"Generated_{regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare strategy metadata
            strategy_metadata = {
                "strategy_id": strategy_id,
                "name": strategy.get("name", f"Generated Strategy for {regime}"),
                "description": strategy.get("description", f"AI-generated strategy optimized for {regime} regime"),
                "creation_timestamp": datetime.now().isoformat(),
                "target_regime": regime,
                "generation_method": strategy.get("method", "ML_Generation"),
                "parameters": strategy.get("parameters", {}),
                "rules": strategy.get("rules", []),
                "initial_weight": 0.05,  # Start with small allocation
                "status": "backtesting"
            }
            
            # Register with Model Registry
            await self.comm_framework.request(
                "model_registry.register_strategy",
                strategy_metadata
            )
            
            # Add to local registry
            self.strategy_registry[strategy_id] = strategy_metadata
            
            return strategy_id
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to register new strategy: {e}"),
                "_register_new_strategy"
            )
            return ""

    async def _initiate_strategy_backtesting(self, strategy_id: str, strategy: Dict[str, Any]):
        """Initiate backtesting for a new strategy"""
        try:
            backtest_request = {
                "strategy_id": strategy_id,
                "strategy_definition": strategy,
                "test_period_days": 7,
                "initial_capital": 10000,
                "performance_thresholds": {
                    "min_sharpe": 0.5,
                    "max_drawdown": 0.1,
                    "min_win_rate": 0.4
                }
            }
            
            await self.comm_framework.publish(
                "backtesting.initiate_test",
                backtest_request
            )
            
            self.logger.info(f"Initiated backtesting for strategy {strategy_id}")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to initiate backtesting: {e}"),
                "_initiate_strategy_backtesting"
            )

    async def _analyze_regime_specific_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Analyze strategy performance in different market regimes"""
        try:
            # Get historical performance data
            performance_history = self.strategy_performance_history.get(strategy_id, [])
            
            regime_performance = {}
            
            # Group performance by regime
            for regime_entry in self.market_regime_history[-100:]:  # Last 100 regime observations
                regime = regime_entry["regime"]
                timestamp = datetime.fromisoformat(regime_entry["timestamp"])
                
                # Find performance data for this time period
                matching_performance = [
                    p for p in performance_history
                    if abs((datetime.fromisoformat(p["timestamp"]) - timestamp).total_seconds()) < 3600
                ]
                
                if matching_performance:
                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    regime_performance[regime].extend(matching_performance)
            
            # Calculate regime-specific metrics
            regime_metrics = {}
            for regime, performances in regime_performance.items():
                if len(performances) >= 3:  # Minimum sample size
                    pnls = [p.get("pnl", 0) for p in performances]
                    regime_metrics[regime] = {
                        "average_pnl": sum(pnls) / len(pnls),
                        "sharpe_ratio": self._calculate_sharpe_ratio(pnls),
                        "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                        "max_drawdown": self._calculate_max_drawdown(pnls),
                        "sample_size": len(performances)
                    }
            
            return regime_metrics
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to analyze regime-specific performance: {e}"),
                "_analyze_regime_specific_performance"
            )
            return {}

    async def _calculate_risk_adjusted_metrics(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        try:
            pnl_history = strategy.get('pnl_history', [])
            if len(pnl_history) < 10:
                return {}
            
            returns = [p for p in pnl_history if p is not None]
            
            return {
                "sharpe_ratio": self._calculate_sharpe_ratio(returns),
                "sortino_ratio": self._calculate_sortino_ratio(returns),
                "max_drawdown": self._calculate_max_drawdown(returns),
                "calmar_ratio": self._calculate_calmar_ratio(returns),
                "volatility": np.std(returns) if returns else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate risk-adjusted metrics: {e}")
            return {}

    async def _identify_optimal_regimes(self, strategy_id: str) -> List[str]:
        """Identify optimal market regimes for a strategy"""
        try:
            regime_performance = await self._analyze_regime_specific_performance(strategy_id)
            
            optimal_regimes = []
            for regime, metrics in regime_performance.items():
                if (metrics.get("sharpe_ratio", 0) > 0.8 and 
                    metrics.get("average_pnl", 0) > 0.001 and
                    metrics.get("win_rate", 0) > 0.5):
                    optimal_regimes.append(regime)
            
            return optimal_regimes
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to identify optimal regimes: {e}"),
                "_identify_optimal_regimes"
            )
            return []

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return / std_return) * np.sqrt(252)  # Annualized

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        
        return (mean_return / downside_deviation) * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio"""
        if not returns:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return float('inf')
        
        return annual_return / max_dd

    async def _get_inactive_strategies(self) -> List[str]:
        """Get list of inactive strategies"""
        try:
            inactive_list = await self.comm_framework.request(
                "model_registry.get_inactive_strategies", {}
            )
            return inactive_list.get('strategy_ids', [])
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to get inactive strategies: {e}"),
                "_get_inactive_strategies"
            )
            return []

    async def _should_reactivate_strategy(self, strategy_id: str, current_regime: str) -> bool:
        """Determine if an inactive strategy should be reactivated"""
        try:
            strategy_info = self.strategy_registry.get(strategy_id, {})
            target_regime = strategy_info.get('target_regime')
            
            # Reactivate if strategy was designed for current regime
            if target_regime == current_regime:
                return True
            
            # Check historical performance in current regime
            regime_performance = await self._analyze_regime_specific_performance(strategy_id)
            current_regime_perf = regime_performance.get(current_regime, {})
            
            if (current_regime_perf.get('sharpe_ratio', 0) > 1.0 and
                current_regime_perf.get('average_pnl', 0) > 0):
                return True
            
            return False
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to determine reactivation for {strategy_id}: {e}"),
                "_should_reactivate_strategy"
            )
            return False

    async def _handle_performance_update(self, message: Dict[str, Any]):
        """Handle performance updates from Analytics Framework"""
        try:
            performance_data = message.get('data', {})
            strategy_id = performance_data.get('strategy_id')
            
            if not strategy_id:
                return
            
            # Update performance history
            if strategy_id not in self.strategy_performance_history:
                self.strategy_performance_history[strategy_id] = []
            
            performance_entry = {
                "timestamp": datetime.now().isoformat(),
                "pnl": performance_data.get('pnl', 0),
                "trades": performance_data.get('trades', 0),
                "win_rate": performance_data.get('win_rate', 0),
                "drawdown": performance_data.get('drawdown', 0)
            }
            
            self.strategy_performance_history[strategy_id].append(performance_entry)
            
            # Keep only recent history
            max_history = 1000
            if len(self.strategy_performance_history[strategy_id]) > max_history:
                self.strategy_performance_history[strategy_id] = \
                    self.strategy_performance_history[strategy_id][-max_history:]
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to handle performance update: {e}"),
                "_handle_performance_update"
            )

    async def _handle_regime_change(self, message: Dict[str, Any]):
        """Handle market regime change notifications"""
        try:
            regime_data = message.get('data', {})
            new_regime = regime_data.get('regime')
            confidence = regime_data.get('confidence', 0)
            
            if not new_regime:
                return
            
            self.logger.info(f"Market regime change detected: {new_regime} (confidence: {confidence:.2f})")
            
            # Trigger immediate adaptation if high confidence
            if confidence > 0.8:
                self.adaptation_metrics["regime_switches_detected"] += 1
                
                # Get current performance data
                performance_data = await self._get_comprehensive_performance_data()
                
                # Generate emergency recommendations
                recommendations = await self._generate_adaptation_recommendations(
                    performance_data, new_regime
                )
                
                if recommendations:
                    await self._apply_recommendations(recommendations)
                    self.logger.info(f"Applied {len(recommendations)} emergency adaptations for regime change.")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to handle regime change: {e}"),
                "_handle_regime_change"
            )

    async def _continuous_monitoring(self):
        """Continuous monitoring of strategy performance and system health"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Check for critical performance issues
                await self._check_critical_performance_issues()
                
                # Update adaptation metrics
                await self._update_adaptation_metrics()
                
                # Publish health status
                await self._publish_health_status()
                
            except Exception as e:
                self.error_system.handle_error(
                    MLError(f"Error in continuous monitoring: {e}"),
                    "_continuous_monitoring"
                )

    async def _check_critical_performance_issues(self):
        """Check for critical performance issues requiring immediate action"""
        try:
            # Check if any strategy is experiencing severe losses
            for strategy_id, history in self.strategy_performance_history.items():
                if len(history) >= 5:
                    recent_pnls = [h.get('pnl', 0) for h in history[-5:]]
                    if all(pnl < -0.01 for pnl in recent_pnls):  # 5 consecutive losses > 1%
                        # Emergency deactivation
                        await self.comm_framework.publish(
                            "ai_model.emergency_deactivation",
                            {
                                "model_id": strategy_id,
                                "reason": "Critical consecutive losses detected",
                                "priority": "critical"
                            }
                        )
                        self.logger.critical(f"Emergency deactivation triggered for {strategy_id}")
                        
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to check critical performance issues: {e}"),
                "_check_critical_performance_issues"
            )

    async def _update_adaptation_metrics(self):
        """Update internal adaptation performance metrics"""
        try:
            # Calculate adaptation success rate
            if self.adaptation_metrics["total_adaptations"] > 0:
                success_rate = (self.adaptation_metrics["successful_adaptations"] / 
                              self.adaptation_metrics["total_adaptations"])
                self.adaptation_metrics["adaptation_success_rate"] = success_rate
            
            # Calculate average performance improvement
            # This would require tracking before/after performance
            # Placeholder implementation
            self.adaptation_metrics["last_update"] = datetime.now().isoformat()
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to update adaptation metrics: {e}"),
                "_update_adaptation_metrics"
            )

    async def _publish_health_status(self):
        """Publish service health status and metrics"""
        try:
            health_status = {
                "service": "AdaptiveStrategyGenerator",
                "status": "healthy" if self.is_running else "stopped",
                "active_strategies": len(self.strategy_registry),
                "performance_history_size": sum(len(h) for h in self.strategy_performance_history.values()),
                "regime_history_size": len(self.market_regime_history),
                "adaptation_metrics": self.adaptation_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.comm_framework.publish(
                "monitoring.service_health",
                health_status
            )
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to publish health status: {e}"),
                "_publish_health_status"
            )

    async def _save_strategy_state(self):
        """Save current strategy state for persistence"""
        try:
            state_data = {
                "strategy_registry": self.strategy_registry,
                "adaptation_metrics": self.adaptation_metrics,
                "last_save": datetime.now().isoformat()
            }
            
            # Save to persistent storage
            await self.comm_framework.request(
                "persistence.save_state",
                {
                    "service": "AdaptiveStrategyGenerator",
                    "state_data": state_data
                }
            )
            
            self.logger.info("Strategy state saved successfully.")
            
        except Exception as e:
            self.error_system.handle_error(
                MLError(f"Failed to save strategy state: {e}"),
                "_save_strategy_state"
            )


# Placeholder ML Model Classes
class MarketRegimeDetector:
    """Placeholder for sophisticated market regime detection model"""
    def detect_regime(self, features: List[float]) -> str:
        # Simplified regime detection logic
        if not features or len(features) < 4:
            return "NORMAL"
        
        regime_indicator = features[0]
        volatility = features[1]
        
        if volatility > 0.008:
            return "VOLATILE"
        elif regime_indicator > 0.7:
            return "TRENDING"
        elif regime_indicator < 0.3:
            return "RANGE_BOUND"
        else:
            return "NORMAL"
    
    def get_confidence(self) -> float:
        return 0.85  # Placeholder confidence

class StrategyPerformanceEvaluator:
    """Placeholder for strategy performance evaluation model"""
    def evaluate_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, float]:
        # Simplified evaluation logic
        return {
            "expected_return": 0.001,
            "risk_score": 0.5,
            "complexity_score": 0.3
        }

class StrategyGenerationEngine:
    """Placeholder for advanced strategy generation using RL/Genetic Programming"""
    async def generate_strategy(self, regime: str, training_data: Dict[str, Any], 
                              existing_strategies: List[str]) -> Dict[str, Any]:
        # Simplified strategy generation
        await asyncio.sleep(1)  # Simulate generation time
        
        strategy_name = f"Generated_{regime}_{len(existing_strategies)}"
        
        return {
            "name": strategy_name,
            "description": f"AI-generated strategy for {regime} market conditions",
            "method": "Genetic_Programming",
            "parameters": {
                "lookback_period": 20,
                "entry_threshold": 0.02,
                "exit_threshold": 0.01,
                "stop_loss": 0.015,
                "take_profit": 0.03
            },
            "rules": [
                f"Enter long when momentum > 0.02 in {regime} regime",
                f"Exit when profit > 3% or loss > 1.5%",
                "Use adaptive position sizing based on volatility"
            ],
            "indicators_used": ["MOMENTUM", "VOLATILITY_REGIME", "TREND_STRENGTH"],
            "expected_performance": {
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.55
            }
        }


# Microservice entry point
async def main():
    """Main entry point for the Adaptive Strategy Generator microservice"""
    try:
        # Initialize communication framework
        comm_framework = Platform3CommunicationFramework()
        await comm_framework.initialize()
        
        # Create and start the strategy generator
        strategy_generator = AdaptiveStrategyGenerator(comm_framework)
        await strategy_generator.start()
        
        # Keep the service running
        while strategy_generator.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down Adaptive Strategy Generator...")
    except Exception as e:
        print(f"Fatal error in Adaptive Strategy Generator: {e}")
    finally:
        if 'strategy_generator' in locals():
            await strategy_generator.stop()

if __name__ == "__main__":
    asyncio.run(main())
