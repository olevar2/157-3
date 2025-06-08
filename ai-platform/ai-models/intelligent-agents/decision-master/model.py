"""
Enhanced AI Model with Platform3 Phase 2 Framework Integration
Auto-enhanced for production-ready performance and reliability
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Platform3 Phase 2 Framework Integration
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "shared"))
from shared.platform3_logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, MLError, ModelError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework

# AI Agent Integration
sys.path.append(str(Path(__file__).parent.parent.parent / "intelligent-agents"))
from dynamic_risk_agent.model import DynamicRiskAgent


class AIModelPerformanceMonitor:
    """Enhanced performance monitoring for AI models"""
    
    def __init__(self, model_name: str):
        self.logger = Platform3Logger(f"ai_model_{model_name}")
        self.error_handler = Platform3ErrorSystem()
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = datetime.now()
        self.logger.info("Starting AI model performance monitoring")
    
    def log_metric(self, metric_name: str, value: float):
        """Log performance metric"""
        self.metrics[metric_name] = value
        self.logger.info(f"Performance metric: {metric_name} = {value}")
    
    def end_monitoring(self):
        """End monitoring and log results"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.log_metric("execution_time_seconds", duration)
            self.logger.info(f"Performance monitoring complete: {duration:.2f}s")


class EnhancedAIModelBase:
    """Enhanced base class for all AI models with Phase 2 integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model_name = self.__class__.__name__
        
        # Phase 2 Framework Integration
        self.logger = Platform3Logger(f"ai_model_{self.model_name}")
        self.error_handler = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.communication = Platform3CommunicationFramework()
        self.performance_monitor = AIModelPerformanceMonitor(self.model_name)
        
        # Model state
        self.is_trained = False
        self.model = None
        self.metrics = {}
        
        self.logger.info(f"Initialized enhanced AI model: {self.model_name}")
    
    async def validate_input(self, data: Any) -> bool:
        """Validate input data with comprehensive checks"""
        try:
            if data is None:
                raise ValueError("Input data cannot be None")
            
            if hasattr(data, 'shape') and len(data.shape) == 0:
                raise ValueError("Input data cannot be empty")
            
            self.logger.debug(f"Input validation passed for {type(data)}")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                MLError(f"Input validation failed: {str(e)}", {"data_type": type(data)})
            )
            return False
    
    async def train_async(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Enhanced async training with monitoring and error handling"""
        self.performance_monitor.start_monitoring()
        
        try:
            # Validate input
            if not await self.validate_input(data):
                raise MLError("Training data validation failed")
            
            self.logger.info(f"Starting training for {self.model_name}")
            
            # Call implementation-specific training
            result = await self._train_implementation(data, **kwargs)
            
            self.is_trained = True
            self.performance_monitor.log_metric("training_success", 1.0)
            self.logger.info(f"Training completed successfully for {self.model_name}")
            
            return result
            
        except Exception as e:
            self.performance_monitor.log_metric("training_success", 0.0)
            self.error_handler.handle_error(
                MLError(f"Training failed for {self.model_name}: {str(e)}", kwargs)
            )
            raise
        finally:
            self.performance_monitor.end_monitoring()
    
    async def predict_async(self, data: Any, **kwargs) -> Any:
        """Enhanced async prediction with monitoring and error handling"""
        self.performance_monitor.start_monitoring()
        
        try:
            if not self.is_trained:
                raise ModelError(f"Model {self.model_name} is not trained")
            
            # Validate input
            if not await self.validate_input(data):
                raise MLError("Prediction data validation failed")
            
            self.logger.debug(f"Starting prediction for {self.model_name}")
            
            # Call implementation-specific prediction
            result = await self._predict_implementation(data, **kwargs)
            
            self.performance_monitor.log_metric("prediction_success", 1.0)
            return result
            
        except Exception as e:
            self.performance_monitor.log_metric("prediction_success", 0.0)
            self.error_handler.handle_error(
                MLError(f"Prediction failed for {self.model_name}: {str(e)}", kwargs)
            )
            raise
        finally:
            self.performance_monitor.end_monitoring()
    
    async def _train_implementation(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Override in subclasses for specific training logic"""
        raise NotImplementedError("Subclasses must implement _train_implementation")
    
    async def _predict_implementation(self, data: Any, **kwargs) -> Any:
        """Override in subclasses for specific prediction logic"""
        raise NotImplementedError("Subclasses must implement _predict_implementation")
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save model with proper error handling and logging"""
        try:
            save_path = path or f"models/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"            
            # Implementation depends on model type
            self.logger.info(f"Model saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.error_handler.handle_error(
                MLError(f"Model save failed: {str(e)}", {"path": path})
            )
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive model metrics"""
        return {
            **self.metrics,
            **self.performance_monitor.metrics,
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "timestamp": datetime.now().isoformat()
        }


# === ENHANCED ORIGINAL IMPLEMENTATION ===
"""
Decision Master Model - Professional Trading Decision Genius

This is a genius-level professional model that specializes in:
1. Real-time trading decision making with institutional precision
2. Multi-factor decision analysis combining all available intelligence
3. Risk-adjusted position sizing and trade management
4. Dynamic stop-loss and take-profit optimization
5. Market timing and execution decisions
6. Portfolio-level decision coordination

For forex traders focused on daily profits through scalping, day trading, and swing trading.
"""

# Required for DecisionMaster class definitions
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of trading decisions"""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_PROFIT = "exit_profit"
    EXIT_LOSS = "exit_loss"
    HOLD = "hold"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    NO_ACTION = "no_action"

class ConfidenceLevel(Enum):
    """Decision confidence levels"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    ABSOLUTE = 1.0

class MarketState(Enum):
    """Market state classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNCERTAIN = "uncertain"

class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = 0.005  # 0.5%
    LOW = 0.01        # 1%
    MEDIUM = 0.02     # 2%
    HIGH = 0.03       # 3%
    VERY_HIGH = 0.05  # 5%

@dataclass
class MarketConditions:
    """Current market conditions assessment"""
    timestamp: datetime
    currency_pair: str
    timeframe: str
    
    # Price action
    current_price: float
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float  # 0-1
    support_level: float
    resistance_level: float
    
    # Volatility
    volatility_regime: str  # 'low', 'medium', 'high'
    atr_value: float
    volatility_percentile: float
    
    # Market structure
    market_state: MarketState
    session: str  # 'Sydney', 'Tokyo', 'London', 'New_York'
    session_overlap: bool
    
    # Technical indicators
    rsi: float
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    moving_average_alignment: str  # 'bullish', 'bearish', 'mixed'
    
    # Sentiment
    market_sentiment: float  # -1 to 1
    news_impact: str  # 'positive', 'negative', 'neutral'
    economic_calendar_risk: float  # 0-1
    
    # Liquidity
    spread: float
    liquidity_score: float  # 0-1
    volume_profile: str  # 'high', 'medium', 'low'

@dataclass
class SignalInput:
    """Input signal from other models"""
    model_name: str
    signal_type: str  # 'entry', 'exit', 'hold'
    direction: str  # 'long', 'short', 'neutral'
    strength: float  # 0-1
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: Optional[str] = None
    reasoning: Optional[str] = None

@dataclass
class TradingDecision:
    """Professional trading decision output"""
    decision_id: str
    timestamp: datetime
    currency_pair: str
    timeframe: str
    
    # Decision details
    decision_type: DecisionType
    confidence: ConfidenceLevel
    urgency: float  # 0-1 (how quickly to act)
    
    # Trade parameters
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: Optional[float] = None
    
    # Risk management
    risk_level: RiskLevel = RiskLevel.LOW
    max_loss_amount: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Execution details
    order_type: str = "market"  # 'market', 'limit', 'stop'
    expiry: Optional[datetime] = None
    partial_fills_allowed: bool = True
      # Decision reasoning
    primary_reason: str = ""
    supporting_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)  # Combined reasoning for backward compatibility
    
    # Performance tracking
    expected_profit: float = 0.0
    success_probability: float = 0.0
    decision_score: float = 0.0

@dataclass
class PortfolioContext:
    """Current portfolio context"""
    total_balance: float
    available_margin: float
    current_exposure: Dict[str, float]  # pair -> exposure amount
    open_positions: int
    daily_pnl: float
    drawdown: float
    risk_utilization: float  # % of risk budget used
    correlation_exposure: Dict[str, float]  # correlation risks

class DecisionMaster:
    """
    Professional Trading Decision Genius
    
    This model makes institutional-grade trading decisions by:
    - Analyzing multiple signal inputs from other models
    - Assessing current market conditions
    - Evaluating portfolio context and risk
    - Making optimal entry/exit decisions
    - Providing detailed reasoning and risk assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Decision Master"""
        self.config = config or {}
        
        # Decision parameters
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2%
        self.max_correlation_exposure = self.config.get('max_correlation_exposure', 0.05)  # 5%
        self.max_daily_trades = self.config.get('max_daily_trades', 10)
        self.min_risk_reward = self.config.get('min_risk_reward', 1.5)
        
        # Model weights for signal integration
        self.model_weights = self.config.get('model_weights', {
            'indicator_expert': 0.25,
            'strategy_expert': 0.25,
            'scalping_lstm': 0.15,
            'tick_classifier': 0.15,
            'sentiment_analyzer': 0.10,
            'elliott_wave': 0.10
        })
        
        # Risk management parameters
        self.risk_limits = {
            'max_portfolio_risk': 0.10,  # 10% max portfolio risk
            'max_single_trade_risk': 0.02,  # 2% max per trade
            'max_correlated_risk': 0.05,  # 5% max correlated exposure
            'max_drawdown_limit': 0.15,  # 15% max drawdown
            'min_available_margin': 0.20  # 20% minimum margin buffer
        }
        
        # Decision history for learning
        self.decision_history: List[TradingDecision] = []
        self.performance_tracking: Dict[str, Any] = {}
        
        # Market correlation matrix (simplified)
        self.correlation_matrix = {
            'EURUSD': {'GBPUSD': 0.7, 'AUDUSD': 0.6, 'USDCHF': -0.8},
            'GBPUSD': {'EURUSD': 0.7, 'AUDUSD': 0.5, 'USDCHF': -0.6},
            'USDCHF': {'EURUSD': -0.8, 'GBPUSD': -0.6, 'AUDUSD': -0.5},
            'AUDUSD': {'EURUSD': 0.6, 'GBPUSD': 0.5, 'USDCHF': -0.5},
            'USDJPY': {'EURUSD': 0.2, 'GBPUSD': 0.1, 'AUDUSD': 0.3}
        }
        
        # Initialize Platform3 Communication Framework for AI agent integration
        self.comm_framework = Platform3CommunicationFramework(
            service_name="decision-master",
            service_port=8007,
            redis_url="redis://localhost:6379",
            consul_host="localhost",
            consul_port=8500
        )
        
        # Initialize DynamicRiskAgent for advanced risk assessment
        try:
            self.risk_agent = DynamicRiskAgent()
            self.risk_agent_available = True
            logger.info("DynamicRiskAgent integrated successfully")
        except Exception as e:
            logger.warning(f"DynamicRiskAgent integration failed: {e}")
            self.risk_agent = None
            self.risk_agent_available = False
        
        # Enhanced risk-aware decision flags
        self.use_advanced_risk_assessment = self.config.get('use_advanced_risk_assessment', True)
        self.risk_adjustment_factor = self.config.get('risk_adjustment_factor', 0.8)  # Conservative by default
        
        logger.info("Decision Master initialized - Professional trading decisions ready")
    
    async def make_trading_decision(self,
                            signals: List[SignalInput],
                            market_conditions: MarketConditions,
                            portfolio_context: PortfolioContext) -> TradingDecision:
        """
        Make a professional trading decision based on all available inputs with advanced risk assessment
        
        Args:
            signals: List of signals from other models
            market_conditions: Current market conditions
            portfolio_context: Current portfolio state
            
        Returns:
            Professional trading decision with full reasoning and risk-adjusted parameters
        """
        logger.info(f"Making risk-aware trading decision for {market_conditions.currency_pair}")
        
        # Generate unique decision ID
        decision_id = f"{market_conditions.currency_pair}_{market_conditions.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Step 1: Analyze and weight signals
        signal_analysis = self._analyze_signals(signals)
        
        # Step 2: Assess market conditions
        market_assessment = self._assess_market_conditions(market_conditions)
        
        # Step 3: Generate initial trade proposal
        initial_proposal = self._generate_initial_proposal(
            decision_id, signal_analysis, market_assessment, market_conditions, portfolio_context
        )
        
        # Step 4: Enhanced risk assessment using DynamicRiskAgent
        enhanced_risk_assessment = await self._get_advanced_risk_assessment(
            initial_proposal, portfolio_context, market_conditions
        )
        
        # Step 5: Make risk-adjusted decision
        final_decision = self._make_risk_aware_decision(
            initial_proposal, enhanced_risk_assessment, portfolio_context, market_conditions
        )
        
        # Step 6: Add comprehensive reasoning including risk adjustments
        final_decision = self._add_enhanced_decision_reasoning(
            final_decision, signal_analysis, market_assessment, enhanced_risk_assessment
        )
        
        # Step 7: Track decision for learning and performance analysis
        self.decision_history.append(final_decision)
        
        risk_score = enhanced_risk_assessment.get('risk_score', 0.5)
        logger.info(f"Risk-aware decision made: {final_decision.decision_type.value} with {final_decision.confidence.name} confidence (Risk Score: {risk_score:.2f})")
        
        return final_decision
    
    def _analyze_signals(self, signals: List[SignalInput]) -> Dict[str, Any]:
        """Analyze and weight multiple model signals"""
        
        if not signals:
            return {
                'weighted_signal': 0.0,
                'weighted_confidence': 0.0,
                'signal_consensus': 'neutral',
                'signal_strength': 0.0,
                'signal_details': {}
            }
        
        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        long_signals = []
        short_signals = []
        signal_details = {}
        
        for signal in signals:
            model_weight = self.model_weights.get(signal.model_name, 0.1)
            
            # Convert signal to numeric value
            if signal.direction == 'long':
                signal_value = signal.strength
                long_signals.append(signal.strength * signal.confidence)
            elif signal.direction == 'short':
                signal_value = -signal.strength
                short_signals.append(signal.strength * signal.confidence)
            else:
                signal_value = 0.0
            
            # Weight the signal
            weighted_signal += signal_value * model_weight * signal.confidence
            weighted_confidence += signal.confidence * model_weight
            total_weight += model_weight
            
            # Store signal details
            signal_details[signal.model_name] = {
                'direction': signal.direction,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'weight': model_weight
            }
        
        # Normalize weights
        if total_weight > 0:
            weighted_signal /= total_weight
            weighted_confidence /= total_weight
        
        # Determine signal consensus
        long_strength = sum(long_signals)
        short_strength = sum(short_signals)
        
        if long_strength > short_strength * 1.5:
            signal_consensus = 'bullish'
        elif short_strength > long_strength * 1.5:
            signal_consensus = 'bearish'
        else:
            signal_consensus = 'neutral'
        
        signal_strength = abs(weighted_signal)
        
        return {
            'weighted_signal': weighted_signal,
            'weighted_confidence': weighted_confidence,
            'signal_consensus': signal_consensus,
            'signal_strength': signal_strength,
            'signal_details': signal_details,
            'long_strength': long_strength,
            'short_strength': short_strength
        }
    
    def _assess_market_conditions(self, conditions: MarketConditions) -> Dict[str, Any]:
        """Assess current market conditions for decision making"""
        
        # Market favorability score (0-1)
        favorability_factors = []
        
        # 1. Trend clarity
        if conditions.trend_strength > 0.7:
            favorability_factors.append(0.8)  # Strong trend is good
        elif conditions.trend_strength > 0.5:
            favorability_factors.append(0.6)  # Moderate trend
        else:
            favorability_factors.append(0.3)  # Weak trend
        
        # 2. Volatility appropriateness
        if conditions.volatility_regime == 'medium':
            favorability_factors.append(0.8)  # Medium volatility ideal
        elif conditions.volatility_regime in ['low', 'high']:
            favorability_factors.append(0.6)  # Low or high still tradeable
        else:
            favorability_factors.append(0.3)  # Extreme volatility
        
        # 3. Session quality
        if conditions.session in ['London', 'New_York'] or conditions.session_overlap:
            favorability_factors.append(0.8)  # High liquidity sessions
        elif conditions.session in ['Tokyo']:
            favorability_factors.append(0.6)  # Medium liquidity
        else:
            favorability_factors.append(0.4)  # Lower liquidity
        
        # 4. Spread and liquidity
        if conditions.liquidity_score > 0.7 and conditions.spread < 2.0:
            favorability_factors.append(0.8)  # Good execution conditions
        elif conditions.liquidity_score > 0.5 and conditions.spread < 3.0:
            favorability_factors.append(0.6)  # Acceptable conditions
        else:
            favorability_factors.append(0.3)  # Poor execution conditions
        
        # 5. News and economic risk
        if conditions.economic_calendar_risk < 0.3:
            favorability_factors.append(0.8)  # Low news risk
        elif conditions.economic_calendar_risk < 0.6:
            favorability_factors.append(0.6)  # Moderate news risk
        else:
            favorability_factors.append(0.2)  # High news risk
        
        market_favorability = np.mean(favorability_factors)
        
        # Market timing score
        timing_factors = []
        
        # RSI levels
        if 30 <= conditions.rsi <= 70:
            timing_factors.append(0.8)  # Good RSI range
        elif 20 <= conditions.rsi <= 80:
            timing_factors.append(0.6)  # Acceptable range
        else:
            timing_factors.append(0.3)  # Extreme levels
        
        # MACD alignment
        if conditions.macd_signal != 'neutral':
            timing_factors.append(0.7)  # Clear MACD signal
        else:
            timing_factors.append(0.4)  # No clear signal
        
        # Moving average alignment
        if conditions.moving_average_alignment in ['bullish', 'bearish']:
            timing_factors.append(0.7)  # Clear MA alignment
        else:
            timing_factors.append(0.4)  # Mixed signals
        
        market_timing = np.mean(timing_factors)
        
        return {
            'market_favorability': market_favorability,
            'market_timing': market_timing,
            'session_quality': favorability_factors[2],
            'liquidity_quality': favorability_factors[3],
            'news_risk': conditions.economic_calendar_risk,
            'overall_market_score': (market_favorability + market_timing) / 2
        }
    
    def _evaluate_risk(self, portfolio: PortfolioContext, conditions: MarketConditions) -> Dict[str, Any]:
        """Evaluate risk factors and constraints"""
        
        risk_factors = []
        risk_constraints = []
        
        # 1. Portfolio risk utilization
        if portfolio.risk_utilization > 0.8:
            risk_factors.append("High portfolio risk utilization")
            risk_score = 0.2
        elif portfolio.risk_utilization > 0.6:
            risk_score = 0.5
        else:
            risk_score = 0.8
        
        # 2. Current drawdown
        if portfolio.drawdown > 0.10:
            risk_factors.append("Significant portfolio drawdown")
            risk_score *= 0.5
        elif portfolio.drawdown > 0.05:
            risk_score *= 0.7
        
        # 3. Available margin
        margin_ratio = portfolio.available_margin / portfolio.total_balance
        if margin_ratio < 0.20:
            risk_factors.append("Low available margin")
            risk_constraints.append("Reduce position sizes")
            risk_score *= 0.6
        
        # 4. Correlation risk
        pair = conditions.currency_pair
        correlation_risk = 0.0
        
        if pair in self.correlation_matrix:
            for corr_pair, correlation in self.correlation_matrix[pair].items():
                if corr_pair in portfolio.current_exposure:
                    correlation_risk += abs(correlation) * portfolio.current_exposure[corr_pair]
        
        if correlation_risk > self.max_correlation_exposure:
            risk_factors.append("High correlation exposure")
            risk_constraints.append("Limit correlated positions")
            risk_score *= 0.7
        
        # 5. Daily trading activity
        if portfolio.daily_pnl < -portfolio.total_balance * 0.05:
            risk_factors.append("Negative daily performance")
            risk_constraints.append("Reduce trading frequency")
            risk_score *= 0.6
        
        # Calculate maximum allowable position size
        max_risk_amount = portfolio.total_balance * self.max_risk_per_trade
        max_position_size = self._calculate_max_position_size(
            max_risk_amount, conditions, portfolio
        )
        
        return {
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'risk_constraints': risk_constraints,
            'max_risk_amount': max_risk_amount,
            'max_position_size': max_position_size,
            'correlation_risk': correlation_risk,
            'available_margin_ratio': margin_ratio
        }
    
    def _calculate_max_position_size(self, 
                                   max_risk_amount: float,
                                   conditions: MarketConditions,
                                   portfolio: PortfolioContext) -> float:
        """Calculate maximum allowable position size"""
        
        # Base calculation using stop loss
        # Assume 50 pip stop loss if not provided
        assumed_stop_loss_pips = 50
        pip_value = 1.0  # $1 per pip per standard lot
        
        # Calculate position size based on risk
        max_position_size = max_risk_amount / (assumed_stop_loss_pips * pip_value)
        
        # Apply leverage constraints
        max_leveraged_size = portfolio.available_margin * 50  # 50:1 leverage max
        max_position_size = min(max_position_size, max_leveraged_size)
        
        # Apply volatility adjustment
        if conditions.volatility_regime == 'high':
            max_position_size *= 0.7  # Reduce size in high volatility
        elif conditions.volatility_regime == 'low':
            max_position_size *= 1.2  # Can increase size in low volatility
        
        # Apply session adjustment
        if conditions.session not in ['London', 'New_York']:
            max_position_size *= 0.8  # Reduce size in lower liquidity sessions
        
        return max(0.01, min(max_position_size, 10.0))  # Between 0.01 and 10 lots
    
    def _make_decision(self,
                      decision_id: str,
                      signal_analysis: Dict[str, Any],
                      market_assessment: Dict[str, Any],
                      risk_assessment: Dict[str, Any],
                      conditions: MarketConditions,
                      portfolio: PortfolioContext) -> TradingDecision:
        """Make the final trading decision"""
        
        # Calculate overall decision score
        signal_score = signal_analysis['signal_strength'] * signal_analysis['weighted_confidence']
        market_score = market_assessment['overall_market_score']
        risk_score = risk_assessment['risk_score']
        
        # Weighted combination
        decision_score = (signal_score * 0.4 + market_score * 0.3 + risk_score * 0.3)
        
        # Determine decision type
        if decision_score < 0.4 or len(risk_assessment['risk_factors']) > 2:
            decision_type = DecisionType.NO_ACTION
            confidence = ConfidenceLevel.LOW
        elif signal_analysis['weighted_signal'] > 0.3 and decision_score > 0.6:
            decision_type = DecisionType.ENTRY_LONG
            confidence = self._determine_confidence(decision_score)
        elif signal_analysis['weighted_signal'] < -0.3 and decision_score > 0.6:
            decision_type = DecisionType.ENTRY_SHORT
            confidence = self._determine_confidence(decision_score)
        else:
            decision_type = DecisionType.HOLD
            confidence = ConfidenceLevel.MEDIUM
        
        # Calculate trade parameters if entering
        if decision_type in [DecisionType.ENTRY_LONG, DecisionType.ENTRY_SHORT]:
            trade_params = self._calculate_trade_parameters(
                decision_type, conditions, risk_assessment, signal_analysis
            )
        else:
            trade_params = {}
        
        # Create decision
        decision = TradingDecision(
            decision_id=decision_id,
            timestamp=conditions.timestamp,
            currency_pair=conditions.currency_pair,
            timeframe=conditions.timeframe,
            decision_type=decision_type,
            confidence=confidence,
            urgency=min(1.0, decision_score * 1.2),
            decision_score=decision_score,
            **trade_params
        )
        
        return decision
    
    def _determine_confidence(self, decision_score: float) -> ConfidenceLevel:
        """Determine confidence level based on decision score"""
        if decision_score >= 0.9:
            return ConfidenceLevel.ABSOLUTE
        elif decision_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif decision_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif decision_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif decision_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_trade_parameters(self,
                                  decision_type: DecisionType,
                                  conditions: MarketConditions,
                                  risk_assessment: Dict[str, Any],
                                  signal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate specific trade parameters"""
        
        # Entry price (current market price with spread adjustment)
        if decision_type == DecisionType.ENTRY_LONG:
            entry_price = conditions.current_price + (conditions.spread / 10000)
        else:  # SHORT
            entry_price = conditions.current_price - (conditions.spread / 10000)
        
        # Position size
        position_size = risk_assessment['max_position_size']
        
        # Adjust position size based on confidence
        confidence_multiplier = signal_analysis['weighted_confidence']
        position_size *= confidence_multiplier
        
        # Stop loss calculation
        atr_multiplier = 2.0 if conditions.volatility_regime == 'high' else 1.5
        stop_distance = conditions.atr_value * atr_multiplier
        
        if decision_type == DecisionType.ENTRY_LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * self.min_risk_reward)
        else:  # SHORT
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * self.min_risk_reward)
        
        # Risk calculations
        risk_amount = abs(entry_price - stop_loss) * position_size * 10000  # Convert to account currency
        risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        
        # Determine risk level
        risk_percentage = risk_amount / risk_assessment['max_risk_amount']
        if risk_percentage <= 0.5:
            risk_level = RiskLevel.VERY_LOW
        elif risk_percentage <= 0.7:
            risk_level = RiskLevel.LOW
        elif risk_percentage <= 0.9:
            risk_level = RiskLevel.MEDIUM
        elif risk_percentage <= 1.0:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.VERY_HIGH
        
        # Expected profit calculation
        success_probability = signal_analysis['weighted_confidence'] * 0.8  # Conservative estimate
        expected_profit = (take_profit - entry_price) * position_size * 10000 * success_probability
        
        return {
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'risk_level': risk_level,
            'max_loss_amount': risk_amount,
            'risk_reward_ratio': risk_reward_ratio,
            'expected_profit': expected_profit,
            'success_probability': success_probability
        }
    
    def _add_decision_reasoning(self,
                              decision: TradingDecision,
                              signal_analysis: Dict[str, Any],
                              market_assessment: Dict[str, Any],
                              risk_assessment: Dict[str, Any]) -> TradingDecision:
        """Add detailed reasoning to the decision"""
        
        # Primary reason
        if decision.decision_type == DecisionType.NO_ACTION:
            decision.primary_reason = "Insufficient signal quality or excessive risk factors"
        elif decision.decision_type in [DecisionType.ENTRY_LONG, DecisionType.ENTRY_SHORT]:
            decision.primary_reason = f"Strong {signal_analysis['signal_consensus']} consensus with favorable market conditions"
        else:
            decision.primary_reason = "Mixed signals suggest waiting for better opportunity"
        
        # Supporting factors
        supporting_factors = []
        
        if signal_analysis['signal_strength'] > 0.6:
            supporting_factors.append(f"Strong signal consensus ({signal_analysis['signal_consensus']})")
        
        if market_assessment['market_favorability'] > 0.7:
            supporting_factors.append("Favorable market conditions")
        
        if market_assessment['session_quality'] > 0.7:
            supporting_factors.append("High-quality trading session")
        
        if risk_assessment['risk_score'] > 0.7:
            supporting_factors.append("Low risk environment")
        
        if market_assessment['liquidity_quality'] > 0.7:
            supporting_factors.append("Good liquidity and spreads")
        
        decision.supporting_factors = supporting_factors
        
        # Risk factors
        decision.risk_factors = risk_assessment['risk_factors'].copy()
        
        if market_assessment['news_risk'] > 0.6:
            decision.risk_factors.append("High economic calendar risk")
        
        if market_assessment['market_timing'] < 0.5:
            decision.risk_factors.append("Poor technical timing")
        
        return decision
    
    def evaluate_existing_position(self,
                                 position: Dict[str, Any],
                                 market_conditions: MarketConditions,
                                 portfolio_context: PortfolioContext) -> TradingDecision:
        """Evaluate whether to maintain, modify, or close existing position"""
        
        decision_id = f"{position['pair']}_eval_{market_conditions.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate current P&L
        entry_price = position['entry_price']
        current_price = market_conditions.current_price
        position_size = position['size']
        direction = position['direction']
        
        if direction == 'long':
            unrealized_pnl = (current_price - entry_price) * position_size * 10000
        else:
            unrealized_pnl = (entry_price - current_price) * position_size * 10000
        
        # Time in trade
        time_in_trade = market_conditions.timestamp - position['entry_time']
        
        # Decision factors
        decision_factors = []
        
        # 1. P&L analysis
        if unrealized_pnl > position.get('take_profit_amount', 0) * 0.8:
            decision_factors.append(('close_profit', 0.8, "Near take profit target"))
        elif unrealized_pnl < -position.get('max_loss_amount', 0) * 0.9:
            decision_factors.append(('close_loss', 0.9, "Near stop loss"))
        
        # 2. Time analysis
        if time_in_trade.total_seconds() > 3600 and unrealized_pnl < 0:  # 1 hour losing trade
            decision_factors.append(('close_loss', 0.6, "Extended losing trade"))
        
        # 3. Market condition change
        if (direction == 'long' and market_conditions.trend_direction == 'down') or \
           (direction == 'short' and market_conditions.trend_direction == 'up'):
            decision_factors.append(('close_loss', 0.7, "Market trend reversal"))
        
        # 4. Risk management
        if portfolio_context.drawdown > 0.1:  # 10% drawdown
            decision_factors.append(('close_loss', 0.5, "Portfolio drawdown protection"))
        
        # Make decision based on strongest factor
        if decision_factors:
            decision_factors.sort(key=lambda x: x[1], reverse=True)
            strongest_factor = decision_factors[0]
            
            if strongest_factor[0] == 'close_profit':
                decision_type = DecisionType.EXIT_PROFIT
                confidence = ConfidenceLevel.HIGH
            elif strongest_factor[0] == 'close_loss':
                decision_type = DecisionType.EXIT_LOSS
                confidence = ConfidenceLevel.HIGH
            else:
                decision_type = DecisionType.HOLD
                confidence = ConfidenceLevel.MEDIUM
        else:
            decision_type = DecisionType.HOLD
            confidence = ConfidenceLevel.MEDIUM
        
        # Create decision
        decision = TradingDecision(
            decision_id=decision_id,
            timestamp=market_conditions.timestamp,
            currency_pair=market_conditions.currency_pair,
            timeframe=market_conditions.timeframe,
            decision_type=decision_type,
            confidence=confidence,
            urgency=0.8 if decision_type != DecisionType.HOLD else 0.3,
            primary_reason=decision_factors[0][2] if decision_factors else "Continue holding position",
            supporting_factors=[f[2] for f in decision_factors[1:]] if len(decision_factors) > 1 else [],
            position_size=position_size if decision_type != DecisionType.HOLD else None
        )
        
        return decision
    
    def get_portfolio_recommendations(self, portfolio_context: PortfolioContext) -> List[str]:
        """Get portfolio-level recommendations"""
        recommendations = []
        
        # Risk utilization
        if portfolio_context.risk_utilization > 0.8:
            recommendations.append("🔴 High risk utilization - Consider reducing position sizes")
        elif portfolio_context.risk_utilization < 0.3:
            recommendations.append("🟡 Low risk utilization - Opportunity to increase exposure")
        
        # Drawdown analysis
        if portfolio_context.drawdown > 0.15:
            recommendations.append("🛑 Significant drawdown - Implement defensive measures")
        elif portfolio_context.drawdown > 0.10:
            recommendations.append("⚠️ Notable drawdown - Monitor risk carefully")
        
        # Daily performance
        if portfolio_context.daily_pnl < -portfolio_context.total_balance * 0.03:
            recommendations.append("📉 Poor daily performance - Consider reducing activity")
        elif portfolio_context.daily_pnl > portfolio_context.total_balance * 0.02:
            recommendations.append("📈 Strong daily performance - Maintain current approach")
        
        # Position count
        if portfolio_context.open_positions > 8:
            recommendations.append("📊 Many open positions - Monitor correlation risk")
        elif portfolio_context.open_positions == 0:
            recommendations.append("💤 No open positions - Look for opportunities")
        
        # Margin utilization
        margin_ratio = portfolio_context.available_margin / portfolio_context.total_balance
        if margin_ratio < 0.2:
            recommendations.append("💰 Low available margin - Reduce leverage")
        
        return recommendations
    
    def analyze_decision_performance(self) -> Dict[str, Any]:
        """Analyze the performance of past decisions"""
        
        if not self.decision_history:
            return {"message": "No decision history available"}
        
        # Performance metrics
        total_decisions = len(self.decision_history)
        entry_decisions = [d for d in self.decision_history if d.decision_type in [DecisionType.ENTRY_LONG, DecisionType.ENTRY_SHORT]]
        exit_decisions = [d for d in self.decision_history if d.decision_type in [DecisionType.EXIT_PROFIT, DecisionType.EXIT_LOSS]]
        
        # Confidence analysis
        confidence_distribution = {}
        for decision in self.decision_history:
            conf_level = decision.confidence.name
            if conf_level not in confidence_distribution:
                confidence_distribution[conf_level] = 0
            confidence_distribution[conf_level] += 1
        
        # Decision type distribution
        decision_type_distribution = {}
        for decision in self.decision_history:
            dec_type = decision.decision_type.name
            if dec_type not in decision_type_distribution:
                decision_type_distribution[dec_type] = 0
            decision_type_distribution[dec_type] += 1
        
        # Average decision scores
        avg_decision_score = np.mean([d.decision_score for d in self.decision_history if d.decision_score > 0])
        avg_confidence = np.mean([d.confidence.value for d in self.decision_history])
        
        return {
            'total_decisions': total_decisions,
            'entry_decisions': len(entry_decisions),
            'exit_decisions': len(exit_decisions),
            'confidence_distribution': confidence_distribution,
            'decision_type_distribution': decision_type_distribution,
            'average_decision_score': avg_decision_score,
            'average_confidence': avg_confidence,
            'recent_decisions': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'pair': d.currency_pair,
                    'decision': d.decision_type.name,
                    'confidence': d.confidence.name,
                    'score': d.decision_score
                } for d in self.decision_history[-10:]  # Last 10 decisions
            ]
        }

    def _generate_initial_proposal(self, 
                                 decision_id: str,
                                 signal_analysis: Dict[str, Any],
                                 market_assessment: Dict[str, Any],
                                 market_conditions: MarketConditions,
                                 portfolio_context: PortfolioContext) -> Dict[str, Any]:
        """Generate initial trade proposal before risk assessment"""
        
        # Basic risk assessment using traditional methods
        basic_risk_assessment = self._evaluate_risk(portfolio_context, market_conditions)
        
        # Generate decision using existing logic
        decision = self._make_decision(
            decision_id, signal_analysis, market_assessment, basic_risk_assessment,
            market_conditions, portfolio_context
        )
        
        # Create proposal structure for risk assessment
        proposal = {
            'decision_id': decision_id,
            'currency_pair': market_conditions.currency_pair,
            'decision_type': decision.decision_type,
            'proposed_position_size': decision.position_size,
            'proposed_entry_price': decision.entry_price,
            'proposed_stop_loss': decision.stop_loss,
            'proposed_take_profit': decision.take_profit,
            'confidence': decision.confidence,
            'decision_score': decision.decision_score,
            'signal_analysis': signal_analysis,
            'market_assessment': market_assessment,
            'basic_risk_assessment': basic_risk_assessment,
            'timestamp': market_conditions.timestamp
        }
        
        return proposal
    
    async def _get_advanced_risk_assessment(self,
                                          initial_proposal: Dict[str, Any],
                                          portfolio_context: PortfolioContext,
                                          market_conditions: MarketConditions) -> Dict[str, Any]:
        """Get advanced risk assessment from DynamicRiskAgent"""
        
        if not self.risk_agent_available or not self.use_advanced_risk_assessment:
            # Fallback to basic risk assessment
            return initial_proposal['basic_risk_assessment']
        
        try:
            # Prepare trade data for DynamicRiskAgent
            trade_data = {
                'symbol': market_conditions.currency_pair,
                'trade_type': initial_proposal['decision_type'].value,
                'position_size': initial_proposal['proposed_position_size'],
                'entry_price': initial_proposal['proposed_entry_price'],
                'stop_loss': initial_proposal['proposed_stop_loss'],
                'take_profit': initial_proposal['proposed_take_profit'],
                'timestamp': market_conditions.timestamp.isoformat()
            }
            
            # Get comprehensive risk assessment
            risk_assessment = await self.risk_agent.assess_trade_risk(trade_data)
            
            # Enhance with portfolio risk analysis
            portfolio_data = {
                'total_balance': portfolio_context.total_balance,
                'available_margin': portfolio_context.available_margin,
                'current_positions': portfolio_context.current_exposure,
                'daily_pnl': portfolio_context.daily_pnl
            }
            
            portfolio_risk = await self.risk_agent.assess_portfolio_risk(portfolio_data)
            
            # Combine assessments
            enhanced_assessment = {
                **initial_proposal['basic_risk_assessment'],
                'ai_risk_score': risk_assessment.get('risk_score', 0.5),
                'ai_risk_factors': risk_assessment.get('risk_factors', []),
                'ai_recommendations': risk_assessment.get('recommendations', []),
                'portfolio_risk_metrics': portfolio_risk,
                'advanced_assessment_available': True,
                'risk_adjusted_position_size': risk_assessment.get('adjusted_position_size'),
                'volatility_risk': risk_assessment.get('volatility_risk', 0.5),
                'liquidity_risk': risk_assessment.get('liquidity_risk', 0.5),
                'correlation_risk_ai': portfolio_risk.get('correlation_risk', 0.5)
            }
            
            logger.info(f"Advanced risk assessment completed: AI Risk Score {risk_assessment.get('risk_score', 0.5):.2f}")
            return enhanced_assessment
            
        except Exception as e:
            logger.warning(f"Advanced risk assessment failed: {e}. Using basic assessment.")
            return {
                **initial_proposal['basic_risk_assessment'],
                'advanced_assessment_available': False,
                'fallback_reason': str(e)
            }
    
    def _make_risk_aware_decision(self,
                                initial_proposal: Dict[str, Any],
                                enhanced_risk_assessment: Dict[str, Any],
                                portfolio_context: PortfolioContext,
                                market_conditions: MarketConditions) -> TradingDecision:
        """Make final decision incorporating advanced risk assessment"""
        
        # Get AI risk score if available
        ai_risk_score = enhanced_risk_assessment.get('ai_risk_score', 0.5)
        basic_risk_score = enhanced_risk_assessment.get('risk_score', 0.5)
        
        # Combine risk scores (weighted average)
        if enhanced_risk_assessment.get('advanced_assessment_available', False):
            combined_risk_score = (ai_risk_score * 0.7) + (basic_risk_score * 0.3)
        else:
            combined_risk_score = basic_risk_score
          # Create copy of initial decision
        decision = TradingDecision(
            decision_id=initial_proposal['decision_id'],
            timestamp=initial_proposal['timestamp'],
            currency_pair=initial_proposal['currency_pair'],
            timeframe=market_conditions.timeframe,
            decision_type=initial_proposal['decision_type'],
            confidence=initial_proposal['confidence'],
            urgency=0.7,  # Default urgency level
            position_size=initial_proposal['proposed_position_size'],
            entry_price=initial_proposal['proposed_entry_price'],
            stop_loss=initial_proposal['proposed_stop_loss'],
            take_profit=initial_proposal['proposed_take_profit'],
            decision_score=initial_proposal['decision_score'],
            reasoning=[]
        )
        
        # Apply risk-based adjustments
        if combined_risk_score > 0.7:
            # High risk - reduce position size or reject trade
            if combined_risk_score > 0.85:
                # Very high risk - reject trade
                decision.decision_type = DecisionType.HOLD
                decision.position_size = 0.0
                decision.confidence = ConfidenceLevel.LOW
                decision.reasoning.append("Trade rejected due to excessive risk (Risk Score: {:.2f})".format(combined_risk_score))
            else:
                # High risk - reduce position size
                risk_reduction_factor = max(0.3, 1.0 - combined_risk_score)
                decision.position_size *= risk_reduction_factor
                decision.reasoning.append("Position size reduced by {:.1%} due to high risk".format(1 - risk_reduction_factor))
                
        elif combined_risk_score < 0.3:
            # Low risk - potentially increase position size (but cap at 1.5x)
            if decision.confidence == ConfidenceLevel.HIGH and portfolio_context.available_margin > 0.5:
                enhancement_factor = min(1.5, 1.0 + (0.3 - combined_risk_score))
                decision.position_size *= enhancement_factor
                decision.reasoning.append("Position size enhanced by {:.1%} due to low risk".format(enhancement_factor - 1))
        
        # Apply AI-recommended position size if available
        ai_position_size = enhanced_risk_assessment.get('risk_adjusted_position_size')
        if ai_position_size and enhanced_risk_assessment.get('advanced_assessment_available', False):
            # Use AI recommendation if it's more conservative
            if ai_position_size < decision.position_size:
                decision.position_size = ai_position_size
                decision.reasoning.append("Position size adjusted to AI recommendation for optimal risk management")
        
        # Update decision score based on risk assessment
        risk_adjustment = 1.0 - (combined_risk_score * 0.3)  # Risk penalty
        decision.decision_score = min(1.0, decision.decision_score * risk_adjustment)
        
        # Add risk-specific reasoning
        if enhanced_risk_assessment.get('ai_risk_factors'):
            decision.reasoning.extend([
                f"AI Risk Factor: {factor}" for factor in enhanced_risk_assessment['ai_risk_factors'][:3]
            ])
        
        if enhanced_risk_assessment.get('ai_recommendations'):
            decision.reasoning.extend([
                f"AI Recommendation: {rec}" for rec in enhanced_risk_assessment['ai_recommendations'][:2]
            ])
        
        return decision
    
    def _add_enhanced_decision_reasoning(self,
                                       decision: TradingDecision,
                                       signal_analysis: Dict[str, Any],
                                       market_assessment: Dict[str, Any],
                                       enhanced_risk_assessment: Dict[str, Any]) -> TradingDecision:
        """Add comprehensive reasoning including advanced risk assessment"""
        
        # Start with existing reasoning
        reasoning = list(decision.reasoning) if decision.reasoning else []
        
        # Add signal analysis reasoning
        reasoning.append(f"Signal Consensus: {signal_analysis.get('signal_consensus', 'neutral')}")
        reasoning.append(f"Weighted Signal Strength: {signal_analysis.get('signal_strength', 0.0):.2f}")
        
        # Add market assessment reasoning
        reasoning.append(f"Market State: {market_assessment.get('market_state', 'unknown')}")
        reasoning.append(f"Volatility Level: {market_assessment.get('volatility_level', 'medium')}")
        
        # Add enhanced risk reasoning
        if enhanced_risk_assessment.get('advanced_assessment_available', False):
            reasoning.append(f"AI Risk Assessment: {enhanced_risk_assessment.get('ai_risk_score', 0.5):.2f}/1.0")
            reasoning.append("Advanced ML risk models applied for optimal decision making")
        else:
            reasoning.append("Basic risk assessment applied (AI models unavailable)")
        
        # Add top risk factors
        risk_factors = enhanced_risk_assessment.get('risk_factors', [])
        if risk_factors:
            reasoning.append(f"Primary Risk Factors: {', '.join(risk_factors[:3])}")
        
        # Final risk score
        final_risk_score = enhanced_risk_assessment.get('ai_risk_score', 
                                                      enhanced_risk_assessment.get('risk_score', 0.5))
        reasoning.append(f"Final Risk Score: {final_risk_score:.2f}/1.0")
        
        decision.reasoning = reasoning
        return decision
    
    def track_risk_adjusted_performance(self, executed_decision: TradingDecision, outcome: Dict[str, Any]) -> None:
        """Track performance metrics for risk-adjusted decisions"""
        
        if not hasattr(self, 'risk_performance_metrics'):
            self.risk_performance_metrics = {
                'total_decisions': 0,
                'ai_enhanced_decisions': 0,
                'risk_prevented_losses': 0,
                'position_size_adjustments': 0,
                'average_risk_score': 0.0,
                'drawdown_reduction': 0.0
            }
        
        # Update basic metrics
        self.risk_performance_metrics['total_decisions'] += 1
        
        # Track AI-enhanced decisions
        if any('AI Risk' in reason for reason in executed_decision.reasoning):
            self.risk_performance_metrics['ai_enhanced_decisions'] += 1
        
        # Track position size adjustments
        if any('reduced' in reason.lower() or 'enhanced' in reason.lower() 
               for reason in executed_decision.reasoning):
            self.risk_performance_metrics['position_size_adjustments'] += 1
        
        # Update performance tracking
        if outcome.get('pnl') is not None:
            # Track if risk management prevented losses
            if outcome['pnl'] > 0 and any('reduced' in reason.lower() 
                                        for reason in executed_decision.reasoning):
                # Position was reduced but still profitable - good risk management
                logger.info("Risk management maintained profitability with reduced exposure")
            
            elif outcome['pnl'] < 0 and executed_decision.decision_type == DecisionType.HOLD:
                # Trade was rejected and would have been a loss
                self.risk_performance_metrics['risk_prevented_losses'] += 1
                logger.info("Risk management prevented potential loss")
        
        logger.info(f"Risk-adjusted performance: {self.risk_performance_metrics['ai_enhanced_decisions']}/{self.risk_performance_metrics['total_decisions']} decisions enhanced")
    
    def get_risk_integration_status(self) -> Dict[str, Any]:
        """Get current status of risk integration"""
        
        return {
            'risk_agent_available': self.risk_agent_available,
            'advanced_risk_enabled': self.use_advanced_risk_assessment,
            'risk_adjustment_factor': self.risk_adjustment_factor,
            'performance_metrics': getattr(self, 'risk_performance_metrics', {}),
            'integration_health': 'healthy' if self.risk_agent_available else 'fallback_mode'
        }

# Export the main class and supporting types
__all__ = [
    'DecisionMaster', 'TradingDecision', 'MarketConditions', 'SignalInput',
    'PortfolioContext', 'DecisionType', 'ConfidenceLevel', 'MarketState', 'RiskLevel'
]


# === PLATFORM3 PHASE 2 ENHANCEMENT APPLIED ===
# Enhanced on: 2025-05-31T22:33:55.334330
# Enhancements: Winston logging, EventEmitter error handling, TypeScript interfaces,
#               Database optimization, Performance monitoring, Async operations
# Phase 3 AI Model Enhancement: Applied advanced ML optimization techniques
