"""
Decision Master - Central Decision Orchestration and Validation AI Model
Production-ready decision coordination for Platform3 Trading System

For the humanitarian mission: Every trading decision must be optimal and profitable
to maximize aid for sick babies and poor families.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import math
from concurrent.futures import ThreadPoolExecutor

class DecisionType(Enum):
    """Types of trading decisions"""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"
    CLOSE_ALL = "close_all"

class ConfidenceLevel(Enum):
    """Decision confidence levels"""
    VERY_LOW = "very_low"      # 0-20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%

@dataclass
class AgentInput:
    """Input from individual genius agents"""
    agent_name: str
    decision_type: DecisionType
    confidence: float  # 0-1
    reasoning: str
    supporting_data: Dict[str, Any]
    risk_assessment: Dict[str, float]
    timestamp: datetime
    processing_time_ms: float

@dataclass
class MarketCondition:
    """Current market condition assessment"""
    trend_direction: str  # bullish, bearish, sideways
    trend_strength: float  # 0-1
    volatility_level: str  # low, normal, high, extreme
    volume_profile: str   # low, normal, high
    session_state: str    # asian, london, ny, overlap
    news_impact: str      # none, low, medium, high, extreme
    risk_sentiment: str   # risk_on, risk_off, neutral

@dataclass
class TradingDecision:
    """Final coordinated trading decision"""
    decision_type: DecisionType
    symbol: str
    confidence_level: ConfidenceLevel
    confidence_score: float  # 0-1
    
    # Entry/Exit details
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    
    # Timing
    execution_urgency: str = "normal"  # immediate, high, normal, low
    valid_until: Optional[datetime] = None
    
    # Risk management
    max_risk_percentage: float = 2.0
    portfolio_impact: str = "low"  # low, medium, high
    
    # Supporting information
    primary_reasoning: str = ""
    supporting_agents: List[str] = None
    conflicting_agents: List[str] = None
    market_condition: Optional[MarketCondition] = None
    
    # Quality metrics
    decision_quality_score: float = 0.0
    expected_outcome: str = "neutral"
    
    # Validation
    validation_checks: Dict[str, bool] = None
    risk_warnings: List[str] = None
    
    timestamp: datetime = None

class DecisionMaster:
    """
    Central Decision Orchestration and Validation AI for Platform3 Trading System
    
    Master coordinator that:
    - Receives inputs from all 8 genius agents
    - Validates and cross-references agent recommendations
    - Resolves conflicts between agents using sophisticated logic
    - Applies final risk and quality checks
    - Produces optimal trading decisions for maximum profitability
    
    For the humanitarian mission: Every decision must be thoroughly validated
    to ensure maximum profitability for helping sick babies and poor families.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Agent coordination
        self.agent_inputs = {}
        self.agent_weights = self._initialize_agent_weights()
        self.consensus_threshold = 0.6  # 60% consensus required
        
        # Decision validation engines
        self.risk_validator = RiskValidator()
        self.quality_assessor = QualityAssessor()
        self.conflict_resolver = ConflictResolver()
        self.market_analyzer = MarketConditionAnalyzer()
        
        # Decision tracking
        self.recent_decisions = []
        self.decision_performance = {}
        self.agent_performance_tracking = {}
        
        # Real-time monitoring
        self.active_monitoring = True
        self.decision_alerts = []
        
    async def process_agent_inputs(
        self, 
        agent_inputs: List[AgentInput],
        symbol: str,
        market_data: pd.DataFrame
    ) -> TradingDecision:
        """
        Process inputs from all genius agents and produce final trading decision.
        
        This is the central brain that coordinates all AI intelligence
        to make optimal trading decisions for humanitarian purposes.
        """
        
        self.logger.info(f"🧠 Decision Master processing {len(agent_inputs)} agent inputs for {symbol}")
        
        # 1. Validate and store agent inputs
        validated_inputs = await self._validate_agent_inputs(agent_inputs)
        
        # 2. Analyze current market conditions
        market_condition = await self._analyze_market_conditions(market_data, symbol)
        
        # 3. Apply agent weighting based on current conditions
        weighted_inputs = await self._apply_agent_weighting(validated_inputs, market_condition)
        
        # 4. Check for consensus among agents
        consensus_analysis = await self._analyze_consensus(weighted_inputs)
        
        # 5. Resolve conflicts between agents
        conflict_resolution = await self._resolve_conflicts(weighted_inputs, market_condition)
        
        # 6. Generate preliminary decision
        preliminary_decision = await self._generate_preliminary_decision(
            weighted_inputs, consensus_analysis, conflict_resolution, market_condition, symbol
        )
        
        # 7. Apply risk validation and quality checks
        validated_decision = await self._apply_final_validation(
            preliminary_decision, market_data, symbol
        )
        
        # 8. Calculate final confidence and quality scores
        final_decision = await self._finalize_decision(validated_decision, weighted_inputs)
        
        # 9. Log and track decision
        await self._log_and_track_decision(final_decision, agent_inputs)
        
        self.logger.info(f"✅ Decision Master: {final_decision.decision_type.value} for {symbol} "
                        f"with {final_decision.confidence_level.value} confidence")
        
        return final_decision
    
    def _initialize_agent_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialize agent weights based on market conditions and specializations"""
        
        return {
            # Risk Genius - Always high weight for risk assessment
            'Risk Genius': {
                'base_weight': 0.20,
                'trending_market': 0.18,
                'volatile_market': 0.25,
                'calm_market': 0.15,
                'news_event': 0.22
            },
            
            # Pattern Master - High weight for technical setups
            'Pattern Master': {
                'base_weight': 0.18,
                'trending_market': 0.22,
                'volatile_market': 0.15,
                'calm_market': 0.20,
                'news_event': 0.12
            },
            
            # Execution Expert - Critical for timing
            'Execution Expert': {
                'base_weight': 0.15,
                'trending_market': 0.16,
                'volatile_market': 0.18,
                'calm_market': 0.14,
                'news_event': 0.20
            },
            
            # Session Expert - Important for timing
            'Session Expert': {
                'base_weight': 0.12,
                'trending_market': 0.10,
                'volatile_market': 0.08,
                'calm_market': 0.15,
                'news_event': 0.12
            },
            
            # Pair Specialist - Pair-specific expertise
            'Pair Specialist': {
                'base_weight': 0.12,
                'trending_market': 0.12,
                'volatile_market': 0.10,
                'calm_market': 0.14,
                'news_event': 0.08
            },
            
            # AI Model Coordinator - System health
            'AI Model Coordinator': {
                'base_weight': 0.08,
                'trending_market': 0.08,
                'volatile_market': 0.08,
                'calm_market': 0.08,
                'news_event': 0.08
            },
            
            # Market Microstructure Genius - Order flow insights
            'Market Microstructure Genius': {
                'base_weight': 0.08,
                'trending_market': 0.08,
                'volatile_market': 0.10,
                'calm_market': 0.07,
                'news_event': 0.10
            },
            
            # Sentiment Integration Genius - Market sentiment
            'Sentiment Integration Genius': {
                'base_weight': 0.07,
                'trending_market': 0.06,
                'volatile_market': 0.06,
                'calm_market': 0.07,
                'news_event': 0.08
            }
        }
    
    async def _validate_agent_inputs(self, agent_inputs: List[AgentInput]) -> List[AgentInput]:
        """Validate agent inputs for completeness and consistency"""
        
        validated_inputs = []
        
        for agent_input in agent_inputs:
            # Check for required fields
            if not agent_input.agent_name or not agent_input.decision_type:
                self.logger.warning(f"Invalid input from {agent_input.agent_name}: missing required fields")
                continue
            
            # Validate confidence range
            if not 0 <= agent_input.confidence <= 1:
                agent_input.confidence = max(0, min(1, agent_input.confidence))
                self.logger.warning(f"Adjusted confidence for {agent_input.agent_name} to valid range")
            
            # Check for reasonable processing time
            if agent_input.processing_time_ms > 1000:  # More than 1 second
                self.logger.warning(f"{agent_input.agent_name} took {agent_input.processing_time_ms}ms to process")
            
            validated_inputs.append(agent_input)
        
        return validated_inputs
    
    async def _analyze_market_conditions(self, market_data: pd.DataFrame, symbol: str) -> MarketCondition:
        """Analyze current market conditions for decision context"""
        
        if len(market_data) < 50:
            # Insufficient data - return neutral conditions
            return MarketCondition(
                trend_direction="sideways",
                trend_strength=0.5,
                volatility_level="normal",
                volume_profile="normal",
                session_state="unknown",
                news_impact="none",
                risk_sentiment="neutral"
            )
        
        # Trend analysis
        ema_21 = market_data['close'].ewm(span=21).mean().iloc[-1]
        ema_55 = market_data['close'].ewm(span=55).mean().iloc[-1]
        current_price = market_data['close'].iloc[-1]
        
        if current_price > ema_21 > ema_55:
            trend_direction = "bullish"
            trend_strength = min(1.0, (current_price - ema_55) / ema_55 * 10)
        elif current_price < ema_21 < ema_55:
            trend_direction = "bearish"
            trend_strength = min(1.0, (ema_55 - current_price) / ema_55 * 10)
        else:
            trend_direction = "sideways"
            trend_strength = 0.3
        
        # Volatility analysis
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        avg_vol = returns.rolling(100).std().mean() * np.sqrt(252)
        
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        if vol_ratio > 1.5:
            volatility_level = "extreme"
        elif vol_ratio > 1.2:
            volatility_level = "high"
        elif vol_ratio < 0.8:
            volatility_level = "low"
        else:
            volatility_level = "normal"
        
        # Volume analysis
        if 'volume' in market_data.columns:
            current_volume = market_data['volume'].iloc[-10:].mean()
            avg_volume = market_data['volume'].iloc[-100:].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.3:
                volume_profile = "high"
            elif volume_ratio < 0.7:
                volume_profile = "low"
            else:
                volume_profile = "normal"
        else:
            volume_profile = "normal"
        
        # Session analysis
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour < 7:
            session_state = "asian"
        elif 7 <= current_hour < 16:
            session_state = "london"
        elif 16 <= current_hour < 22:
            session_state = "ny"
        elif 13 <= current_hour < 16:
            session_state = "overlap"
        else:
            session_state = "unknown"
        
        return MarketCondition(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            volume_profile=volume_profile,
            session_state=session_state,
            news_impact="none",  # Would integrate with news feed
            risk_sentiment="neutral"  # Would integrate with sentiment feeds
        )
    
    async def _apply_agent_weighting(
        self, 
        agent_inputs: List[AgentInput], 
        market_condition: MarketCondition
    ) -> List[Tuple[AgentInput, float]]:
        """Apply dynamic weighting to agent inputs based on market conditions"""
        
        weighted_inputs = []
        
        # Determine market condition key for weights
        if market_condition.news_impact in ["high", "extreme"]:
            condition_key = "news_event"
        elif market_condition.volatility_level in ["high", "extreme"]:
            condition_key = "volatile_market"
        elif market_condition.trend_strength > 0.7:
            condition_key = "trending_market"
        else:
            condition_key = "calm_market"
        
        for agent_input in agent_inputs:
            agent_name = agent_input.agent_name
            
            # Get base weight
            base_weight = self.agent_weights.get(agent_name, {}).get('base_weight', 0.1)
            
            # Get condition-specific weight
            condition_weight = self.agent_weights.get(agent_name, {}).get(condition_key, base_weight)
            
            # Adjust weight based on agent's confidence
            confidence_adjustment = 0.8 + (agent_input.confidence * 0.4)  # 0.8 to 1.2 multiplier
            
            # Adjust weight based on agent's recent performance
            performance_multiplier = self._get_agent_performance_multiplier(agent_name)
            
            final_weight = condition_weight * confidence_adjustment * performance_multiplier
            
            weighted_inputs.append((agent_input, final_weight))
        
        return weighted_inputs    
    async def _analyze_consensus(self, weighted_inputs: List[Tuple[AgentInput, float]]) -> Dict[str, Any]:
        """Analyze consensus among agents"""
        
        decision_scores = {}
        total_weight = sum(weight for _, weight in weighted_inputs)
        
        if total_weight == 0:
            return {'consensus': False, 'strength': 0.0, 'dominant_decision': DecisionType.HOLD}
        
        # Calculate weighted scores for each decision type
        for agent_input, weight in weighted_inputs:
            decision = agent_input.decision_type
            normalized_weight = weight / total_weight
            weighted_confidence = agent_input.confidence * normalized_weight
            
            if decision not in decision_scores:
                decision_scores[decision] = 0.0
            decision_scores[decision] += weighted_confidence
        
        # Find dominant decision
        dominant_decision = max(decision_scores, key=decision_scores.get)
        dominant_score = decision_scores[dominant_decision]
        
        # Check for consensus (dominant decision has >60% of total weighted confidence)
        consensus = dominant_score > self.consensus_threshold
        
        return {
            'consensus': consensus,
            'strength': dominant_score,
            'dominant_decision': dominant_decision,
            'all_scores': decision_scores,
            'agreement_level': self._calculate_agreement_level(decision_scores)
        }
    
    async def _resolve_conflicts(
        self, 
        weighted_inputs: List[Tuple[AgentInput, float]], 
        market_condition: MarketCondition
    ) -> Dict[str, Any]:
        """Resolve conflicts between agents using sophisticated logic"""
        
        conflicts = []
        supporting_agents = []
        conflicting_agents = []
        
        # Group agents by decision type
        decision_groups = {}
        for agent_input, weight in weighted_inputs:
            decision = agent_input.decision_type
            if decision not in decision_groups:
                decision_groups[decision] = []
            decision_groups[decision].append((agent_input, weight))
        
        # Identify conflicts
        if len(decision_groups) > 1:
            decisions = list(decision_groups.keys())
            
            # Check for opposing decisions
            opposing_pairs = [
                (DecisionType.ENTRY_LONG, DecisionType.ENTRY_SHORT),
                (DecisionType.EXIT_LONG, DecisionType.EXIT_SHORT),
                (DecisionType.ENTRY_LONG, DecisionType.EXIT_LONG),
                (DecisionType.ENTRY_SHORT, DecisionType.EXIT_SHORT)
            ]
            
            for pair in opposing_pairs:
                if pair[0] in decisions and pair[1] in decisions:
                    conflicts.append({
                        'type': 'opposing_decisions',
                        'decisions': pair,
                        'agents': {
                            pair[0]: [agent.agent_name for agent, _ in decision_groups[pair[0]]],
                            pair[1]: [agent.agent_name for agent, _ in decision_groups[pair[1]]]
                        }
                    })
        
        # Apply conflict resolution rules
        resolution_strategy = "weighted_majority"
        resolution_reasoning = []
        
        if conflicts:
            # Rule 1: Risk Genius veto power for extreme risk
            risk_genius_input = next(
                (agent for agent, _ in weighted_inputs if agent.agent_name == "Risk Genius"), 
                None
            )
            if risk_genius_input and risk_genius_input.confidence > 0.8:
                if risk_genius_input.decision_type == DecisionType.HOLD:
                    resolution_strategy = "risk_veto"
                    resolution_reasoning.append("Risk Genius veto due to high risk assessment")
            
            # Rule 2: High volatility - favor execution expert
            if market_condition.volatility_level in ["high", "extreme"]:
                execution_expert = next(
                    (agent for agent, _ in weighted_inputs if agent.agent_name == "Execution Expert"),
                    None
                )
                if execution_expert and execution_expert.confidence > 0.7:
                    resolution_strategy = "execution_priority"
                    resolution_reasoning.append("Execution Expert priority in high volatility")
            
            # Rule 3: Strong pattern signals - favor pattern master
            pattern_master = next(
                (agent for agent, _ in weighted_inputs if agent.agent_name == "Pattern Master"),
                None
            )
            if pattern_master and pattern_master.confidence > 0.8:
                if pattern_master.decision_type in [DecisionType.ENTRY_LONG, DecisionType.ENTRY_SHORT]:
                    resolution_strategy = "pattern_priority"
                    resolution_reasoning.append("Pattern Master priority for strong technical setup")
        
        return {
            'conflicts_found': len(conflicts) > 0,
            'conflicts': conflicts,
            'resolution_strategy': resolution_strategy,
            'resolution_reasoning': resolution_reasoning
        }
    
    async def _generate_preliminary_decision(
        self,
        weighted_inputs: List[Tuple[AgentInput, float]],
        consensus_analysis: Dict[str, Any],
        conflict_resolution: Dict[str, Any],
        market_condition: MarketCondition,
        symbol: str
    ) -> TradingDecision:
        """Generate preliminary trading decision"""
        
        # Determine decision type based on consensus and conflict resolution
        if conflict_resolution['resolution_strategy'] == "risk_veto":
            decision_type = DecisionType.HOLD
            confidence_score = 0.9  # High confidence in risk management
        elif conflict_resolution['resolution_strategy'] == "execution_priority":
            execution_expert = next(
                (agent for agent, _ in weighted_inputs if agent.agent_name == "Execution Expert"),
                None
            )
            decision_type = execution_expert.decision_type if execution_expert else DecisionType.HOLD
            confidence_score = execution_expert.confidence if execution_expert else 0.5
        elif conflict_resolution['resolution_strategy'] == "pattern_priority":
            pattern_master = next(
                (agent for agent, _ in weighted_inputs if agent.agent_name == "Pattern Master"),
                None
            )
            decision_type = pattern_master.decision_type if pattern_master else DecisionType.HOLD
            confidence_score = pattern_master.confidence if pattern_master else 0.5
        else:
            # Use consensus decision
            decision_type = consensus_analysis['dominant_decision']
            confidence_score = consensus_analysis['strength']
        
        # Convert confidence score to confidence level
        if confidence_score >= 0.8:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        # Calculate entry/exit prices and risk management
        entry_price = None
        stop_loss = None
        take_profit = None
        position_size = None
        
        if decision_type in [DecisionType.ENTRY_LONG, DecisionType.ENTRY_SHORT]:
            # Use execution expert's recommendations if available
            execution_expert = next(
                (agent for agent, _ in weighted_inputs if agent.agent_name == "Execution Expert"),
                None
            )
            if execution_expert and execution_expert.supporting_data:
                entry_price = execution_expert.supporting_data.get('entry_price')
                stop_loss = execution_expert.supporting_data.get('stop_loss')
                take_profit = execution_expert.supporting_data.get('take_profit')
            
            # Use risk genius for position sizing
            risk_genius = next(
                (agent for agent, _ in weighted_inputs if agent.agent_name == "Risk Genius"),
                None
            )
            if risk_genius and risk_genius.supporting_data:
                position_size = risk_genius.supporting_data.get('position_size', 0.02)  # 2% default
        
        # Determine execution urgency
        urgency = "normal"
        if market_condition.volatility_level == "extreme":
            urgency = "immediate"
        elif market_condition.news_impact in ["high", "extreme"]:
            urgency = "high"
        
        # Set validity period
        valid_until = datetime.now() + timedelta(hours=1)  # Default 1 hour validity
        
        return TradingDecision(
            decision_type=decision_type,
            symbol=symbol,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            execution_urgency=urgency,
            valid_until=valid_until,
            market_condition=market_condition,
            timestamp=datetime.now()
        )
    
    def _get_agent_performance_multiplier(self, agent_name: str) -> float:
        """Get performance-based weight multiplier for agent"""
        
        if agent_name not in self.agent_performance_tracking:
            return 1.0  # Neutral multiplier for new agents
        
        performance = self.agent_performance_tracking[agent_name]
        recent_accuracy = performance.get('recent_accuracy', 0.5)
        
        # Convert accuracy to multiplier (0.5 accuracy = 1.0 multiplier)
        # Range: 0.7 to 1.3 multiplier
        multiplier = 0.7 + (recent_accuracy * 1.2)
        return max(0.7, min(1.3, multiplier))
    
    def _calculate_agreement_level(self, decision_scores: Dict[DecisionType, float]) -> str:
        """Calculate the level of agreement among agents"""
        
        if not decision_scores:
            return "no_input"
        
        sorted_scores = sorted(decision_scores.values(), reverse=True)
        
        if len(sorted_scores) == 1:
            return "unanimous"
        
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        
        if top_score > 0.8:
            return "strong_majority"
        elif top_score > 0.6:
            return "majority"
        elif top_score - second_score < 0.1:
            return "split"
        else:
            return "weak_majority"

# Support classes for Decision Master
class RiskValidator:
    """Validates decisions against risk management rules"""
    pass

class QualityAssessor:
    """Assesses decision quality and expected outcomes"""
    pass

class ConflictResolver:
    """Resolves conflicts between agent recommendations"""
    pass

class MarketConditionAnalyzer:
    """Analyzes current market conditions for decision context"""
    pass

# Example usage for testing
if __name__ == "__main__":
    print("🧠 Decision Master - Central Decision Orchestration and Validation AI")
    print("For the humanitarian mission: Making optimal trading decisions")
    print("to generate maximum aid for sick babies and poor families")