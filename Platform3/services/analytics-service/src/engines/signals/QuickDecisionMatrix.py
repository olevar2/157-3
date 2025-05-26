"""
Platform3 Forex Trading Platform
Quick Decision Matrix - Ultra-Fast Trading Decision Engine

This module provides rapid decision making for high-frequency trading
with optimized algorithms for sub-second execution decisions.

Author: Platform3 Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import json
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Trading decision types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_ACTION = "NO_ACTION"

class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

class UrgencyLevel(Enum):
    """Decision urgency levels"""
    IMMEDIATE = 1    # Execute within seconds
    URGENT = 2       # Execute within 1 minute
    NORMAL = 3       # Execute within 5 minutes
    DELAYED = 4      # Execute within 15 minutes
    OPTIONAL = 5     # Execute when convenient

@dataclass
class MarketCondition:
    """Current market condition snapshot"""
    volatility: float  # 0.0 to 1.0
    volume: float     # Relative volume
    spread: float     # Bid-ask spread
    trend_strength: float  # -1.0 to 1.0
    session: str      # Trading session
    timestamp: datetime

@dataclass
class DecisionInput:
    """Input data for decision matrix"""
    signal_confidence: float
    signal_type: str
    timeframe_alignment: float
    market_condition: MarketCondition
    current_price: float
    position_size: Optional[float] = None
    account_risk: Optional[float] = None

@dataclass
class TradingDecision:
    """Final trading decision output"""
    decision_type: DecisionType
    confidence: float
    urgency: UrgencyLevel
    risk_level: RiskLevel
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    execution_price: float
    reasoning: List[str]
    timestamp: datetime
    validity_duration: timedelta

class QuickDecisionMatrix:
    """
    Ultra-fast decision matrix for high-frequency trading
    
    Features:
    - Sub-second decision making
    - Multi-factor analysis integration
    - Risk-adjusted position sizing
    - Dynamic stop-loss and take-profit calculation
    - Market condition adaptation
    - Real-time decision optimization
    """
    
    def __init__(self):
        """Initialize the quick decision matrix"""
        self.decision_weights = {
            'signal_confidence': 0.35,
            'timeframe_alignment': 0.25,
            'market_volatility': 0.20,
            'volume_confirmation': 0.15,
            'risk_assessment': 0.05
        }
        
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 0.2,
            RiskLevel.LOW: 0.4,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.VERY_HIGH: 1.0
        }
        
        self.session_multipliers = {
            'Asian': 0.8,      # Lower volatility
            'London': 1.2,     # High volatility
            'NY': 1.1,         # High volatility
            'Overlap': 1.3     # Highest volatility
        }
        
        self.decision_history = deque(maxlen=1000)
        self.performance_metrics = {}
        
    async def make_decision(
        self,
        decision_input: DecisionInput
    ) -> TradingDecision:
        """
        Make ultra-fast trading decision
        
        Args:
            decision_input: All input data for decision making
            
        Returns:
            TradingDecision with complete execution plan
        """
        try:
            start_time = datetime.now()
            
            # Quick validation
            if not self._validate_input(decision_input):
                return self._create_no_action_decision("Invalid input data")
            
            # Calculate decision score
            decision_score = await self._calculate_decision_score(decision_input)
            
            # Determine decision type
            decision_type = self._determine_decision_type(
                decision_score, decision_input.signal_type
            )
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(decision_input)
            
            # Determine urgency
            urgency = self._determine_urgency(
                decision_score, decision_input.market_condition
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                decision_input, risk_level, decision_score
            )
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_sl_tp(
                decision_input, decision_type, risk_level
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                decision_input, decision_score, risk_level
            )
            
            # Create decision
            decision = TradingDecision(
                decision_type=decision_type,
                confidence=decision_score,
                urgency=urgency,
                risk_level=risk_level,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                execution_price=decision_input.current_price,
                reasoning=reasoning,
                timestamp=start_time,
                validity_duration=self._calculate_validity_duration(urgency)
            )
            
            # Store decision
            self.decision_history.append(decision)
            
            # Log performance
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Decision made in {execution_time:.2f}ms: {decision_type.value}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return self._create_no_action_decision(f"Error: {str(e)}")
    
    def _validate_input(self, input_data: DecisionInput) -> bool:
        """Validate input data quickly"""
        return (
            0.0 <= input_data.signal_confidence <= 1.0 and
            0.0 <= input_data.timeframe_alignment <= 1.0 and
            input_data.current_price > 0 and
            input_data.market_condition is not None
        )
    
    async def _calculate_decision_score(self, input_data: DecisionInput) -> float:
        """Calculate overall decision score"""
        # Base score from signal confidence
        base_score = input_data.signal_confidence
        
        # Timeframe alignment bonus
        alignment_bonus = input_data.timeframe_alignment * 0.2
        
        # Market condition adjustment
        market_condition = input_data.market_condition
        volatility_factor = self._get_volatility_factor(market_condition.volatility)
        volume_factor = self._get_volume_factor(market_condition.volume)
        session_factor = self.session_multipliers.get(market_condition.session, 1.0)
        
        # Calculate weighted score
        weighted_score = (
            base_score * self.decision_weights['signal_confidence'] +
            alignment_bonus * self.decision_weights['timeframe_alignment'] +
            volatility_factor * self.decision_weights['market_volatility'] +
            volume_factor * self.decision_weights['volume_confirmation']
        )
        
        # Apply session multiplier
        final_score = min(1.0, weighted_score * session_factor)
        
        return final_score
    
    def _get_volatility_factor(self, volatility: float) -> float:
        """Get volatility adjustment factor"""
        if volatility < 0.2:
            return 0.7  # Low volatility penalty
        elif volatility > 0.8:
            return 0.8  # High volatility caution
        else:
            return 1.0  # Normal volatility
    
    def _get_volume_factor(self, volume: float) -> float:
        """Get volume confirmation factor"""
        if volume > 1.5:
            return 1.2  # High volume confirmation
        elif volume < 0.5:
            return 0.8  # Low volume penalty
        else:
            return 1.0  # Normal volume
    
    def _determine_decision_type(self, score: float, signal_type: str) -> DecisionType:
        """Determine decision type based on score and signal"""
        if score < 0.3:
            return DecisionType.NO_ACTION
        
        # Map signal types to decisions based on score
        if signal_type.upper() in ['BUY', 'STRONG_BUY']:
            if score >= 0.8:
                return DecisionType.STRONG_BUY
            elif score >= 0.5:
                return DecisionType.BUY
            else:
                return DecisionType.HOLD
        
        elif signal_type.upper() in ['SELL', 'STRONG_SELL']:
            if score >= 0.8:
                return DecisionType.STRONG_SELL
            elif score >= 0.5:
                return DecisionType.SELL
            else:
                return DecisionType.HOLD
        
        else:
            return DecisionType.HOLD
    
    def _calculate_risk_level(self, input_data: DecisionInput) -> RiskLevel:
        """Calculate risk level for the decision"""
        risk_factors = []
        
        # Market volatility risk
        volatility = input_data.market_condition.volatility
        risk_factors.append(volatility)
        
        # Spread risk
        spread = input_data.market_condition.spread
        risk_factors.append(min(1.0, spread * 10))  # Normalize spread
        
        # Signal confidence risk (inverse)
        risk_factors.append(1.0 - input_data.signal_confidence)
        
        # Timeframe alignment risk (inverse)
        risk_factors.append(1.0 - input_data.timeframe_alignment)
        
        # Calculate average risk
        avg_risk = np.mean(risk_factors)
        
        # Map to risk levels
        for risk_level, threshold in self.risk_thresholds.items():
            if avg_risk <= threshold:
                return risk_level
        
        return RiskLevel.VERY_HIGH
    
    def _determine_urgency(
        self,
        score: float,
        market_condition: MarketCondition
    ) -> UrgencyLevel:
        """Determine execution urgency"""
        if score >= 0.9 and market_condition.volatility > 0.7:
            return UrgencyLevel.IMMEDIATE
        elif score >= 0.8:
            return UrgencyLevel.URGENT
        elif score >= 0.6:
            return UrgencyLevel.NORMAL
        elif score >= 0.4:
            return UrgencyLevel.DELAYED
        else:
            return UrgencyLevel.OPTIONAL
    
    def _calculate_position_size(
        self,
        input_data: DecisionInput,
        risk_level: RiskLevel,
        score: float
    ) -> float:
        """Calculate optimal position size"""
        # Base position size (as percentage of account)
        base_size = 0.02  # 2% base risk
        
        # Risk adjustment
        risk_multiplier = {
            RiskLevel.VERY_LOW: 1.5,
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.7,
            RiskLevel.VERY_HIGH: 0.3
        }[risk_level]
        
        # Confidence adjustment
        confidence_multiplier = min(2.0, score * 2)
        
        # Calculate final size
        position_size = base_size * risk_multiplier * confidence_multiplier
        
        # Apply limits
        return max(0.001, min(0.1, position_size))  # 0.1% to 10% max
    
    def _calculate_sl_tp(
        self,
        input_data: DecisionInput,
        decision_type: DecisionType,
        risk_level: RiskLevel
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        if decision_type in [DecisionType.NO_ACTION, DecisionType.HOLD]:
            return None, None
        
        current_price = input_data.current_price
        volatility = input_data.market_condition.volatility
        
        # Calculate ATR-based levels
        atr_multiplier = max(1.0, volatility * 3)
        
        # Risk-adjusted distances
        risk_multipliers = {
            RiskLevel.VERY_LOW: 0.5,
            RiskLevel.LOW: 0.7,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 1.3,
            RiskLevel.VERY_HIGH: 1.5
        }
        
        distance_multiplier = risk_multipliers[risk_level] * atr_multiplier
        
        if decision_type in [DecisionType.BUY, DecisionType.STRONG_BUY]:
            # Long position
            sl_distance = current_price * 0.001 * distance_multiplier  # 0.1% base
            tp_distance = sl_distance * 2  # 2:1 reward:risk
            
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        
        else:  # SELL or STRONG_SELL
            # Short position
            sl_distance = current_price * 0.001 * distance_multiplier
            tp_distance = sl_distance * 2
            
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        
        return stop_loss, take_profit
    
    def _generate_reasoning(
        self,
        input_data: DecisionInput,
        score: float,
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate human-readable reasoning for the decision"""
        reasoning = []
        
        # Signal confidence
        if input_data.signal_confidence >= 0.8:
            reasoning.append("High signal confidence detected")
        elif input_data.signal_confidence >= 0.6:
            reasoning.append("Moderate signal confidence")
        else:
            reasoning.append("Low signal confidence - caution advised")
        
        # Timeframe alignment
        if input_data.timeframe_alignment >= 0.8:
            reasoning.append("Strong multi-timeframe alignment")
        elif input_data.timeframe_alignment >= 0.5:
            reasoning.append("Partial timeframe alignment")
        else:
            reasoning.append("Weak timeframe alignment")
        
        # Market conditions
        market = input_data.market_condition
        if market.volatility > 0.7:
            reasoning.append("High market volatility detected")
        if market.volume > 1.5:
            reasoning.append("Above-average volume confirmation")
        
        # Risk assessment
        reasoning.append(f"Risk level: {risk_level.name}")
        
        return reasoning
    
    def _calculate_validity_duration(self, urgency: UrgencyLevel) -> timedelta:
        """Calculate how long the decision remains valid"""
        durations = {
            UrgencyLevel.IMMEDIATE: timedelta(seconds=30),
            UrgencyLevel.URGENT: timedelta(minutes=1),
            UrgencyLevel.NORMAL: timedelta(minutes=5),
            UrgencyLevel.DELAYED: timedelta(minutes=15),
            UrgencyLevel.OPTIONAL: timedelta(hours=1)
        }
        return durations[urgency]
    
    def _create_no_action_decision(self, reason: str) -> TradingDecision:
        """Create a no-action decision"""
        return TradingDecision(
            decision_type=DecisionType.NO_ACTION,
            confidence=0.0,
            urgency=UrgencyLevel.OPTIONAL,
            risk_level=RiskLevel.VERY_HIGH,
            position_size=0.0,
            stop_loss=None,
            take_profit=None,
            execution_price=0.0,
            reasoning=[reason],
            timestamp=datetime.now(),
            validity_duration=timedelta(minutes=1)
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get decision matrix performance metrics"""
        if not self.decision_history:
            return {}
        
        recent_decisions = list(self.decision_history)[-100:]
        
        return {
            'total_decisions': len(recent_decisions),
            'decision_distribution': {
                decision_type.value: len([d for d in recent_decisions if d.decision_type == decision_type])
                for decision_type in DecisionType
            },
            'average_confidence': np.mean([d.confidence for d in recent_decisions]),
            'risk_distribution': {
                risk_level.name: len([d for d in recent_decisions if d.risk_level == risk_level])
                for risk_level in RiskLevel
            },
            'urgency_distribution': {
                urgency.name: len([d for d in recent_decisions if d.urgency == urgency])
                for urgency in UrgencyLevel
            }
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_decision_matrix():
        matrix = QuickDecisionMatrix()
        
        # Create test input
        market_condition = MarketCondition(
            volatility=0.6,
            volume=1.2,
            spread=0.0002,
            trend_strength=0.7,
            session="London",
            timestamp=datetime.now()
        )
        
        decision_input = DecisionInput(
            signal_confidence=0.8,
            signal_type="BUY",
            timeframe_alignment=0.7,
            market_condition=market_condition,
            current_price=1.2500
        )
        
        # Make decision
        decision = await matrix.make_decision(decision_input)
        
        print(f"Decision: {decision.decision_type.value}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Urgency: {decision.urgency.name}")
        print(f"Risk Level: {decision.risk_level.name}")
        print(f"Position Size: {decision.position_size:.4f}")
        print(f"Stop Loss: {decision.stop_loss}")
        print(f"Take Profit: {decision.take_profit}")
        print(f"Reasoning: {decision.reasoning}")
    
    # Run test
    asyncio.run(test_decision_matrix())
