"""
Quick Decision Matrix Engine
Fast buy/sell/hold decision making for rapid execution.
Implements decision matrix logic for ultra-fast trading decisions.
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

from .SignalAggregator import SignalInput, Timeframe
from .ConflictResolver import ResolvedSignal
from .ConfidenceCalculator import ConfidenceMetrics


class DecisionType(Enum):
    """Types of trading decisions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class RiskLevel(Enum):
    """Risk levels for position sizing"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TradingDecision:
    """Final trading decision with execution parameters"""
    decision_type: DecisionType
    confidence_level: float  # 0-1
    position_size: float  # Percentage of capital to risk
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    max_risk_per_trade: float
    execution_urgency: str  # 'immediate', 'normal', 'patient'
    decision_factors: Dict[str, float]
    risk_assessment: Dict[str, Any]
    execution_instructions: Dict[str, Any]


@dataclass
class DecisionMatrix:
    """Decision matrix configuration"""
    confidence_thresholds: Dict[str, float]
    strength_thresholds: Dict[str, float]
    risk_thresholds: Dict[str, float]
    position_sizing_rules: Dict[str, float]
    execution_rules: Dict[str, Any]


class QuickDecisionMatrix:
    """
    Quick Decision Matrix Engine
    Makes rapid buy/sell/hold decisions based on aggregated signals
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Decision matrix configuration
        self.decision_matrix = DecisionMatrix(
            confidence_thresholds={
                'strong_buy': 0.85,
                'buy': 0.65,
                'hold': 0.45,
                'sell': 0.65,
                'strong_sell': 0.85
            },
            strength_thresholds={
                'strong': 80.0,
                'medium': 60.0,
                'weak': 40.0
            },
            risk_thresholds={
                'very_low': 0.5,
                'low': 1.0,
                'medium': 2.0,
                'high': 3.0,
                'very_high': 5.0
            },
            position_sizing_rules={
                'very_low_risk': 0.5,   # 0.5% of capital
                'low_risk': 1.0,        # 1% of capital
                'medium_risk': 2.0,     # 2% of capital
                'high_risk': 3.0,       # 3% of capital
                'very_high_risk': 5.0   # 5% of capital
            },
            execution_rules={
                'immediate_threshold': 0.9,  # Execute immediately if confidence > 90%
                'normal_threshold': 0.7,     # Normal execution if confidence > 70%
                'patient_threshold': 0.5     # Wait for better entry if confidence > 50%
            }
        )
        
        # Risk management parameters
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 2.0)  # 2% max risk
        self.default_risk_reward_ratio = self.config.get('default_risk_reward_ratio', 2.0)
        
        # Performance tracking
        self.decision_count = 0
        self.total_decision_time = 0.0
        
    async def make_decision(
        self, 
        resolved_signal: ResolvedSignal, 
        confidence_metrics: ConfidenceMetrics
    ) -> TradingDecision:
        """
        Make a rapid trading decision based on resolved signal and confidence metrics
        """
        start_time = time.time()
        
        try:
            # Determine base decision type
            decision_type = await self._determine_decision_type(resolved_signal, confidence_metrics)
            
            # Calculate position size based on risk assessment
            position_size, risk_level = await self._calculate_position_size(resolved_signal, confidence_metrics)
            
            # Calculate entry, stop loss, and take profit levels
            entry_price, stop_loss, take_profit = await self._calculate_execution_levels(
                resolved_signal, confidence_metrics, decision_type
            )
            
            # Calculate risk-reward ratio
            risk_reward_ratio = await self._calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profit, decision_type
            )
            
            # Determine execution urgency
            execution_urgency = await self._determine_execution_urgency(confidence_metrics)
            
            # Create decision factors breakdown
            decision_factors = await self._create_decision_factors(resolved_signal, confidence_metrics)
            
            # Create risk assessment
            risk_assessment = await self._create_risk_assessment(
                resolved_signal, confidence_metrics, risk_level, position_size
            )
            
            # Create execution instructions
            execution_instructions = await self._create_execution_instructions(
                decision_type, execution_urgency, entry_price, stop_loss, take_profit
            )
            
            # Update performance tracking
            decision_time = time.time() - start_time
            self.decision_count += 1
            self.total_decision_time += decision_time
            
            self.logger.debug(f"Made {decision_type.value} decision with {confidence_metrics.overall_confidence:.2f} confidence")
            
            return TradingDecision(
                decision_type=decision_type,
                confidence_level=confidence_metrics.overall_confidence,
                position_size=position_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                max_risk_per_trade=self.max_risk_per_trade,
                execution_urgency=execution_urgency,
                decision_factors=decision_factors,
                risk_assessment=risk_assessment,
                execution_instructions=execution_instructions
            )
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            raise
    
    async def _determine_decision_type(
        self, 
        resolved_signal: ResolvedSignal, 
        confidence_metrics: ConfidenceMetrics
    ) -> DecisionType:
        """Determine the type of trading decision"""
        
        signal_type = resolved_signal.signal_type
        confidence = confidence_metrics.overall_confidence
        strength = resolved_signal.strength
        
        # Strong decisions require high confidence and strength
        if confidence >= self.decision_matrix.confidence_thresholds['strong_buy'] and \
           strength >= self.decision_matrix.strength_thresholds['strong']:
            if signal_type == 'buy':
                return DecisionType.STRONG_BUY
            elif signal_type == 'sell':
                return DecisionType.STRONG_SELL
        
        # Regular decisions require medium confidence
        if confidence >= self.decision_matrix.confidence_thresholds['buy']:
            if signal_type == 'buy':
                return DecisionType.BUY
            elif signal_type == 'sell':
                return DecisionType.SELL
        
        # Default to hold for low confidence or hold signals
        return DecisionType.HOLD
    
    async def _calculate_position_size(
        self, 
        resolved_signal: ResolvedSignal, 
        confidence_metrics: ConfidenceMetrics
    ) -> Tuple[float, RiskLevel]:
        """Calculate position size based on confidence and risk assessment"""
        
        confidence = confidence_metrics.overall_confidence
        strength = resolved_signal.strength
        
        # Determine risk level
        if confidence >= 0.9 and strength >= 85:
            risk_level = RiskLevel.VERY_LOW
            base_size = self.decision_matrix.position_sizing_rules['very_low_risk']
        elif confidence >= 0.8 and strength >= 75:
            risk_level = RiskLevel.LOW
            base_size = self.decision_matrix.position_sizing_rules['low_risk']
        elif confidence >= 0.7 and strength >= 65:
            risk_level = RiskLevel.MEDIUM
            base_size = self.decision_matrix.position_sizing_rules['medium_risk']
        elif confidence >= 0.6 and strength >= 55:
            risk_level = RiskLevel.HIGH
            base_size = self.decision_matrix.position_sizing_rules['high_risk']
        else:
            risk_level = RiskLevel.VERY_HIGH
            base_size = self.decision_matrix.position_sizing_rules['very_high_risk']
        
        # Adjust position size based on additional factors
        size_multiplier = 1.0
        
        # Reduce size if many conflicting signals
        conflict_ratio = len(resolved_signal.discarded_signals) / (
            len(resolved_signal.contributing_signals) + len(resolved_signal.discarded_signals)
        ) if (len(resolved_signal.contributing_signals) + len(resolved_signal.discarded_signals)) > 0 else 0
        
        if conflict_ratio > 0.3:
            size_multiplier *= 0.8
        
        # Increase size for high trend confidence
        if confidence_metrics.trend_confidence > 0.8:
            size_multiplier *= 1.1
        
        # Reduce size for high volatility
        if confidence_metrics.volatility_confidence < 0.5:
            size_multiplier *= 0.9
        
        final_position_size = min(base_size * size_multiplier, self.max_risk_per_trade)
        
        return final_position_size, risk_level
    
    async def _calculate_execution_levels(
        self, 
        resolved_signal: ResolvedSignal, 
        confidence_metrics: ConfidenceMetrics, 
        decision_type: DecisionType
    ) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        
        # Use the first contributing signal as reference for price levels
        if resolved_signal.contributing_signals:
            reference_signal = resolved_signal.contributing_signals[0]
            entry_price = reference_signal.entry_price
            
            # Use signal's stop loss and take profit if available
            if reference_signal.stop_loss and reference_signal.take_profit:
                stop_loss = reference_signal.stop_loss
                take_profit = reference_signal.take_profit
            else:
                # Calculate default levels
                entry_price, stop_loss, take_profit = await self._calculate_default_levels(
                    entry_price, decision_type, confidence_metrics
                )
        else:
            # Fallback to basic calculation
            entry_price = 1.0  # This should be replaced with current market price
            entry_price, stop_loss, take_profit = await self._calculate_default_levels(
                entry_price, decision_type, confidence_metrics
            )
        
        return entry_price, stop_loss, take_profit
    
    async def _calculate_default_levels(
        self, 
        entry_price: float, 
        decision_type: DecisionType, 
        confidence_metrics: ConfidenceMetrics
    ) -> Tuple[float, float, float]:
        """Calculate default stop loss and take profit levels"""
        
        # Base risk percentage (distance from entry to stop loss)
        base_risk_pct = 0.01  # 1%
        
        # Adjust risk based on confidence
        if confidence_metrics.overall_confidence > 0.8:
            risk_pct = base_risk_pct * 0.8  # Tighter stop for high confidence
        elif confidence_metrics.overall_confidence < 0.6:
            risk_pct = base_risk_pct * 1.2  # Wider stop for low confidence
        else:
            risk_pct = base_risk_pct
        
        # Calculate stop loss and take profit
        if decision_type in [DecisionType.BUY, DecisionType.STRONG_BUY]:
            stop_loss = entry_price * (1 - risk_pct)
            take_profit = entry_price * (1 + risk_pct * self.default_risk_reward_ratio)
        elif decision_type in [DecisionType.SELL, DecisionType.STRONG_SELL]:
            stop_loss = entry_price * (1 + risk_pct)
            take_profit = entry_price * (1 - risk_pct * self.default_risk_reward_ratio)
        else:  # HOLD
            stop_loss = entry_price
            take_profit = entry_price
        
        return entry_price, stop_loss, take_profit
    
    async def _calculate_risk_reward_ratio(
        self, 
        entry_price: float, 
        stop_loss: float, 
        take_profit: float, 
        decision_type: DecisionType
    ) -> float:
        """Calculate risk-reward ratio"""
        
        if decision_type == DecisionType.HOLD:
            return 0.0
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    async def _determine_execution_urgency(self, confidence_metrics: ConfidenceMetrics) -> str:
        """Determine execution urgency based on confidence"""
        
        confidence = confidence_metrics.overall_confidence
        
        if confidence >= self.decision_matrix.execution_rules['immediate_threshold']:
            return 'immediate'
        elif confidence >= self.decision_matrix.execution_rules['normal_threshold']:
            return 'normal'
        else:
            return 'patient'
    
    async def _create_decision_factors(
        self, 
        resolved_signal: ResolvedSignal, 
        confidence_metrics: ConfidenceMetrics
    ) -> Dict[str, float]:
        """Create breakdown of decision factors"""
        
        return {
            'signal_strength': resolved_signal.strength / 100.0,
            'overall_confidence': confidence_metrics.overall_confidence,
            'timeframe_confidence': confidence_metrics.timeframe_confidence,
            'source_confidence': confidence_metrics.source_confidence,
            'trend_alignment': confidence_metrics.trend_confidence,
            'volume_confirmation': confidence_metrics.volume_confidence,
            'signal_consistency': confidence_metrics.consistency_confidence,
            'resolution_confidence': resolved_signal.resolution_confidence,
            'contributing_signals': len(resolved_signal.contributing_signals),
            'conflicting_signals': len(resolved_signal.discarded_signals)
        }
    
    async def _create_risk_assessment(
        self, 
        resolved_signal: ResolvedSignal, 
        confidence_metrics: ConfidenceMetrics, 
        risk_level: RiskLevel, 
        position_size: float
    ) -> Dict[str, Any]:
        """Create comprehensive risk assessment"""
        
        return {
            'risk_level': risk_level.value,
            'position_size_pct': position_size,
            'max_risk_per_trade': self.max_risk_per_trade,
            'confidence_risk': 1.0 - confidence_metrics.overall_confidence,
            'volatility_risk': 1.0 - confidence_metrics.volatility_confidence,
            'trend_risk': 1.0 - confidence_metrics.trend_confidence,
            'signal_conflict_risk': len(resolved_signal.discarded_signals) / 
                                  (len(resolved_signal.contributing_signals) + len(resolved_signal.discarded_signals))
                                  if (len(resolved_signal.contributing_signals) + len(resolved_signal.discarded_signals)) > 0 else 0,
            'execution_risk_factors': [
                'low_confidence' if confidence_metrics.overall_confidence < 0.7 else None,
                'high_volatility' if confidence_metrics.volatility_confidence < 0.5 else None,
                'trend_misalignment' if confidence_metrics.trend_confidence < 0.6 else None,
                'signal_conflicts' if len(resolved_signal.discarded_signals) > len(resolved_signal.contributing_signals) else None
            ]
        }
    
    async def _create_execution_instructions(
        self, 
        decision_type: DecisionType, 
        execution_urgency: str, 
        entry_price: float, 
        stop_loss: float, 
        take_profit: float
    ) -> Dict[str, Any]:
        """Create detailed execution instructions"""
        
        instructions = {
            'action': decision_type.value,
            'urgency': execution_urgency,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'order_type': 'market' if execution_urgency == 'immediate' else 'limit',
            'time_in_force': 'IOC' if execution_urgency == 'immediate' else 'GTC',
            'execution_notes': []
        }
        
        # Add execution notes based on decision type and urgency
        if decision_type in [DecisionType.STRONG_BUY, DecisionType.STRONG_SELL]:
            instructions['execution_notes'].append('High confidence signal - consider larger position')
        
        if execution_urgency == 'immediate':
            instructions['execution_notes'].append('Execute immediately - strong signal detected')
        elif execution_urgency == 'patient':
            instructions['execution_notes'].append('Wait for better entry - signal confidence moderate')
        
        return instructions
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get decision making performance metrics"""
        return {
            'total_decisions': self.decision_count,
            'average_decision_time_ms': (self.total_decision_time / self.decision_count * 1000) 
                                      if self.decision_count > 0 else 0
        }
