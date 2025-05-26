"""
Platform3 Forex Trading Platform
Signal Confidence Calculator - Advanced Signal Strength Scoring

This module provides sophisticated confidence scoring for trading signals
across multiple timeframes and indicators for optimal decision making.

Author: Platform3 Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types for confidence calculation"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class TimeFrame(Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

@dataclass
class SignalData:
    """Individual signal data structure"""
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    timeframe: TimeFrame
    indicator_name: str
    timestamp: datetime
    price: float
    volume: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConfidenceScore:
    """Confidence score result"""
    overall_confidence: float  # 0.0 to 1.0
    signal_type: SignalType
    contributing_signals: List[SignalData]
    timeframe_weights: Dict[TimeFrame, float]
    risk_adjusted_confidence: float
    execution_priority: int  # 1-5 (1=highest)
    timestamp: datetime

class ConfidenceCalculator:
    """
    Advanced signal confidence calculator for multi-timeframe analysis
    
    Features:
    - Multi-timeframe signal aggregation
    - Weighted confidence scoring
    - Risk-adjusted confidence calculation
    - Real-time signal strength assessment
    - Adaptive learning from historical performance
    """
    
    def __init__(self):
        """Initialize the confidence calculator"""
        self.timeframe_weights = {
            TimeFrame.M1: 0.05,   # Low weight for noise
            TimeFrame.M5: 0.10,   # Scalping signals
            TimeFrame.M15: 0.15,  # Day trading
            TimeFrame.M30: 0.20,  # Intraday confirmation
            TimeFrame.H1: 0.25,   # Strong trend signals
            TimeFrame.H4: 0.20,   # Swing trading
            TimeFrame.D1: 0.05    # Long-term bias
        }
        
        self.indicator_weights = {
            'momentum': 0.25,
            'trend': 0.30,
            'volume': 0.20,
            'volatility': 0.15,
            'support_resistance': 0.10
        }
        
        self.signal_history = []
        self.performance_metrics = {}
        
    async def calculate_confidence(
        self,
        signals: List[SignalData],
        current_price: float,
        market_volatility: float = 0.5
    ) -> ConfidenceScore:
        """
        Calculate overall confidence score for a set of signals
        
        Args:
            signals: List of individual signals
            current_price: Current market price
            market_volatility: Current market volatility (0.0-1.0)
            
        Returns:
            ConfidenceScore object with detailed analysis
        """
        try:
            if not signals:
                return self._create_neutral_confidence()
            
            # Group signals by type
            signal_groups = self._group_signals_by_type(signals)
            
            # Calculate base confidence for each signal type
            confidence_scores = {}
            for signal_type, signal_list in signal_groups.items():
                confidence_scores[signal_type] = await self._calculate_type_confidence(
                    signal_list, current_price
                )
            
            # Determine dominant signal type
            dominant_signal = max(confidence_scores.items(), key=lambda x: x[1])
            signal_type, base_confidence = dominant_signal
            
            # Apply timeframe weighting
            timeframe_adjusted = self._apply_timeframe_weighting(signals, base_confidence)
            
            # Apply volume confirmation
            volume_adjusted = self._apply_volume_confirmation(signals, timeframe_adjusted)
            
            # Apply risk adjustment based on market volatility
            risk_adjusted = self._apply_risk_adjustment(volume_adjusted, market_volatility)
            
            # Calculate execution priority
            execution_priority = self._calculate_execution_priority(
                risk_adjusted, signal_type, market_volatility
            )
            
            # Create confidence score result
            confidence_score = ConfidenceScore(
                overall_confidence=volume_adjusted,
                signal_type=signal_type,
                contributing_signals=signals,
                timeframe_weights=self.timeframe_weights,
                risk_adjusted_confidence=risk_adjusted,
                execution_priority=execution_priority,
                timestamp=datetime.now()
            )
            
            # Store for learning
            self.signal_history.append(confidence_score)
            
            logger.info(f"Confidence calculated: {signal_type.value} - {volume_adjusted:.3f}")
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return self._create_neutral_confidence()
    
    def _group_signals_by_type(self, signals: List[SignalData]) -> Dict[SignalType, List[SignalData]]:
        """Group signals by their type"""
        groups = {}
        for signal in signals:
            if signal.signal_type not in groups:
                groups[signal.signal_type] = []
            groups[signal.signal_type].append(signal)
        return groups
    
    async def _calculate_type_confidence(
        self,
        signals: List[SignalData],
        current_price: float
    ) -> float:
        """Calculate confidence for a specific signal type"""
        if not signals:
            return 0.0
        
        # Weight by signal strength and recency
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            # Time decay factor (newer signals have higher weight)
            time_diff = (datetime.now() - signal.timestamp).total_seconds()
            time_weight = max(0.1, 1.0 - (time_diff / 3600))  # 1-hour decay
            
            # Timeframe weight
            tf_weight = self.timeframe_weights.get(signal.timeframe, 0.1)
            
            # Combined weight
            signal_weight = signal.strength * time_weight * tf_weight
            
            weighted_confidence += signal_weight
            total_weight += signal_weight
        
        return min(1.0, weighted_confidence / max(total_weight, 0.001))
    
    def _apply_timeframe_weighting(self, signals: List[SignalData], base_confidence: float) -> float:
        """Apply timeframe-based weighting to confidence"""
        timeframe_bonus = 0.0
        
        # Check for multi-timeframe confluence
        timeframes_present = set(signal.timeframe for signal in signals)
        
        if len(timeframes_present) >= 3:
            timeframe_bonus += 0.15  # Multi-timeframe confluence bonus
        
        if TimeFrame.H1 in timeframes_present and TimeFrame.M15 in timeframes_present:
            timeframe_bonus += 0.10  # Strong intraday confluence
        
        return min(1.0, base_confidence + timeframe_bonus)
    
    def _apply_volume_confirmation(self, signals: List[SignalData], confidence: float) -> float:
        """Apply volume-based confirmation to confidence"""
        volume_signals = [s for s in signals if s.volume is not None]
        
        if not volume_signals:
            return confidence * 0.9  # Slight penalty for no volume confirmation
        
        # Calculate average volume strength
        avg_volume_strength = np.mean([s.volume for s in volume_signals])
        
        if avg_volume_strength > 0.7:
            return min(1.0, confidence * 1.1)  # Volume confirmation bonus
        elif avg_volume_strength < 0.3:
            return confidence * 0.85  # Low volume penalty
        
        return confidence
    
    def _apply_risk_adjustment(self, confidence: float, market_volatility: float) -> float:
        """Apply risk adjustment based on market conditions"""
        if market_volatility > 0.8:
            # High volatility - reduce confidence
            return confidence * 0.8
        elif market_volatility < 0.2:
            # Low volatility - slight confidence boost
            return min(1.0, confidence * 1.05)
        
        return confidence
    
    def _calculate_execution_priority(
        self,
        confidence: float,
        signal_type: SignalType,
        market_volatility: float
    ) -> int:
        """Calculate execution priority (1-5, 1=highest)"""
        if confidence >= 0.8:
            return 1  # Highest priority
        elif confidence >= 0.6:
            return 2  # High priority
        elif confidence >= 0.4:
            return 3  # Medium priority
        elif confidence >= 0.2:
            return 4  # Low priority
        else:
            return 5  # Lowest priority
    
    def _create_neutral_confidence(self) -> ConfidenceScore:
        """Create a neutral confidence score for error cases"""
        return ConfidenceScore(
            overall_confidence=0.0,
            signal_type=SignalType.HOLD,
            contributing_signals=[],
            timeframe_weights=self.timeframe_weights,
            risk_adjusted_confidence=0.0,
            execution_priority=5,
            timestamp=datetime.now()
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get historical performance metrics"""
        if not self.signal_history:
            return {}
        
        recent_signals = self.signal_history[-100:]  # Last 100 signals
        
        return {
            'average_confidence': np.mean([s.overall_confidence for s in recent_signals]),
            'high_confidence_ratio': len([s for s in recent_signals if s.overall_confidence > 0.7]) / len(recent_signals),
            'signal_distribution': {
                signal_type.value: len([s for s in recent_signals if s.signal_type == signal_type])
                for signal_type in SignalType
            }
        }
    
    def update_weights(self, performance_feedback: Dict[str, float]):
        """Update weights based on performance feedback"""
        # Adaptive learning implementation
        for timeframe, performance in performance_feedback.items():
            if timeframe in self.timeframe_weights:
                # Adjust weights based on performance
                adjustment = (performance - 0.5) * 0.1  # Small adjustments
                self.timeframe_weights[timeframe] = max(0.01, min(0.5, 
                    self.timeframe_weights[timeframe] + adjustment))
        
        # Normalize weights
        total_weight = sum(self.timeframe_weights.values())
        for tf in self.timeframe_weights:
            self.timeframe_weights[tf] /= total_weight

# Example usage and testing
if __name__ == "__main__":
    async def test_confidence_calculator():
        calculator = ConfidenceCalculator()
        
        # Create test signals
        test_signals = [
            SignalData(
                signal_type=SignalType.BUY,
                strength=0.8,
                timeframe=TimeFrame.H1,
                indicator_name="RSI",
                timestamp=datetime.now(),
                price=1.2500,
                volume=0.7
            ),
            SignalData(
                signal_type=SignalType.BUY,
                strength=0.6,
                timeframe=TimeFrame.M15,
                indicator_name="MACD",
                timestamp=datetime.now(),
                price=1.2501,
                volume=0.5
            )
        ]
        
        # Calculate confidence
        confidence = await calculator.calculate_confidence(
            signals=test_signals,
            current_price=1.2502,
            market_volatility=0.4
        )
        
        print(f"Confidence Score: {confidence.overall_confidence:.3f}")
        print(f"Signal Type: {confidence.signal_type.value}")
        print(f"Execution Priority: {confidence.execution_priority}")
        print(f"Risk Adjusted: {confidence.risk_adjusted_confidence:.3f}")
    
    # Run test
    asyncio.run(test_confidence_calculator())
