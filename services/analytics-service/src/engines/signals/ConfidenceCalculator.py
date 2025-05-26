"""
Confidence Calculator Engine
Calculates signal strength scoring and confidence metrics.
Provides sophisticated confidence assessment for trading signals.
"""

import asyncio
import time
import logging
import numpy as np
import statistics
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum

from .SignalAggregator import SignalInput, Timeframe
from .ConflictResolver import ResolvedSignal


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for signals"""
    overall_confidence: float  # 0-1 overall confidence score
    timeframe_confidence: float  # Confidence based on timeframe agreement
    source_confidence: float  # Confidence based on source reliability
    strength_confidence: float  # Confidence based on signal strength
    consistency_confidence: float  # Confidence based on signal consistency
    volume_confidence: float  # Confidence based on volume confirmation
    trend_confidence: float  # Confidence based on trend alignment
    volatility_confidence: float  # Confidence based on volatility context
    historical_confidence: float  # Confidence based on historical performance
    risk_adjusted_confidence: float  # Risk-adjusted confidence score
    confidence_breakdown: Dict[str, float]  # Detailed breakdown of confidence factors


class ConfidenceFactors(Enum):
    """Factors that influence signal confidence"""
    TIMEFRAME_AGREEMENT = "timeframe_agreement"
    SOURCE_RELIABILITY = "source_reliability"
    SIGNAL_STRENGTH = "signal_strength"
    SIGNAL_CONSISTENCY = "signal_consistency"
    VOLUME_CONFIRMATION = "volume_confirmation"
    TREND_ALIGNMENT = "trend_alignment"
    VOLATILITY_CONTEXT = "volatility_context"
    HISTORICAL_PERFORMANCE = "historical_performance"
    MARKET_CONDITIONS = "market_conditions"


class ConfidenceCalculator:
    """
    Confidence Calculator Engine
    Calculates comprehensive confidence metrics for trading signals
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Confidence factor weights
        self.confidence_weights = {
            ConfidenceFactors.TIMEFRAME_AGREEMENT: 0.2,
            ConfidenceFactors.SOURCE_RELIABILITY: 0.15,
            ConfidenceFactors.SIGNAL_STRENGTH: 0.15,
            ConfidenceFactors.SIGNAL_CONSISTENCY: 0.15,
            ConfidenceFactors.VOLUME_CONFIRMATION: 0.1,
            ConfidenceFactors.TREND_ALIGNMENT: 0.1,
            ConfidenceFactors.VOLATILITY_CONTEXT: 0.05,
            ConfidenceFactors.HISTORICAL_PERFORMANCE: 0.05,
            ConfidenceFactors.MARKET_CONDITIONS: 0.05
        }
        
        # Source reliability scores
        self.source_reliability = {
            'trend': 0.9,
            'momentum': 0.8,
            'volume': 0.75,
            'pattern': 0.7,
            'volatility': 0.6,
            'sentiment': 0.5
        }
        
        # Timeframe reliability scores
        self.timeframe_reliability = {
            Timeframe.H4: 0.95,
            Timeframe.H1: 0.9,
            Timeframe.M30: 0.8,
            Timeframe.M15: 0.7,
            Timeframe.M5: 0.6,
            Timeframe.M1: 0.5
        }
        
        # Historical performance tracking
        self.historical_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        
    async def calculate_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics for a resolved signal
        """
        start_time = time.time()
        
        try:
            # Calculate individual confidence factors
            timeframe_confidence = await self._calculate_timeframe_confidence(signals, resolved_signal)
            source_confidence = await self._calculate_source_confidence(signals, resolved_signal)
            strength_confidence = await self._calculate_strength_confidence(signals, resolved_signal)
            consistency_confidence = await self._calculate_consistency_confidence(signals, resolved_signal)
            volume_confidence = await self._calculate_volume_confidence(signals, resolved_signal)
            trend_confidence = await self._calculate_trend_confidence(signals, resolved_signal)
            volatility_confidence = await self._calculate_volatility_confidence(signals, resolved_signal)
            historical_confidence = await self._calculate_historical_confidence(signals, resolved_signal)
            
            # Calculate overall confidence using weighted average
            confidence_scores = {
                ConfidenceFactors.TIMEFRAME_AGREEMENT: timeframe_confidence,
                ConfidenceFactors.SOURCE_RELIABILITY: source_confidence,
                ConfidenceFactors.SIGNAL_STRENGTH: strength_confidence,
                ConfidenceFactors.SIGNAL_CONSISTENCY: consistency_confidence,
                ConfidenceFactors.VOLUME_CONFIRMATION: volume_confidence,
                ConfidenceFactors.TREND_ALIGNMENT: trend_confidence,
                ConfidenceFactors.VOLATILITY_CONTEXT: volatility_confidence,
                ConfidenceFactors.HISTORICAL_PERFORMANCE: historical_confidence
            }
            
            overall_confidence = sum(
                score * self.confidence_weights[factor]
                for factor, score in confidence_scores.items()
            )
            
            # Calculate risk-adjusted confidence
            risk_adjusted_confidence = await self._calculate_risk_adjusted_confidence(
                overall_confidence, signals, resolved_signal
            )
            
            # Create confidence breakdown
            confidence_breakdown = {factor.value: score for factor, score in confidence_scores.items()}
            
            # Update performance tracking
            calculation_time = time.time() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            return ConfidenceMetrics(
                overall_confidence=overall_confidence,
                timeframe_confidence=timeframe_confidence,
                source_confidence=source_confidence,
                strength_confidence=strength_confidence,
                consistency_confidence=consistency_confidence,
                volume_confidence=volume_confidence,
                trend_confidence=trend_confidence,
                volatility_confidence=volatility_confidence,
                historical_confidence=historical_confidence,
                risk_adjusted_confidence=risk_adjusted_confidence,
                confidence_breakdown=confidence_breakdown
            )
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            raise
    
    async def _calculate_timeframe_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on timeframe agreement"""
        
        if not signals:
            return 0.0
        
        # Group signals by timeframe
        timeframe_signals = defaultdict(list)
        for signal in signals:
            timeframe_signals[signal.timeframe].append(signal)
        
        # Calculate agreement within each timeframe
        timeframe_agreements = []
        timeframe_weights = []
        
        for timeframe, tf_signals in timeframe_signals.items():
            # Count signals that agree with resolved signal
            agreeing_signals = sum(1 for s in tf_signals if s.signal_type == resolved_signal.signal_type)
            agreement_ratio = agreeing_signals / len(tf_signals)
            
            # Weight by timeframe reliability
            weight = self.timeframe_reliability[timeframe]
            
            timeframe_agreements.append(agreement_ratio)
            timeframe_weights.append(weight)
        
        # Calculate weighted average agreement
        if not timeframe_agreements:
            return 0.0
        
        weighted_agreement = np.average(timeframe_agreements, weights=timeframe_weights)
        return min(weighted_agreement, 1.0)
    
    async def _calculate_source_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on source reliability"""
        
        if not signals:
            return 0.0
        
        # Get signals that agree with resolved signal
        agreeing_signals = [s for s in signals if s.signal_type == resolved_signal.signal_type]
        
        if not agreeing_signals:
            return 0.0
        
        # Calculate weighted reliability of agreeing sources
        total_weight = 0.0
        weighted_reliability = 0.0
        
        for signal in agreeing_signals:
            reliability = self.source_reliability.get(signal.source, 0.5)
            weight = signal.confidence  # Use signal's own confidence as weight
            
            weighted_reliability += reliability * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_reliability / total_weight
    
    async def _calculate_strength_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on signal strength"""
        
        # Get signals that agree with resolved signal
        agreeing_signals = [s for s in signals if s.signal_type == resolved_signal.signal_type]
        
        if not agreeing_signals:
            return 0.0
        
        # Calculate average strength of agreeing signals
        strengths = [s.strength for s in agreeing_signals]
        avg_strength = np.mean(strengths)
        
        # Normalize to 0-1 range
        normalized_strength = avg_strength / 100.0
        
        # Apply confidence boost for high strength signals
        if normalized_strength > 0.8:
            return min(normalized_strength * 1.1, 1.0)
        elif normalized_strength > 0.6:
            return normalized_strength
        else:
            return normalized_strength * 0.9
    
    async def _calculate_consistency_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on signal consistency"""
        
        # Get signals that agree with resolved signal
        agreeing_signals = [s for s in signals if s.signal_type == resolved_signal.signal_type]
        
        if len(agreeing_signals) < 2:
            return 0.5  # Neutral confidence for single signal
        
        # Calculate consistency metrics
        strengths = [s.strength for s in agreeing_signals]
        confidences = [s.confidence for s in agreeing_signals]
        
        # Calculate coefficient of variation (lower = more consistent)
        strength_cv = statistics.stdev(strengths) / statistics.mean(strengths) if statistics.mean(strengths) > 0 else 1.0
        confidence_cv = statistics.stdev(confidences) / statistics.mean(confidences) if statistics.mean(confidences) > 0 else 1.0
        
        # Convert to consistency score (higher = more consistent)
        strength_consistency = max(0, 1 - strength_cv)
        confidence_consistency = max(0, 1 - confidence_cv)
        
        # Average the consistency scores
        overall_consistency = (strength_consistency + confidence_consistency) / 2
        
        return overall_consistency
    
    async def _calculate_volume_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on volume confirmation"""
        
        # Look for volume-based signals
        volume_signals = [s for s in signals if 'volume' in s.source.lower()]
        
        if not volume_signals:
            return 0.5  # Neutral confidence if no volume data
        
        # Check if volume signals agree with resolved signal
        agreeing_volume_signals = [s for s in volume_signals if s.signal_type == resolved_signal.signal_type]
        
        if not agreeing_volume_signals:
            return 0.3  # Lower confidence if volume disagrees
        
        # Calculate volume confirmation strength
        volume_strength = np.mean([s.strength for s in agreeing_volume_signals])
        volume_confidence = np.mean([s.confidence for s in agreeing_volume_signals])
        
        # Combine volume strength and confidence
        volume_confirmation = (volume_strength / 100.0 + volume_confidence) / 2
        
        return min(volume_confirmation * 1.2, 1.0)  # Boost for volume confirmation
    
    async def _calculate_trend_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on trend alignment"""
        
        # Look for trend-based signals
        trend_signals = [s for s in signals if 'trend' in s.source.lower()]
        
        if not trend_signals:
            return 0.5  # Neutral confidence if no trend data
        
        # Check if trend signals agree with resolved signal
        agreeing_trend_signals = [s for s in trend_signals if s.signal_type == resolved_signal.signal_type]
        
        if not agreeing_trend_signals:
            return 0.2  # Very low confidence if trend disagrees
        
        # Calculate trend alignment strength
        trend_strength = np.mean([s.strength for s in agreeing_trend_signals])
        trend_confidence = np.mean([s.confidence for s in agreeing_trend_signals])
        
        # Combine trend strength and confidence
        trend_alignment = (trend_strength / 100.0 + trend_confidence) / 2
        
        return min(trend_alignment * 1.3, 1.0)  # Strong boost for trend alignment
    
    async def _calculate_volatility_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on volatility context"""
        
        # Look for volatility-based signals
        volatility_signals = [s for s in signals if 'volatility' in s.source.lower()]
        
        if not volatility_signals:
            return 0.7  # Slightly positive confidence if no volatility concerns
        
        # Analyze volatility context
        avg_volatility_strength = np.mean([s.strength for s in volatility_signals])
        
        # High volatility reduces confidence, low volatility increases it
        if avg_volatility_strength > 80:
            return 0.4  # High volatility = lower confidence
        elif avg_volatility_strength > 60:
            return 0.6  # Medium volatility = medium confidence
        else:
            return 0.8  # Low volatility = higher confidence
    
    async def _calculate_historical_confidence(
        self, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate confidence based on historical performance"""
        
        # For now, return a baseline confidence
        # In a real implementation, this would analyze historical signal performance
        return 0.7
    
    async def _calculate_risk_adjusted_confidence(
        self, 
        base_confidence: float, 
        signals: List[SignalInput], 
        resolved_signal: ResolvedSignal
    ) -> float:
        """Calculate risk-adjusted confidence score"""
        
        # Calculate risk factors
        signal_count = len(signals)
        agreement_ratio = len(resolved_signal.contributing_signals) / signal_count if signal_count > 0 else 0
        
        # Risk adjustments
        risk_adjustments = []
        
        # Low signal count increases risk
        if signal_count < 3:
            risk_adjustments.append(-0.1)
        elif signal_count > 10:
            risk_adjustments.append(0.05)  # More signals = slightly more confidence
        
        # Low agreement increases risk
        if agreement_ratio < 0.5:
            risk_adjustments.append(-0.2)
        elif agreement_ratio > 0.8:
            risk_adjustments.append(0.1)
        
        # Apply risk adjustments
        risk_adjusted = base_confidence + sum(risk_adjustments)
        
        return max(0.0, min(risk_adjusted, 1.0))
    
    async def update_historical_performance(self, symbol: str, signal_type: str, was_correct: bool):
        """Update historical performance tracking"""
        key = f"{symbol}_{signal_type}"
        self.historical_performance[key]['total'] += 1
        if was_correct:
            self.historical_performance[key]['correct'] += 1
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get confidence calculation performance metrics"""
        return {
            'total_calculations': self.calculation_count,
            'average_calculation_time_ms': (self.total_calculation_time / self.calculation_count * 1000) 
                                         if self.calculation_count > 0 else 0
        }
