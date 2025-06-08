"""
Signal Confidence AI Module for Platform3
Provides confidence scoring and signal validation for trading decisions
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalConfidenceAI:
    """
    AI-based signal confidence assessment and validation
    Evaluates the reliability and strength of trading signals
    """
    
    def __init__(self):
        """Initialize the Signal Confidence AI"""
        self.initialized = True
        self.confidence_models = {
            'signal_strength': self._assess_signal_strength,
            'market_context': self._assess_market_context,
            'indicator_consensus': self._assess_indicator_consensus,
            'historical_accuracy': self._assess_historical_accuracy,
            'volatility_adjustment': self._assess_volatility_adjustment
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.75,
            'medium': 0.50,
            'low': 0.25
        }
        
        logger.info("SignalConfidenceAI initialized successfully")
    
    def assess_signal_confidence(self, signal_data: Dict[str, Any], 
                               market_data: Dict[str, Any],
                               indicator_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess confidence in a trading signal
        
        Args:
            signal_data: Signal information (direction, strength, etc.)
            market_data: Current market data
            indicator_results: Results from various indicators
            
        Returns:
            Comprehensive confidence assessment
        """
        try:
            confidence_assessment = {
                'overall_confidence': 0.0,
                'confidence_level': 'LOW',
                'confidence_components': {},
                'signal_validation': {},
                'recommendations': [],
                'risk_adjusted_confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate individual confidence components
            total_confidence = 0.0
            component_count = 0
            
            for confidence_type, confidence_model in self.confidence_models.items():
                try:
                    component_confidence = confidence_model(signal_data, market_data, indicator_results)
                    confidence_assessment['confidence_components'][confidence_type] = component_confidence
                    total_confidence += component_confidence['score']
                    component_count += 1
                except Exception as e:
                    logger.warning(f"Error in confidence component {confidence_type}: {e}")
                    # Use default confidence for failed components
                    confidence_assessment['confidence_components'][confidence_type] = {
                        'score': 0.5,
                        'description': f"Default confidence for {confidence_type}"
                    }
                    total_confidence += 0.5
                    component_count += 1
            
            # Calculate overall confidence (0-1 scale)
            confidence_assessment['overall_confidence'] = total_confidence / component_count if component_count > 0 else 0.5
            
            # Determine confidence level
            confidence_assessment['confidence_level'] = self._determine_confidence_level(
                confidence_assessment['overall_confidence']
            )
            
            # Validate signal quality
            confidence_assessment['signal_validation'] = self._validate_signal_quality(
                signal_data, market_data, confidence_assessment['overall_confidence']
            )
            
            # Calculate risk-adjusted confidence
            confidence_assessment['risk_adjusted_confidence'] = self._calculate_risk_adjusted_confidence(
                confidence_assessment['overall_confidence'], market_data
            )
            
            # Generate recommendations
            confidence_assessment['recommendations'] = self._generate_confidence_recommendations(
                confidence_assessment
            )
            
            return confidence_assessment
            
        except Exception as e:
            logger.error(f"Error in signal confidence assessment: {e}")
            return self._default_confidence_assessment()
    
    def assess_multi_signal_confidence(self, signals: List[Dict[str, Any]], 
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess confidence across multiple signals
        
        Args:
            signals: List of trading signals
            market_data: Current market data
            
        Returns:
            Multi-signal confidence assessment
        """
        try:
            multi_confidence = {
                'consensus_confidence': 0.0,
                'signal_agreement': 0.0,
                'conflicting_signals': 0,
                'dominant_direction': 'NEUTRAL',
                'individual_confidences': [],
                'weighted_confidence': 0.0,
                'recommendations': []
            }
            
            if not signals:
                return multi_confidence
            
            # Assess individual signal confidences
            individual_assessments = []
            for signal in signals:
                assessment = self.assess_signal_confidence(signal, market_data)
                individual_assessments.append(assessment)
                multi_confidence['individual_confidences'].append(assessment['overall_confidence'])
            
            # Calculate consensus metrics
            multi_confidence['consensus_confidence'] = np.mean(multi_confidence['individual_confidences'])
            
            # Analyze signal agreement
            directions = [sig.get('direction', 'NEUTRAL') for sig in signals]
            buy_signals = directions.count('BUY')
            sell_signals = directions.count('SELL')
            neutral_signals = directions.count('NEUTRAL')
            
            total_signals = len(signals)
            if buy_signals > sell_signals and buy_signals > neutral_signals:
                multi_confidence['dominant_direction'] = 'BUY'
                agreement_ratio = buy_signals / total_signals
            elif sell_signals > buy_signals and sell_signals > neutral_signals:
                multi_confidence['dominant_direction'] = 'SELL'
                agreement_ratio = sell_signals / total_signals
            else:
                multi_confidence['dominant_direction'] = 'NEUTRAL'
                agreement_ratio = max(buy_signals, sell_signals, neutral_signals) / total_signals
            
            multi_confidence['signal_agreement'] = agreement_ratio
            multi_confidence['conflicting_signals'] = total_signals - max(buy_signals, sell_signals, neutral_signals)
            
            # Calculate weighted confidence (higher weight for more confident signals)
            weights = [conf for conf in multi_confidence['individual_confidences']]
            if sum(weights) > 0:
                weighted_confidences = [conf * weight for conf, weight in zip(multi_confidence['individual_confidences'], weights)]
                multi_confidence['weighted_confidence'] = sum(weighted_confidences) / sum(weights)
            else:
                multi_confidence['weighted_confidence'] = multi_confidence['consensus_confidence']
            
            # Generate multi-signal recommendations
            multi_confidence['recommendations'] = self._generate_multi_signal_recommendations(multi_confidence)
            
            return multi_confidence
            
        except Exception as e:
            logger.error(f"Error in multi-signal confidence assessment: {e}")
            return {'error': str(e)}
    
    def _assess_signal_strength(self, signal_data: Dict[str, Any], 
                              market_data: Dict[str, Any], 
                              indicator_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess the intrinsic strength of the signal"""
        try:
            signal_strength = signal_data.get('strength', 0.5)
            signal_magnitude = abs(signal_data.get('magnitude', 0.5))
            
            # Normalize signal strength (0-1)
            normalized_strength = min(max(signal_strength, 0), 1)
            normalized_magnitude = min(max(signal_magnitude, 0), 1)
            
            # Combine strength and magnitude
            combined_score = (normalized_strength + normalized_magnitude) / 2
            
            return {
                'score': combined_score,
                'signal_strength': normalized_strength,
                'signal_magnitude': normalized_magnitude,
                'description': f"Signal strength assessment: {combined_score:.3f}"
            }
        except:
            return {
                'score': 0.5,
                'description': "Default signal strength assessment"
            }
    
    def _assess_market_context(self, signal_data: Dict[str, Any], 
                             market_data: Dict[str, Any], 
                             indicator_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess how well the signal fits current market context"""
        try:
            # Analyze market volatility
            if 'price_series' in market_data and len(market_data['price_series']) > 1:
                prices = market_data['price_series']
                returns = np.diff(np.log(prices))
                volatility = np.std(returns)
            else:
                high = market_data.get('high', market_data.get('close', 1.0))
                low = market_data.get('low', market_data.get('close', 1.0))
                close = market_data.get('close', 1.0)
                volatility = (high - low) / close if close > 0 else 0.01
            
            # Market context score based on volatility regime
            if volatility < 0.005:  # Low volatility
                context_score = 0.7  # Moderate confidence in trending signals
            elif volatility < 0.015:  # Normal volatility
                context_score = 0.8  # High confidence
            else:  # High volatility
                context_score = 0.6  # Lower confidence due to noise
            
            return {
                'score': context_score,
                'volatility': volatility,
                'volatility_regime': 'LOW' if volatility < 0.005 else 'NORMAL' if volatility < 0.015 else 'HIGH',
                'description': f"Market context assessment based on volatility: {volatility:.4f}"
            }
        except:
            return {
                'score': 0.6,
                'description': "Default market context assessment"
            }
    
    def _assess_indicator_consensus(self, signal_data: Dict[str, Any], 
                                  market_data: Dict[str, Any], 
                                  indicator_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess consensus among multiple indicators"""
        try:
            if not indicator_results:
                return {
                    'score': 0.5,
                    'description': "No indicator results available for consensus"
                }
            
            # Count indicators supporting the signal direction
            signal_direction = signal_data.get('direction', 'NEUTRAL')
            supporting_indicators = 0
            total_indicators = 0
            
            for indicator_name, result in indicator_results.items():
                if isinstance(result, dict) and 'signal' in result:
                    total_indicators += 1
                    if result['signal'] == signal_direction:
                        supporting_indicators += 1
            
            # Calculate consensus score
            if total_indicators > 0:
                consensus_score = supporting_indicators / total_indicators
            else:
                consensus_score = 0.5
            
            return {
                'score': consensus_score,
                'supporting_indicators': supporting_indicators,
                'total_indicators': total_indicators,
                'consensus_ratio': consensus_score,
                'description': f"Indicator consensus: {supporting_indicators}/{total_indicators} supporting"
            }
        except:
            return {
                'score': 0.5,
                'description': "Default indicator consensus assessment"
            }
    
    def _assess_historical_accuracy(self, signal_data: Dict[str, Any], 
                                  market_data: Dict[str, Any], 
                                  indicator_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess historical accuracy of similar signals"""
        try:
            # Simplified historical accuracy model
            # In production, this would query historical performance database
            
            signal_type = signal_data.get('type', 'UNKNOWN')
            timeframe = market_data.get('timeframe', '1H')
            
            # Default accuracy estimates by signal type and timeframe
            accuracy_estimates = {
                'TREND': {'1H': 0.65, '4H': 0.72, '1D': 0.78},
                'REVERSAL': {'1H': 0.55, '4H': 0.62, '1D': 0.68},
                'BREAKOUT': {'1H': 0.60, '4H': 0.68, '1D': 0.75},
                'MOMENTUM': {'1H': 0.58, '4H': 0.65, '1D': 0.70}
            }
            
            estimated_accuracy = accuracy_estimates.get(signal_type, {}).get(timeframe, 0.60)
            
            return {
                'score': estimated_accuracy,
                'estimated_accuracy': estimated_accuracy,
                'signal_type': signal_type,
                'timeframe': timeframe,
                'description': f"Historical accuracy estimate: {estimated_accuracy:.3f}"
            }
        except:
            return {
                'score': 0.6,
                'description': "Default historical accuracy assessment"
            }
    
    def _assess_volatility_adjustment(self, signal_data: Dict[str, Any], 
                                    market_data: Dict[str, Any], 
                                    indicator_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adjust confidence based on current volatility conditions"""
        try:
            # Calculate current volatility
            if 'price_series' in market_data and len(market_data['price_series']) > 1:
                prices = market_data['price_series']
                returns = np.diff(np.log(prices))
                current_vol = np.std(returns)
            else:
                high = market_data.get('high', market_data.get('close', 1.0))
                low = market_data.get('low', market_data.get('close', 1.0))
                close = market_data.get('close', 1.0)
                current_vol = (high - low) / close if close > 0 else 0.01
            
            # Volatility adjustment factor
            # Lower volatility = higher confidence for trend signals
            # Higher volatility = lower confidence due to noise
            if current_vol < 0.005:
                vol_adjustment = 0.85  # High confidence in low vol
            elif current_vol < 0.010:
                vol_adjustment = 0.75  # Normal confidence
            elif current_vol < 0.020:
                vol_adjustment = 0.65  # Reduced confidence
            else:
                vol_adjustment = 0.55  # Low confidence in high vol
            
            return {
                'score': vol_adjustment,
                'current_volatility': current_vol,
                'adjustment_factor': vol_adjustment,
                'description': f"Volatility adjustment: {vol_adjustment:.3f} (vol: {current_vol:.4f})"
            }
        except:
            return {
                'score': 0.7,
                'description': "Default volatility adjustment"
            }
    
    def _determine_confidence_level(self, overall_confidence: float) -> str:
        """Determine categorical confidence level"""
        if overall_confidence >= self.confidence_thresholds['high']:
            return 'HIGH'
        elif overall_confidence >= self.confidence_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _validate_signal_quality(self, signal_data: Dict[str, Any], 
                               market_data: Dict[str, Any], 
                               confidence: float) -> Dict[str, Any]:
        """Validate overall signal quality"""
        validation = {
            'is_valid': True,
            'quality_score': confidence,
            'quality_issues': [],
            'quality_level': 'GOOD'
        }
        
        # Check for basic signal requirements
        if not signal_data.get('direction'):
            validation['quality_issues'].append("Missing signal direction")
            validation['is_valid'] = False
        
        if signal_data.get('strength', 0) < 0.1:
            validation['quality_issues'].append("Very weak signal strength")
            validation['quality_score'] *= 0.8
        
        # Determine quality level
        if validation['quality_score'] >= 0.8:
            validation['quality_level'] = 'EXCELLENT'
        elif validation['quality_score'] >= 0.6:
            validation['quality_level'] = 'GOOD'
        elif validation['quality_score'] >= 0.4:
            validation['quality_level'] = 'FAIR'
        else:
            validation['quality_level'] = 'POOR'
        
        return validation
    
    def _calculate_risk_adjusted_confidence(self, base_confidence: float, 
                                          market_data: Dict[str, Any]) -> float:
        """Calculate risk-adjusted confidence score"""
        try:
            # Get volatility measure
            if 'price_series' in market_data and len(market_data['price_series']) > 1:
                prices = market_data['price_series']
                returns = np.diff(np.log(prices))
                volatility = np.std(returns)
            else:
                high = market_data.get('high', market_data.get('close', 1.0))
                low = market_data.get('low', market_data.get('close', 1.0))
                close = market_data.get('close', 1.0)
                volatility = (high - low) / close if close > 0 else 0.01
            
            # Risk adjustment factor based on volatility
            risk_factor = max(0.5, 1 - (volatility * 50))  # Higher vol = lower confidence
            
            return base_confidence * risk_factor
        except:
            return base_confidence * 0.8  # Conservative adjustment
    
    def _generate_confidence_recommendations(self, confidence_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on confidence assessment"""
        recommendations = []
        
        confidence_level = confidence_assessment['confidence_level']
        overall_confidence = confidence_assessment['overall_confidence']
        
        if confidence_level == 'HIGH':
            recommendations.append("✅ High confidence signal - consider full position sizing")
            recommendations.append("📈 Strong signal quality - monitor for execution")
        elif confidence_level == 'MEDIUM':
            recommendations.append("⚠️ Moderate confidence - use standard position sizing")
            recommendations.append("📊 Monitor signal development for confirmation")
        else:
            recommendations.append("🚨 Low confidence signal - consider reduced position or wait")
            recommendations.append("🔍 Look for additional confirmation before acting")
        
        # Component-specific recommendations
        components = confidence_assessment.get('confidence_components', {})
        
        indicator_consensus = components.get('indicator_consensus', {})
        if indicator_consensus.get('score', 0.5) < 0.4:
            recommendations.append("📉 Low indicator consensus - wait for clearer signals")
        
        market_context = components.get('market_context', {})
        if market_context.get('volatility_regime') == 'HIGH':
            recommendations.append("🌪️ High volatility environment - use tighter risk management")
        
        return recommendations
    
    def _generate_multi_signal_recommendations(self, multi_confidence: Dict[str, Any]) -> List[str]:
        """Generate recommendations for multi-signal scenarios"""
        recommendations = []
        
        consensus_confidence = multi_confidence['consensus_confidence']
        signal_agreement = multi_confidence['signal_agreement']
        conflicting_signals = multi_confidence['conflicting_signals']
        
        if signal_agreement >= 0.8:
            recommendations.append("🎯 Strong signal consensus - high probability setup")
        elif signal_agreement >= 0.6:
            recommendations.append("📊 Good signal agreement - proceed with caution")
        else:
            recommendations.append("⚠️ Mixed signals - wait for clearer consensus")
        
        if conflicting_signals > 2:
            recommendations.append("🔄 Multiple conflicting signals - avoid trading until clarity emerges")
        
        if consensus_confidence >= 0.75:
            recommendations.append("🚀 High consensus confidence - consider scaling up")
        elif consensus_confidence < 0.4:
            recommendations.append("🛑 Low consensus confidence - avoid trading")
        
        return recommendations
    
    def _default_confidence_assessment(self) -> Dict[str, Any]:
        """Return default confidence assessment when calculation fails"""
        return {
            'overall_confidence': 0.5,
            'confidence_level': 'MEDIUM',
            'confidence_components': {},
            'signal_validation': {'is_valid': True, 'quality_level': 'FAIR'},
            'recommendations': ['⚠️ Default confidence assessment - limited analysis available'],
            'risk_adjusted_confidence': 0.4,
            'timestamp': datetime.now().isoformat()
        }

# Global instance for easy access
_signal_confidence_ai = None

def get_signal_confidence_ai() -> SignalConfidenceAI:
    """Get the global SignalConfidenceAI instance"""
    global _signal_confidence_ai
    if _signal_confidence_ai is None:
        _signal_confidence_ai = SignalConfidenceAI()
    return _signal_confidence_ai
