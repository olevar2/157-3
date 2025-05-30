"""
AI-Powered Signal Confidence Assessment
Advanced confidence scoring for trading signals using ensemble methods,
Bayesian inference, and meta-learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SignalMetrics:
    """Signal performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float
    max_drawdown: float
    
@dataclass
class ConfidenceFactors:
    """Factors contributing to signal confidence"""
    model_agreement: float      # Agreement between different models
    historical_performance: float  # Past performance of similar signals
    market_regime_fit: float    # How well signal fits current regime
    data_quality: float         # Quality and completeness of input data
    volatility_adjustment: float  # Adjustment for market volatility
    time_decay: float          # Adjustment for signal age
    ensemble_variance: float    # Variance in ensemble predictions
    
@dataclass
class SignalConfidenceResult:
    """Results from signal confidence assessment"""
    confidence_score: float     # Overall confidence (0-1)
    confidence_level: str       # 'very_low', 'low', 'medium', 'high', 'very_high'
    signal_strength: float      # Adjusted signal strength
    confidence_factors: ConfidenceFactors
    model_predictions: Dict[str, float]  # Individual model predictions
    risk_adjusted_confidence: float  # Confidence adjusted for risk
    expected_performance: Dict[str, float]  # Expected metrics
    calibrated_probability: float  # Calibrated success probability
    
@dataclass
class TradingSignal:
    """Enhanced trading signal with confidence"""
    signal_id: str
    signal_type: str           # 'buy', 'sell', 'hold'
    original_strength: float   # Original signal strength
    confidence_score: float    # AI confidence assessment
    final_strength: float      # Confidence-adjusted strength
    timestamp: pd.Timestamp
    source_indicators: List[str]  # Source indicators
    metadata: Dict[str, Any]   # Additional signal metadata

class SignalConfidenceAI:
    """
    AI-powered signal confidence assessment with:
    - Ensemble model confidence scoring
    - Bayesian confidence updating
    - Historical performance tracking
    - Market regime adaptation
    - Risk-adjusted confidence metrics
    - Meta-learning for signal quality
    """
    
    def __init__(self, 
                 confidence_window: int = 100,
                 min_samples_for_training: int = 50,
                 calibration_window: int = 200,
                 confidence_threshold: float = 0.7):
        """
        Initialize Signal Confidence AI system
        
        Args:
            confidence_window: Window for confidence calculation
            min_samples_for_training: Minimum samples needed for training
            calibration_window: Window for probability calibration
            confidence_threshold: Threshold for high confidence signals
        """
        self.confidence_window = confidence_window
        self.min_samples_for_training = min_samples_for_training
        self.calibration_window = calibration_window
        self.confidence_threshold = confidence_threshold
        
        # Ensemble models for confidence prediction
        self.rf_confidence = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_confidence = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lr_confidence = LogisticRegression(random_state=42, max_iter=1000)
        
        # Calibrated models for probability estimation
        self.calibrated_rf = CalibratedClassifierCV(self.rf_confidence, method='sigmoid', cv=3)
        self.calibrated_gb = CalibratedClassifierCV(self.gb_confidence, method='sigmoid', cv=3)
        
        self.scaler = StandardScaler()
        
        # Historical tracking
        self.signal_history = []
        self.performance_history = []
        self.confidence_features_history = []
        
        # Model state
        self.models_trained = False
        self.confidence_baseline = 0.5
        
        # Performance tracking by signal type
        self.signal_performance_tracker = {}
        
    def assess_signal_confidence(self, 
                                signal: Dict[str, Any],
                                market_data: Dict[str, float],
                                indicator_outputs: Dict[str, Any]) -> SignalConfidenceResult:
        """
        Assess confidence for a trading signal
        
        Args:
            signal: Trading signal information
            market_data: Current market data
            indicator_outputs: Outputs from various indicators
            
        Returns:
            SignalConfidenceResult with comprehensive confidence assessment
        """
        # Extract confidence features
        confidence_features = self._extract_confidence_features(signal, market_data, indicator_outputs)
        
        # Calculate confidence factors
        confidence_factors = self._calculate_confidence_factors(signal, market_data, indicator_outputs)
        
        # Get model predictions if trained
        model_predictions = {}
        if self.models_trained:
            model_predictions = self._get_model_predictions(confidence_features)
        
        # Calculate overall confidence score
        confidence_score = self._calculate_overall_confidence(confidence_factors, model_predictions)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(confidence_score)
        
        # Calculate risk-adjusted confidence
        risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(
            confidence_score, market_data, signal
        )
        
        # Calculate calibrated probability
        calibrated_probability = self._calculate_calibrated_probability(confidence_features)
        
        # Predict expected performance
        expected_performance = self._predict_expected_performance(confidence_score, signal)
        
        # Adjust signal strength based on confidence
        original_strength = signal.get('strength', 0.5)
        final_strength = self._adjust_signal_strength(original_strength, confidence_score)
        
        result = SignalConfidenceResult(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            signal_strength=final_strength,
            confidence_factors=confidence_factors,
            model_predictions=model_predictions,
            risk_adjusted_confidence=risk_adjusted_confidence,
            expected_performance=expected_performance,
            calibrated_probability=calibrated_probability
        )
        
        # Store for learning
        self._store_signal_assessment(signal, result, confidence_features)
        
        return result
    
    def _extract_confidence_features(self, 
                                   signal: Dict[str, Any],
                                   market_data: Dict[str, float],
                                   indicator_outputs: Dict[str, Any]) -> np.ndarray:
        """Extract features for confidence assessment"""
        try:
            features = []
            
            # Signal-based features
            signal_strength = signal.get('strength', 0.5)
            signal_type = signal.get('type', 'neutral')
            signal_age = signal.get('age', 0)  # Time since signal generation
            
            features.extend([
                signal_strength,
                1.0 if signal_type == 'buy' else -1.0 if signal_type == 'sell' else 0.0,
                min(signal_age / 10.0, 1.0),  # Normalized age
            ])
            
            # Market condition features
            volatility = market_data.get('volatility', 0.02)
            volume = market_data.get('volume', 1000000)
            price_change = market_data.get('price_change', 0.0)
            
            features.extend([
                min(volatility * 100, 10),  # Volatility (%)
                np.log(volume / 1000000),   # Log-normalized volume
                np.clip(price_change * 100, -10, 10),  # Price change (%)
            ])
            
            # Indicator consensus features
            indicator_agreement = self._calculate_indicator_agreement(indicator_outputs)
            indicator_strength = self._calculate_indicator_strength(indicator_outputs)
            indicator_diversity = self._calculate_indicator_diversity(indicator_outputs)
            
            features.extend([
                indicator_agreement,
                indicator_strength,
                indicator_diversity,
            ])
            
            # Market microstructure features
            bid_ask_spread = market_data.get('bid_ask_spread', 0.001)
            market_depth = market_data.get('market_depth', 1.0)
            order_flow = market_data.get('order_flow', 0.0)
            
            features.extend([
                min(bid_ask_spread * 1000, 10),  # Spread in basis points
                np.clip(market_depth, 0, 5),
                np.clip(order_flow, -1, 1),
            ])
            
            # Historical context features
            recent_signal_performance = self._get_recent_signal_performance(signal_type)
            market_regime_consistency = self._assess_market_regime_consistency(market_data)
            signal_frequency = self._calculate_signal_frequency(signal_type)
            
            features.extend([
                recent_signal_performance,
                market_regime_consistency,
                min(signal_frequency / 10.0, 1.0),
            ])
            
            # Technical features
            trend_alignment = self._assess_trend_alignment(indicator_outputs)
            momentum_consistency = self._assess_momentum_consistency(indicator_outputs)
            support_resistance = self._assess_support_resistance_proximity(market_data)
            
            features.extend([
                trend_alignment,
                momentum_consistency,
                support_resistance,
            ])
            
            # Risk features
            drawdown_risk = market_data.get('current_drawdown', 0.0)
            correlation_risk = market_data.get('correlation_risk', 0.3)
            liquidity_risk = 1.0 - min(market_data.get('liquidity_score', 0.8), 1.0)
            
            features.extend([
                min(drawdown_risk, 0.5),
                correlation_risk,
                liquidity_risk,
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception:
            return np.zeros(21)  # Default feature vector
    
    def _calculate_indicator_agreement(self, indicator_outputs: Dict[str, Any]) -> float:
        """Calculate agreement between different indicators"""
        try:
            signals = []
            for indicator_name, output in indicator_outputs.items():
                if isinstance(output, dict) and 'signal' in output:
                    signal_value = output['signal']
                    if isinstance(signal_value, (int, float)):
                        signals.append(signal_value)
                    elif isinstance(signal_value, str):
                        # Convert string signals to numeric
                        if signal_value.lower() in ['buy', 'bullish']:
                            signals.append(1.0)
                        elif signal_value.lower() in ['sell', 'bearish']:
                            signals.append(-1.0)
                        else:
                            signals.append(0.0)
            
            if len(signals) < 2:
                return 0.5
            
            # Calculate agreement as 1 - normalized standard deviation
            agreement = 1.0 - (np.std(signals) / 2.0)  # Assuming signals are in [-1, 1]
            return np.clip(agreement, 0, 1)
            
        except Exception:
            return 0.5
    
    def _calculate_indicator_strength(self, indicator_outputs: Dict[str, Any]) -> float:
        """Calculate average strength of indicator signals"""
        try:
            strengths = []
            for indicator_name, output in indicator_outputs.items():
                if isinstance(output, dict) and 'strength' in output:
                    strength = output['strength']
                    if isinstance(strength, (int, float)):
                        strengths.append(abs(strength))
            
            return np.mean(strengths) if strengths else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_indicator_diversity(self, indicator_outputs: Dict[str, Any]) -> float:
        """Calculate diversity of indicator types"""
        try:
            indicator_types = set()
            for indicator_name, output in indicator_outputs.items():
                # Categorize indicators by type
                if 'trend' in indicator_name.lower():
                    indicator_types.add('trend')
                elif 'momentum' in indicator_name.lower():
                    indicator_types.add('momentum')
                elif 'volatility' in indicator_name.lower():
                    indicator_types.add('volatility')
                elif 'volume' in indicator_name.lower():
                    indicator_types.add('volume')
                else:
                    indicator_types.add('other')
            
            # Diversity score based on number of different types
            diversity = len(indicator_types) / 5.0  # Max 5 types
            return min(diversity, 1.0)
            
        except Exception:
            return 0.5
    
    def _get_recent_signal_performance(self, signal_type: str) -> float:
        """Get recent performance for this signal type"""
        try:
            if signal_type not in self.signal_performance_tracker:
                return 0.5
            
            recent_performances = self.signal_performance_tracker[signal_type][-10:]  # Last 10 signals
            if not recent_performances:
                return 0.5
            
            return np.mean(recent_performances)
            
        except Exception:
            return 0.5
    
    def _assess_market_regime_consistency(self, market_data: Dict[str, float]) -> float:
        """Assess consistency with current market regime"""
        try:
            # Simple regime assessment based on volatility and trend
            volatility = market_data.get('volatility', 0.02)
            trend = market_data.get('trend', 0.0)
            
            # High consistency in stable, trending markets
            # Low consistency in volatile, choppy markets
            vol_factor = 1.0 - min(volatility * 50, 1.0)  # Lower vol = higher consistency
            trend_factor = min(abs(trend) * 2, 1.0)       # Stronger trend = higher consistency
            
            consistency = (vol_factor + trend_factor) / 2.0
            return consistency
            
        except Exception:
            return 0.5
    
    def _calculate_signal_frequency(self, signal_type: str) -> float:
        """Calculate recent frequency of this signal type"""
        try:
            recent_signals = [s for s in self.signal_history[-50:] if s.get('type') == signal_type]
            return len(recent_signals)
            
        except Exception:
            return 5.0
    
    def _assess_trend_alignment(self, indicator_outputs: Dict[str, Any]) -> float:
        """Assess alignment with overall trend"""
        try:
            trend_signals = []
            for indicator_name, output in indicator_outputs.items():
                if 'trend' in indicator_name.lower() and isinstance(output, dict):
                    signal = output.get('signal', 0)
                    if isinstance(signal, (int, float)):
                        trend_signals.append(signal)
            
            if not trend_signals:
                return 0.5
            
            # High alignment if all trend signals agree
            return 1.0 - (np.std(trend_signals) / 2.0)
            
        except Exception:
            return 0.5
    
    def _assess_momentum_consistency(self, indicator_outputs: Dict[str, Any]) -> float:
        """Assess momentum consistency"""
        try:
            momentum_signals = []
            for indicator_name, output in indicator_outputs.items():
                if 'momentum' in indicator_name.lower() and isinstance(output, dict):
                    signal = output.get('signal', 0)
                    if isinstance(signal, (int, float)):
                        momentum_signals.append(signal)
            
            if not momentum_signals:
                return 0.5
            
            return 1.0 - (np.std(momentum_signals) / 2.0)
            
        except Exception:
            return 0.5
    
    def _assess_support_resistance_proximity(self, market_data: Dict[str, float]) -> float:
        """Assess proximity to support/resistance levels"""
        try:
            current_price = market_data.get('price', 100)
            support_level = market_data.get('support', current_price * 0.98)
            resistance_level = market_data.get('resistance', current_price * 1.02)
            
            # Distance to nearest level
            support_distance = abs(current_price - support_level) / current_price
            resistance_distance = abs(current_price - resistance_level) / current_price
            
            min_distance = min(support_distance, resistance_distance)
            
            # Closer to levels = higher significance
            proximity_score = 1.0 - min(min_distance * 50, 1.0)
            return proximity_score
            
        except Exception:
            return 0.5
    
    def _calculate_confidence_factors(self, 
                                    signal: Dict[str, Any],
                                    market_data: Dict[str, float],
                                    indicator_outputs: Dict[str, Any]) -> ConfidenceFactors:
        """Calculate individual confidence factors"""
        try:
            # Model agreement (if models are trained)
            model_agreement = 0.5
            if self.models_trained and len(self.confidence_features_history) > 10:
                recent_predictions = []
                for features in self.confidence_features_history[-5:]:
                    predictions = self._get_model_predictions(features)
                    recent_predictions.append(list(predictions.values()))
                
                if recent_predictions:
                    model_agreement = 1.0 - np.std(recent_predictions)
            
            # Historical performance
            signal_type = signal.get('type', 'neutral')
            historical_performance = self._get_recent_signal_performance(signal_type)
            
            # Market regime fit
            market_regime_fit = self._assess_market_regime_consistency(market_data)
            
            # Data quality (based on completeness and freshness)
            data_quality = self._assess_data_quality(market_data, indicator_outputs)
            
            # Volatility adjustment
            volatility = market_data.get('volatility', 0.02)
            volatility_adjustment = 1.0 - min(volatility * 30, 0.8)  # Higher vol = lower confidence
            
            # Time decay
            signal_age = signal.get('age', 0)
            time_decay = np.exp(-signal_age / 10.0)  # Exponential decay
            
            # Ensemble variance (uncertainty)
            ensemble_variance = 0.1  # Default low variance
            if self.models_trained:
                ensemble_variance = model_agreement  # Use agreement as inverse of variance
            
            return ConfidenceFactors(
                model_agreement=model_agreement,
                historical_performance=historical_performance,
                market_regime_fit=market_regime_fit,
                data_quality=data_quality,
                volatility_adjustment=volatility_adjustment,
                time_decay=time_decay,
                ensemble_variance=ensemble_variance
            )
            
        except Exception:
            return ConfidenceFactors(0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.1)
    
    def _assess_data_quality(self, 
                           market_data: Dict[str, float],
                           indicator_outputs: Dict[str, Any]) -> float:
        """Assess quality of input data"""
        try:
            quality_score = 0.0
            total_checks = 0
            
            # Check market data completeness
            expected_market_fields = ['price', 'volume', 'volatility']
            for field in expected_market_fields:
                total_checks += 1
                if field in market_data and market_data[field] is not None:
                    quality_score += 1.0
            
            # Check indicator output quality
            valid_indicators = 0
            total_indicators = len(indicator_outputs)
            
            for indicator_name, output in indicator_outputs.items():
                if isinstance(output, dict) and ('signal' in output or 'value' in output):
                    valid_indicators += 1
            
            if total_indicators > 0:
                total_checks += 1
                quality_score += valid_indicators / total_indicators
            
            return quality_score / total_checks if total_checks > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _get_model_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from trained models"""
        try:
            if not self.models_trained:
                return {'ensemble': 0.5}
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            predictions = {}
            
            # Random Forest prediction
            try:
                rf_prob = self.calibrated_rf.predict_proba(features_scaled)[0][1]  # Probability of success
                predictions['random_forest'] = rf_prob
            except:
                predictions['random_forest'] = 0.5
            
            # Gradient Boosting prediction
            try:
                gb_prob = self.calibrated_gb.predict_proba(features_scaled)[0][1]
                predictions['gradient_boosting'] = gb_prob
            except:
                predictions['gradient_boosting'] = 0.5
            
            # Logistic Regression prediction
            try:
                lr_prob = self.lr_confidence.predict_proba(features_scaled)[0][1]
                predictions['logistic_regression'] = lr_prob
            except:
                predictions['logistic_regression'] = 0.5
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            predictions['ensemble'] = ensemble_pred
            
            return predictions
            
        except Exception:
            return {'ensemble': 0.5}
    
    def _calculate_overall_confidence(self, 
                                    confidence_factors: ConfidenceFactors,
                                    model_predictions: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        try:
            # Weight different factors
            factor_weights = {
                'model_agreement': 0.15,
                'historical_performance': 0.20,
                'market_regime_fit': 0.15,
                'data_quality': 0.10,
                'volatility_adjustment': 0.15,
                'time_decay': 0.10,
                'ensemble_variance': 0.15
            }
            
            # Calculate weighted factor score
            factor_score = (
                confidence_factors.model_agreement * factor_weights['model_agreement'] +
                confidence_factors.historical_performance * factor_weights['historical_performance'] +
                confidence_factors.market_regime_fit * factor_weights['market_regime_fit'] +
                confidence_factors.data_quality * factor_weights['data_quality'] +
                confidence_factors.volatility_adjustment * factor_weights['volatility_adjustment'] +
                confidence_factors.time_decay * factor_weights['time_decay'] +
                confidence_factors.ensemble_variance * factor_weights['ensemble_variance']
            )
            
            # Combine with model predictions if available
            if model_predictions and 'ensemble' in model_predictions:
                model_score = model_predictions['ensemble']
                # Weighted combination: 60% factors, 40% models
                overall_confidence = 0.6 * factor_score + 0.4 * model_score
            else:
                overall_confidence = factor_score
            
            return np.clip(overall_confidence, 0, 1)
            
        except Exception:
            return 0.5
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level category"""
        if confidence_score >= 0.8:
            return 'very_high'
        elif confidence_score >= 0.65:
            return 'high'
        elif confidence_score >= 0.5:
            return 'medium'
        elif confidence_score >= 0.35:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_risk_adjusted_confidence(self, 
                                          confidence_score: float,
                                          market_data: Dict[str, float],
                                          signal: Dict[str, Any]) -> float:
        """Calculate risk-adjusted confidence"""
        try:
            # Risk factors
            volatility = market_data.get('volatility', 0.02)
            drawdown = market_data.get('current_drawdown', 0.0)
            liquidity = market_data.get('liquidity_score', 0.8)
            
            # Risk adjustment factor (lower risk = higher confidence)
            vol_penalty = min(volatility * 20, 0.3)
            drawdown_penalty = min(drawdown * 2, 0.2)
            liquidity_bonus = (liquidity - 0.5) * 0.1
            
            risk_adjustment = 1.0 - vol_penalty - drawdown_penalty + liquidity_bonus
            risk_adjustment = np.clip(risk_adjustment, 0.5, 1.2)
            
            return confidence_score * risk_adjustment
            
        except Exception:
            return confidence_score * 0.9  # Default slight reduction
    
    def _calculate_calibrated_probability(self, features: np.ndarray) -> float:
        """Calculate calibrated probability of signal success"""
        try:
            if not self.models_trained:
                return 0.5
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Use calibrated models for better probability estimates
            rf_prob = self.calibrated_rf.predict_proba(features_scaled)[0][1]
            gb_prob = self.calibrated_gb.predict_proba(features_scaled)[0][1]
            
            # Average calibrated probabilities
            calibrated_prob = (rf_prob + gb_prob) / 2.0
            
            return np.clip(calibrated_prob, 0.1, 0.9)
            
        except Exception:
            return 0.5
    
    def _predict_expected_performance(self, 
                                    confidence_score: float,
                                    signal: Dict[str, Any]) -> Dict[str, float]:
        """Predict expected performance metrics"""
        try:
            # Base performance estimates
            base_win_rate = 0.55
            base_avg_return = 0.01
            base_sharpe = 0.5
            base_max_dd = 0.05
            
            # Adjust based on confidence
            confidence_multiplier = confidence_score / 0.5  # Normalized to 0.5 baseline
            
            expected_performance = {
                'win_rate': min(base_win_rate * confidence_multiplier, 0.8),
                'avg_return': base_avg_return * confidence_multiplier,
                'sharpe_ratio': base_sharpe * confidence_multiplier,
                'max_drawdown': base_max_dd / confidence_multiplier,
                'expected_profit': signal.get('strength', 0.5) * confidence_score * 0.02
            }
            
            return expected_performance
            
        except Exception:
            return {'win_rate': 0.55, 'avg_return': 0.01, 'sharpe_ratio': 0.5, 'max_drawdown': 0.05}
    
    def _adjust_signal_strength(self, original_strength: float, confidence_score: float) -> float:
        """Adjust signal strength based on confidence"""
        try:
            # Non-linear confidence adjustment
            if confidence_score > 0.7:
                # High confidence: amplify strong signals
                amplification = 1.0 + (confidence_score - 0.7) * 0.5
                adjusted_strength = original_strength * amplification
            elif confidence_score < 0.4:
                # Low confidence: dampen signals
                dampening = confidence_score / 0.4
                adjusted_strength = original_strength * dampening
            else:
                # Medium confidence: linear adjustment
                adjusted_strength = original_strength * (0.5 + confidence_score)
            
            return np.clip(adjusted_strength, 0, 1)
            
        except Exception:
            return original_strength * confidence_score
    
    def _store_signal_assessment(self, 
                               signal: Dict[str, Any],
                               result: SignalConfidenceResult,
                               features: np.ndarray):
        """Store signal assessment for learning"""
        try:
            # Store signal
            signal_record = {
                'signal': signal,
                'confidence_result': result,
                'timestamp': pd.Timestamp.now()
            }
            self.signal_history.append(signal_record)
            
            # Store features
            self.confidence_features_history.append(features)
            
            # Maintain history size
            if len(self.signal_history) > self.confidence_window * 2:
                self.signal_history = self.signal_history[-self.confidence_window:]
                self.confidence_features_history = self.confidence_features_history[-self.confidence_window:]
            
        except Exception:
            pass
    
    def update_signal_performance(self, 
                                signal_id: str, 
                                actual_performance: Dict[str, float]):
        """Update performance tracking for a signal"""
        try:
            # Find the signal in history
            for record in self.signal_history:
                if record['signal'].get('id') == signal_id:
                    record['actual_performance'] = actual_performance
                    
                    # Update performance tracker
                    signal_type = record['signal'].get('type', 'unknown')
                    if signal_type not in self.signal_performance_tracker:
                        self.signal_performance_tracker[signal_type] = []
                    
                    # Store performance metric (could be return, accuracy, etc.)
                    performance_score = actual_performance.get('return', 0.0)
                    self.signal_performance_tracker[signal_type].append(performance_score)
                    
                    # Maintain tracker size
                    if len(self.signal_performance_tracker[signal_type]) > 50:
                        self.signal_performance_tracker[signal_type] = self.signal_performance_tracker[signal_type][-50:]
                    
                    break
            
            # Retrain models if enough data
            if len(self.signal_history) >= self.min_samples_for_training:
                self._retrain_confidence_models()
                
        except Exception:
            pass
    
    def _retrain_confidence_models(self):
        """Retrain confidence models with updated performance data"""
        try:
            # Prepare training data
            X = []
            y = []
            
            for record in self.signal_history:
                if 'actual_performance' in record:
                    features = self._extract_confidence_features(
                        record['signal'], 
                        {}, 
                        {}
                    )
                    X.append(features)
                    
                    # Label: 1 if performance was good, 0 otherwise
                    performance = record['actual_performance'].get('return', 0.0)
                    label = 1 if performance > 0 else 0
                    y.append(label)
            
            if len(X) >= self.min_samples_for_training:
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train models
                self.rf_confidence.fit(X_scaled, y)
                self.gb_confidence.fit(X_scaled, y)
                self.lr_confidence.fit(X_scaled, y)
                
                # Recalibrate
                self.calibrated_rf.fit(X_scaled, y)
                self.calibrated_gb.fit(X_scaled, y)
                
                self.models_trained = True
                
        except Exception:
            pass
    
    def create_enhanced_signal(self, 
                             signal: Dict[str, Any],
                             confidence_result: SignalConfidenceResult) -> TradingSignal:
        """Create enhanced trading signal with confidence"""
        return TradingSignal(
            signal_id=signal.get('id', f"signal_{pd.Timestamp.now().timestamp()}"),
            signal_type=signal.get('type', 'neutral'),
            original_strength=signal.get('strength', 0.5),
            confidence_score=confidence_result.confidence_score,
            final_strength=confidence_result.signal_strength,
            timestamp=pd.Timestamp.now(),
            source_indicators=signal.get('source_indicators', []),
            metadata={
                'confidence_level': confidence_result.confidence_level,
                'risk_adjusted_confidence': confidence_result.risk_adjusted_confidence,
                'calibrated_probability': confidence_result.calibrated_probability,
                'expected_performance': confidence_result.expected_performance
            }
        )
    
    def get_confidence_summary(self, confidence_result: SignalConfidenceResult) -> Dict[str, Any]:
        """Get confidence assessment summary"""
        summary = {
            'confidence_score': confidence_result.confidence_score,
            'confidence_level': confidence_result.confidence_level,
            'risk_adjusted_confidence': confidence_result.risk_adjusted_confidence,
            'calibrated_probability': confidence_result.calibrated_probability,
            'key_factors': {
                'historical_performance': confidence_result.confidence_factors.historical_performance,
                'market_regime_fit': confidence_result.confidence_factors.market_regime_fit,
                'data_quality': confidence_result.confidence_factors.data_quality
            },
            'expected_metrics': confidence_result.expected_performance,
            'model_consensus': confidence_result.model_predictions.get('ensemble', 0.5),
            'recommendation': self._get_recommendation(confidence_result.confidence_score)
        }
        
        return summary
    
    def _get_recommendation(self, confidence_score: float) -> str:
        """Get trading recommendation based on confidence"""
        if confidence_score >= 0.8:
            return "HIGH CONFIDENCE: Consider increased position size"
        elif confidence_score >= 0.65:
            return "GOOD CONFIDENCE: Normal position size recommended"
        elif confidence_score >= 0.5:
            return "MODERATE CONFIDENCE: Reduced position size suggested"
        elif confidence_score >= 0.35:
            return "LOW CONFIDENCE: Minimal position or paper trade"
        else:
            return "VERY LOW CONFIDENCE: Avoid this signal"
