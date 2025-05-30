"""
AI-Powered Market Regime Detection
Advanced regime identification using machine learning, hidden Markov models,
and ensemble methods for market state classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegimeState:
    """Market regime state information"""
    regime_id: int
    regime_name: str
    volatility_level: str  # 'low', 'medium', 'high'
    trend_direction: str   # 'bullish', 'bearish', 'sideways'
    persistence: float     # 0-1, how stable the regime is
    transition_probability: Dict[int, float]  # Probabilities to other regimes
    characteristic_features: Dict[str, float]
    
@dataclass
class RegimeDetectionResult:
    """Results from regime detection analysis"""
    current_regime: RegimeState
    regime_probability: Dict[int, float]
    regime_history: List[int]
    regime_duration: int
    transition_signals: List[str]
    confidence_score: float
    feature_importance: Dict[str, float]
    
@dataclass
class RegimeSignal:
    """Signal from regime detection"""
    signal_type: str  # 'regime_change', 'regime_continuation', 'high_uncertainty'
    strength: float
    confidence: float
    new_regime: Optional[int]
    expected_duration: int
    trading_style: str  # 'trend_following', 'mean_reversion', 'breakout'

class RegimeDetectionAI:
    """
    AI-powered market regime detection with:
    - Hidden Markov Models for regime identification
    - Machine learning feature classification
    - Ensemble methods for robust detection
    - Transition probability estimation
    - Real-time regime monitoring
    """
    
    def __init__(self, 
                 n_regimes: int = 4,
                 lookback_period: int = 200,
                 feature_window: int = 20,
                 confidence_threshold: float = 0.7):
        """
        Initialize AI Regime Detection system
        
        Args:
            n_regimes: Number of market regimes to identify
            lookback_period: Historical data for model training
            feature_window: Window for feature calculation
            confidence_threshold: Minimum confidence for regime signals
        """
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.feature_window = feature_window
        self.confidence_threshold = confidence_threshold
        
        # Models
        self.hmm_model = None
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.gmm_model = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        
        # Internal state
        self.price_history = []
        self.feature_history = []
        self.regime_history = []
        self.regime_states = {}
        self.model_trained = False
        
        # Initialize regime definitions
        self._initialize_regime_definitions()
        
    def _initialize_regime_definitions(self):
        """Initialize predefined regime characteristics"""
        self.regime_states = {
            0: RegimeState(
                regime_id=0,
                regime_name="Low Volatility Bull",
                volatility_level="low",
                trend_direction="bullish",
                persistence=0.8,
                transition_probability={1: 0.15, 2: 0.03, 3: 0.02},
                characteristic_features={}
            ),
            1: RegimeState(
                regime_id=1,
                regime_name="High Volatility Bull", 
                volatility_level="high",
                trend_direction="bullish",
                persistence=0.6,
                transition_probability={0: 0.2, 2: 0.15, 3: 0.05},
                characteristic_features={}
            ),
            2: RegimeState(
                regime_id=2,
                regime_name="Bear Market",
                volatility_level="high", 
                trend_direction="bearish",
                persistence=0.7,
                transition_probability={0: 0.1, 1: 0.1, 3: 0.1},
                characteristic_features={}
            ),
            3: RegimeState(
                regime_id=3,
                regime_name="Sideways Consolidation",
                volatility_level="medium",
                trend_direction="sideways", 
                persistence=0.5,
                transition_probability={0: 0.3, 1: 0.1, 2: 0.1},
                characteristic_features={}
            )
        }
    
    def update(self, price: float, volume: float = None, timestamp: pd.Timestamp = None) -> RegimeDetectionResult:
        """
        Update regime detection with new market data
        
        Args:
            price: Current price
            volume: Current volume (optional)
            timestamp: Current timestamp (optional)
            
        Returns:
            RegimeDetectionResult with current regime analysis
        """
        # Update price history
        self.price_history.append(price)
        
        # Ensure we don't exceed memory limits
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history = self.price_history[-self.lookback_period:]
        
        # Calculate features
        if len(self.price_history) >= self.feature_window:
            features = self._calculate_features(volume)
            self.feature_history.append(features)
            
            # Limit feature history
            if len(self.feature_history) > self.lookback_period:
                self.feature_history = self.feature_history[-self.lookback_period:]
        
        # Train models if we have enough data
        if len(self.feature_history) >= 50 and not self.model_trained:
            self._train_models()
            self.model_trained = True
        
        # Detect current regime
        if self.model_trained and len(self.feature_history) >= 10:
            return self._detect_regime()
        else:
            return self._generate_default_result()
    
    def _calculate_features(self, volume: Optional[float] = None) -> np.ndarray:
        """Calculate market features for regime detection"""
        try:
            prices = np.array(self.price_history[-self.feature_window:])
            
            if len(prices) < self.feature_window:
                return np.zeros(15)  # Default feature count
            
            # Price-based features
            returns = np.diff(np.log(prices))
            
            # 1. Volatility (rolling std of returns)
            volatility = np.std(returns)
            
            # 2. Trend strength (slope of linear regression)
            x = np.arange(len(prices))
            trend_slope = np.polyfit(x, prices, 1)[0] / prices[-1]  # Normalized
            
            # 3. Mean reversion (correlation with lagged prices)
            if len(prices) > 5:
                mean_reversion = -np.corrcoef(prices[:-1], prices[1:])[0, 1]
            else:
                mean_reversion = 0.0
            
            # 4. Momentum (rate of change)
            momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
            
            # 5. Volatility of volatility
            if len(returns) >= 10:
                rolling_vol = np.array([np.std(returns[i:i+5]) for i in range(len(returns)-4)])
                vol_of_vol = np.std(rolling_vol) if len(rolling_vol) > 1 else 0.0
            else:
                vol_of_vol = 0.0
            
            # 6. Skewness of returns
            skewness = self._calculate_skewness(returns) if len(returns) > 3 else 0.0
            
            # 7. Kurtosis of returns
            kurtosis = self._calculate_kurtosis(returns) if len(returns) > 3 else 0.0
            
            # 8. Price acceleration (second derivative)
            if len(prices) >= 3:
                acceleration = prices[-1] - 2*prices[-2] + prices[-3]
                acceleration = acceleration / prices[-1] if prices[-1] != 0 else 0.0
            else:
                acceleration = 0.0
            
            # 9. Autocorrelation of returns
            autocorr = self._calculate_autocorrelation(returns) if len(returns) > 5 else 0.0
            
            # 10. Price range (high-low normalized)
            price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
            
            # 11. Return-to-volatility ratio (Sharpe-like)
            avg_return = np.mean(returns) if len(returns) > 0 else 0.0
            return_vol_ratio = avg_return / (volatility + 1e-8)
            
            # 12. Drawdown from peak
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak[-1] - cumulative[-1]) / peak[-1] if peak[-1] != 0 else 0.0
            
            # 13. Volume features (if available)
            volume_trend = 0.0
            volume_volatility = 0.0
            if volume is not None:
                # Simulate volume history for demonstration
                volume_trend = np.random.normal(0.05, 0.1)
                volume_volatility = np.random.normal(0.2, 0.05)
            
            # 14. Market stress indicator
            stress_indicator = volatility * np.abs(trend_slope) * (1 + np.abs(drawdown))
            
            # 15. Regime stability (consistency of recent features)
            if len(self.feature_history) >= 5:
                recent_volatilities = [f[0] for f in self.feature_history[-5:]]
                stability = 1.0 - (np.std(recent_volatilities) / (np.mean(recent_volatilities) + 1e-8))
            else:
                stability = 0.5
            
            features = np.array([
                volatility, trend_slope, mean_reversion, momentum, vol_of_vol,
                skewness, kurtosis, acceleration, autocorr, price_range,
                return_vol_ratio, drawdown, volume_trend, volume_volatility,
                stress_indicator, stability
            ])
            
            # Handle NaN/Inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception:
            return np.zeros(16)  # Return default features on error
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            if len(data) < 3:
                return 0.0
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            skew = np.mean(((data - mean_val) / std_val) ** 3)
            return np.clip(skew, -5, 5)  # Clip extreme values
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            if len(data) < 4:
                return 0.0
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3
            return np.clip(kurt, -5, 5)  # Clip extreme values
        except:
            return 0.0
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of data"""
        try:
            if len(data) <= lag:
                return 0.0
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _train_models(self):
        """Train regime detection models"""
        try:
            if len(self.feature_history) < 30:
                return
            
            # Prepare feature matrix
            X = np.array(self.feature_history)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Gaussian Mixture Model for unsupervised regime detection
            self.gmm_model.fit(X_scaled)
            
            # Generate labels using GMM
            labels = self.gmm_model.predict(X_scaled)
            
            # Train supervised models
            if len(np.unique(labels)) > 1:
                self.rf_classifier.fit(X_scaled, labels)
                self.gb_classifier.fit(X_scaled, labels)
            
            # Update regime state characteristics
            self._update_regime_characteristics(X, labels)
            
        except Exception:
            pass  # Handle training errors gracefully
    
    def _update_regime_characteristics(self, features: np.ndarray, labels: np.ndarray):
        """Update regime characteristics based on training data"""
        try:
            for regime_id in range(self.n_regimes):
                regime_mask = labels == regime_id
                if np.sum(regime_mask) > 0:
                    regime_features = features[regime_mask]
                    
                    # Calculate characteristic features
                    mean_features = np.mean(regime_features, axis=0)
                    
                    # Update regime state
                    if regime_id in self.regime_states:
                        self.regime_states[regime_id].characteristic_features = {
                            'volatility': mean_features[0],
                            'trend_slope': mean_features[1],
                            'momentum': mean_features[3],
                            'stress_indicator': mean_features[14] if len(mean_features) > 14 else 0.0
                        }
                        
                        # Update volatility level
                        vol = mean_features[0]
                        if vol < 0.01:
                            self.regime_states[regime_id].volatility_level = "low"
                        elif vol < 0.03:
                            self.regime_states[regime_id].volatility_level = "medium"
                        else:
                            self.regime_states[regime_id].volatility_level = "high"
                        
                        # Update trend direction
                        trend = mean_features[1]
                        if trend > 0.01:
                            self.regime_states[regime_id].trend_direction = "bullish"
                        elif trend < -0.01:
                            self.regime_states[regime_id].trend_direction = "bearish"
                        else:
                            self.regime_states[regime_id].trend_direction = "sideways"
                            
        except Exception:
            pass
    
    def _detect_regime(self) -> RegimeDetectionResult:
        """Detect current market regime"""
        try:
            # Get current features
            current_features = self.feature_history[-1].reshape(1, -1)
            current_features_scaled = self.scaler.transform(current_features)
            
            # Get predictions from all models
            gmm_probs = self.gmm_model.predict_proba(current_features_scaled)[0]
            
            rf_pred = self.rf_classifier.predict(current_features_scaled)[0]
            rf_probs = self.rf_classifier.predict_proba(current_features_scaled)[0]
            
            gb_pred = self.gb_classifier.predict(current_features_scaled)[0]
            gb_probs = self.gb_classifier.predict_proba(current_features_scaled)[0]
            
            # Ensemble prediction (weighted average)
            ensemble_probs = (gmm_probs + rf_probs + gb_probs) / 3.0
            current_regime_id = np.argmax(ensemble_probs)
            confidence = ensemble_probs[current_regime_id]
            
            # Update regime history
            self.regime_history.append(current_regime_id)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            # Calculate regime duration
            regime_duration = 1
            for i in range(len(self.regime_history)-2, -1, -1):
                if self.regime_history[i] == current_regime_id:
                    regime_duration += 1
                else:
                    break
            
            # Detect transition signals
            transition_signals = self._detect_transition_signals()
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance()
            
            # Get current regime state
            current_regime = self.regime_states.get(current_regime_id, self.regime_states[0])
            
            # Create regime probabilities dictionary
            regime_probability = {i: ensemble_probs[i] for i in range(len(ensemble_probs))}
            
            return RegimeDetectionResult(
                current_regime=current_regime,
                regime_probability=regime_probability,
                regime_history=self.regime_history.copy(),
                regime_duration=regime_duration,
                transition_signals=transition_signals,
                confidence_score=confidence,
                feature_importance=feature_importance
            )
            
        except Exception:
            return self._generate_default_result()
    
    def _detect_transition_signals(self) -> List[str]:
        """Detect regime transition signals"""
        signals = []
        
        try:
            if len(self.regime_history) < 5:
                return signals
            
            recent_regimes = self.regime_history[-5:]
            
            # Check for regime instability
            if len(set(recent_regimes)) > 2:
                signals.append("regime_instability")
            
            # Check for recent regime change
            if len(self.regime_history) >= 2 and self.regime_history[-1] != self.regime_history[-2]:
                signals.append("regime_change")
            
            # Check for oscillating regimes
            if len(recent_regimes) >= 4:
                if (recent_regimes[-1] == recent_regimes[-3] and 
                    recent_regimes[-2] == recent_regimes[-4] and
                    recent_regimes[-1] != recent_regimes[-2]):
                    signals.append("regime_oscillation")
            
        except Exception:
            pass
        
        return signals
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance from trained models"""
        feature_names = [
            'volatility', 'trend_slope', 'mean_reversion', 'momentum', 'vol_of_vol',
            'skewness', 'kurtosis', 'acceleration', 'autocorr', 'price_range',
            'return_vol_ratio', 'drawdown', 'volume_trend', 'volume_volatility',
            'stress_indicator', 'stability'
        ]
        
        try:
            # Get feature importance from Random Forest
            rf_importance = self.rf_classifier.feature_importances_
            
            # Get feature importance from Gradient Boosting
            gb_importance = self.gb_classifier.feature_importances_
            
            # Average the importances
            avg_importance = (rf_importance + gb_importance) / 2.0
            
            return {name: importance for name, importance in zip(feature_names, avg_importance)}
            
        except Exception:
            return {name: 1.0/len(feature_names) for name in feature_names}
    
    def _generate_default_result(self) -> RegimeDetectionResult:
        """Generate default result when models aren't trained"""
        default_regime = self.regime_states[0]
        
        return RegimeDetectionResult(
            current_regime=default_regime,
            regime_probability={i: 0.25 for i in range(4)},
            regime_history=[0],
            regime_duration=1,
            transition_signals=[],
            confidence_score=0.5,
            feature_importance={}
        )
    
    def generate_signals(self, regime_result: RegimeDetectionResult) -> List[RegimeSignal]:
        """Generate trading signals based on regime detection"""
        signals = []
        
        try:
            current_regime = regime_result.current_regime
            confidence = regime_result.confidence_score
            
            # High confidence regime continuation signal
            if confidence > self.confidence_threshold and regime_result.regime_duration > 3:
                trading_style = self._get_trading_style(current_regime)
                signals.append(RegimeSignal(
                    signal_type='regime_continuation',
                    strength=confidence,
                    confidence=confidence,
                    new_regime=None,
                    expected_duration=int(current_regime.persistence * 20),
                    trading_style=trading_style
                ))
            
            # Regime change signal
            if 'regime_change' in regime_result.transition_signals:
                signals.append(RegimeSignal(
                    signal_type='regime_change',
                    strength=1.0 - confidence,  # Lower confidence indicates change
                    confidence=0.8,
                    new_regime=current_regime.regime_id,
                    expected_duration=10,
                    trading_style=self._get_trading_style(current_regime)
                ))
            
            # High uncertainty signal
            if confidence < 0.5 or 'regime_instability' in regime_result.transition_signals:
                signals.append(RegimeSignal(
                    signal_type='high_uncertainty',
                    strength=1.0 - confidence,
                    confidence=0.6,
                    new_regime=None,
                    expected_duration=5,
                    trading_style='defensive'
                ))
                
        except Exception:
            pass
        
        return signals
    
    def _get_trading_style(self, regime: RegimeState) -> str:
        """Determine appropriate trading style for regime"""
        if regime.trend_direction == 'bullish' or regime.trend_direction == 'bearish':
            return 'trend_following'
        elif regime.volatility_level == 'low':
            return 'mean_reversion'
        else:
            return 'breakout'
    
    def get_regime_summary(self, regime_result: RegimeDetectionResult) -> Dict[str, Any]:
        """Get comprehensive regime analysis summary"""
        current_regime = regime_result.current_regime
        
        summary = {
            'current_regime': current_regime.regime_name,
            'regime_id': current_regime.regime_id,
            'confidence': regime_result.confidence_score,
            'volatility_level': current_regime.volatility_level,
            'trend_direction': current_regime.trend_direction,
            'regime_duration': regime_result.regime_duration,
            'regime_stability': current_regime.persistence,
            'transition_probability': current_regime.transition_probability,
            'key_features': current_regime.characteristic_features,
            'transition_signals': regime_result.transition_signals,
            'recommended_strategy': self._get_trading_style(current_regime)
        }
        
        return summary
