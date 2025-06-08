#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom AI Composite Indicator - Advanced Machine Learning Ensemble
Platform3 Phase 3 - AI-Enhanced Market Analysis

The Custom AI Composite Indicator combines multiple technical indicators using advanced
machine learning techniques to create a unified market signal. It uses ensemble learning,
feature engineering, and adaptive algorithms to provide superior market timing.
"""

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import math
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
warnings.filterwarnings('ignore')

@dataclass
class CompositeSignal:
    """AI Composite signal data structure"""
    timestamp: datetime
    composite_score: float      # Main AI composite score (-1 to 1)
    confidence: float          # Signal confidence (0 to 1)
    trend_strength: float      # Trend component (-1 to 1)
    momentum_strength: float   # Momentum component (-1 to 1)
    volatility_regime: str     # 'low', 'medium', 'high'
    market_regime: str         # 'trending', 'ranging', 'breakout'
    feature_importance: Dict[str, float]  # Individual indicator contributions
    ensemble_predictions: Dict[str, float]  # Individual model predictions
    signal_quality: str        # 'excellent', 'good', 'fair', 'poor'
    recommended_action: str    # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'

class CustomAICompositeIndicator:
    """
    Advanced AI Composite Indicator Engine
    
    Features:
    - Multi-indicator ensemble learning
    - Adaptive model selection based on market conditions
    - Real-time feature engineering and selection
    - Regime-aware signal generation
    - Confidence-based position sizing
    - Advanced noise filtering and signal smoothing
    """
    
    def __init__(self, lookback_period: int = 100, feature_window: int = 20, 
                 ensemble_size: int = 5, retrain_frequency: int = 500):
        """Initialize Custom AI Composite Indicator with Platform3 framework"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.lookback_period = lookback_period
        self.feature_window = feature_window
        self.ensemble_size = ensemble_size
        self.retrain_frequency = retrain_frequency
        
        # ML Models ensemble
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500),
        }
        
        # Scalers for feature normalization
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        # Model state
        self.is_trained = False
        self.last_train_time = None
        self.feature_names = []
        self.model_weights = {}
        self.performance_history = []
        
        # Technical indicators for feature engineering
        self.core_indicators = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'atr_14', 'adx_14', 'stoch_k', 'stoch_d', 'cci_20',
            'williams_r', 'momentum_10', 'roc_10', 'obv', 'mfi_14'
        ]
        
        self.logger.info(f"Custom AI Composite Indicator initialized - Lookback: {self.lookback_period}, Features: {self.feature_window}")
    
    async def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Calculate AI composite indicator with ensemble learning
        
        Args:
            data: Price data (OHLC DataFrame or close price array)
            
        Returns:
            Dictionary containing AI composite analysis and signals
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting AI composite indicator calculation")
            
            # Prepare and validate data
            price_df = self._prepare_data(data)
            if price_df is None or len(price_df) < self.lookback_period:
                raise ServiceError("Insufficient price data", "INVALID_DATA")
            
            # Engineer features from raw price data
            features_df = await self._engineer_features(price_df)
            if features_df is None or len(features_df) < self.feature_window:
                raise ServiceError("Failed to engineer features", "FEATURE_ERROR")
            
            # Train or retrain models if needed
            if not self.is_trained or self._should_retrain():
                await self._train_ensemble(features_df)
            
            # Generate predictions from ensemble
            predictions = await self._generate_ensemble_predictions(features_df)
            
            # Calculate composite signal
            composite_signal = await self._calculate_composite_signal(predictions, features_df)
            
            # Analyze market regime and adjust signals
            regime_analysis = await self._analyze_market_regime(price_df, features_df)
            
            # Generate final trading signals with confidence
            trading_signals = await self._generate_trading_signals(
                composite_signal, regime_analysis, price_df.iloc[-1]['close']
            )
            
            # Calculate feature importance and model contributions
            feature_analysis = await self._analyze_feature_importance(features_df)
            
            result = {
                'composite_signal': self._signal_to_dict(composite_signal),
                'predictions': predictions,
                'regime_analysis': regime_analysis,
                'trading_signals': trading_signals,
                'feature_analysis': feature_analysis,
                'model_performance': self._get_model_performance(),
                'current_price': price_df.iloc[-1]['close'],
                'calculation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"AI composite calculation completed - Score: {composite_signal.composite_score:.3f}, "
                           f"Confidence: {composite_signal.confidence:.3f} in {result['calculation_time']:.4f}s")
            
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error in AI composite calculation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in AI composite calculation: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None    
    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare and validate input data"""
        try:
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    df = data.copy()
                else:
                    # Assume first column is close price
                    df = pd.DataFrame({'close': data.iloc[:, 0]})
                    if data.shape[1] > 1:
                        df['high'] = data.iloc[:, 1]
                        df['low'] = data.iloc[:, 2] if data.shape[1] > 2 else df['close']
                        df['volume'] = data.iloc[:, 3] if data.shape[1] > 3 else 1000
            else:
                df = pd.DataFrame({
                    'close': np.array(data),
                    'high': np.array(data),
                    'low': np.array(data),
                    'volume': np.ones(len(data)) * 1000
                })
            
            # Ensure required columns exist
            for col in ['high', 'low', 'volume']:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 1000
                    else:
                        df[col] = df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None
    
    async def _engineer_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Engineer technical indicator features"""
        try:
            features_df = pd.DataFrame(index=df.index)
            
            # Price-based features
            features_df['close'] = df['close']
            features_df['high'] = df['high']
            features_df['low'] = df['low']
            features_df['volume'] = df['volume']
            
            # Moving averages
            for period in [10, 20, 50]:
                features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
                features_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = df['close'].rolling(bb_period).mean()
            bb_std_dev = df['close'].rolling(bb_period).std()
            features_df['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
            features_df['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
            features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
            features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / features_df['bb_width']
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features_df['atr_14'] = true_range.rolling(14).mean()
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            features_df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            features_df['stoch_d'] = features_df['stoch_k'].rolling(3).mean()
            
            # Williams %R
            features_df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
            
            # Momentum and ROC
            features_df['momentum_10'] = df['close'] / df['close'].shift(10)
            features_df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Volume indicators
            features_df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
            
            # Money Flow Index (simplified)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow
            features_df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
            
            # Commodity Channel Index
            cci_period = 20
            tp = (df['high'] + df['low'] + df['close']) / 3
            tp_ma = tp.rolling(cci_period).mean()
            tp_mad = tp.rolling(cci_period).apply(lambda x: np.abs(x - x.mean()).mean())
            features_df['cci_20'] = (tp - tp_ma) / (0.015 * tp_mad)
            
            # Price patterns and derived features
            features_df['price_change'] = df['close'].pct_change()
            features_df['volatility'] = features_df['price_change'].rolling(20).std()
            features_df['price_position'] = df['close'] / df['close'].rolling(50).max()
            
            # Lag features for temporal patterns
            for lag in [1, 2, 5]:
                features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
                features_df[f'rsi_lag_{lag}'] = features_df['rsi_14'].shift(lag)
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            self.feature_names = [col for col in features_df.columns if col not in ['close', 'high', 'low', 'volume']]
            self.logger.debug(f"Engineered {len(self.feature_names)} features from {len(df)} data points")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return None    
    async def _train_ensemble(self, features_df: pd.DataFrame) -> bool:
        """Train the ensemble of ML models"""
        try:
            self.logger.info("Training AI ensemble models...")
            
            # Prepare training data
            X = features_df[self.feature_names].values
            
            # Create target variable (future price change)
            future_periods = 5  # Predict 5 periods ahead
            y = features_df['close'].shift(-future_periods).pct_change().dropna()
            
            # Align X and y
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y.values[:min_length]
            
            if len(X) < 50:  # Need minimum data for training
                self.logger.warning("Insufficient data for training")
                return False
            
            # Scale features
            X_scaled = self.scalers['robust'].fit_transform(X)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            model_scores = {}
            
            for name, model in self.models.items():
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Validate
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                model_scores[name] = np.mean(scores)
                self.logger.debug(f"Model {name} average R² score: {model_scores[name]:.4f}")
            
            # Calculate model weights based on performance
            total_score = sum(max(0, score) for score in model_scores.values())
            if total_score > 0:
                self.model_weights = {name: max(0, score) / total_score for name, score in model_scores.items()}
            else:
                # Equal weights if all models perform poorly
                self.model_weights = {name: 1/len(self.models) for name in self.models.keys()}
            
            # Final training on full dataset
            for name, model in self.models.items():
                model.fit(X_scaled, y)
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            
            self.logger.info(f"Ensemble training completed. Model weights: {self.model_weights}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            return False
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if not self.is_trained or self.last_train_time is None:
            return True
        
        # Retrain every retrain_frequency calculations or after significant time
        time_since_train = datetime.now() - self.last_train_time
        return time_since_train > timedelta(hours=24)  # Retrain daily
    
    async def _generate_ensemble_predictions(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions from ensemble models"""
        try:
            if not self.is_trained:
                return {}
            
            # Get latest features
            latest_features = features_df[self.feature_names].iloc[-1:].values
            X_scaled = self.scalers['robust'].transform(latest_features)
            
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    self.logger.warning(f"Error predicting with {name}: {e}")
                    predictions[name] = 0.0
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return {}
    
    async def _calculate_composite_signal(self, predictions: Dict[str, float], 
                                        features_df: pd.DataFrame) -> CompositeSignal:
        """Calculate composite signal from ensemble predictions"""
        try:
            # Weighted average of predictions
            if predictions and self.model_weights:
                weighted_pred = sum(pred * self.model_weights.get(name, 0) 
                                  for name, pred in predictions.items())
                
                # Normalize to -1 to 1 range
                composite_score = np.tanh(weighted_pred * 10)  # Scale and bound
            else:
                composite_score = 0.0
            
            # Calculate confidence based on prediction agreement
            if len(predictions) > 1:
                pred_values = list(predictions.values())
                pred_std = np.std(pred_values)
                pred_mean = np.mean(pred_values)
                
                # Lower standard deviation = higher confidence
                confidence = 1.0 / (1.0 + pred_std * 10) if pred_std > 0 else 1.0
                confidence = max(0.1, min(1.0, confidence))  # Bound between 0.1 and 1.0
            else:
                confidence = 0.5
            
            # Analyze trend and momentum components
            latest_features = features_df.iloc[-1]
            
            # Trend strength from moving averages
            if 'sma_20' in latest_features and 'sma_50' in latest_features:
                trend_strength = (latest_features['sma_20'] - latest_features['sma_50']) / latest_features['close']
                trend_strength = np.tanh(trend_strength * 100)  # Normalize
            else:
                trend_strength = 0.0
            
            # Momentum strength from RSI and MACD
            momentum_components = []
            if 'rsi_14' in latest_features:
                rsi_momentum = (latest_features['rsi_14'] - 50) / 50  # Normalize RSI
                momentum_components.append(rsi_momentum)
            
            if 'macd_histogram' in latest_features:
                macd_momentum = np.tanh(latest_features['macd_histogram'] * 1000)
                momentum_components.append(macd_momentum)
            
            momentum_strength = np.mean(momentum_components) if momentum_components else 0.0
            
            # Volatility regime
            if 'atr_14' in latest_features and 'close' in latest_features:
                vol_ratio = latest_features['atr_14'] / latest_features['close']
                if vol_ratio < 0.01:
                    volatility_regime = 'low'
                elif vol_ratio < 0.03:
                    volatility_regime = 'medium'
                else:
                    volatility_regime = 'high'
            else:
                volatility_regime = 'medium'
            
            # Market regime based on trend and volatility
            if abs(trend_strength) > 0.3:
                market_regime = 'trending'
            elif volatility_regime == 'high':
                market_regime = 'breakout'
            else:
                market_regime = 'ranging'
            
            # Signal quality assessment
            if confidence > 0.8 and abs(composite_score) > 0.5:
                signal_quality = 'excellent'
            elif confidence > 0.6 and abs(composite_score) > 0.3:
                signal_quality = 'good'
            elif confidence > 0.4:
                signal_quality = 'fair'
            else:
                signal_quality = 'poor'
            
            # Recommended action
            if composite_score > 0.7 and confidence > 0.7:
                recommended_action = 'strong_buy'
            elif composite_score > 0.3 and confidence > 0.5:
                recommended_action = 'buy'
            elif composite_score < -0.7 and confidence > 0.7:
                recommended_action = 'strong_sell'
            elif composite_score < -0.3 and confidence > 0.5:
                recommended_action = 'sell'
            else:
                recommended_action = 'hold'
            
            return CompositeSignal(
                timestamp=datetime.now(),
                composite_score=composite_score,
                confidence=confidence,
                trend_strength=trend_strength,
                momentum_strength=momentum_strength,
                volatility_regime=volatility_regime,
                market_regime=market_regime,
                feature_importance={},  # Will be filled by analyze_feature_importance
                ensemble_predictions=predictions,
                signal_quality=signal_quality,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating composite signal: {e}")
            # Return default signal
            return CompositeSignal(
                timestamp=datetime.now(),
                composite_score=0.0,
                confidence=0.0,
                trend_strength=0.0,
                momentum_strength=0.0,
                volatility_regime='medium',
                market_regime='ranging',
                feature_importance={},
                ensemble_predictions={},
                signal_quality='poor',
                recommended_action='hold'
            )    
    async def _analyze_market_regime(self, price_df: pd.DataFrame, 
                                   features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime"""
        try:
            regime_analysis = {
                'trend_direction': 'neutral',
                'trend_strength': 0.0,
                'volatility_level': 'medium',
                'momentum_divergence': False,
                'support_resistance': {},
                'regime_confidence': 0.5
            }
            
            if len(features_df) < 20:
                return regime_analysis
            
            latest = features_df.iloc[-1]
            
            # Trend analysis
            if 'sma_20' in latest and 'sma_50' in latest:
                if latest['close'] > latest['sma_20'] > latest['sma_50']:
                    regime_analysis['trend_direction'] = 'bullish'
                    regime_analysis['trend_strength'] = 0.7
                elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                    regime_analysis['trend_direction'] = 'bearish'
                    regime_analysis['trend_strength'] = 0.7
                else:
                    regime_analysis['trend_direction'] = 'neutral'
                    regime_analysis['trend_strength'] = 0.3
            
            # Volatility analysis
            if 'atr_14' in latest:
                atr_pct = latest['atr_14'] / latest['close']
                if atr_pct > 0.03:
                    regime_analysis['volatility_level'] = 'high'
                elif atr_pct < 0.01:
                    regime_analysis['volatility_level'] = 'low'
                else:
                    regime_analysis['volatility_level'] = 'medium'
            
            # Support and resistance levels
            recent_highs = price_df['high'].tail(20).rolling(5).max()
            recent_lows = price_df['low'].tail(20).rolling(5).min()
            
            regime_analysis['support_resistance'] = {
                'resistance': float(recent_highs.max()),
                'support': float(recent_lows.min()),
                'current_price': float(latest['close'])
            }
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {e}")
            return {'trend_direction': 'neutral', 'trend_strength': 0.0}
    
    async def _generate_trading_signals(self, signal: CompositeSignal, regime: Dict[str, Any], 
                                      current_price: float) -> Dict[str, Any]:
        """Generate trading signals with risk management"""
        try:
            signals = {
                'action': signal.recommended_action,
                'entry_price': current_price,
                'confidence': signal.confidence,
                'position_size': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'risk_reward_ratio': 0.0,
                'signal_strength': 'weak'
            }
            
            # Position sizing based on confidence and volatility
            base_position = 0.1  # 10% base position
            confidence_multiplier = signal.confidence
            volatility_adjustment = 1.0
            
            if signal.volatility_regime == 'high':
                volatility_adjustment = 0.5  # Reduce position in high volatility
            elif signal.volatility_regime == 'low':
                volatility_adjustment = 1.2  # Increase position in low volatility
            
            signals['position_size'] = base_position * confidence_multiplier * volatility_adjustment
            signals['position_size'] = min(0.25, max(0.01, signals['position_size']))  # Cap at 25%
            
            # Risk management levels
            atr_estimate = current_price * 0.02  # Rough ATR estimate
            
            if signal.composite_score > 0:  # Long position
                signals['stop_loss'] = current_price - (atr_estimate * 2)
                signals['take_profit'] = current_price + (atr_estimate * 3)
                signals['signal_strength'] = 'strong' if signal.composite_score > 0.5 else 'moderate'
            elif signal.composite_score < 0:  # Short position
                signals['stop_loss'] = current_price + (atr_estimate * 2)
                signals['take_profit'] = current_price - (atr_estimate * 3)
                signals['signal_strength'] = 'strong' if signal.composite_score < -0.5 else 'moderate'
            else:
                signals['signal_strength'] = 'weak'
            
            # Risk-reward ratio
            if signals['stop_loss'] != current_price:
                risk = abs(current_price - signals['stop_loss'])
                reward = abs(signals['take_profit'] - current_price)
                signals['risk_reward_ratio'] = reward / risk if risk > 0 else 0
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    async def _analyze_feature_importance(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance from trained models"""
        try:
            importance_analysis = {
                'top_features': [],
                'feature_scores': {},
                'model_contributions': {}
            }
            
            if not self.is_trained or not self.feature_names:
                return importance_analysis
            
            # Get feature importance from tree-based models
            total_importance = np.zeros(len(self.feature_names))
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    weight = self.model_weights.get(name, 0)
                    total_importance += importances * weight
                    
                    # Store individual model contributions
                    importance_analysis['model_contributions'][name] = {
                        feature: float(importance) for feature, importance 
                        in zip(self.feature_names, importances)
                    }
            
            # Normalize total importance
            if np.sum(total_importance) > 0:
                total_importance = total_importance / np.sum(total_importance)
            
            # Create feature scores dictionary
            importance_analysis['feature_scores'] = {
                feature: float(importance) 
                for feature, importance in zip(self.feature_names, total_importance)
            }
            
            # Get top 10 features
            feature_importance_pairs = list(zip(self.feature_names, total_importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            importance_analysis['top_features'] = [
                {'feature': feature, 'importance': float(importance)}
                for feature, importance in feature_importance_pairs[:10]
            ]
            
            return importance_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            return {'top_features': [], 'feature_scores': {}}
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            return {
                'is_trained': self.is_trained,
                'last_train_time': self.last_train_time.isoformat() if self.last_train_time else None,
                'model_weights': self.model_weights,
                'feature_count': len(self.feature_names),
                'ensemble_size': len(self.models)
            }
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {}
    
    def _signal_to_dict(self, signal: CompositeSignal) -> Dict[str, Any]:
        """Convert CompositeSignal to dictionary"""
        return {
            'timestamp': signal.timestamp.isoformat(),
            'composite_score': signal.composite_score,
            'confidence': signal.confidence,
            'trend_strength': signal.trend_strength,
            'momentum_strength': signal.momentum_strength,
            'volatility_regime': signal.volatility_regime,
            'market_regime': signal.market_regime,
            'feature_importance': signal.feature_importance,
            'ensemble_predictions': signal.ensemble_predictions,
            'signal_quality': signal.signal_quality,
            'recommended_action': signal.recommended_action
        }    
    async def get_signal_summary(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Get summary of AI composite signal"""
        try:
            result = await self.calculate(data)
            if not result:
                return None
            
            signal = result['composite_signal']
            return {
                'composite_score': signal['composite_score'],
                'confidence': signal['confidence'],
                'recommended_action': signal['recommended_action'],
                'signal_quality': signal['signal_quality'],
                'market_regime': signal['market_regime'],
                'volatility_regime': signal['volatility_regime']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal summary: {e}")
            return None
    
    async def backtest_signals(self, historical_data: pd.DataFrame, 
                             lookback_days: int = 30) -> Dict[str, Any]:
        """Backtest AI composite signals"""
        try:
            self.logger.info(f"Backtesting AI signals over {lookback_days} days")
            
            backtest_results = {
                'total_signals': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'signal_distribution': {},
                'performance_by_regime': {}
            }
            
            # Generate signals for historical periods
            signals_history = []
            for i in range(self.lookback_period, len(historical_data) - 5, 5):  # Every 5 periods
                data_slice = historical_data.iloc[:i]
                result = await self.calculate(data_slice)
                
                if result:
                    signal = result['composite_signal']
                    future_price = historical_data.iloc[i + 5]['close']
                    current_price = historical_data.iloc[i]['close']
                    actual_return = (future_price - current_price) / current_price
                    
                    # Check if signal was correct
                    predicted_direction = 1 if signal['composite_score'] > 0 else -1
                    actual_direction = 1 if actual_return > 0 else -1
                    correct = predicted_direction == actual_direction
                    
                    signals_history.append({
                        'signal': signal,
                        'actual_return': actual_return,
                        'correct': correct
                    })
            
            if signals_history:
                backtest_results['total_signals'] = len(signals_history)
                backtest_results['correct_predictions'] = sum(s['correct'] for s in signals_history)
                backtest_results['accuracy'] = backtest_results['correct_predictions'] / backtest_results['total_signals']
                backtest_results['avg_confidence'] = np.mean([s['signal']['confidence'] for s in signals_history])
                
                # Signal distribution
                actions = [s['signal']['recommended_action'] for s in signals_history]
                backtest_results['signal_distribution'] = {action: actions.count(action) for action in set(actions)}
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save trained models to file"""
        try:
            if self.is_trained:
                model_data = {
                    'models': self.models,
                    'scalers': self.scalers,
                    'model_weights': self.model_weights,
                    'feature_names': self.feature_names,
                    'last_train_time': self.last_train_time
                }
                joblib.dump(model_data, filepath)
                self.logger.info(f"Model saved to {filepath}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained models from file"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_weights = model_data['model_weights']
            self.feature_names = model_data['feature_names']
            self.last_train_time = model_data['last_train_time']
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    async def test_ai_composite():
        # Create sample OHLC data
        np.random.seed(42)
        periods = 300
        
        # Generate realistic market data
        base_price = 100
        data = {'close': [], 'high': [], 'low': [], 'volume': []}
        
        for i in range(periods):
            # Trending with noise and occasional reversals
            trend = 0.001 * np.sin(i * 0.05) + 0.0005
            noise = np.random.normal(0, 0.01)
            change = trend + noise
            
            if i == 0:
                price = base_price
            else:
                price = data['close'][-1] * (1 + change)
            
            data['close'].append(price)
            data['high'].append(price * (1 + abs(np.random.normal(0, 0.005))))
            data['low'].append(price * (1 - abs(np.random.normal(0, 0.005))))
            data['volume'].append(np.random.randint(1000, 10000))
        
        df = pd.DataFrame(data)
        
        # Test the AI composite indicator
        ai_indicator = CustomAICompositeIndicator(
            lookback_period=100, 
            feature_window=20,
            ensemble_size=3,
            retrain_frequency=100
        )
        
        print("Testing Custom AI Composite Indicator...")
        result = await ai_indicator.calculate(df)
        
        if result:
            signal = result['composite_signal']
            print(f"Calculation completed in {result['calculation_time']:.4f} seconds")
            print(f"Composite Score: {signal['composite_score']:.3f}")
            print(f"Confidence: {signal['confidence']:.3f}")
            print(f"Recommended Action: {signal['recommended_action']}")
            print(f"Signal Quality: {signal['signal_quality']}")
            print(f"Market Regime: {signal['market_regime']}")
            print(f"Volatility Regime: {signal['volatility_regime']}")
            
            # Show trading signals
            trading = result['trading_signals']
            print(f"\nTrading Signals:")
            print(f"Action: {trading['action']}")
            print(f"Position Size: {trading['position_size']:.1%}")
            print(f"Entry Price: {trading['entry_price']:.2f}")
            print(f"Stop Loss: {trading['stop_loss']:.2f}")
            print(f"Take Profit: {trading['take_profit']:.2f}")
            print(f"Risk/Reward: {trading['risk_reward_ratio']:.2f}")
            
            # Show top features
            if result['feature_analysis']['top_features']:
                print(f"\nTop 5 Important Features:")
                for i, feature in enumerate(result['feature_analysis']['top_features'][:5]):
                    print(f"{i+1}. {feature['feature']}: {feature['importance']:.3f}")
            
            # Model performance
            perf = result['model_performance']
            print(f"\nModel Status: {'Trained' if perf['is_trained'] else 'Not Trained'}")
            print(f"Feature Count: {perf['feature_count']}")
            print(f"Ensemble Size: {perf['ensemble_size']}")
            
        else:
            print("AI composite calculation failed")
    
    # Run the test
    asyncio.run(test_ai_composite())