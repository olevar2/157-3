"""
Spread Predictor
Bid/ask spread forecasting using machine learning for scalping optimization.
Predicts optimal entry timing based on spread dynamics and market conditions.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import pickle
import os

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Using mock implementation.")


@dataclass
class SpreadPrediction:
    """Spread prediction result"""
    timestamp: float
    symbol: str
    current_spread: float
    predicted_spread: float
    spread_direction: str  # 'widening', 'tightening', 'stable'
    confidence: float  # 0-1
    optimal_entry_timing: str  # 'immediate', 'wait_short', 'wait_long'
    expected_spread_change: float
    prediction_horizon_seconds: int
    model_accuracy: float


@dataclass
class SpreadFeatures:
    """Feature set for spread prediction"""
    spread_history: List[float]  # Recent spread values
    volume_profile: List[float]  # Volume characteristics
    volatility_indicators: List[float]  # Market volatility
    time_features: List[float]  # Time-based features
    market_conditions: List[float]  # Overall market state
    liquidity_indicators: List[float]  # Liquidity measures


@dataclass
class SpreadAnalysis:
    """Comprehensive spread analysis"""
    current_spread: float
    average_spread: float
    spread_volatility: float
    spread_percentile: float  # Current spread vs historical
    liquidity_score: float
    market_impact_cost: float
    optimal_side: str  # 'bid', 'ask', 'mid'


class SpreadPredictor:
    """
    Spread Predictor
    ML-based bid/ask spread forecasting for scalping optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Prediction configuration
        self.lookback_periods = self.config.get('lookback_periods', 50)
        self.prediction_horizon = self.config.get('prediction_horizon', 60)  # seconds
        self.feature_count = self.config.get('feature_count', 20)
        
        # Model ensemble
        self.models = {}  # symbol -> {model_name: model}
        self.scalers = {}  # symbol -> scaler
        self.model_accuracy = {}  # symbol -> accuracy metrics
        
        # Model weights for ensemble
        self.model_weights = {
            'random_forest': 0.35,
            'xgboost': 0.35,
            'gradient_boost': 0.2,
            'linear': 0.1
        }
        
        # Data buffers
        self.spread_buffers = {}  # symbol -> deque of spread data
        self.volume_buffers = {}  # symbol -> deque of volume data
        self.volatility_buffers = {}  # symbol -> deque of volatility data
        self.max_buffer_size = 1000
        
        # Spread thresholds for classification
        self.spread_thresholds = {
            'tight': 0.5,    # Below 50th percentile
            'normal': 0.8,   # 50th-80th percentile
            'wide': 1.0      # Above 80th percentile
        }
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.training_count = 0
        
        # Model storage
        self.model_dir = self.config.get('model_dir', 'models/spread_predictor')
        os.makedirs(self.model_dir, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the spread predictor"""
        try:
            if not ML_AVAILABLE:
                self.logger.warning("ML libraries not available. Using mock implementation.")
                return
            
            # Load existing models
            await self._load_existing_models()
            
            self.logger.info("Spread Predictor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Spread Predictor: {e}")
            raise
    
    async def predict_spread(self, symbol: str, market_data: List[Dict]) -> SpreadPrediction:
        """
        Predict future spread and optimal entry timing
        """
        start_time = time.time()
        
        try:
            # Extract features
            features = await self._extract_spread_features(symbol, market_data)
            
            if len(features) < self.feature_count:
                raise ValueError(f"Insufficient features for prediction. Need {self.feature_count}, got {len(features)}")
            
            # Get or train models
            models = await self._get_or_train_models(symbol, market_data)
            
            # Make prediction
            prediction_result = await self._make_spread_prediction(
                models, features, symbol, market_data[-1]
            )
            
            # Update performance tracking
            prediction_time = time.time() - start_time
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            self.logger.debug(f"Spread prediction for {symbol} completed in {prediction_time:.3f}s")
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Spread prediction failed for {symbol}: {e}")
            raise
    
    async def analyze_current_spread(self, symbol: str, market_data: List[Dict]) -> SpreadAnalysis:
        """
        Analyze current spread conditions
        """
        try:
            if not market_data:
                raise ValueError("No market data provided")
            
            current_data = market_data[-1]
            current_spread = float(current_data.get('spread', 0))
            
            # Calculate historical spread statistics
            recent_spreads = [float(data.get('spread', 0)) for data in market_data[-100:]]
            average_spread = np.mean(recent_spreads) if recent_spreads else current_spread
            spread_volatility = np.std(recent_spreads) if len(recent_spreads) > 1 else 0
            
            # Calculate spread percentile
            if len(recent_spreads) > 10:
                spread_percentile = np.percentile(recent_spreads, 
                                                [p for p in range(101) if np.percentile(recent_spreads, p) <= current_spread][-1] 
                                                if any(np.percentile(recent_spreads, p) <= current_spread for p in range(101)) else 100)
            else:
                spread_percentile = 50.0
            
            # Calculate liquidity score
            recent_volumes = [float(data.get('volume', 0)) for data in market_data[-20:]]
            avg_volume = np.mean(recent_volumes) if recent_volumes else 1
            liquidity_score = min(100, avg_volume / max(current_spread, 0.0001) * 100)
            
            # Estimate market impact cost
            market_impact_cost = current_spread * 0.5  # Simplified: half spread as impact
            
            # Determine optimal side
            bid = float(current_data.get('bid', 0))
            ask = float(current_data.get('ask', 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(current_data.get('price', 0))
            
            # Simple heuristic for optimal side
            if current_spread < average_spread * 0.8:  # Tight spread
                optimal_side = 'mid'  # Can afford to cross spread
            elif liquidity_score > 70:  # High liquidity
                optimal_side = 'bid'  # Wait for better fill
            else:
                optimal_side = 'ask'  # Take liquidity
            
            return SpreadAnalysis(
                current_spread=current_spread,
                average_spread=average_spread,
                spread_volatility=spread_volatility,
                spread_percentile=spread_percentile,
                liquidity_score=liquidity_score,
                market_impact_cost=market_impact_cost,
                optimal_side=optimal_side
            )
            
        except Exception as e:
            self.logger.error(f"Spread analysis failed for {symbol}: {e}")
            raise
    
    async def _extract_spread_features(self, symbol: str, market_data: List[Dict]) -> np.ndarray:
        """Extract features for spread prediction"""
        
        if len(market_data) < self.lookback_periods:
            raise ValueError(f"Insufficient data. Need {self.lookback_periods}, got {len(market_data)}")
        
        features = []
        
        # Use recent data for feature extraction
        recent_data = market_data[-self.lookback_periods:]
        
        # Spread history features
        spreads = [float(data.get('spread', 0)) for data in recent_data]
        spread_features = self._calculate_spread_features(spreads)
        features.extend(spread_features)
        
        # Volume profile features
        volumes = [float(data.get('volume', 0)) for data in recent_data]
        volume_features = self._calculate_volume_features(volumes, spreads)
        features.extend(volume_features)
        
        # Volatility indicators
        prices = [float(data.get('close', 0)) for data in recent_data]
        volatility_features = self._calculate_volatility_features(prices)
        features.extend(volatility_features)
        
        # Time-based features
        timestamps = [float(data.get('timestamp', time.time())) for data in recent_data]
        time_features = self._calculate_time_features(timestamps)
        features.extend(time_features)
        
        # Market condition features
        market_features = self._calculate_market_features(recent_data)
        features.extend(market_features)
        
        # Liquidity indicators
        liquidity_features = self._calculate_liquidity_features(recent_data)
        features.extend(liquidity_features)
        
        return np.array(features)
    
    def _calculate_spread_features(self, spreads: List[float]) -> List[float]:
        """Calculate spread-based features"""
        if len(spreads) < 5:
            return [0.0] * 6
        
        # Current spread characteristics
        current_spread = spreads[-1]
        avg_spread = np.mean(spreads)
        spread_volatility = np.std(spreads)
        
        # Spread momentum
        spread_momentum = (spreads[-1] - spreads[-5]) / max(spreads[-5], 0.0001) if len(spreads) >= 5 else 0
        
        # Spread trend
        if len(spreads) >= 10:
            spread_trend = np.polyfit(range(10), spreads[-10:], 1)[0]
        else:
            spread_trend = 0
        
        # Spread percentile
        spread_percentile = (current_spread - min(spreads)) / max(max(spreads) - min(spreads), 0.0001)
        
        return [current_spread, avg_spread, spread_volatility, spread_momentum, spread_trend, spread_percentile]
    
    def _calculate_volume_features(self, volumes: List[float], spreads: List[float]) -> List[float]:
        """Calculate volume-related features"""
        if len(volumes) < 3:
            return [0.0] * 4
        
        # Volume characteristics
        avg_volume = np.mean(volumes)
        volume_volatility = np.std(volumes)
        current_volume = volumes[-1]
        
        # Volume-spread relationship
        if len(volumes) == len(spreads) and len(volumes) >= 10:
            correlation = np.corrcoef(volumes[-10:], spreads[-10:])[0, 1] if np.std(volumes[-10:]) > 0 and np.std(spreads[-10:]) > 0 else 0
        else:
            correlation = 0
        
        return [avg_volume, volume_volatility, current_volume, correlation]
    
    def _calculate_volatility_features(self, prices: List[float]) -> List[float]:
        """Calculate volatility-based features"""
        if len(prices) < 5:
            return [0.0] * 3
        
        # Price volatility
        price_volatility = np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices)
        
        # Realized volatility (short-term)
        if len(prices) >= 10:
            returns = np.diff(np.log(prices[-10:]))
            realized_vol = np.std(returns) * np.sqrt(len(returns))
        else:
            realized_vol = 0
        
        # Volatility trend
        if len(prices) >= 20:
            vol_short = np.std(prices[-10:])
            vol_long = np.std(prices[-20:])
            vol_trend = (vol_short - vol_long) / max(vol_long, 0.0001)
        else:
            vol_trend = 0
        
        return [price_volatility, realized_vol, vol_trend]
    
    def _calculate_time_features(self, timestamps: List[float]) -> List[float]:
        """Calculate time-based features"""
        if len(timestamps) < 2:
            return [0.0] * 3
        
        # Time of day effect (normalized to 0-1)
        current_time = datetime.fromtimestamp(timestamps[-1])
        hour_of_day = current_time.hour / 24.0
        day_of_week = current_time.weekday() / 6.0
        
        # Trading session intensity
        # Simplified: higher intensity during overlap hours
        if 8 <= current_time.hour <= 17:  # Business hours
            session_intensity = 1.0
        elif 0 <= current_time.hour <= 6:  # Asian session
            session_intensity = 0.7
        else:  # Off hours
            session_intensity = 0.3
        
        return [hour_of_day, day_of_week, session_intensity]
    
    def _calculate_market_features(self, market_data: List[Dict]) -> List[float]:
        """Calculate overall market condition features"""
        if len(market_data) < 5:
            return [0.0] * 3
        
        # Market direction
        prices = [float(data.get('close', 0)) for data in market_data[-10:]]
        if len(prices) >= 2:
            market_direction = (prices[-1] - prices[0]) / max(prices[0], 0.0001)
        else:
            market_direction = 0
        
        # Market activity
        volumes = [float(data.get('volume', 0)) for data in market_data[-10:]]
        avg_volume = np.mean(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        activity_ratio = current_volume / max(avg_volume, 1)
        
        # Market stress (simplified)
        spreads = [float(data.get('spread', 0)) for data in market_data[-10:]]
        avg_spread = np.mean(spreads) if spreads else 0
        current_spread = spreads[-1] if spreads else 0
        stress_indicator = current_spread / max(avg_spread, 0.0001)
        
        return [market_direction, activity_ratio, stress_indicator]
    
    def _calculate_liquidity_features(self, market_data: List[Dict]) -> List[float]:
        """Calculate liquidity-related features"""
        if len(market_data) < 3:
            return [0.0] * 2
        
        # Bid-ask spread as liquidity proxy
        spreads = [float(data.get('spread', 0)) for data in market_data[-10:]]
        avg_spread = np.mean(spreads) if spreads else 0
        
        # Volume-weighted liquidity
        volumes = [float(data.get('volume', 0)) for data in market_data[-10:]]
        avg_volume = np.mean(volumes) if volumes else 1
        
        liquidity_proxy = avg_volume / max(avg_spread, 0.0001)
        
        # Market depth proxy (simplified)
        depth_proxy = sum(float(data.get('market_depth', 0)) for data in market_data[-5:]) / 5
        
        return [liquidity_proxy, depth_proxy]
    
    async def _get_or_train_models(self, symbol: str, market_data: List[Dict]) -> Dict[str, Any]:
        """Get existing models or train new ones"""
        
        if symbol in self.models and self.models[symbol]:
            return self.models[symbol]
        
        # Train new models
        return await self._train_models(symbol, market_data)
    
    async def _train_models(self, symbol: str, market_data: List[Dict]) -> Dict[str, Any]:
        """Train ensemble of spread prediction models"""
        start_time = time.time()
        
        try:
            if not ML_AVAILABLE:
                # Mock models
                return {
                    'random_forest': {'predict': lambda x: np.array([0.0001])},
                    'xgboost': {'predict': lambda x: np.array([0.0001])},
                    'gradient_boost': {'predict': lambda x: np.array([0.0001])},
                    'linear': {'predict': lambda x: np.array([0.0001])}
                }
            
            # Prepare training data
            X, y = await self._prepare_training_data(symbol, market_data)
            
            if len(X) < 50:
                raise ValueError(f"Insufficient training data. Need at least 50 samples, got {len(X)}")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train models
            models = {}
            
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            models['random_forest'] = rf
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
            gb.fit(X_train, y_train)
            models['gradient_boost'] = gb
            
            # Linear Regression
            lr = Ridge(alpha=1.0)
            lr.fit(X_train, y_train)
            models['linear'] = lr
            
            # Evaluate models
            accuracy_metrics = {}
            for name, model in models.items():
                y_pred = model.predict(X_test)
                accuracy_metrics[name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
            
            # Store models and preprocessors
            self.models[symbol] = models
            self.scalers[symbol] = scaler
            self.model_accuracy[symbol] = accuracy_metrics
            
            # Save models
            await self._save_models(symbol, models, scaler)
            
            training_time = time.time() - start_time
            self.training_count += 1
            
            self.logger.info(f"Trained spread predictor models for {symbol} in {training_time:.2f}s")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Model training failed for {symbol}: {e}")
            raise
    
    async def _prepare_training_data(self, symbol: str, market_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from market history"""
        X, y = [], []
        
        for i in range(self.lookback_periods, len(market_data) - 1):
            # Extract features for current window
            window_data = market_data[:i+1]
            features = await self._extract_spread_features(symbol, window_data)
            
            # Target: next spread value
            next_spread = float(market_data[i+1].get('spread', 0))
            
            X.append(features)
            y.append(next_spread)
        
        return np.array(X), np.array(y)
    
    async def _make_spread_prediction(
        self, 
        models: Dict[str, Any], 
        features: np.ndarray, 
        symbol: str, 
        current_data: Dict
    ) -> SpreadPrediction:
        """Make spread prediction using ensemble"""
        
        # Scale features
        scaler = self.scalers.get(symbol)
        if scaler and hasattr(scaler, 'transform'):
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Get predictions from all models
        predictions = []
        for model_name, model in models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(features_scaled)[0]
                weight = self.model_weights.get(model_name, 0.25)
                predictions.append(pred * weight)
            else:
                # Mock prediction
                predictions.append(0.0001 * self.model_weights.get(model_name, 0.25))
        
        # Ensemble prediction
        predicted_spread = sum(predictions)
        current_spread = float(current_data.get('spread', 0))
        
        # Determine spread direction
        spread_change = predicted_spread - current_spread
        if abs(spread_change) < current_spread * 0.05:  # Less than 5% change
            spread_direction = 'stable'
        elif spread_change > 0:
            spread_direction = 'widening'
        else:
            spread_direction = 'tightening'
        
        # Calculate confidence based on model agreement
        pred_std = np.std([p / self.model_weights.get(list(models.keys())[i], 0.25) 
                          for i, p in enumerate(predictions)])
        confidence = max(0.1, min(0.9, 1.0 - (pred_std / max(predicted_spread, 0.0001))))
        
        # Determine optimal entry timing
        if spread_direction == 'tightening' and confidence > 0.7:
            optimal_timing = 'wait_short'  # Wait for tighter spread
        elif spread_direction == 'widening' and confidence > 0.7:
            optimal_timing = 'immediate'  # Enter before spread widens
        else:
            optimal_timing = 'immediate'  # Neutral timing
        
        # Get model accuracy
        accuracy_metrics = self.model_accuracy.get(symbol, {})
        avg_r2 = np.mean([metrics.get('r2', 0) for metrics in accuracy_metrics.values()]) if accuracy_metrics else 0.5
        
        return SpreadPrediction(
            timestamp=time.time(),
            symbol=symbol,
            current_spread=current_spread,
            predicted_spread=predicted_spread,
            spread_direction=spread_direction,
            confidence=confidence,
            optimal_entry_timing=optimal_timing,
            expected_spread_change=spread_change,
            prediction_horizon_seconds=self.prediction_horizon,
            model_accuracy=avg_r2
        )
    
    async def _save_models(self, symbol: str, models: Dict, scaler: Any) -> None:
        """Save models to disk"""
        try:
            # Save each model
            for model_name, model in models.items():
                model_path = os.path.join(self.model_dir, f"{symbol}_{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
        except Exception as e:
            self.logger.warning(f"Failed to save models for {symbol}: {e}")
    
    async def _load_existing_models(self) -> None:
        """Load existing models from disk"""
        if not os.path.exists(self.model_dir):
            return
        
        # Group files by symbol
        symbols = set()
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pkl') and '_' in filename:
                symbol = filename.split('_')[0]
                symbols.add(symbol)
        
        for symbol in symbols:
            try:
                models = {}
                
                # Load each model type
                for model_name in self.model_weights.keys():
                    model_path = os.path.join(self.model_dir, f"{symbol}_{model_name}.pkl")
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            models[model_name] = pickle.load(f)
                
                if models:
                    self.models[symbol] = models
                
                # Load scaler
                scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[symbol] = pickle.load(f)
                
                self.logger.debug(f"Loaded spread predictor models for {symbol}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load models for {symbol}: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get spread predictor performance metrics"""
        return {
            'total_predictions': self.prediction_count,
            'average_prediction_time_ms': (self.total_prediction_time / self.prediction_count * 1000) 
                                        if self.prediction_count > 0 else 0,
            'models_trained': self.training_count,
            'active_symbols': len(self.models),
            'ml_available': ML_AVAILABLE,
            'model_weights': self.model_weights,
            'model_accuracy': self.model_accuracy
        }
