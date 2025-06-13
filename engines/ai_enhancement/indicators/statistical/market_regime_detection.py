"""
Market Regime Detection Indicator for Platform3

An advanced indicator that identifies different market regimes (bull, bear, 
sideways, volatile, calm) using multiple statistical and technical methods.
This is crucial for adaptive trading strategies.

Author: Platform3 Development Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MarketRegimeDetection:
    """
    Market Regime Detection Indicator
    
    This indicator uses multiple methods to detect market regimes:
    - Trend analysis (bull/bear/sideways)
    - Volatility analysis (high/low)
    - Volume analysis (participation levels)
    - Statistical clustering methods
    - Hidden Markov Models approximation
    
    Regimes identified:
    0: Bear Market (downtrend, high volatility)
    1: Bull Market (uptrend, moderate volatility)
    2: Sideways Market (no clear trend, low volatility)
    3: Volatile Market (high volatility, mixed signals)
    4: Transition (regime change occurring)
    """
    
    def __init__(self, 
                 trend_window: int = 20,
                 volatility_window: int = 20,
                 volume_window: int = 20,
                 regime_window: int = 50,
                 n_regimes: int = 5,
                 smoothing_factor: float = 0.7):
        """
        Initialize Market Regime Detection
        
        Args:
            trend_window: Window for trend calculation
            volatility_window: Window for volatility calculation
            volume_window: Window for volume analysis
            regime_window: Window for regime detection
            n_regimes: Number of regimes to detect
            smoothing_factor: Smoothing factor for regime transitions
        """
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.regime_window = regime_window
        self.n_regimes = n_regimes
        self.smoothing_factor = smoothing_factor
        
        self.regime_history = []
        self.regime_features = []
        self.is_trained = False
        
        if SKLEARN_AVAILABLE:
            self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            self.scaler = StandardScaler()
            self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        else:
            print("Warning: scikit-learn not available. Using simplified regime detection.")
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength and direction
        
        Args:
            data: Market data
            
        Returns:
            Trend strength (-1 to 1, negative = downtrend, positive = uptrend)
        """
        if len(data) < self.trend_window:
            return 0.0
        
        # Linear regression slope
        close_prices = data['close'].values[-self.trend_window:]
        x = np.arange(len(close_prices))
        
        # Calculate slope
        slope = np.polyfit(x, close_prices, 1)[0]
        
        # Normalize by average price
        avg_price = np.mean(close_prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0.0
        
        # Scale to [-1, 1]
        trend_strength = np.tanh(normalized_slope * 100)
        
        return trend_strength
    
    def _calculate_volatility_regime(self, data: pd.DataFrame) -> float:
        """
        Calculate volatility regime
        
        Args:
            data: Market data
            
        Returns:
            Volatility level (0 to 1)
        """
        if len(data) < self.volatility_window:
            return 0.5
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < self.volatility_window:
            return 0.5
        
        # Rolling volatility
        vol = returns.rolling(window=self.volatility_window).std()
        current_vol = vol.iloc[-1] if not vol.empty else 0.0
        
        # Historical volatility percentile
        if len(vol) > self.volatility_window:
            vol_percentile = (vol.iloc[-1] > vol.iloc[:-1]).sum() / len(vol.iloc[:-1])
        else:
            vol_percentile = 0.5
        
        return vol_percentile
    
    def _calculate_volume_regime(self, data: pd.DataFrame) -> float:
        """
        Calculate volume regime
        
        Args:
            data: Market data
            
        Returns:
            Volume participation level (0 to 1)
        """
        if len(data) < self.volume_window or 'volume' not in data.columns:
            return 0.5
        
        # Volume moving average
        vol_ma = data['volume'].rolling(window=self.volume_window).mean()
        current_vol = data['volume'].iloc[-1]
        avg_vol = vol_ma.iloc[-1] if not vol_ma.empty else current_vol
        
        # Volume ratio
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Normalize to [0, 1]
        vol_regime = min(1.0, max(0.0, (vol_ratio - 0.5) * 2))
        
        return vol_regime
    
    def _calculate_momentum_regime(self, data: pd.DataFrame) -> float:
        """
        Calculate momentum regime using RSI and MACD
        
        Args:
            data: Market data
            
        Returns:
            Momentum strength (-1 to 1)
        """
        if len(data) < 26:  # Need enough data for MACD
            return 0.0
        
        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # MACD calculation
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        current_histogram = histogram.iloc[-1] if not histogram.empty else 0
        
        # Combine RSI and MACD
        rsi_signal = (current_rsi - 50) / 50  # Normalize to [-1, 1]
        macd_signal = np.tanh(current_histogram / data['close'].iloc[-1] * 1000)
        
        momentum = (rsi_signal + macd_signal) / 2
        
        return momentum
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime classification
        
        Args:
            data: Market data
            
        Returns:
            Feature vector for regime detection
        """
        features = []
        
        # Trend features
        trend_strength = self._calculate_trend_strength(data)
        features.append(trend_strength)
        
        # Volatility features
        vol_regime = self._calculate_volatility_regime(data)
        features.append(vol_regime)
        
        # Volume features
        volume_regime = self._calculate_volume_regime(data)
        features.append(volume_regime)
        
        # Momentum features
        momentum = self._calculate_momentum_regime(data)
        features.append(momentum)
        
        # Price position features
        if len(data) >= 20:
            high_20 = data['high'].rolling(window=20).max().iloc[-1]
            low_20 = data['low'].rolling(window=20).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if high_20 > low_20:
                price_position = (current_price - low_20) / (high_20 - low_20)
            else:
                price_position = 0.5
            
            features.append(price_position)
        else:
            features.append(0.5)
        
        # Correlation with market (simplified as price autocorrelation)
        if len(data) >= 10:
            returns = data['close'].pct_change().dropna()
            if len(returns) >= 10:
                autocorr = returns.autocorr(lag=1)
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0
            features.append(autocorr)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def _classify_regime_simple(self, features: np.ndarray) -> int:
        """
        Simple rule-based regime classification (fallback)
        
        Args:
            features: Feature vector [trend, volatility, volume, momentum, price_pos, autocorr]
            
        Returns:
            Regime class (0-4)
        """
        trend, volatility, volume, momentum, price_pos, autocorr = features
        
        # High volatility regime
        if volatility > 0.8:
            return 3  # Volatile Market
        
        # Strong uptrend
        if trend > 0.3 and momentum > 0.2:
            return 1  # Bull Market
        
        # Strong downtrend
        if trend < -0.3 and momentum < -0.2:
            return 0  # Bear Market
        
        # Low volatility, no clear trend
        if volatility < 0.3 and abs(trend) < 0.2:
            return 2  # Sideways Market
        
        # Default to transition
        return 4  # Transition
    
    def _train_regime_models(self, feature_matrix: np.ndarray):
        """
        Train regime detection models
        
        Args:
            feature_matrix: Matrix of regime features
        """
        if not SKLEARN_AVAILABLE or len(feature_matrix) < self.n_regimes:
            return
        
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            # Train K-means
            self.kmeans.fit(scaled_features)
            
            # Train Gaussian Mixture Model
            self.gmm.fit(scaled_features)
            
            self.is_trained = True
            
        except Exception as e:
            print(f"Warning: Could not train regime models: {e}")
    
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect current market regime
        
        Args:
            data: Market data
            
        Returns:
            Regime detection results
        """
        if len(data) < max(self.trend_window, self.volatility_window):
            return {
                'regime': 2,  # Default to sideways
                'confidence': 0.5,
                'features': np.zeros(6),
                'probabilities': np.ones(self.n_regimes) / self.n_regimes
            }
        
        # Extract features
        features = self._extract_regime_features(data)
        
        # Store features for training
        self.regime_features.append(features)
        
        # Keep only recent features for training
        if len(self.regime_features) > self.regime_window:
            self.regime_features = self.regime_features[-self.regime_window:]
        
        # Train models if we have enough data
        if len(self.regime_features) >= self.regime_window and not self.is_trained:
            feature_matrix = np.array(self.regime_features)
            self._train_regime_models(feature_matrix)
        
        # Classify regime
        if SKLEARN_AVAILABLE and self.is_trained:
            try:
                # Scale features
                scaled_features = self.scaler.transform(features.reshape(1, -1))
                
                # K-means prediction
                kmeans_regime = self.kmeans.predict(scaled_features)[0]
                
                # GMM prediction
                gmm_regime = self.gmm.predict(scaled_features)[0]
                gmm_probs = self.gmm.predict_proba(scaled_features)[0]
                
                # Combine predictions
                regime = int((kmeans_regime + gmm_regime) / 2)
                confidence = np.max(gmm_probs)
                
                return {
                    'regime': regime,
                    'confidence': confidence,
                    'features': features,
                    'probabilities': gmm_probs,
                    'kmeans_regime': kmeans_regime,
                    'gmm_regime': gmm_regime
                }
                
            except Exception as e:
                print(f"Warning: ML regime detection failed: {e}")
        
        # Fallback to simple classification
        regime = self._classify_regime_simple(features)
        
        return {
            'regime': regime,
            'confidence': 0.7,  # Moderate confidence for rule-based
            'features': features,
            'probabilities': np.ones(self.n_regimes) / self.n_regimes
        }
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate regime signals for entire dataset
        
        Args:
            data: Historical market data
            
        Returns:
            Array of regime classifications
        """
        regimes = np.full(len(data), 2)  # Default to sideways
        
        # Calculate regimes for each point
        for i in range(max(self.trend_window, self.volatility_window), len(data)):
            # Get data window
            window_data = data.iloc[max(0, i-self.regime_window):i+1]
            
            # Detect regime
            result = self.detect_regime(window_data)
            regime = result['regime']
            
            # Apply smoothing
            if i > 0 and len(self.regime_history) > 0:
                prev_regime = self.regime_history[-1]
                smoothed_regime = (self.smoothing_factor * prev_regime + 
                                 (1 - self.smoothing_factor) * regime)
                regime = int(round(smoothed_regime))
            
            regimes[i] = regime
            self.regime_history.append(regime)
            
            # Keep regime history manageable
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
        
        return regimes
    
    def get_regime_description(self, regime: int) -> str:
        """
        Get human-readable regime description
        
        Args:
            regime: Regime number
            
        Returns:
            Regime description
        """
        descriptions = {
            0: "Bear Market (downtrend, high volatility)",
            1: "Bull Market (uptrend, moderate volatility)", 
            2: "Sideways Market (no clear trend, low volatility)",
            3: "Volatile Market (high volatility, mixed signals)",
            4: "Transition (regime change occurring)"
        }
        
        return descriptions.get(regime, f"Unknown Regime ({regime})")
    
    def get_regime_statistics(self) -> Dict[str, any]:
        """
        Get statistics about detected regimes
        
        Returns:
            Regime statistics
        """
        if not self.regime_history:
            return {}
        
        regimes = np.array(self.regime_history)
        unique, counts = np.unique(regimes, return_counts=True)
        
        stats = {
            'regime_counts': dict(zip(unique.astype(int), counts)),
            'regime_percentages': dict(zip(unique.astype(int), counts / len(regimes) * 100)),
            'total_periods': len(regimes),
            'current_regime': int(regimes[-1]) if len(regimes) > 0 else 2,
            'regime_changes': np.sum(np.diff(regimes) != 0),
            'stability': 1 - (np.sum(np.diff(regimes) != 0) / len(regimes))
        }
        
        return stats


# Test and example usage
if __name__ == "__main__":
    print("Testing Market Regime Detection Indicator...")
    
    # Generate sample data with different market regimes
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Create synthetic market data with regime changes
    price = 100
    prices = []
    volumes = []
    
    for i in range(300):
        # Different regimes at different periods
        if i < 75:  # Bull market
            trend = 0.002
            vol_factor = 0.015
        elif i < 150:  # Bear market
            trend = -0.0015
            vol_factor = 0.025
        elif i < 225:  # Sideways
            trend = 0.0002
            vol_factor = 0.008
        else:  # Volatile
            trend = 0.001 * np.sin(i * 0.2)
            vol_factor = 0.03
        
        # Price movement
        noise = np.random.normal(0, vol_factor)
        price = price * (1 + trend + noise)
        prices.append(price)
        
        # Volume
        volume = 1000000 + np.random.normal(0, 200000)
        volumes.append(max(volume, 500000))
    
    data = pd.DataFrame({
        'date': dates,
        'open': np.array(prices) * np.random.uniform(0.995, 1.005, 300),
        'high': np.array(prices) * np.random.uniform(1.005, 1.02, 300),
        'low': np.array(prices) * np.random.uniform(0.98, 0.995, 300),
        'close': prices,
        'volume': volumes
    })
    
    # Initialize regime detector
    regime_detector = MarketRegimeDetection(
        trend_window=20,
        volatility_window=20,
        regime_window=50
    )
    
    print(f"Initialized Market Regime Detector")
    print(f"Using {'ML models' if SKLEARN_AVAILABLE else 'rule-based classification'}")
    
    # Test regime detection on recent data
    print("\nTesting regime detection...")
    recent_data = data.iloc[-100:]
    regime_result = regime_detector.detect_regime(recent_data)
    
    print("Current Regime Analysis:")
    print(f"  Regime: {regime_result['regime']} - {regime_detector.get_regime_description(regime_result['regime'])}")
    print(f"  Confidence: {regime_result['confidence']:.3f}")
    print(f"  Features: {regime_result['features']}")
    
    if 'probabilities' in regime_result:
        print("  Regime Probabilities:")
        for i, prob in enumerate(regime_result['probabilities']):
            print(f"    {i} ({regime_detector.get_regime_description(i)}): {prob:.3f}")
    
    # Test full calculation
    print("\nCalculating regimes for full dataset...")
    regimes = regime_detector.calculate(data)
    
    print(f"Generated {len(regimes)} regime classifications")
    print(f"Unique regimes found: {np.unique(regimes)}")
    
    # Regime statistics
    stats = regime_detector.get_regime_statistics()
    if stats:
        print("\nRegime Statistics:")
        print(f"  Total periods analyzed: {stats['total_periods']}")
        print(f"  Current regime: {stats['current_regime']} - {regime_detector.get_regime_description(stats['current_regime'])}")
        print(f"  Regime changes: {stats['regime_changes']}")
        print(f"  Market stability: {stats['stability']:.3f}")
        print("  Regime distribution:")
        for regime, percentage in stats['regime_percentages'].items():
            print(f"    {regime} ({regime_detector.get_regime_description(regime)}): {percentage:.1f}%")
    
    # Show regime transitions
    print("\nLast 20 regime classifications:")
    for i, regime in enumerate(regimes[-20:]):
        period = len(regimes) - 20 + i
        print(f"  Period {period}: {int(regime)} ({regime_detector.get_regime_description(int(regime))})")
    
    print("\nMarket Regime Detection test completed successfully!")