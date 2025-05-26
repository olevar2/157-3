"""
Platform3 Forex Trading Platform
Noise Filter - ML-Based Market Noise Filtering

This module provides advanced machine learning algorithms for filtering
market noise and extracting clean trading signals from high-frequency data.

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
from scipy import signal as scipy_signal
from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseType(Enum):
    """Types of market noise"""
    MICROSTRUCTURE = "microstructure"
    ALGORITHMIC = "algorithmic"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    SPREAD = "spread"

class FilterMethod(Enum):
    """Noise filtering methods"""
    KALMAN = "kalman"
    WAVELET = "wavelet"
    PCA = "pca"
    ICA = "ica"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"
    ADAPTIVE = "adaptive"

@dataclass
class TickData:
    """Individual tick data point"""
    timestamp: datetime
    bid: float
    ask: float
    price: float
    volume: float
    spread: float

@dataclass
class FilteredData:
    """Filtered data result"""
    original_data: List[TickData]
    filtered_data: List[TickData]
    noise_level: float
    confidence: float
    filter_method: FilterMethod
    noise_types_detected: List[NoiseType]
    timestamp: datetime

@dataclass
class NoiseMetrics:
    """Noise analysis metrics"""
    signal_to_noise_ratio: float
    noise_variance: float
    price_volatility: float
    spread_stability: float
    volume_consistency: float

class NoiseFilter:
    """
    Advanced ML-based noise filtering for high-frequency trading data
    
    Features:
    - Multiple filtering algorithms (Kalman, Wavelet, PCA, ICA)
    - Real-time noise detection and classification
    - Adaptive filtering based on market conditions
    - Signal-to-noise ratio optimization
    - Multi-timeframe noise analysis
    - Anomaly detection for outlier removal
    """
    
    def __init__(self):
        """Initialize the noise filter"""
        self.filter_methods = {
            FilterMethod.KALMAN: self._kalman_filter,
            FilterMethod.WAVELET: self._wavelet_filter,
            FilterMethod.PCA: self._pca_filter,
            FilterMethod.ICA: self._ica_filter,
            FilterMethod.ISOLATION_FOREST: self._isolation_forest_filter,
            FilterMethod.DBSCAN: self._dbscan_filter,
            FilterMethod.ADAPTIVE: self._adaptive_filter
        }
        
        self.noise_thresholds = {
            NoiseType.MICROSTRUCTURE: 0.0001,  # 1 pip for major pairs
            NoiseType.ALGORITHMIC: 0.0002,     # 2 pips
            NoiseType.LIQUIDITY: 0.0005,       # 5 pips
            NoiseType.VOLATILITY: 0.001,       # 10 pips
            NoiseType.SPREAD: 0.0003           # 3 pips
        }
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.ica = FastICA(n_components=5, random_state=42)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        self.filter_history = []
        self.performance_metrics = {}
        
    async def filter_noise(
        self,
        tick_data: List[TickData],
        method: FilterMethod = FilterMethod.ADAPTIVE,
        noise_threshold: float = 0.0002
    ) -> FilteredData:
        """
        Filter noise from tick data using specified method
        
        Args:
            tick_data: List of tick data points
            method: Filtering method to use
            noise_threshold: Noise detection threshold
            
        Returns:
            FilteredData with cleaned signals
        """
        try:
            if len(tick_data) < 10:
                logger.warning("Insufficient data for noise filtering")
                return self._create_passthrough_result(tick_data, method)
            
            # Analyze noise characteristics
            noise_metrics = await self._analyze_noise(tick_data)
            
            # Detect noise types
            noise_types = self._detect_noise_types(tick_data, noise_metrics)
            
            # Apply filtering method
            filter_func = self.filter_methods.get(method, self._adaptive_filter)
            filtered_ticks = await filter_func(tick_data, noise_threshold)
            
            # Calculate filtering confidence
            confidence = self._calculate_filter_confidence(
                tick_data, filtered_ticks, noise_metrics
            )
            
            # Create result
            result = FilteredData(
                original_data=tick_data,
                filtered_data=filtered_ticks,
                noise_level=noise_metrics.signal_to_noise_ratio,
                confidence=confidence,
                filter_method=method,
                noise_types_detected=noise_types,
                timestamp=datetime.now()
            )
            
            # Store for analysis
            self.filter_history.append(result)
            
            logger.info(f"Noise filtered: {len(tick_data)} -> {len(filtered_ticks)} ticks "
                       f"(SNR: {noise_metrics.signal_to_noise_ratio:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error filtering noise: {e}")
            return self._create_passthrough_result(tick_data, method)
    
    async def _analyze_noise(self, tick_data: List[TickData]) -> NoiseMetrics:
        """Analyze noise characteristics in the data"""
        prices = np.array([tick.price for tick in tick_data])
        spreads = np.array([tick.spread for tick in tick_data])
        volumes = np.array([tick.volume for tick in tick_data])
        
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Signal-to-noise ratio
        signal_power = np.var(prices)
        noise_power = np.var(price_changes)
        snr = signal_power / max(noise_power, 1e-10)
        
        # Noise variance
        noise_variance = np.var(price_changes)
        
        # Price volatility
        price_volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        # Spread stability
        spread_stability = 1.0 - (np.std(spreads) / max(np.mean(spreads), 1e-10))
        
        # Volume consistency
        volume_consistency = 1.0 - (np.std(volumes) / max(np.mean(volumes), 1e-10))
        
        return NoiseMetrics(
            signal_to_noise_ratio=snr,
            noise_variance=noise_variance,
            price_volatility=price_volatility,
            spread_stability=max(0, spread_stability),
            volume_consistency=max(0, volume_consistency)
        )
    
    def _detect_noise_types(
        self,
        tick_data: List[TickData],
        metrics: NoiseMetrics
    ) -> List[NoiseType]:
        """Detect types of noise present in the data"""
        noise_types = []
        
        # Microstructure noise (high-frequency oscillations)
        if metrics.signal_to_noise_ratio < 2.0:
            noise_types.append(NoiseType.MICROSTRUCTURE)
        
        # Algorithmic noise (regular patterns)
        prices = [tick.price for tick in tick_data]
        if self._detect_algorithmic_patterns(prices):
            noise_types.append(NoiseType.ALGORITHMIC)
        
        # Liquidity noise (irregular spreads)
        if metrics.spread_stability < 0.7:
            noise_types.append(NoiseType.LIQUIDITY)
        
        # Volatility noise (excessive price movements)
        if metrics.price_volatility > 0.01:  # 1% volatility threshold
            noise_types.append(NoiseType.VOLATILITY)
        
        # Spread noise (inconsistent bid-ask spreads)
        spreads = [tick.spread for tick in tick_data]
        if np.std(spreads) / max(np.mean(spreads), 1e-10) > 0.5:
            noise_types.append(NoiseType.SPREAD)
        
        return noise_types
    
    def _detect_algorithmic_patterns(self, prices: List[float]) -> bool:
        """Detect algorithmic trading patterns"""
        if len(prices) < 20:
            return False
        
        # Check for regular oscillations
        price_array = np.array(prices)
        autocorr = np.correlate(price_array, price_array, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Look for periodic patterns
        peaks = scipy_signal.find_peaks(autocorr[1:10])[0]  # Check first 10 lags
        
        return len(peaks) > 2  # Multiple peaks suggest algorithmic patterns
    
    async def _kalman_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply Kalman filtering to remove noise"""
        if len(tick_data) < 5:
            return tick_data
        
        prices = np.array([tick.price for tick in tick_data])
        
        # Simple Kalman filter implementation
        n = len(prices)
        filtered_prices = np.zeros(n)
        
        # Initialize
        x = prices[0]  # Initial state
        P = 1.0        # Initial covariance
        Q = noise_threshold ** 2  # Process noise
        R = (noise_threshold * 2) ** 2  # Measurement noise
        
        filtered_prices[0] = x
        
        for i in range(1, n):
            # Predict
            x_pred = x
            P_pred = P + Q
            
            # Update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (prices[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered_prices[i] = x
        
        # Create filtered tick data
        filtered_ticks = []
        for i, tick in enumerate(tick_data):
            filtered_tick = TickData(
                timestamp=tick.timestamp,
                bid=tick.bid,
                ask=tick.ask,
                price=filtered_prices[i],
                volume=tick.volume,
                spread=tick.spread
            )
            filtered_ticks.append(filtered_tick)
        
        return filtered_ticks
    
    async def _wavelet_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply wavelet denoising"""
        try:
            import pywt
            
            prices = np.array([tick.price for tick in tick_data])
            
            # Wavelet denoising
            coeffs = pywt.wavedec(prices, 'db4', level=3)
            
            # Threshold coefficients
            threshold = noise_threshold * np.sqrt(2 * np.log(len(prices)))
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            
            # Reconstruct signal
            filtered_prices = pywt.waverec(coeffs_thresh, 'db4')
            
            # Ensure same length
            if len(filtered_prices) != len(prices):
                filtered_prices = filtered_prices[:len(prices)]
            
            # Create filtered tick data
            filtered_ticks = []
            for i, tick in enumerate(tick_data):
                filtered_tick = TickData(
                    timestamp=tick.timestamp,
                    bid=tick.bid,
                    ask=tick.ask,
                    price=filtered_prices[i],
                    volume=tick.volume,
                    spread=tick.spread
                )
                filtered_ticks.append(filtered_tick)
            
            return filtered_ticks
            
        except ImportError:
            logger.warning("PyWavelets not available, using Kalman filter")
            return await self._kalman_filter(tick_data, noise_threshold)
    
    async def _pca_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply PCA-based noise filtering"""
        if len(tick_data) < 10:
            return tick_data
        
        # Create feature matrix
        features = np.array([
            [tick.price, tick.bid, tick.ask, tick.volume, tick.spread]
            for tick in tick_data
        ])
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Reconstruct with reduced noise
        features_reconstructed = self.pca.inverse_transform(features_pca)
        features_original = self.scaler.inverse_transform(features_reconstructed)
        
        # Create filtered tick data
        filtered_ticks = []
        for i, tick in enumerate(tick_data):
            filtered_tick = TickData(
                timestamp=tick.timestamp,
                bid=features_original[i, 1],
                ask=features_original[i, 2],
                price=features_original[i, 0],
                volume=max(0, features_original[i, 3]),
                spread=max(0.0001, features_original[i, 4])
            )
            filtered_ticks.append(filtered_tick)
        
        return filtered_ticks
    
    async def _ica_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply ICA-based noise filtering"""
        if len(tick_data) < 20:
            return tick_data
        
        # Create feature matrix
        features = np.array([
            [tick.price, tick.bid, tick.ask, tick.volume, tick.spread]
            for tick in tick_data
        ])
        
        # Apply ICA
        features_ica = self.ica.fit_transform(features)
        
        # Remove noisy components (keep first 3 components)
        features_ica_filtered = features_ica[:, :3]
        
        # Reconstruct (approximate)
        mixing_matrix = self.ica.mixing_[:, :3]
        features_reconstructed = features_ica_filtered @ mixing_matrix.T
        
        # Create filtered tick data
        filtered_ticks = []
        for i, tick in enumerate(tick_data):
            filtered_tick = TickData(
                timestamp=tick.timestamp,
                bid=features_reconstructed[i, 1],
                ask=features_reconstructed[i, 2],
                price=features_reconstructed[i, 0],
                volume=max(0, features_reconstructed[i, 3]),
                spread=max(0.0001, features_reconstructed[i, 4])
            )
            filtered_ticks.append(filtered_tick)
        
        return filtered_ticks
    
    async def _isolation_forest_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply Isolation Forest for outlier removal"""
        if len(tick_data) < 20:
            return tick_data
        
        # Create feature matrix
        features = np.array([
            [tick.price, tick.volume, tick.spread]
            for tick in tick_data
        ])
        
        # Detect outliers
        outlier_labels = self.isolation_forest.fit_predict(features)
        
        # Keep only inliers
        filtered_ticks = [
            tick for i, tick in enumerate(tick_data)
            if outlier_labels[i] == 1
        ]
        
        return filtered_ticks
    
    async def _dbscan_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply DBSCAN clustering for noise removal"""
        if len(tick_data) < 10:
            return tick_data
        
        # Create feature matrix
        features = np.array([
            [tick.price, tick.volume]
            for tick in tick_data
        ])
        
        # Standardize features
        features_scaled = StandardScaler().fit_transform(features)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Keep points in the largest cluster
        if len(set(cluster_labels)) > 1:
            largest_cluster = max(set(cluster_labels), key=list(cluster_labels).count)
            filtered_ticks = [
                tick for i, tick in enumerate(tick_data)
                if cluster_labels[i] == largest_cluster
            ]
        else:
            filtered_ticks = tick_data
        
        return filtered_ticks
    
    async def _adaptive_filter(
        self,
        tick_data: List[TickData],
        noise_threshold: float
    ) -> List[TickData]:
        """Apply adaptive filtering based on data characteristics"""
        # Analyze data to choose best method
        metrics = await self._analyze_noise(tick_data)
        
        if metrics.signal_to_noise_ratio < 1.0:
            # High noise - use Kalman filter
            return await self._kalman_filter(tick_data, noise_threshold)
        elif len(tick_data) > 50:
            # Sufficient data - use PCA
            return await self._pca_filter(tick_data, noise_threshold)
        else:
            # Limited data - use simple outlier removal
            return await self._isolation_forest_filter(tick_data, noise_threshold)
    
    def _calculate_filter_confidence(
        self,
        original: List[TickData],
        filtered: List[TickData],
        metrics: NoiseMetrics
    ) -> float:
        """Calculate confidence in filtering results"""
        if not filtered or len(filtered) == 0:
            return 0.0
        
        # Data retention ratio
        retention_ratio = len(filtered) / len(original)
        
        # Signal-to-noise improvement
        snr_factor = min(1.0, metrics.signal_to_noise_ratio / 5.0)
        
        # Combine factors
        confidence = (retention_ratio * 0.6 + snr_factor * 0.4)
        
        return max(0.0, min(1.0, confidence))
    
    def _create_passthrough_result(
        self,
        tick_data: List[TickData],
        method: FilterMethod
    ) -> FilteredData:
        """Create passthrough result when filtering fails"""
        return FilteredData(
            original_data=tick_data,
            filtered_data=tick_data,
            noise_level=0.5,
            confidence=0.3,
            filter_method=method,
            noise_types_detected=[],
            timestamp=datetime.now()
        )
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering performance statistics"""
        if not self.filter_history:
            return {}
        
        recent_filters = self.filter_history[-100:]
        
        return {
            'total_filters': len(recent_filters),
            'average_confidence': np.mean([f.confidence for f in recent_filters]),
            'average_noise_level': np.mean([f.noise_level for f in recent_filters]),
            'method_distribution': {
                method.value: len([f for f in recent_filters if f.filter_method == method])
                for method in FilterMethod
            },
            'noise_type_frequency': {
                noise_type.value: sum(1 for f in recent_filters for nt in f.noise_types_detected if nt == noise_type)
                for noise_type in NoiseType
            }
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_noise_filter():
        noise_filter = NoiseFilter()
        
        # Create test data with noise
        base_price = 1.2500
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(100)]
        
        test_data = []
        for i, ts in enumerate(timestamps):
            # Add some noise
            noise = np.random.normal(0, 0.0001)
            price = base_price + 0.0001 * np.sin(i * 0.1) + noise
            
            tick = TickData(
                timestamp=ts,
                bid=price - 0.00010,
                ask=price + 0.00010,
                price=price,
                volume=1000 + np.random.randint(-100, 100),
                spread=0.00020
            )
            test_data.append(tick)
        
        # Filter noise
        result = await noise_filter.filter_noise(test_data, FilterMethod.ADAPTIVE)
        
        print(f"Original ticks: {len(result.original_data)}")
        print(f"Filtered ticks: {len(result.filtered_data)}")
        print(f"Noise level: {result.noise_level:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Filter method: {result.filter_method.value}")
        print(f"Noise types: {[nt.value for nt in result.noise_types_detected]}")
    
    # Run test
    asyncio.run(test_noise_filter())
