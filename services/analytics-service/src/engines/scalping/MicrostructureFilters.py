"""
Microstructure Filters Module
Noise filtering for M1 data and market microstructure analysis for scalping.
Provides ultra-fast noise detection and filtering for clean scalping signals.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics
from scipy import signal as scipy_signal


@dataclass
class NoiseMetrics:
    """Market noise measurement metrics"""
    timestamp: float
    symbol: str
    noise_level: float  # 0-100 scale
    signal_to_noise_ratio: float
    volatility_noise: float
    price_efficiency: float
    microstructure_quality: str  # 'clean', 'noisy', 'very_noisy'


@dataclass
class FilteredSignal:
    """Filtered market signal"""
    timestamp: float
    original_price: float
    filtered_price: float
    noise_removed: float
    confidence: float  # 0-1
    filter_type: str


@dataclass
class MicrostructureAnalysis:
    """Complete microstructure analysis result"""
    symbol: str
    timestamp: float
    noise_metrics: NoiseMetrics
    filtered_signals: List[FilteredSignal]
    market_quality: Dict[str, float]
    trading_conditions: Dict[str, str]
    execution_recommendations: Dict[str, float]


class MicrostructureFilters:
    """
    Market Microstructure Filters for Scalping
    Provides noise filtering and market quality analysis for M1 data
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for noise filtering
        self.noise_threshold = 0.3  # Noise level threshold
        self.filter_window = 20  # Moving window for filtering
        self.volatility_window = 50  # Window for volatility calculation
        self.efficiency_threshold = 0.7  # Price efficiency threshold
        
        # Filter parameters
        self.kalman_q = 0.001  # Process noise
        self.kalman_r = 0.01   # Measurement noise
        self.ema_alpha = 0.3   # EMA smoothing factor
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.noise_history: Dict[str, deque] = {}
        self.filtered_history: Dict[str, deque] = {}
        
        # Kalman filter states
        self.kalman_states: Dict[str, Dict] = {}
        
        # Performance tracking
        self.filter_count = 0
        self.total_filter_time = 0.0

    async def initialize(self) -> bool:
        """Initialize microstructure filters engine"""
        try:
            self.logger.info("Initializing Microstructure Filters Engine...")
            
            # Test noise calculation
            test_prices = [1.1000, 1.1002, 1.0999, 1.1003, 1.1001, 1.1004, 1.0998]
            test_noise = self._calculate_noise_level(test_prices)
            
            if test_noise is not None:
                self.ready = True
                self.logger.info("✅ Microstructure Filters Engine initialized successfully")
                return True
            else:
                raise ValueError("Noise calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Microstructure Filters Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_microstructure(self, symbol: str, price_data: List[Dict], 
                                   volume_data: List[Dict] = None) -> MicrostructureAnalysis:
        """
        Main microstructure analysis function
        """
        if not self.ready:
            raise RuntimeError("Microstructure Filters Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.price_history:
                self._initialize_symbol_buffers(symbol)
            
            # Extract price series
            prices = [float(data.get('close', 0)) for data in price_data]
            
            # Calculate noise metrics
            noise_metrics = await self._calculate_noise_metrics(symbol, prices)
            
            # Apply various filters
            filtered_signals = await self._apply_filters(symbol, prices)
            
            # Analyze market quality
            market_quality = await self._analyze_market_quality(prices, volume_data)
            
            # Determine trading conditions
            trading_conditions = await self._assess_trading_conditions(noise_metrics, market_quality)
            
            # Generate execution recommendations
            execution_recommendations = await self._generate_execution_recommendations(
                noise_metrics, market_quality, trading_conditions
            )
            
            # Update performance tracking
            filter_time = time.time() - start_time
            self.filter_count += 1
            self.total_filter_time += filter_time
            
            return MicrostructureAnalysis(
                symbol=symbol,
                timestamp=time.time(),
                noise_metrics=noise_metrics,
                filtered_signals=filtered_signals,
                market_quality=market_quality,
                trading_conditions=trading_conditions,
                execution_recommendations=execution_recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Microstructure analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.price_history[symbol] = deque(maxlen=1000)
        self.noise_history[symbol] = deque(maxlen=1000)
        self.filtered_history[symbol] = deque(maxlen=1000)
        
        # Initialize Kalman filter state
        self.kalman_states[symbol] = {
            'x': 0.0,  # State estimate
            'P': 1.0,  # Error covariance
            'initialized': False
        }

    def _calculate_noise_level(self, prices: List[float]) -> Optional[float]:
        """Calculate market noise level"""
        if len(prices) < 10:
            return None
        
        # Calculate price changes
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Calculate absolute price changes
        abs_changes = [abs(change) for change in price_changes]
        
        # Calculate noise as ratio of small changes to total changes
        small_changes = [change for change in abs_changes if change < statistics.mean(abs_changes) * 0.5]
        
        if len(abs_changes) > 0:
            noise_ratio = len(small_changes) / len(abs_changes)
            return noise_ratio * 100  # Convert to percentage
        
        return None

    async def _calculate_noise_metrics(self, symbol: str, prices: List[float]) -> NoiseMetrics:
        """Calculate comprehensive noise metrics"""
        # Basic noise level
        noise_level = self._calculate_noise_level(prices) or 0.0
        
        # Signal-to-noise ratio
        if len(prices) >= self.volatility_window:
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            signal_strength = abs(statistics.mean(price_changes)) if price_changes else 0
            noise_strength = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
            
            snr = signal_strength / noise_strength if noise_strength > 0 else 0
        else:
            snr = 0.0
        
        # Volatility-based noise
        volatility_noise = self._calculate_volatility_noise(prices)
        
        # Price efficiency (how much price moves in trending direction)
        price_efficiency = self._calculate_price_efficiency(prices)
        
        # Determine market quality
        if noise_level < 20 and price_efficiency > 0.7:
            quality = 'clean'
        elif noise_level < 50 and price_efficiency > 0.5:
            quality = 'noisy'
        else:
            quality = 'very_noisy'
        
        return NoiseMetrics(
            timestamp=time.time(),
            symbol=symbol,
            noise_level=noise_level,
            signal_to_noise_ratio=snr,
            volatility_noise=volatility_noise,
            price_efficiency=price_efficiency,
            microstructure_quality=quality
        )

    def _calculate_volatility_noise(self, prices: List[float]) -> float:
        """Calculate volatility-based noise measure"""
        if len(prices) < 20:
            return 0.0
        
        # Calculate rolling volatility
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices)) if prices[i-1] != 0]
        
        if len(returns) < 10:
            return 0.0
        
        # Calculate volatility of volatility (noise in volatility)
        window_size = min(10, len(returns) // 2)
        volatilities = []
        
        for i in range(window_size, len(returns)):
            window_returns = returns[i-window_size:i]
            vol = statistics.stdev(window_returns) if len(window_returns) > 1 else 0
            volatilities.append(vol)
        
        if len(volatilities) > 1:
            vol_of_vol = statistics.stdev(volatilities)
            avg_vol = statistics.mean(volatilities)
            return (vol_of_vol / avg_vol * 100) if avg_vol > 0 else 0
        
        return 0.0

    def _calculate_price_efficiency(self, prices: List[float]) -> float:
        """Calculate price efficiency (trending vs random walk)"""
        if len(prices) < 20:
            return 0.5
        
        # Calculate net price movement
        net_movement = abs(prices[-1] - prices[0])
        
        # Calculate total price movement (sum of absolute changes)
        total_movement = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))
        
        # Efficiency is ratio of net to total movement
        efficiency = net_movement / total_movement if total_movement > 0 else 0
        
        return min(efficiency, 1.0)

    async def _apply_filters(self, symbol: str, prices: List[float]) -> List[FilteredSignal]:
        """Apply various noise filters to price data"""
        filtered_signals = []
        
        if len(prices) < 5:
            return filtered_signals
        
        current_price = prices[-1]
        
        # 1. Exponential Moving Average Filter
        ema_filtered = self._apply_ema_filter(prices)
        if ema_filtered is not None:
            noise_removed = abs(current_price - ema_filtered)
            confidence = 1.0 - min(noise_removed / current_price, 0.5) if current_price > 0 else 0.5
            
            filtered_signals.append(FilteredSignal(
                timestamp=time.time(),
                original_price=current_price,
                filtered_price=ema_filtered,
                noise_removed=noise_removed,
                confidence=confidence,
                filter_type='EMA'
            ))
        
        # 2. Kalman Filter
        kalman_filtered = self._apply_kalman_filter(symbol, current_price)
        if kalman_filtered is not None:
            noise_removed = abs(current_price - kalman_filtered)
            confidence = 1.0 - min(noise_removed / current_price, 0.3) if current_price > 0 else 0.7
            
            filtered_signals.append(FilteredSignal(
                timestamp=time.time(),
                original_price=current_price,
                filtered_price=kalman_filtered,
                noise_removed=noise_removed,
                confidence=confidence,
                filter_type='Kalman'
            ))
        
        # 3. Median Filter
        median_filtered = self._apply_median_filter(prices)
        if median_filtered is not None:
            noise_removed = abs(current_price - median_filtered)
            confidence = 1.0 - min(noise_removed / current_price, 0.4) if current_price > 0 else 0.6
            
            filtered_signals.append(FilteredSignal(
                timestamp=time.time(),
                original_price=current_price,
                filtered_price=median_filtered,
                noise_removed=noise_removed,
                confidence=confidence,
                filter_type='Median'
            ))
        
        return filtered_signals

    def _apply_ema_filter(self, prices: List[float]) -> Optional[float]:
        """Apply Exponential Moving Average filter"""
        if not prices:
            return None
        
        ema = prices[0]
        for price in prices[1:]:
            ema = self.ema_alpha * price + (1 - self.ema_alpha) * ema
        
        return ema

    def _apply_kalman_filter(self, symbol: str, measurement: float) -> Optional[float]:
        """Apply Kalman filter for noise reduction"""
        state = self.kalman_states[symbol]
        
        if not state['initialized']:
            state['x'] = measurement
            state['P'] = 1.0
            state['initialized'] = True
            return measurement
        
        # Prediction step
        x_pred = state['x']  # No process model, just maintain state
        P_pred = state['P'] + self.kalman_q
        
        # Update step
        K = P_pred / (P_pred + self.kalman_r)  # Kalman gain
        state['x'] = x_pred + K * (measurement - x_pred)
        state['P'] = (1 - K) * P_pred
        
        return state['x']

    def _apply_median_filter(self, prices: List[float]) -> Optional[float]:
        """Apply median filter for spike removal"""
        if len(prices) < 5:
            return None
        
        window_size = min(5, len(prices))
        window = prices[-window_size:]
        
        return statistics.median(window)

    async def _analyze_market_quality(self, prices: List[float], 
                                    volume_data: List[Dict] = None) -> Dict[str, float]:
        """Analyze overall market quality metrics"""
        quality_metrics = {}
        
        # Price-based quality metrics
        if len(prices) >= 20:
            # Bid-ask spread proxy (using price volatility)
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            avg_spread_proxy = statistics.mean(price_changes) if price_changes else 0
            
            # Market depth proxy (inverse of volatility)
            volatility = statistics.stdev(price_changes) if len(price_changes) > 1 else 0
            depth_proxy = 1 / (1 + volatility) if volatility > 0 else 1
            
            quality_metrics.update({
                'spread_proxy': avg_spread_proxy,
                'depth_proxy': depth_proxy,
                'price_stability': 1 - min(volatility, 1),
                'trend_consistency': self._calculate_price_efficiency(prices)
            })
        
        # Volume-based quality metrics (if available)
        if volume_data and len(volume_data) >= 10:
            volumes = [float(data.get('volume', 0)) for data in volume_data]
            avg_volume = statistics.mean(volumes) if volumes else 0
            volume_consistency = 1 - (statistics.stdev(volumes) / avg_volume) if avg_volume > 0 else 0
            
            quality_metrics.update({
                'volume_consistency': max(0, min(volume_consistency, 1)),
                'liquidity_proxy': min(avg_volume / 1000, 1)  # Normalized liquidity
            })
        
        return quality_metrics

    async def _assess_trading_conditions(self, noise_metrics: NoiseMetrics, 
                                       market_quality: Dict[str, float]) -> Dict[str, str]:
        """Assess current trading conditions"""
        conditions = {}
        
        # Overall market condition
        if noise_metrics.microstructure_quality == 'clean':
            conditions['market_condition'] = 'excellent'
        elif noise_metrics.microstructure_quality == 'noisy':
            conditions['market_condition'] = 'acceptable'
        else:
            conditions['market_condition'] = 'poor'
        
        # Scalping suitability
        if noise_metrics.noise_level < 25 and noise_metrics.price_efficiency > 0.6:
            conditions['scalping_suitability'] = 'high'
        elif noise_metrics.noise_level < 50 and noise_metrics.price_efficiency > 0.4:
            conditions['scalping_suitability'] = 'medium'
        else:
            conditions['scalping_suitability'] = 'low'
        
        # Execution quality expectation
        spread_proxy = market_quality.get('spread_proxy', 0.001)
        if spread_proxy < 0.0005:  # Very tight spreads
            conditions['execution_quality'] = 'excellent'
        elif spread_proxy < 0.001:
            conditions['execution_quality'] = 'good'
        else:
            conditions['execution_quality'] = 'fair'
        
        return conditions

    async def _generate_execution_recommendations(self, noise_metrics: NoiseMetrics,
                                                market_quality: Dict[str, float],
                                                trading_conditions: Dict[str, str]) -> Dict[str, float]:
        """Generate execution recommendations based on microstructure analysis"""
        recommendations = {}
        
        # Position sizing recommendation (0-1 scale)
        if trading_conditions.get('market_condition') == 'excellent':
            recommendations['position_size_factor'] = 1.0
        elif trading_conditions.get('market_condition') == 'acceptable':
            recommendations['position_size_factor'] = 0.7
        else:
            recommendations['position_size_factor'] = 0.3
        
        # Stop loss adjustment factor
        noise_adjustment = 1 + (noise_metrics.noise_level / 100)
        recommendations['stop_loss_multiplier'] = noise_adjustment
        
        # Take profit adjustment factor
        efficiency_factor = noise_metrics.price_efficiency
        recommendations['take_profit_multiplier'] = 0.8 + (efficiency_factor * 0.4)
        
        # Entry timing confidence
        if noise_metrics.signal_to_noise_ratio > 2:
            recommendations['entry_confidence'] = 0.9
        elif noise_metrics.signal_to_noise_ratio > 1:
            recommendations['entry_confidence'] = 0.7
        else:
            recommendations['entry_confidence'] = 0.5
        
        # Market timing score
        timing_score = (noise_metrics.price_efficiency * 0.4 + 
                       (1 - noise_metrics.noise_level / 100) * 0.6)
        recommendations['market_timing_score'] = max(0, min(timing_score, 1))
        
        return recommendations

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'total_filters_applied': self.filter_count,
            'average_filter_time_ms': (self.total_filter_time / self.filter_count * 1000) 
                                    if self.filter_count > 0 else 0,
            'filters_per_second': self.filter_count / self.total_filter_time 
                                if self.total_filter_time > 0 else 0
        }
