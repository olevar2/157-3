"""
Volatility Spikes Detector Module
Sudden volatility changes detection for quick profits in day trading.
Optimized for M15-H1 timeframes with real-time volatility monitoring.
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
import math


@dataclass
class VolatilityData:
    """Volatility measurement data"""
    timestamp: float
    symbol: str
    current_volatility: float
    average_volatility: float
    volatility_ratio: float  # current/average
    volatility_percentile: float  # 0-100
    spike_detected: bool


@dataclass
class VolatilitySpike:
    """Volatility spike event data"""
    timestamp: float
    symbol: str
    spike_magnitude: float  # Multiple of average volatility
    duration: int  # Number of periods
    price_change: float
    volume_increase: float
    spike_type: str  # 'sudden', 'gradual', 'extreme'
    direction: str  # 'up', 'down', 'both'


@dataclass
class VolatilitySignal:
    """Volatility-based trading signal"""
    timestamp: float
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    volatility_context: str  # 'spike_start', 'spike_continuation', 'spike_end'
    strength: float  # 0-100
    confidence: float  # 0-1
    expected_duration: int  # Expected periods for opportunity
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class VolatilitySpikesResult:
    """Complete volatility spikes analysis result"""
    symbol: str
    timestamp: float
    current_volatility: VolatilityData
    detected_spikes: List[VolatilitySpike]
    signals: List[VolatilitySignal]
    volatility_forecast: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_metrics: Dict[str, float]


class VolatilitySpikesDetector:
    """
    Volatility Spikes Detector Engine for Day Trading
    Provides real-time volatility spike detection and trading opportunities
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for volatility detection
        self.volatility_window = 20  # Periods for volatility calculation
        self.spike_threshold = 2.0  # Multiple of average volatility for spike detection
        self.extreme_spike_threshold = 3.5  # Extreme spike threshold
        self.min_spike_duration = 2  # Minimum periods for spike confirmation
        self.max_spike_duration = 10  # Maximum expected spike duration
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.volatility_history: Dict[str, deque] = {}
        self.spike_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.detection_count = 0
        self.total_detection_time = 0.0

    async def initialize(self) -> bool:
        """Initialize volatility spikes detector engine"""
        try:
            self.logger.info("Initializing Volatility Spikes Detector Engine...")
            
            # Test volatility calculation
            test_prices = [1.1000, 1.1005, 1.1002, 1.1015, 1.0995, 1.1020, 1.0990]
            test_volatility = self._calculate_volatility(test_prices)
            
            if test_volatility is not None and test_volatility >= 0:
                self.ready = True
                self.logger.info("✅ Volatility Spikes Detector Engine initialized successfully")
                return True
            else:
                raise ValueError("Volatility calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Volatility Spikes Detector Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def detect_volatility_spikes(self, symbol: str, price_data: List[Dict], 
                                     volume_data: List[Dict] = None) -> VolatilitySpikesResult:
        """
        Main volatility spikes detection function
        """
        if not self.ready:
            raise RuntimeError("Volatility Spikes Detector Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.price_history:
                self._initialize_symbol_buffers(symbol)
            
            # Extract price and volume data
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
            
            volumes = []
            if volume_data:
                volumes = [float(data.get('volume', 0)) for data in volume_data]
            else:
                volumes = [float(data.get('volume', 0)) for data in price_data]
            
            # Calculate current volatility
            current_volatility = await self._calculate_current_volatility(symbol, closes, highs, lows)
            
            # Detect volatility spikes
            detected_spikes = await self._detect_spikes(symbol, closes, highs, lows, volumes, timestamps)
            
            # Generate volatility-based signals
            signals = await self._generate_volatility_signals(symbol, closes[-1], current_volatility, detected_spikes)
            
            # Forecast volatility
            volatility_forecast = await self._forecast_volatility(symbol, current_volatility, detected_spikes)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(current_volatility, detected_spikes)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(signals, current_volatility)
            
            # Update performance tracking
            detection_time = time.time() - start_time
            self.detection_count += 1
            self.total_detection_time += detection_time
            
            return VolatilitySpikesResult(
                symbol=symbol,
                timestamp=time.time(),
                current_volatility=current_volatility,
                detected_spikes=detected_spikes,
                signals=signals,
                volatility_forecast=volatility_forecast,
                risk_metrics=risk_metrics,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Volatility spikes detection failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        buffer_size = 200  # Keep enough data for volatility analysis
        self.price_history[symbol] = deque(maxlen=buffer_size)
        self.volatility_history[symbol] = deque(maxlen=buffer_size)
        self.spike_history[symbol] = deque(maxlen=buffer_size)
        self.volume_history[symbol] = deque(maxlen=buffer_size)

    def _calculate_volatility(self, prices: List[float], method: str = 'returns') -> Optional[float]:
        """Calculate volatility using different methods"""
        if len(prices) < 2:
            return None
        
        if method == 'returns':
            # Calculate returns-based volatility
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            
            if len(returns) > 1:
                return statistics.stdev(returns)
            else:
                return 0.0
        
        elif method == 'range':
            # Calculate range-based volatility (simplified)
            ranges = []
            for i in range(1, len(prices)):
                price_range = abs(prices[i] - prices[i-1])
                ranges.append(price_range)
            
            if ranges:
                return statistics.mean(ranges)
            else:
                return 0.0
        
        return None

    async def _calculate_current_volatility(self, symbol: str, closes: List[float], 
                                          highs: List[float], lows: List[float]) -> VolatilityData:
        """Calculate current volatility metrics"""
        if len(closes) < self.volatility_window:
            return VolatilityData(time.time(), symbol, 0.0, 0.0, 1.0, 50.0, False)
        
        # Calculate current volatility (recent period)
        recent_closes = closes[-self.volatility_window:]
        current_vol = self._calculate_volatility(recent_closes, 'returns') or 0.0
        
        # Calculate average volatility (longer period)
        if len(closes) >= self.volatility_window * 2:
            avg_vol = self._calculate_volatility(closes[-self.volatility_window*2:], 'returns') or 0.0
        else:
            avg_vol = current_vol
        
        # Calculate volatility ratio
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Calculate volatility percentile
        if len(self.volatility_history[symbol]) >= 20:
            historical_vols = list(self.volatility_history[symbol])
            percentile = sum(1 for v in historical_vols if v <= current_vol) / len(historical_vols) * 100
        else:
            percentile = 50.0
        
        # Detect spike
        spike_detected = vol_ratio >= self.spike_threshold
        
        # Store current volatility
        self.volatility_history[symbol].append(current_vol)
        
        return VolatilityData(
            timestamp=time.time(),
            symbol=symbol,
            current_volatility=current_vol,
            average_volatility=avg_vol,
            volatility_ratio=vol_ratio,
            volatility_percentile=percentile,
            spike_detected=spike_detected
        )

    async def _detect_spikes(self, symbol: str, closes: List[float], highs: List[float], 
                           lows: List[float], volumes: List[float], timestamps: List[float]) -> List[VolatilitySpike]:
        """Detect volatility spikes in the data"""
        spikes = []
        
        if len(closes) < self.volatility_window + self.min_spike_duration:
            return spikes
        
        # Calculate rolling volatility
        volatilities = []
        for i in range(self.volatility_window, len(closes)):
            window_closes = closes[i-self.volatility_window:i]
            vol = self._calculate_volatility(window_closes, 'returns') or 0.0
            volatilities.append(vol)
        
        if len(volatilities) < self.min_spike_duration:
            return spikes
        
        # Calculate average volatility for comparison
        avg_volatility = statistics.mean(volatilities[:-self.min_spike_duration]) if len(volatilities) > self.min_spike_duration else statistics.mean(volatilities)
        
        # Detect spikes
        i = 0
        while i < len(volatilities) - self.min_spike_duration:
            current_vol = volatilities[i]
            
            # Check if current volatility exceeds threshold
            if current_vol >= avg_volatility * self.spike_threshold:
                # Found potential spike start
                spike_start = i
                spike_duration = 1
                max_vol = current_vol
                
                # Determine spike duration
                j = i + 1
                while j < len(volatilities) and j < i + self.max_spike_duration:
                    if volatilities[j] >= avg_volatility * self.spike_threshold:
                        spike_duration += 1
                        max_vol = max(max_vol, volatilities[j])
                        j += 1
                    else:
                        break
                
                # Confirm spike if duration meets minimum requirement
                if spike_duration >= self.min_spike_duration:
                    spike_magnitude = max_vol / avg_volatility
                    
                    # Calculate price change during spike
                    start_idx = spike_start + self.volatility_window
                    end_idx = min(start_idx + spike_duration, len(closes) - 1)
                    price_change = abs(closes[end_idx] - closes[start_idx])
                    
                    # Calculate volume increase
                    volume_increase = 1.0
                    if volumes and len(volumes) > end_idx:
                        spike_volumes = volumes[start_idx:end_idx+1]
                        avg_volume = statistics.mean(volumes[max(0, start_idx-10):start_idx]) if start_idx >= 10 else statistics.mean(volumes[:start_idx])
                        if avg_volume > 0:
                            current_avg_volume = statistics.mean(spike_volumes)
                            volume_increase = current_avg_volume / avg_volume
                    
                    # Determine spike type
                    if spike_magnitude >= self.extreme_spike_threshold:
                        spike_type = 'extreme'
                    elif spike_duration <= 3:
                        spike_type = 'sudden'
                    else:
                        spike_type = 'gradual'
                    
                    # Determine direction
                    direction = 'both'  # Default for volatility spikes
                    if end_idx < len(closes):
                        if closes[end_idx] > closes[start_idx]:
                            direction = 'up'
                        elif closes[end_idx] < closes[start_idx]:
                            direction = 'down'
                    
                    spike = VolatilitySpike(
                        timestamp=timestamps[start_idx] if start_idx < len(timestamps) else time.time(),
                        symbol=symbol,
                        spike_magnitude=spike_magnitude,
                        duration=spike_duration,
                        price_change=price_change,
                        volume_increase=volume_increase,
                        spike_type=spike_type,
                        direction=direction
                    )
                    
                    spikes.append(spike)
                    
                    # Skip ahead to avoid overlapping spikes
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        return spikes

    async def _generate_volatility_signals(self, symbol: str, current_price: float,
                                         current_volatility: VolatilityData,
                                         detected_spikes: List[VolatilitySpike]) -> List[VolatilitySignal]:
        """Generate trading signals based on volatility analysis"""
        signals = []
        
        # Determine volatility context
        volatility_context = 'normal'
        if current_volatility.spike_detected:
            volatility_context = 'spike_start'
        elif detected_spikes:
            # Check if we're in continuation of a recent spike
            recent_spike = detected_spikes[-1]
            time_since_spike = time.time() - recent_spike.timestamp
            if time_since_spike < 3600:  # Within 1 hour
                volatility_context = 'spike_continuation'
        
        # Generate signals based on volatility context
        signal_type = 'hold'
        strength = 0.0
        confidence = 0.5
        expected_duration = 5  # Default 5 periods
        
        if volatility_context == 'spike_start':
            # High volatility spike starting - potential breakout
            if current_volatility.volatility_ratio >= self.extreme_spike_threshold:
                signal_type = 'buy'  # Assume upward breakout (could be refined with price direction)
                strength = min(current_volatility.volatility_ratio * 30, 100)
                confidence = 0.7
                expected_duration = 3
            elif current_volatility.volatility_ratio >= self.spike_threshold:
                signal_type = 'buy'
                strength = min(current_volatility.volatility_ratio * 25, 100)
                confidence = 0.6
                expected_duration = 5
        
        elif volatility_context == 'spike_continuation':
            # In middle of volatility spike - trend continuation
            recent_spike = detected_spikes[-1]
            if recent_spike.direction == 'up':
                signal_type = 'buy'
                strength = min(recent_spike.spike_magnitude * 20, 100)
                confidence = 0.6
            elif recent_spike.direction == 'down':
                signal_type = 'sell'
                strength = min(recent_spike.spike_magnitude * 20, 100)
                confidence = 0.6
            expected_duration = max(1, recent_spike.duration - 2)
        
        # Calculate stop loss and take profit based on volatility
        stop_loss, take_profit = self._calculate_volatility_levels(current_price, signal_type, current_volatility)
        
        signal = VolatilitySignal(
            timestamp=time.time(),
            symbol=symbol,
            signal_type=signal_type,
            volatility_context=volatility_context,
            strength=strength,
            confidence=confidence,
            expected_duration=expected_duration,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        signals.append(signal)
        return signals

    def _calculate_volatility_levels(self, price: float, signal_type: str, 
                                   volatility_data: VolatilityData) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on volatility"""
        # Base pip value
        pip_value = 0.0001
        
        # Adjust levels based on volatility
        volatility_multiplier = max(1.0, volatility_data.volatility_ratio)
        
        if signal_type == 'buy':
            stop_loss = price - (15 * pip_value * volatility_multiplier)  # Wider stops in high volatility
            take_profit = price + (30 * pip_value * volatility_multiplier)  # Wider targets
        elif signal_type == 'sell':
            stop_loss = price + (15 * pip_value * volatility_multiplier)
            take_profit = price - (30 * pip_value * volatility_multiplier)
        else:
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit

    async def _forecast_volatility(self, symbol: str, current_volatility: VolatilityData,
                                 detected_spikes: List[VolatilitySpike]) -> Dict[str, float]:
        """Forecast future volatility based on current conditions"""
        forecast = {}
        
        # Base volatility forecast
        base_forecast = current_volatility.average_volatility
        
        # Adjust based on current spike conditions
        if current_volatility.spike_detected:
            # High volatility likely to continue short-term but revert medium-term
            forecast['next_1_period'] = current_volatility.current_volatility * 0.8
            forecast['next_5_periods'] = current_volatility.average_volatility * 1.2
            forecast['next_10_periods'] = current_volatility.average_volatility
        else:
            # Normal volatility conditions
            forecast['next_1_period'] = current_volatility.current_volatility
            forecast['next_5_periods'] = current_volatility.average_volatility
            forecast['next_10_periods'] = current_volatility.average_volatility
        
        # Spike probability
        if detected_spikes:
            recent_spikes = [s for s in detected_spikes if time.time() - s.timestamp < 3600]  # Last hour
            spike_frequency = len(recent_spikes)
            forecast['spike_probability_1h'] = min(spike_frequency * 0.2, 1.0)
        else:
            forecast['spike_probability_1h'] = 0.1  # Base probability
        
        # Volatility regime
        if current_volatility.volatility_percentile > 80:
            forecast['volatility_regime'] = 1.0  # High volatility regime
        elif current_volatility.volatility_percentile < 20:
            forecast['volatility_regime'] = 0.0  # Low volatility regime
        else:
            forecast['volatility_regime'] = 0.5  # Normal regime
        
        return forecast

    async def _calculate_risk_metrics(self, current_volatility: VolatilityData,
                                    detected_spikes: List[VolatilitySpike]) -> Dict[str, float]:
        """Calculate risk metrics based on volatility analysis"""
        risk_metrics = {}
        
        # Volatility risk score
        vol_risk = min(current_volatility.volatility_ratio, 5.0) / 5.0  # Normalize to 0-1
        risk_metrics['volatility_risk_score'] = vol_risk
        
        # Spike risk
        if detected_spikes:
            recent_spike = detected_spikes[-1]
            spike_risk = min(recent_spike.spike_magnitude / 5.0, 1.0)  # Normalize
            risk_metrics['spike_risk_score'] = spike_risk
        else:
            risk_metrics['spike_risk_score'] = 0.0
        
        # Overall risk level
        overall_risk = (vol_risk * 0.6 + risk_metrics['spike_risk_score'] * 0.4)
        risk_metrics['overall_risk_level'] = overall_risk
        
        # Risk category
        if overall_risk > 0.7:
            risk_metrics['risk_category'] = 'high'
        elif overall_risk > 0.4:
            risk_metrics['risk_category'] = 'medium'
        else:
            risk_metrics['risk_category'] = 'low'
        
        # Position sizing recommendation
        risk_metrics['position_size_factor'] = max(0.2, 1.0 - overall_risk)
        
        return risk_metrics

    async def _calculate_execution_metrics(self, signals: List[VolatilitySignal],
                                         current_volatility: VolatilityData) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        return {
            'signal_strength': latest_signal.strength,
            'signal_confidence': latest_signal.confidence,
            'expected_duration': latest_signal.expected_duration,
            'volatility_ratio': current_volatility.volatility_ratio,
            'volatility_percentile': current_volatility.volatility_percentile,
            'spike_detected': 1.0 if current_volatility.spike_detected else 0.0,
            'analysis_speed_ms': (self.total_detection_time / self.detection_count * 1000) 
                               if self.detection_count > 0 else 0
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'total_detections': self.detection_count,
            'average_detection_time_ms': (self.total_detection_time / self.detection_count * 1000) 
                                       if self.detection_count > 0 else 0,
            'detections_per_second': self.detection_count / self.total_detection_time 
                                   if self.total_detection_time > 0 else 0
        }
