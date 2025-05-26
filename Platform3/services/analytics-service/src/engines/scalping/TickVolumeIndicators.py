"""
Tick Volume Indicators Module
Tick volume momentum analysis for M1-M5 scalping strategies.
Provides ultra-fast tick volume calculations and momentum signals for daily profit focus.
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


@dataclass
class TickVolumeData:
    """Individual tick volume data point"""
    timestamp: float
    symbol: str
    price: float
    volume: int
    tick_direction: str  # 'up', 'down', 'neutral'
    cumulative_volume: int
    volume_rate: float  # ticks per second


@dataclass
class VolumeIndicator:
    """Volume-based indicator result"""
    name: str
    value: float
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-100
    confidence: float  # 0-1


@dataclass
class TickVolumeSignal:
    """Tick volume-based scalping signal"""
    timestamp: float
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    volume_momentum: float
    tick_pressure: str  # 'buying', 'selling', 'neutral'
    volume_spike: bool
    entry_confidence: float


@dataclass
class TickVolumeResult:
    """Complete tick volume analysis result"""
    symbol: str
    timestamp: float
    current_volume_rate: float
    volume_indicators: List[VolumeIndicator]
    momentum_analysis: Dict[str, float]
    signals: List[TickVolumeSignal]
    volume_profile: Dict[str, float]
    execution_metrics: Dict[str, float]


class TickVolumeIndicators:
    """
    Tick Volume Indicators Engine for Scalping
    Provides tick volume momentum analysis optimized for M1-M5 timeframes
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for tick volume analysis
        self.volume_periods = [10, 20, 50]  # Different periods for volume analysis
        self.momentum_threshold = 1.5  # Volume momentum threshold
        self.spike_threshold = 2.0  # Volume spike detection threshold
        self.tick_buffer_size = 1000  # Keep last 1000 ticks
        
        # Data storage
        self.tick_data: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.momentum_history: Dict[str, deque] = {}
        self.signals_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

    async def initialize(self) -> bool:
        """Initialize tick volume indicators engine"""
        try:
            self.logger.info("Initializing Tick Volume Indicators Engine...")
            
            # Test volume calculations
            test_volumes = [100, 150, 200, 120, 300, 180, 250]
            test_momentum = self._calculate_volume_momentum(test_volumes, 5)
            
            if test_momentum is not None:
                self.ready = True
                self.logger.info("✅ Tick Volume Indicators Engine initialized successfully")
                return True
            else:
                raise ValueError("Volume momentum calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Tick Volume Indicators Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_tick_volume(self, symbol: str, tick_data: List[Dict]) -> TickVolumeResult:
        """
        Main tick volume analysis function
        """
        if not self.ready:
            raise RuntimeError("Tick Volume Indicators Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.tick_data:
                self._initialize_symbol_buffers(symbol)
            
            # Process tick data
            processed_ticks = await self._process_tick_data(symbol, tick_data)
            
            # Calculate volume indicators
            volume_indicators = await self._calculate_volume_indicators(symbol, processed_ticks)
            
            # Analyze volume momentum
            momentum_analysis = await self._analyze_volume_momentum(processed_ticks)
            
            # Generate volume-based signals
            signals = await self._generate_volume_signals(symbol, processed_ticks, momentum_analysis)
            
            # Create volume profile
            volume_profile = await self._create_volume_profile(processed_ticks)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(signals, processed_ticks)
            
            # Update performance tracking
            calculation_time = time.time() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            return TickVolumeResult(
                symbol=symbol,
                timestamp=time.time(),
                current_volume_rate=processed_ticks[-1].volume_rate if processed_ticks else 0.0,
                volume_indicators=volume_indicators,
                momentum_analysis=momentum_analysis,
                signals=signals,
                volume_profile=volume_profile,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Tick volume analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.tick_data[symbol] = deque(maxlen=self.tick_buffer_size)
        self.volume_history[symbol] = deque(maxlen=self.tick_buffer_size)
        self.momentum_history[symbol] = deque(maxlen=self.tick_buffer_size)
        self.signals_history[symbol] = deque(maxlen=self.tick_buffer_size)

    async def _process_tick_data(self, symbol: str, tick_data: List[Dict]) -> List[TickVolumeData]:
        """Process raw tick data into structured format"""
        processed_ticks = []
        cumulative_volume = 0
        
        for i, tick in enumerate(tick_data):
            timestamp = float(tick.get('timestamp', time.time()))
            price = float(tick.get('price', 0))
            volume = int(tick.get('volume', 0))
            
            # Determine tick direction
            tick_direction = 'neutral'
            if i > 0:
                prev_price = float(tick_data[i-1].get('price', 0))
                if price > prev_price:
                    tick_direction = 'up'
                elif price < prev_price:
                    tick_direction = 'down'
            
            cumulative_volume += volume
            
            # Calculate volume rate (simplified)
            volume_rate = volume  # In real implementation, this would be ticks per second
            
            tick_volume_data = TickVolumeData(
                timestamp=timestamp,
                symbol=symbol,
                price=price,
                volume=volume,
                tick_direction=tick_direction,
                cumulative_volume=cumulative_volume,
                volume_rate=volume_rate
            )
            
            processed_ticks.append(tick_volume_data)
            
            # Store in buffer
            self.tick_data[symbol].append(tick_volume_data)
        
        return processed_ticks

    def _calculate_volume_momentum(self, volumes: List[float], period: int) -> Optional[float]:
        """Calculate volume momentum for given period"""
        if len(volumes) < period * 2:
            return None
        
        recent_avg = statistics.mean(volumes[-period:])
        previous_avg = statistics.mean(volumes[-period*2:-period])
        
        if previous_avg > 0:
            return (recent_avg - previous_avg) / previous_avg
        
        return None

    async def _calculate_volume_indicators(self, symbol: str, 
                                         processed_ticks: List[TickVolumeData]) -> List[VolumeIndicator]:
        """Calculate various volume-based indicators"""
        indicators = []
        
        if len(processed_ticks) < max(self.volume_periods):
            return indicators
        
        volumes = [tick.volume for tick in processed_ticks]
        
        # Volume Moving Average indicators
        for period in self.volume_periods:
            if len(volumes) >= period:
                vma = statistics.mean(volumes[-period:])
                current_volume = volumes[-1]
                
                # Determine signal
                signal = 'neutral'
                strength = 0.0
                confidence = 0.5
                
                if current_volume > vma * 1.2:  # 20% above average
                    signal = 'bullish'
                    strength = min((current_volume / vma - 1) * 100, 100)
                    confidence = 0.7
                elif current_volume < vma * 0.8:  # 20% below average
                    signal = 'bearish'
                    strength = min((1 - current_volume / vma) * 100, 100)
                    confidence = 0.6
                
                indicator = VolumeIndicator(
                    name=f'VMA_{period}',
                    value=vma,
                    signal=signal,
                    strength=strength,
                    confidence=confidence
                )
                indicators.append(indicator)
        
        # Volume Rate of Change (VROC)
        if len(volumes) >= 20:
            current_volume = volumes[-1]
            past_volume = volumes[-20]
            
            if past_volume > 0:
                vroc = ((current_volume - past_volume) / past_volume) * 100
                
                signal = 'neutral'
                strength = abs(vroc)
                confidence = 0.6
                
                if vroc > 20:  # 20% increase
                    signal = 'bullish'
                elif vroc < -20:  # 20% decrease
                    signal = 'bearish'
                
                indicator = VolumeIndicator(
                    name='VROC_20',
                    value=vroc,
                    signal=signal,
                    strength=min(strength, 100),
                    confidence=confidence
                )
                indicators.append(indicator)
        
        # Tick Direction Pressure
        if len(processed_ticks) >= 50:
            recent_ticks = processed_ticks[-50:]
            up_ticks = sum(1 for tick in recent_ticks if tick.tick_direction == 'up')
            down_ticks = sum(1 for tick in recent_ticks if tick.tick_direction == 'down')
            total_directional = up_ticks + down_ticks
            
            if total_directional > 0:
                pressure = (up_ticks - down_ticks) / total_directional
                
                signal = 'neutral'
                strength = abs(pressure) * 100
                confidence = 0.7
                
                if pressure > 0.2:  # 20% more up ticks
                    signal = 'bullish'
                elif pressure < -0.2:  # 20% more down ticks
                    signal = 'bearish'
                
                indicator = VolumeIndicator(
                    name='Tick_Pressure',
                    value=pressure,
                    signal=signal,
                    strength=strength,
                    confidence=confidence
                )
                indicators.append(indicator)
        
        return indicators

    async def _analyze_volume_momentum(self, processed_ticks: List[TickVolumeData]) -> Dict[str, float]:
        """Analyze volume momentum characteristics"""
        if len(processed_ticks) < 20:
            return {}
        
        volumes = [tick.volume for tick in processed_ticks]
        
        # Calculate momentum for different periods
        momentum_5 = self._calculate_volume_momentum(volumes, 5)
        momentum_10 = self._calculate_volume_momentum(volumes, 10)
        momentum_20 = self._calculate_volume_momentum(volumes, 20)
        
        # Calculate volume acceleration
        volume_acceleration = 0.0
        if momentum_10 is not None and momentum_20 is not None:
            volume_acceleration = momentum_10 - momentum_20
        
        # Detect volume spikes
        current_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[-20:])
        volume_spike = current_volume > (avg_volume * self.spike_threshold)
        
        return {
            'momentum_5': momentum_5 or 0.0,
            'momentum_10': momentum_10 or 0.0,
            'momentum_20': momentum_20 or 0.0,
            'volume_acceleration': volume_acceleration,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_spike': 1.0 if volume_spike else 0.0,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0
        }

    async def _generate_volume_signals(self, symbol: str, processed_ticks: List[TickVolumeData], 
                                     momentum_analysis: Dict[str, float]) -> List[TickVolumeSignal]:
        """Generate volume-based scalping signals"""
        signals = []
        
        if not processed_ticks or not momentum_analysis:
            return signals
        
        # Get momentum values
        momentum_5 = momentum_analysis.get('momentum_5', 0.0)
        momentum_10 = momentum_analysis.get('momentum_10', 0.0)
        volume_spike = momentum_analysis.get('volume_spike', 0.0) > 0
        volume_ratio = momentum_analysis.get('volume_ratio', 1.0)
        
        # Determine tick pressure
        recent_ticks = processed_ticks[-20:] if len(processed_ticks) >= 20 else processed_ticks
        up_ticks = sum(1 for tick in recent_ticks if tick.tick_direction == 'up')
        down_ticks = sum(1 for tick in recent_ticks if tick.tick_direction == 'down')
        
        tick_pressure = 'neutral'
        if up_ticks > down_ticks * 1.5:
            tick_pressure = 'buying'
        elif down_ticks > up_ticks * 1.5:
            tick_pressure = 'selling'
        
        # Generate signal
        signal_type = 'hold'
        strength = 0.0
        entry_confidence = 0.5
        
        # Volume momentum signals
        if momentum_5 > self.momentum_threshold and tick_pressure == 'buying':
            signal_type = 'buy'
            strength = min(momentum_5 * 50, 100)
            entry_confidence = 0.7 if volume_spike else 0.6
        elif momentum_5 < -self.momentum_threshold and tick_pressure == 'selling':
            signal_type = 'sell'
            strength = min(abs(momentum_5) * 50, 100)
            entry_confidence = 0.7 if volume_spike else 0.6
        
        # Volume spike confirmation
        if volume_spike and signal_type != 'hold':
            strength = min(strength * 1.2, 100)  # Boost signal strength
            entry_confidence = min(entry_confidence * 1.1, 1.0)
        
        signal = TickVolumeSignal(
            timestamp=time.time(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            volume_momentum=momentum_5,
            tick_pressure=tick_pressure,
            volume_spike=volume_spike,
            entry_confidence=entry_confidence
        )
        
        signals.append(signal)
        return signals

    async def _create_volume_profile(self, processed_ticks: List[TickVolumeData]) -> Dict[str, float]:
        """Create volume profile analysis"""
        if not processed_ticks:
            return {}
        
        volumes = [tick.volume for tick in processed_ticks]
        
        return {
            'total_volume': sum(volumes),
            'average_volume': statistics.mean(volumes),
            'max_volume': max(volumes),
            'min_volume': min(volumes),
            'volume_std': statistics.stdev(volumes) if len(volumes) > 1 else 0,
            'volume_trend': 'increasing' if len(volumes) >= 2 and volumes[-1] > volumes[-2] else 'decreasing',
            'high_volume_ticks': sum(1 for v in volumes if v > statistics.mean(volumes) * 1.5),
            'low_volume_ticks': sum(1 for v in volumes if v < statistics.mean(volumes) * 0.5)
        }

    async def _calculate_execution_metrics(self, signals: List[TickVolumeSignal], 
                                         processed_ticks: List[TickVolumeData]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        return {
            'signal_strength': latest_signal.strength,
            'entry_confidence': latest_signal.entry_confidence,
            'volume_momentum': latest_signal.volume_momentum,
            'volume_spike_present': 1.0 if latest_signal.volume_spike else 0.0,
            'tick_pressure_score': 1.0 if latest_signal.tick_pressure != 'neutral' else 0.0,
            'analysis_speed_ms': (self.total_calculation_time / self.calculation_count * 1000) 
                               if self.calculation_count > 0 else 0
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'total_calculations': self.calculation_count,
            'average_calculation_time_ms': (self.total_calculation_time / self.calculation_count * 1000) 
                                         if self.calculation_count > 0 else 0,
            'calculations_per_second': self.calculation_count / self.total_calculation_time 
                                     if self.total_calculation_time > 0 else 0
        }
