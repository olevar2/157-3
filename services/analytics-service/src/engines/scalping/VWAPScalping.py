"""
VWAP Scalping Analysis Module
Volume-Weighted Average Price analysis optimized for M1-M5 scalping strategies.
Provides ultra-fast VWAP calculations and scalping signals for daily profit focus.
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
class VWAPData:
    """VWAP calculation data structure"""
    timestamp: float
    symbol: str
    price: float
    volume: float
    vwap: float
    cumulative_volume: float
    cumulative_pv: float  # price * volume


@dataclass
class VWAPSignal:
    """VWAP-based scalping signal"""
    timestamp: float
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    price_vs_vwap: float  # percentage deviation from VWAP
    volume_confirmation: bool
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float


@dataclass
class VWAPScalpingResult:
    """Complete VWAP scalping analysis result"""
    symbol: str
    timestamp: float
    current_vwap: float
    price_deviation: float  # percentage from VWAP
    volume_profile: Dict[str, float]
    signals: List[VWAPSignal]
    support_resistance: Dict[str, List[float]]
    execution_metrics: Dict[str, float]


class VWAPScalping:
    """
    VWAP Scalping Analysis Engine
    Optimized for M1-M5 timeframes with volume-weighted price analysis
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for VWAP scalping
        self.vwap_periods = [20, 50, 100]  # Different VWAP periods for M1-M5
        self.deviation_threshold = 0.02  # 2% deviation for signals
        self.volume_threshold = 1.5  # Volume spike threshold
        self.signal_buffer_size = 500  # Keep last 500 signals
        
        # Data storage
        self.vwap_data: Dict[str, deque] = {}
        self.price_data: Dict[str, deque] = {}
        self.volume_data: Dict[str, deque] = {}
        self.signals_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

    async def initialize(self) -> bool:
        """Initialize VWAP scalping engine"""
        try:
            self.logger.info("Initializing VWAP Scalping Engine...")
            
            # Test VWAP calculations
            test_prices = [1.1000, 1.1005, 1.1010, 1.1008, 1.1012]
            test_volumes = [100, 150, 200, 120, 180]
            test_vwap = self._calculate_vwap(test_prices, test_volumes)
            
            if test_vwap > 0:
                self.ready = True
                self.logger.info("✅ VWAP Scalping Engine initialized successfully")
                return True
            else:
                raise ValueError("VWAP calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ VWAP Scalping Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_vwap_scalping(self, symbol: str, price_data: List[Dict], 
                                  volume_data: List[Dict]) -> VWAPScalpingResult:
        """
        Main VWAP scalping analysis function
        """
        if not self.ready:
            raise RuntimeError("VWAP Scalping Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.vwap_data:
                self._initialize_symbol_buffers(symbol)
            
            # Calculate VWAP for different periods
            vwap_results = await self._calculate_multi_period_vwap(symbol, price_data, volume_data)
            
            # Analyze price deviation from VWAP
            price_deviation = await self._analyze_price_deviation(symbol, price_data, vwap_results)
            
            # Generate scalping signals
            signals = await self._generate_vwap_signals(symbol, price_data, vwap_results, price_deviation)
            
            # Calculate volume profile
            volume_profile = await self._analyze_volume_profile(volume_data)
            
            # Identify support/resistance levels
            support_resistance = await self._identify_vwap_levels(vwap_results, price_data)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(signals, price_data)
            
            # Update performance tracking
            calculation_time = time.time() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            return VWAPScalpingResult(
                symbol=symbol,
                timestamp=time.time(),
                current_vwap=vwap_results['current_vwap'],
                price_deviation=price_deviation,
                volume_profile=volume_profile,
                signals=signals,
                support_resistance=support_resistance,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"VWAP scalping analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.vwap_data[symbol] = deque(maxlen=self.signal_buffer_size)
        self.price_data[symbol] = deque(maxlen=self.signal_buffer_size)
        self.volume_data[symbol] = deque(maxlen=self.signal_buffer_size)
        self.signals_history[symbol] = deque(maxlen=self.signal_buffer_size)

    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate basic VWAP"""
        if not prices or not volumes or len(prices) != len(volumes):
            return 0.0
        
        total_pv = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        return total_pv / total_volume if total_volume > 0 else 0.0

    async def _calculate_multi_period_vwap(self, symbol: str, price_data: List[Dict], 
                                         volume_data: List[Dict]) -> Dict[str, float]:
        """Calculate VWAP for multiple periods"""
        results = {}
        
        # Extract prices and volumes
        prices = [float(data.get('close', 0)) for data in price_data]
        volumes = [float(data.get('volume', 0)) for data in volume_data]
        
        # Calculate VWAP for different periods
        for period in self.vwap_periods:
            if len(prices) >= period:
                period_prices = prices[-period:]
                period_volumes = volumes[-period:]
                vwap = self._calculate_vwap(period_prices, period_volumes)
                results[f'vwap_{period}'] = vwap
        
        # Current VWAP (shortest period)
        if self.vwap_periods:
            results['current_vwap'] = results.get(f'vwap_{min(self.vwap_periods)}', 0.0)
        
        return results

    async def _analyze_price_deviation(self, symbol: str, price_data: List[Dict], 
                                     vwap_results: Dict[str, float]) -> float:
        """Analyze price deviation from VWAP"""
        if not price_data or 'current_vwap' not in vwap_results:
            return 0.0
        
        current_price = float(price_data[-1].get('close', 0))
        current_vwap = vwap_results['current_vwap']
        
        if current_vwap > 0:
            deviation = ((current_price - current_vwap) / current_vwap) * 100
            return deviation
        
        return 0.0

    async def _generate_vwap_signals(self, symbol: str, price_data: List[Dict], 
                                   vwap_results: Dict[str, float], 
                                   price_deviation: float) -> List[VWAPSignal]:
        """Generate VWAP-based scalping signals"""
        signals = []
        
        if not price_data or 'current_vwap' not in vwap_results:
            return signals
        
        current_price = float(price_data[-1].get('close', 0))
        current_vwap = vwap_results['current_vwap']
        current_volume = float(price_data[-1].get('volume', 0))
        
        # Calculate average volume for comparison
        volumes = [float(data.get('volume', 0)) for data in price_data[-20:]]
        avg_volume = statistics.mean(volumes) if volumes else 0
        
        # Volume confirmation
        volume_confirmation = current_volume > (avg_volume * self.volume_threshold)
        
        # Generate signals based on VWAP deviation
        signal_type = 'hold'
        strength = 0.0
        confidence = 0.0
        
        if abs(price_deviation) > self.deviation_threshold:
            if price_deviation < -self.deviation_threshold:  # Price below VWAP
                signal_type = 'buy'
                strength = min(abs(price_deviation) * 50, 100)
                confidence = 0.7 if volume_confirmation else 0.5
            elif price_deviation > self.deviation_threshold:  # Price above VWAP
                signal_type = 'sell'
                strength = min(abs(price_deviation) * 50, 100)
                confidence = 0.7 if volume_confirmation else 0.5
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_scalping_levels(current_price, signal_type, current_vwap)
        
        signal = VWAPSignal(
            timestamp=time.time(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price_vs_vwap=price_deviation,
            volume_confirmation=volume_confirmation,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence
        )
        
        signals.append(signal)
        return signals

    def _calculate_scalping_levels(self, price: float, signal_type: str, vwap: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit for scalping"""
        pip_value = 0.0001  # For major pairs
        
        if signal_type == 'buy':
            stop_loss = price - (10 * pip_value)  # 10 pip stop
            take_profit = price + (15 * pip_value)  # 15 pip target
        elif signal_type == 'sell':
            stop_loss = price + (10 * pip_value)  # 10 pip stop
            take_profit = price - (15 * pip_value)  # 15 pip target
        else:
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit

    async def _analyze_volume_profile(self, volume_data: List[Dict]) -> Dict[str, float]:
        """Analyze volume profile for scalping"""
        if not volume_data:
            return {}
        
        volumes = [float(data.get('volume', 0)) for data in volume_data]
        
        return {
            'current_volume': volumes[-1] if volumes else 0,
            'average_volume': statistics.mean(volumes) if volumes else 0,
            'volume_spike': max(volumes) / statistics.mean(volumes) if volumes and statistics.mean(volumes) > 0 else 1,
            'volume_trend': 'increasing' if len(volumes) >= 2 and volumes[-1] > volumes[-2] else 'decreasing'
        }

    async def _identify_vwap_levels(self, vwap_results: Dict[str, float], 
                                  price_data: List[Dict]) -> Dict[str, List[float]]:
        """Identify support and resistance levels based on VWAP"""
        support_levels = []
        resistance_levels = []
        
        # Use VWAP values as dynamic support/resistance
        for key, vwap_value in vwap_results.items():
            if key.startswith('vwap_') and vwap_value > 0:
                # VWAP can act as both support and resistance
                support_levels.append(vwap_value * 0.999)  # Slightly below VWAP
                resistance_levels.append(vwap_value * 1.001)  # Slightly above VWAP
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels, reverse=True)
        }

    async def _calculate_execution_metrics(self, signals: List[VWAPSignal], 
                                         price_data: List[Dict]) -> Dict[str, float]:
        """Calculate execution metrics for VWAP scalping"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        return {
            'signal_strength': latest_signal.strength,
            'confidence_score': latest_signal.confidence,
            'volume_confirmation': 1.0 if latest_signal.volume_confirmation else 0.0,
            'risk_reward_ratio': abs(latest_signal.take_profit - latest_signal.entry_price) / 
                               abs(latest_signal.entry_price - latest_signal.stop_loss) 
                               if latest_signal.stop_loss != latest_signal.entry_price else 0,
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
