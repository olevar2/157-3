"""
Fast Momentum Oscillators Module
RSI, Stochastic, Williams %R optimized for M15-H1 day trading strategies.
Provides ultra-fast momentum calculations and intraday trading signals.
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
class RSIData:
    """RSI calculation data"""
    timestamp: float
    value: float
    signal: str  # 'oversold', 'overbought', 'neutral'
    divergence: Optional[str]  # 'bullish', 'bearish', None
    strength: float  # 0-100


@dataclass
class StochasticData:
    """Stochastic oscillator data"""
    timestamp: float
    k_percent: float
    d_percent: float
    signal: str  # 'oversold', 'overbought', 'neutral'
    crossover: Optional[str]  # 'bullish', 'bearish', None
    strength: float


@dataclass
class WilliamsRData:
    """Williams %R data"""
    timestamp: float
    value: float
    signal: str  # 'oversold', 'overbought', 'neutral'
    momentum: str  # 'increasing', 'decreasing', 'stable'
    strength: float


@dataclass
class MomentumSignal:
    """Combined momentum signal"""
    timestamp: float
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    confidence: float  # 0-1
    oscillator_consensus: Dict[str, str]
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class FastMomentumResult:
    """Complete fast momentum analysis result"""
    symbol: str
    timestamp: float
    rsi: RSIData
    stochastic: StochasticData
    williams_r: WilliamsRData
    signals: List[MomentumSignal]
    momentum_consensus: Dict[str, float]
    execution_metrics: Dict[str, float]


class FastMomentumOscillators:
    """
    Fast Momentum Oscillators Engine for Day Trading
    Optimized for M15-H1 timeframes with rapid momentum analysis
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for momentum oscillators
        self.rsi_period = 14
        self.stochastic_k_period = 14
        self.stochastic_d_period = 3
        self.williams_r_period = 14
        
        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.stoch_oversold = 20
        self.stoch_overbought = 80
        self.williams_oversold = -80
        self.williams_overbought = -20
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.rsi_history: Dict[str, deque] = {}
        self.stoch_history: Dict[str, deque] = {}
        self.williams_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.calculation_count = 0
        self.total_calculation_time = 0.0

    async def initialize(self) -> bool:
        """Initialize fast momentum oscillators engine"""
        try:
            self.logger.info("Initializing Fast Momentum Oscillators Engine...")
            
            # Test RSI calculation
            test_prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 47.25, 47.92, 
                          46.23, 44.18, 43.61, 42.00, 42.66, 43.21, 43.42, 43.03, 43.05, 44.06]
            test_rsi = self._calculate_rsi(test_prices)
            
            if test_rsi is not None and 0 <= test_rsi <= 100:
                self.ready = True
                self.logger.info("✅ Fast Momentum Oscillators Engine initialized successfully")
                return True
            else:
                raise ValueError("RSI calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Fast Momentum Oscillators Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_momentum(self, symbol: str, price_data: List[Dict]) -> FastMomentumResult:
        """
        Main momentum analysis function
        """
        if not self.ready:
            raise RuntimeError("Fast Momentum Oscillators Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.price_history:
                self._initialize_symbol_buffers(symbol)
            
            # Extract price data
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            
            # Calculate RSI
            rsi_data = await self._calculate_rsi_data(symbol, closes)
            
            # Calculate Stochastic
            stochastic_data = await self._calculate_stochastic_data(symbol, highs, lows, closes)
            
            # Calculate Williams %R
            williams_data = await self._calculate_williams_r_data(symbol, highs, lows, closes)
            
            # Generate momentum signals
            signals = await self._generate_momentum_signals(symbol, closes[-1], rsi_data, 
                                                          stochastic_data, williams_data)
            
            # Calculate momentum consensus
            momentum_consensus = await self._calculate_momentum_consensus(rsi_data, stochastic_data, williams_data)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(signals)
            
            # Update performance tracking
            calculation_time = time.time() - start_time
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            
            return FastMomentumResult(
                symbol=symbol,
                timestamp=time.time(),
                rsi=rsi_data,
                stochastic=stochastic_data,
                williams_r=williams_data,
                signals=signals,
                momentum_consensus=momentum_consensus,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Momentum analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        buffer_size = 200  # Keep enough data for calculations
        self.price_history[symbol] = deque(maxlen=buffer_size)
        self.rsi_history[symbol] = deque(maxlen=buffer_size)
        self.stoch_history[symbol] = deque(maxlen=buffer_size)
        self.williams_history[symbol] = deque(maxlen=buffer_size)

    def _calculate_rsi(self, prices: List[float], period: int = None) -> Optional[float]:
        """Calculate RSI (Relative Strength Index)"""
        if period is None:
            period = self.rsi_period
            
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gains and losses
        if len(gains) < period or len(losses) < period:
            return None
        
        avg_gain = statistics.mean(gains[-period:])
        avg_loss = statistics.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    async def _calculate_rsi_data(self, symbol: str, closes: List[float]) -> RSIData:
        """Calculate RSI with signal analysis"""
        rsi_value = self._calculate_rsi(closes)
        
        if rsi_value is None:
            return RSIData(time.time(), 50.0, 'neutral', None, 0.0)
        
        # Determine signal
        if rsi_value <= self.rsi_oversold:
            signal = 'oversold'
            strength = (self.rsi_oversold - rsi_value) / self.rsi_oversold * 100
        elif rsi_value >= self.rsi_overbought:
            signal = 'overbought'
            strength = (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought) * 100
        else:
            signal = 'neutral'
            strength = 0.0
        
        # Check for divergence (simplified)
        divergence = await self._check_rsi_divergence(symbol, closes, rsi_value)
        
        return RSIData(
            timestamp=time.time(),
            value=rsi_value,
            signal=signal,
            divergence=divergence,
            strength=min(strength, 100)
        )

    def _calculate_stochastic(self, highs: List[float], lows: List[float], 
                            closes: List[float], k_period: int = None, d_period: int = None) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Stochastic oscillator"""
        if k_period is None:
            k_period = self.stochastic_k_period
        if d_period is None:
            d_period = self.stochastic_d_period
            
        if len(closes) < k_period:
            return None, None
        
        # Calculate %K
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (moving average of %K)
        # For simplicity, using current %K as %D (in real implementation, would use historical %K values)
        d_percent = k_percent
        
        return k_percent, d_percent

    async def _calculate_stochastic_data(self, symbol: str, highs: List[float], 
                                       lows: List[float], closes: List[float]) -> StochasticData:
        """Calculate Stochastic with signal analysis"""
        k_percent, d_percent = self._calculate_stochastic(highs, lows, closes)
        
        if k_percent is None or d_percent is None:
            return StochasticData(time.time(), 50.0, 50.0, 'neutral', None, 0.0)
        
        # Determine signal
        if k_percent <= self.stoch_oversold:
            signal = 'oversold'
            strength = (self.stoch_oversold - k_percent) / self.stoch_oversold * 100
        elif k_percent >= self.stoch_overbought:
            signal = 'overbought'
            strength = (k_percent - self.stoch_overbought) / (100 - self.stoch_overbought) * 100
        else:
            signal = 'neutral'
            strength = 0.0
        
        # Check for crossover
        crossover = None
        if len(self.stoch_history[symbol]) > 0:
            prev_stoch = self.stoch_history[symbol][-1]
            if k_percent > d_percent and prev_stoch['k'] <= prev_stoch['d']:
                crossover = 'bullish'
            elif k_percent < d_percent and prev_stoch['k'] >= prev_stoch['d']:
                crossover = 'bearish'
        
        # Store current values for next comparison
        self.stoch_history[symbol].append({'k': k_percent, 'd': d_percent})
        
        return StochasticData(
            timestamp=time.time(),
            k_percent=k_percent,
            d_percent=d_percent,
            signal=signal,
            crossover=crossover,
            strength=min(strength, 100)
        )

    def _calculate_williams_r(self, highs: List[float], lows: List[float], 
                            closes: List[float], period: int = None) -> Optional[float]:
        """Calculate Williams %R"""
        if period is None:
            period = self.williams_r_period
            
        if len(closes) < period:
            return None
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        
        return williams_r

    async def _calculate_williams_r_data(self, symbol: str, highs: List[float], 
                                       lows: List[float], closes: List[float]) -> WilliamsRData:
        """Calculate Williams %R with signal analysis"""
        williams_value = self._calculate_williams_r(highs, lows, closes)
        
        if williams_value is None:
            return WilliamsRData(time.time(), -50.0, 'neutral', 'stable', 0.0)
        
        # Determine signal
        if williams_value <= self.williams_oversold:
            signal = 'oversold'
            strength = (self.williams_oversold - williams_value) / abs(self.williams_oversold) * 100
        elif williams_value >= self.williams_overbought:
            signal = 'overbought'
            strength = (williams_value - self.williams_overbought) / abs(self.williams_overbought) * 100
        else:
            signal = 'neutral'
            strength = 0.0
        
        # Determine momentum
        momentum = 'stable'
        if len(self.williams_history[symbol]) > 0:
            prev_williams = self.williams_history[symbol][-1]
            if williams_value > prev_williams:
                momentum = 'increasing'
            elif williams_value < prev_williams:
                momentum = 'decreasing'
        
        # Store current value for next comparison
        self.williams_history[symbol].append(williams_value)
        
        return WilliamsRData(
            timestamp=time.time(),
            value=williams_value,
            signal=signal,
            momentum=momentum,
            strength=min(strength, 100)
        )

    async def _check_rsi_divergence(self, symbol: str, closes: List[float], current_rsi: float) -> Optional[str]:
        """Check for RSI divergence (simplified implementation)"""
        # This is a simplified divergence check
        # In a full implementation, this would compare price highs/lows with RSI highs/lows
        if len(closes) < 20 or len(self.rsi_history[symbol]) < 10:
            return None
        
        # Basic divergence detection logic would go here
        # For now, returning None (no divergence detected)
        return None

    async def _generate_momentum_signals(self, symbol: str, current_price: float,
                                       rsi_data: RSIData, stochastic_data: StochasticData,
                                       williams_data: WilliamsRData) -> List[MomentumSignal]:
        """Generate momentum-based trading signals"""
        signals = []
        
        # Count oscillator signals
        buy_signals = 0
        sell_signals = 0
        
        oscillator_consensus = {}
        
        # RSI signals
        if rsi_data.signal == 'oversold':
            buy_signals += 1
            oscillator_consensus['RSI'] = 'buy'
        elif rsi_data.signal == 'overbought':
            sell_signals += 1
            oscillator_consensus['RSI'] = 'sell'
        else:
            oscillator_consensus['RSI'] = 'neutral'
        
        # Stochastic signals
        if stochastic_data.signal == 'oversold' or stochastic_data.crossover == 'bullish':
            buy_signals += 1
            oscillator_consensus['Stochastic'] = 'buy'
        elif stochastic_data.signal == 'overbought' or stochastic_data.crossover == 'bearish':
            sell_signals += 1
            oscillator_consensus['Stochastic'] = 'sell'
        else:
            oscillator_consensus['Stochastic'] = 'neutral'
        
        # Williams %R signals
        if williams_data.signal == 'oversold':
            buy_signals += 1
            oscillator_consensus['Williams_R'] = 'buy'
        elif williams_data.signal == 'overbought':
            sell_signals += 1
            oscillator_consensus['Williams_R'] = 'sell'
        else:
            oscillator_consensus['Williams_R'] = 'neutral'
        
        # Generate final signal
        signal_type = 'hold'
        strength = 0.0
        confidence = 0.5
        
        if buy_signals >= 2:  # At least 2 oscillators agree
            signal_type = 'buy'
            strength = (buy_signals / 3) * 100
            confidence = 0.6 + (buy_signals / 3) * 0.3
        elif sell_signals >= 2:  # At least 2 oscillators agree
            signal_type = 'sell'
            strength = (sell_signals / 3) * 100
            confidence = 0.6 + (sell_signals / 3) * 0.3
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_day_trading_levels(current_price, signal_type)
        
        signal = MomentumSignal(
            timestamp=time.time(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            oscillator_consensus=oscillator_consensus,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        signals.append(signal)
        return signals

    def _calculate_day_trading_levels(self, price: float, signal_type: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit for day trading"""
        pip_value = 0.0001  # For major pairs
        
        if signal_type == 'buy':
            stop_loss = price - (20 * pip_value)  # 20 pip stop
            take_profit = price + (30 * pip_value)  # 30 pip target (1.5:1 R/R)
        elif signal_type == 'sell':
            stop_loss = price + (20 * pip_value)  # 20 pip stop
            take_profit = price - (30 * pip_value)  # 30 pip target
        else:
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit

    async def _calculate_momentum_consensus(self, rsi_data: RSIData, stochastic_data: StochasticData,
                                          williams_data: WilliamsRData) -> Dict[str, float]:
        """Calculate overall momentum consensus"""
        # Calculate average strength
        avg_strength = (rsi_data.strength + stochastic_data.strength + williams_data.strength) / 3
        
        # Calculate momentum score
        momentum_score = 0.0
        if rsi_data.signal in ['oversold', 'overbought']:
            momentum_score += 33.33
        if stochastic_data.signal in ['oversold', 'overbought']:
            momentum_score += 33.33
        if williams_data.signal in ['oversold', 'overbought']:
            momentum_score += 33.33
        
        return {
            'average_strength': avg_strength,
            'momentum_score': momentum_score,
            'rsi_contribution': rsi_data.strength,
            'stochastic_contribution': stochastic_data.strength,
            'williams_contribution': williams_data.strength,
            'consensus_level': min(avg_strength / 100, 1.0)
        }

    async def _calculate_execution_metrics(self, signals: List[MomentumSignal]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        return {
            'signal_strength': latest_signal.strength,
            'signal_confidence': latest_signal.confidence,
            'oscillator_agreement': sum(1 for v in latest_signal.oscillator_consensus.values() 
                                      if v != 'neutral') / len(latest_signal.oscillator_consensus),
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
