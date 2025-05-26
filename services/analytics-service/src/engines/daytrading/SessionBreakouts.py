"""
Session Breakouts Module
Asian/London/NY session breakout detection for day trading strategies.
Optimized for M15-H1 timeframes with session-based analysis.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import deque
import statistics


@dataclass
class TradingSession:
    """Trading session definition"""
    name: str
    start_hour: int  # UTC hour
    end_hour: int    # UTC hour
    timezone_name: str
    volatility_profile: str  # 'high', 'medium', 'low'


@dataclass
class SessionRange:
    """Session price range data"""
    session_name: str
    start_time: float
    end_time: float
    high: float
    low: float
    open: float
    close: float
    range_size: float
    volume: float


@dataclass
class BreakoutSignal:
    """Session breakout signal"""
    timestamp: float
    symbol: str
    session: str
    breakout_type: str  # 'high_break', 'low_break'
    breakout_price: float
    range_size: float
    volume_confirmation: bool
    strength: float  # 0-100
    target_price: float
    stop_loss: float


@dataclass
class SessionBreakoutResult:
    """Complete session breakout analysis result"""
    symbol: str
    timestamp: float
    current_session: str
    session_ranges: Dict[str, SessionRange]
    breakout_signals: List[BreakoutSignal]
    session_analysis: Dict[str, float]
    volatility_forecast: Dict[str, float]
    execution_metrics: Dict[str, float]


class SessionBreakouts:
    """
    Session Breakouts Engine for Day Trading
    Provides Asian/London/NY session breakout detection for M15-H1 strategies
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Define trading sessions (UTC times)
        self.trading_sessions = {
            'Asian': TradingSession('Asian', 0, 9, 'Asia/Tokyo', 'medium'),
            'London': TradingSession('London', 8, 17, 'Europe/London', 'high'),
            'NewYork': TradingSession('NewYork', 13, 22, 'America/New_York', 'high'),
            'Sydney': TradingSession('Sydney', 22, 7, 'Australia/Sydney', 'low')
        }
        
        # Configuration
        self.breakout_threshold = 0.0010  # 10 pips for major pairs
        self.volume_threshold = 1.5  # Volume spike threshold
        self.range_min_size = 0.0005  # Minimum range size (5 pips)
        self.lookback_sessions = 5  # Number of sessions to analyze
        
        # Data storage
        self.session_data: Dict[str, deque] = {}
        self.breakout_history: Dict[str, deque] = {}
        self.range_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0

    async def initialize(self) -> bool:
        """Initialize session breakouts engine"""
        try:
            self.logger.info("Initializing Session Breakouts Engine...")
            
            # Test session detection
            test_time = time.time()
            current_session = self._get_current_session(test_time)
            
            if current_session is not None:
                self.ready = True
                self.logger.info("✅ Session Breakouts Engine initialized successfully")
                return True
            else:
                raise ValueError("Session detection test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Session Breakouts Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_session_breakouts(self, symbol: str, price_data: List[Dict], 
                                      volume_data: List[Dict] = None) -> SessionBreakoutResult:
        """
        Main session breakout analysis function
        """
        if not self.ready:
            raise RuntimeError("Session Breakouts Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.session_data:
                self._initialize_symbol_buffers(symbol)
            
            # Determine current session
            current_session = self._get_current_session(time.time())
            
            # Calculate session ranges
            session_ranges = await self._calculate_session_ranges(symbol, price_data)
            
            # Detect breakouts
            breakout_signals = await self._detect_breakouts(symbol, price_data, session_ranges, volume_data)
            
            # Analyze session characteristics
            session_analysis = await self._analyze_session_characteristics(session_ranges)
            
            # Forecast volatility
            volatility_forecast = await self._forecast_session_volatility(session_ranges)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(breakout_signals)
            
            # Update performance tracking
            analysis_time = time.time() - start_time
            self.analysis_count += 1
            self.total_analysis_time += analysis_time
            
            return SessionBreakoutResult(
                symbol=symbol,
                timestamp=time.time(),
                current_session=current_session or 'Unknown',
                session_ranges=session_ranges,
                breakout_signals=breakout_signals,
                session_analysis=session_analysis,
                volatility_forecast=volatility_forecast,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Session breakout analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        buffer_size = 500  # Keep enough data for session analysis
        self.session_data[symbol] = deque(maxlen=buffer_size)
        self.breakout_history[symbol] = deque(maxlen=buffer_size)
        self.range_history[symbol] = deque(maxlen=buffer_size)

    def _get_current_session(self, timestamp: float) -> Optional[str]:
        """Determine current trading session based on UTC time"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        current_hour = dt.hour
        
        # Check each session
        for session_name, session in self.trading_sessions.items():
            if session.start_hour <= session.end_hour:
                # Normal session (doesn't cross midnight)
                if session.start_hour <= current_hour < session.end_hour:
                    return session_name
            else:
                # Session crosses midnight (like Sydney)
                if current_hour >= session.start_hour or current_hour < session.end_hour:
                    return session_name
        
        return None

    def _get_session_for_timestamp(self, timestamp: float) -> Optional[str]:
        """Get session for a specific timestamp"""
        return self._get_current_session(timestamp)

    async def _calculate_session_ranges(self, symbol: str, price_data: List[Dict]) -> Dict[str, SessionRange]:
        """Calculate price ranges for each trading session"""
        session_ranges = {}
        
        # Group price data by session
        session_groups = {}
        for data in price_data:
            timestamp = float(data.get('timestamp', time.time()))
            session = self._get_session_for_timestamp(timestamp)
            
            if session:
                if session not in session_groups:
                    session_groups[session] = []
                session_groups[session].append(data)
        
        # Calculate ranges for each session
        for session_name, session_data in session_groups.items():
            if not session_data:
                continue
            
            # Extract OHLC data
            highs = [float(data.get('high', 0)) for data in session_data]
            lows = [float(data.get('low', 0)) for data in session_data]
            opens = [float(data.get('open', 0)) for data in session_data]
            closes = [float(data.get('close', 0)) for data in session_data]
            volumes = [float(data.get('volume', 0)) for data in session_data]
            
            if highs and lows and opens and closes:
                session_high = max(highs)
                session_low = min(lows)
                session_open = opens[0]
                session_close = closes[-1]
                range_size = session_high - session_low
                total_volume = sum(volumes)
                
                start_time = float(session_data[0].get('timestamp', time.time()))
                end_time = float(session_data[-1].get('timestamp', time.time()))
                
                session_ranges[session_name] = SessionRange(
                    session_name=session_name,
                    start_time=start_time,
                    end_time=end_time,
                    high=session_high,
                    low=session_low,
                    open=session_open,
                    close=session_close,
                    range_size=range_size,
                    volume=total_volume
                )
        
        return session_ranges

    async def _detect_breakouts(self, symbol: str, price_data: List[Dict], 
                              session_ranges: Dict[str, SessionRange], 
                              volume_data: List[Dict] = None) -> List[BreakoutSignal]:
        """Detect session breakouts"""
        breakout_signals = []
        
        if not price_data or not session_ranges:
            return breakout_signals
        
        current_price = float(price_data[-1].get('close', 0))
        current_volume = float(price_data[-1].get('volume', 0))
        current_timestamp = time.time()
        
        # Calculate average volume for comparison
        volumes = [float(data.get('volume', 0)) for data in price_data[-20:]]
        avg_volume = statistics.mean(volumes) if volumes else 0
        volume_confirmation = current_volume > (avg_volume * self.volume_threshold)
        
        # Check for breakouts in each session
        for session_name, session_range in session_ranges.items():
            if session_range.range_size < self.range_min_size:
                continue  # Skip sessions with too small ranges
            
            # Check for high breakout
            if current_price > session_range.high + self.breakout_threshold:
                strength = min(((current_price - session_range.high) / session_range.range_size) * 100, 100)
                target_price = current_price + (session_range.range_size * 0.5)  # 50% of range as target
                stop_loss = session_range.high - (self.breakout_threshold * 0.5)  # Just below breakout level
                
                signal = BreakoutSignal(
                    timestamp=current_timestamp,
                    symbol=symbol,
                    session=session_name,
                    breakout_type='high_break',
                    breakout_price=current_price,
                    range_size=session_range.range_size,
                    volume_confirmation=volume_confirmation,
                    strength=strength,
                    target_price=target_price,
                    stop_loss=stop_loss
                )
                breakout_signals.append(signal)
            
            # Check for low breakout
            elif current_price < session_range.low - self.breakout_threshold:
                strength = min(((session_range.low - current_price) / session_range.range_size) * 100, 100)
                target_price = current_price - (session_range.range_size * 0.5)  # 50% of range as target
                stop_loss = session_range.low + (self.breakout_threshold * 0.5)  # Just above breakout level
                
                signal = BreakoutSignal(
                    timestamp=current_timestamp,
                    symbol=symbol,
                    session=session_name,
                    breakout_type='low_break',
                    breakout_price=current_price,
                    range_size=session_range.range_size,
                    volume_confirmation=volume_confirmation,
                    strength=strength,
                    target_price=target_price,
                    stop_loss=stop_loss
                )
                breakout_signals.append(signal)
        
        return breakout_signals

    async def _analyze_session_characteristics(self, session_ranges: Dict[str, SessionRange]) -> Dict[str, float]:
        """Analyze characteristics of trading sessions"""
        analysis = {}
        
        if not session_ranges:
            return analysis
        
        # Calculate average range sizes
        range_sizes = [sr.range_size for sr in session_ranges.values()]
        avg_range = statistics.mean(range_sizes) if range_sizes else 0
        
        # Calculate volatility by session
        session_volatilities = {}
        for session_name, session_range in session_ranges.items():
            if session_range.range_size > 0:
                # Volatility as percentage of price
                volatility = (session_range.range_size / session_range.open) * 100 if session_range.open > 0 else 0
                session_volatilities[session_name] = volatility
        
        # Identify most volatile session
        most_volatile_session = max(session_volatilities.items(), key=lambda x: x[1]) if session_volatilities else ('Unknown', 0)
        
        analysis.update({
            'average_range_size': avg_range,
            'total_sessions_analyzed': len(session_ranges),
            'most_volatile_session': most_volatile_session[0],
            'highest_volatility': most_volatile_session[1],
            'range_consistency': 1 - (statistics.stdev(range_sizes) / avg_range) if avg_range > 0 and len(range_sizes) > 1 else 0
        })
        
        # Add individual session metrics
        for session_name, session_range in session_ranges.items():
            analysis[f'{session_name}_range'] = session_range.range_size
            analysis[f'{session_name}_volume'] = session_range.volume
        
        return analysis

    async def _forecast_session_volatility(self, session_ranges: Dict[str, SessionRange]) -> Dict[str, float]:
        """Forecast volatility for upcoming sessions"""
        forecast = {}
        
        # Get current session
        current_session = self._get_current_session(time.time())
        
        # Predict volatility based on historical patterns
        for session_name, session_info in self.trading_sessions.items():
            if session_name in session_ranges:
                current_range = session_ranges[session_name]
                
                # Base forecast on session's volatility profile
                if session_info.volatility_profile == 'high':
                    base_volatility = 0.8
                elif session_info.volatility_profile == 'medium':
                    base_volatility = 0.6
                else:
                    base_volatility = 0.4
                
                # Adjust based on recent range size
                if current_range.range_size > 0:
                    range_factor = min(current_range.range_size / 0.001, 2.0)  # Normalize to 10 pips
                    adjusted_volatility = base_volatility * range_factor
                else:
                    adjusted_volatility = base_volatility
                
                forecast[f'{session_name}_volatility'] = min(adjusted_volatility, 1.0)
                forecast[f'{session_name}_breakout_probability'] = adjusted_volatility * 0.7  # 70% of volatility
        
        # Overall market volatility forecast
        if session_ranges:
            avg_range = statistics.mean([sr.range_size for sr in session_ranges.values()])
            forecast['overall_volatility'] = min(avg_range / 0.001, 1.0)  # Normalize
        
        return forecast

    async def _calculate_execution_metrics(self, breakout_signals: List[BreakoutSignal]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not breakout_signals:
            return {
                'breakout_count': 0,
                'average_strength': 0,
                'volume_confirmation_rate': 0,
                'analysis_speed_ms': (self.total_analysis_time / self.analysis_count * 1000) 
                                   if self.analysis_count > 0 else 0
            }
        
        # Calculate metrics
        strengths = [signal.strength for signal in breakout_signals]
        volume_confirmations = [signal.volume_confirmation for signal in breakout_signals]
        
        return {
            'breakout_count': len(breakout_signals),
            'average_strength': statistics.mean(strengths),
            'max_strength': max(strengths),
            'volume_confirmation_rate': sum(volume_confirmations) / len(volume_confirmations),
            'high_breaks': sum(1 for s in breakout_signals if s.breakout_type == 'high_break'),
            'low_breaks': sum(1 for s in breakout_signals if s.breakout_type == 'low_break'),
            'analysis_speed_ms': (self.total_analysis_time / self.analysis_count * 1000) 
                               if self.analysis_count > 0 else 0
        }

    def get_session_info(self) -> Dict[str, Dict]:
        """Get information about all trading sessions"""
        return {name: {
            'start_hour_utc': session.start_hour,
            'end_hour_utc': session.end_hour,
            'timezone': session.timezone_name,
            'volatility_profile': session.volatility_profile
        } for name, session in self.trading_sessions.items()}

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'total_analyses': self.analysis_count,
            'average_analysis_time_ms': (self.total_analysis_time / self.analysis_count * 1000) 
                                      if self.analysis_count > 0 else 0,
            'analyses_per_second': self.analysis_count / self.total_analysis_time 
                                 if self.total_analysis_time > 0 else 0
        }
