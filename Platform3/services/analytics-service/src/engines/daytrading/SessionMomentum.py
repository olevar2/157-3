"""
Session Momentum Module
Session-specific momentum patterns for day trading strategies.
Optimized for M15-H1 timeframes with session-based momentum analysis.
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
class SessionInfo:
    """Trading session information"""
    name: str
    start_hour: int  # UTC
    end_hour: int    # UTC
    typical_volume: str  # 'high', 'medium', 'low'
    momentum_characteristics: str  # 'trending', 'ranging', 'volatile'


@dataclass
class SessionMomentumData:
    """Session momentum analysis data"""
    session_name: str
    start_time: float
    current_time: float
    momentum_score: float  # -100 to +100
    momentum_direction: str  # 'bullish', 'bearish', 'neutral'
    momentum_strength: float  # 0-100
    momentum_acceleration: float  # Rate of change
    volume_momentum: float
    price_momentum: float


@dataclass
class MomentumPattern:
    """Momentum pattern identification"""
    pattern_name: str
    confidence: float  # 0-1
    expected_duration: int  # Minutes
    target_move: float  # Expected price movement
    success_probability: float  # Historical success rate


@dataclass
class SessionMomentumSignal:
    """Session momentum-based trading signal"""
    timestamp: float
    symbol: str
    session: str
    signal_type: str  # 'buy', 'sell', 'hold'
    momentum_context: str  # 'session_start', 'session_mid', 'session_end'
    strength: float  # 0-100
    confidence: float  # 0-1
    pattern: Optional[MomentumPattern]
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class SessionMomentumResult:
    """Complete session momentum analysis result"""
    symbol: str
    timestamp: float
    current_session: str
    session_momentum: Dict[str, SessionMomentumData]
    detected_patterns: List[MomentumPattern]
    signals: List[SessionMomentumSignal]
    momentum_forecast: Dict[str, float]
    execution_metrics: Dict[str, float]


class SessionMomentum:
    """
    Session Momentum Engine for Day Trading
    Provides session-specific momentum pattern analysis for M15-H1 strategies
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Define trading sessions with momentum characteristics
        self.trading_sessions = {
            'Asian': SessionInfo('Asian', 0, 9, 'medium', 'ranging'),
            'London': SessionInfo('London', 8, 17, 'high', 'trending'),
            'NewYork': SessionInfo('NewYork', 13, 22, 'high', 'volatile'),
            'Overlap_London_NY': SessionInfo('Overlap_London_NY', 13, 17, 'high', 'trending')
        }
        
        # Configuration
        self.momentum_window = 20  # Periods for momentum calculation
        self.pattern_lookback = 50  # Periods to look back for pattern recognition
        self.momentum_threshold = 30  # Minimum momentum score for signals
        self.volume_weight = 0.3  # Weight of volume in momentum calculation
        self.price_weight = 0.7  # Weight of price in momentum calculation
        
        # Data storage
        self.session_data: Dict[str, deque] = {}
        self.momentum_history: Dict[str, deque] = {}
        self.pattern_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0

    async def initialize(self) -> bool:
        """Initialize session momentum engine"""
        try:
            self.logger.info("Initializing Session Momentum Engine...")
            
            # Test momentum calculation
            test_prices = [1.1000, 1.1005, 1.1010, 1.1008, 1.1015, 1.1020, 1.1018]
            test_momentum = self._calculate_price_momentum(test_prices)
            
            if test_momentum is not None:
                self.ready = True
                self.logger.info("✅ Session Momentum Engine initialized successfully")
                return True
            else:
                raise ValueError("Momentum calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Session Momentum Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_session_momentum(self, symbol: str, price_data: List[Dict], 
                                     volume_data: List[Dict] = None) -> SessionMomentumResult:
        """
        Main session momentum analysis function
        """
        if not self.ready:
            raise RuntimeError("Session Momentum Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.session_data:
                self._initialize_symbol_buffers(symbol)
            
            # Determine current session
            current_session = self._get_current_session(time.time())
            
            # Calculate session momentum for all sessions
            session_momentum = await self._calculate_session_momentum(symbol, price_data, volume_data)
            
            # Detect momentum patterns
            detected_patterns = await self._detect_momentum_patterns(symbol, price_data, session_momentum)
            
            # Generate momentum signals
            signals = await self._generate_momentum_signals(symbol, price_data[-1], current_session, 
                                                          session_momentum, detected_patterns)
            
            # Forecast momentum
            momentum_forecast = await self._forecast_session_momentum(current_session, session_momentum)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(signals, session_momentum)
            
            # Update performance tracking
            analysis_time = time.time() - start_time
            self.analysis_count += 1
            self.total_analysis_time += analysis_time
            
            return SessionMomentumResult(
                symbol=symbol,
                timestamp=time.time(),
                current_session=current_session or 'Unknown',
                session_momentum=session_momentum,
                detected_patterns=detected_patterns,
                signals=signals,
                momentum_forecast=momentum_forecast,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Session momentum analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        buffer_size = 500  # Keep enough data for session analysis
        self.session_data[symbol] = deque(maxlen=buffer_size)
        self.momentum_history[symbol] = deque(maxlen=buffer_size)
        self.pattern_history[symbol] = deque(maxlen=buffer_size)

    def _get_current_session(self, timestamp: float) -> Optional[str]:
        """Determine current trading session"""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        current_hour = dt.hour
        
        # Check for overlap sessions first (more specific)
        if 13 <= current_hour < 17:  # London-NY overlap
            return 'Overlap_London_NY'
        
        # Check individual sessions
        for session_name, session in self.trading_sessions.items():
            if session_name == 'Overlap_London_NY':
                continue  # Already checked
                
            if session.start_hour <= session.end_hour:
                if session.start_hour <= current_hour < session.end_hour:
                    return session_name
            else:
                # Session crosses midnight
                if current_hour >= session.start_hour or current_hour < session.end_hour:
                    return session_name
        
        return None

    def _calculate_price_momentum(self, prices: List[float], period: int = None) -> Optional[float]:
        """Calculate price momentum"""
        if period is None:
            period = min(self.momentum_window, len(prices))
        
        if len(prices) < period:
            return None
        
        # Calculate rate of change
        current_price = prices[-1]
        past_price = prices[-period]
        
        if past_price == 0:
            return 0.0
        
        momentum = ((current_price - past_price) / past_price) * 100
        return momentum

    def _calculate_volume_momentum(self, volumes: List[float], period: int = None) -> Optional[float]:
        """Calculate volume momentum"""
        if period is None:
            period = min(self.momentum_window, len(volumes))
        
        if len(volumes) < period:
            return None
        
        # Calculate volume rate of change
        recent_volume = statistics.mean(volumes[-period//2:]) if len(volumes) >= period//2 else volumes[-1]
        past_volume = statistics.mean(volumes[-period:-period//2]) if len(volumes) >= period else volumes[0]
        
        if past_volume == 0:
            return 0.0
        
        volume_momentum = ((recent_volume - past_volume) / past_volume) * 100
        return volume_momentum

    async def _calculate_session_momentum(self, symbol: str, price_data: List[Dict], 
                                        volume_data: List[Dict] = None) -> Dict[str, SessionMomentumData]:
        """Calculate momentum for each trading session"""
        session_momentum = {}
        
        # Group data by session
        session_groups = {}
        for i, data in enumerate(price_data):
            timestamp = float(data.get('timestamp', time.time()))
            session = self._get_session_for_timestamp(timestamp)
            
            if session:
                if session not in session_groups:
                    session_groups[session] = []
                session_groups[session].append((i, data))
        
        # Calculate momentum for each session
        for session_name, session_data_list in session_groups.items():
            if len(session_data_list) < 5:  # Need minimum data points
                continue
            
            # Extract prices and volumes for this session
            session_prices = [float(data[1].get('close', 0)) for data in session_data_list]
            session_volumes = []
            
            if volume_data:
                session_volumes = [float(volume_data[data[0]].get('volume', 0)) 
                                 if data[0] < len(volume_data) else 0 for data in session_data_list]
            else:
                session_volumes = [float(data[1].get('volume', 0)) for data in session_data_list]
            
            # Calculate price momentum
            price_momentum = self._calculate_price_momentum(session_prices) or 0.0
            
            # Calculate volume momentum
            volume_momentum = self._calculate_volume_momentum(session_volumes) or 0.0
            
            # Calculate combined momentum score
            momentum_score = (price_momentum * self.price_weight + 
                            volume_momentum * self.volume_weight)
            
            # Determine momentum direction and strength
            if momentum_score > 5:
                momentum_direction = 'bullish'
                momentum_strength = min(abs(momentum_score), 100)
            elif momentum_score < -5:
                momentum_direction = 'bearish'
                momentum_strength = min(abs(momentum_score), 100)
            else:
                momentum_direction = 'neutral'
                momentum_strength = 0.0
            
            # Calculate momentum acceleration (change in momentum)
            momentum_acceleration = 0.0
            if len(self.momentum_history[symbol]) > 0:
                prev_momentum = self.momentum_history[symbol][-1].get(session_name, {}).get('momentum_score', 0)
                momentum_acceleration = momentum_score - prev_momentum
            
            session_start_time = session_data_list[0][1].get('timestamp', time.time())
            
            session_momentum[session_name] = SessionMomentumData(
                session_name=session_name,
                start_time=float(session_start_time),
                current_time=time.time(),
                momentum_score=momentum_score,
                momentum_direction=momentum_direction,
                momentum_strength=momentum_strength,
                momentum_acceleration=momentum_acceleration,
                volume_momentum=volume_momentum,
                price_momentum=price_momentum
            )
        
        # Store momentum history
        momentum_dict = {name: {'momentum_score': data.momentum_score} 
                        for name, data in session_momentum.items()}
        self.momentum_history[symbol].append(momentum_dict)
        
        return session_momentum

    def _get_session_for_timestamp(self, timestamp: float) -> Optional[str]:
        """Get session for a specific timestamp"""
        return self._get_current_session(timestamp)

    async def _detect_momentum_patterns(self, symbol: str, price_data: List[Dict],
                                      session_momentum: Dict[str, SessionMomentumData]) -> List[MomentumPattern]:
        """Detect momentum patterns in session data"""
        patterns = []
        
        if not session_momentum:
            return patterns
        
        # Pattern 1: Strong Session Start
        for session_name, momentum_data in session_momentum.items():
            session_info = self.trading_sessions.get(session_name)
            if not session_info:
                continue
            
            # Check if we're at session start (within first 30 minutes)
            session_elapsed = (time.time() - momentum_data.start_time) / 60  # Minutes
            
            if session_elapsed <= 30 and abs(momentum_data.momentum_score) > 20:
                pattern = MomentumPattern(
                    pattern_name=f'{session_name}_Strong_Start',
                    confidence=min(abs(momentum_data.momentum_score) / 50, 1.0),
                    expected_duration=60,  # 1 hour
                    target_move=abs(momentum_data.momentum_score) * 0.1,  # 10% of momentum score as pips
                    success_probability=0.7
                )
                patterns.append(pattern)
        
        # Pattern 2: Momentum Acceleration
        for session_name, momentum_data in session_momentum.items():
            if abs(momentum_data.momentum_acceleration) > 10:
                pattern = MomentumPattern(
                    pattern_name=f'{session_name}_Momentum_Acceleration',
                    confidence=min(abs(momentum_data.momentum_acceleration) / 20, 1.0),
                    expected_duration=30,  # 30 minutes
                    target_move=abs(momentum_data.momentum_acceleration) * 0.05,
                    success_probability=0.6
                )
                patterns.append(pattern)
        
        # Pattern 3: Cross-Session Momentum Transfer
        if len(session_momentum) >= 2:
            momentum_values = [data.momentum_score for data in session_momentum.values()]
            if all(score > 15 for score in momentum_values) or all(score < -15 for score in momentum_values):
                pattern = MomentumPattern(
                    pattern_name='Cross_Session_Momentum',
                    confidence=0.8,
                    expected_duration=120,  # 2 hours
                    target_move=statistics.mean([abs(score) for score in momentum_values]) * 0.15,
                    success_probability=0.75
                )
                patterns.append(pattern)
        
        return patterns

    async def _generate_momentum_signals(self, symbol: str, current_data: Dict, current_session: Optional[str],
                                       session_momentum: Dict[str, SessionMomentumData],
                                       detected_patterns: List[MomentumPattern]) -> List[SessionMomentumSignal]:
        """Generate trading signals based on session momentum"""
        signals = []
        
        if not current_session or current_session not in session_momentum:
            return signals
        
        current_price = float(current_data.get('close', 0))
        momentum_data = session_momentum[current_session]
        
        # Determine momentum context
        session_elapsed = (time.time() - momentum_data.start_time) / 60  # Minutes
        if session_elapsed <= 60:
            momentum_context = 'session_start'
        elif session_elapsed <= 300:  # 5 hours
            momentum_context = 'session_mid'
        else:
            momentum_context = 'session_end'
        
        # Generate signal based on momentum
        signal_type = 'hold'
        strength = 0.0
        confidence = 0.5
        pattern = None
        
        # Strong momentum signals
        if abs(momentum_data.momentum_score) > self.momentum_threshold:
            if momentum_data.momentum_direction == 'bullish':
                signal_type = 'buy'
            elif momentum_data.momentum_direction == 'bearish':
                signal_type = 'sell'
            
            strength = min(momentum_data.momentum_strength, 100)
            confidence = min(abs(momentum_data.momentum_score) / 50, 1.0)
            
            # Check for pattern confirmation
            for p in detected_patterns:
                if current_session.lower() in p.pattern_name.lower():
                    pattern = p
                    confidence = min(confidence * 1.2, 1.0)
                    strength = min(strength * 1.1, 100)
                    break
        
        # Momentum acceleration signals
        elif abs(momentum_data.momentum_acceleration) > 15:
            if momentum_data.momentum_acceleration > 0:
                signal_type = 'buy'
            else:
                signal_type = 'sell'
            
            strength = min(abs(momentum_data.momentum_acceleration) * 3, 100)
            confidence = 0.6
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_session_levels(current_price, signal_type, 
                                                              momentum_data, pattern)
        
        signal = SessionMomentumSignal(
            timestamp=time.time(),
            symbol=symbol,
            session=current_session,
            signal_type=signal_type,
            momentum_context=momentum_context,
            strength=strength,
            confidence=confidence,
            pattern=pattern,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        signals.append(signal)
        return signals

    def _calculate_session_levels(self, price: float, signal_type: str,
                                momentum_data: SessionMomentumData,
                                pattern: Optional[MomentumPattern]) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on session momentum"""
        pip_value = 0.0001
        
        # Base levels
        if signal_type == 'buy':
            stop_loss = price - (20 * pip_value)
            take_profit = price + (40 * pip_value)
        elif signal_type == 'sell':
            stop_loss = price + (20 * pip_value)
            take_profit = price - (40 * pip_value)
        else:
            return price, price
        
        # Adjust based on momentum strength
        momentum_multiplier = 1 + (momentum_data.momentum_strength / 200)  # 1.0 to 1.5
        
        if signal_type == 'buy':
            take_profit = price + (40 * pip_value * momentum_multiplier)
        elif signal_type == 'sell':
            take_profit = price - (40 * pip_value * momentum_multiplier)
        
        # Adjust based on pattern
        if pattern and pattern.target_move > 0:
            pattern_target = pattern.target_move * pip_value * 10  # Convert to price
            if signal_type == 'buy':
                take_profit = max(take_profit, price + pattern_target)
            elif signal_type == 'sell':
                take_profit = min(take_profit, price - pattern_target)
        
        return stop_loss, take_profit

    async def _forecast_session_momentum(self, current_session: Optional[str],
                                       session_momentum: Dict[str, SessionMomentumData]) -> Dict[str, float]:
        """Forecast momentum for upcoming sessions"""
        forecast = {}
        
        if not current_session or current_session not in session_momentum:
            return forecast
        
        current_momentum = session_momentum[current_session]
        
        # Forecast momentum continuation
        forecast['momentum_continuation_probability'] = min(abs(current_momentum.momentum_score) / 50, 1.0)
        
        # Forecast momentum reversal
        if abs(current_momentum.momentum_score) > 40:
            forecast['momentum_reversal_probability'] = 0.3  # High momentum often reverses
        else:
            forecast['momentum_reversal_probability'] = 0.1
        
        # Session-specific forecasts
        session_info = self.trading_sessions.get(current_session)
        if session_info:
            if session_info.momentum_characteristics == 'trending':
                forecast['trend_continuation_probability'] = 0.7
            elif session_info.momentum_characteristics == 'ranging':
                forecast['trend_continuation_probability'] = 0.3
            else:  # volatile
                forecast['trend_continuation_probability'] = 0.5
        
        # Next session momentum transfer
        forecast['next_session_momentum_transfer'] = min(abs(current_momentum.momentum_score) / 100, 0.8)
        
        return forecast

    async def _calculate_execution_metrics(self, signals: List[SessionMomentumSignal],
                                         session_momentum: Dict[str, SessionMomentumData]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        metrics = {
            'signal_strength': latest_signal.strength,
            'signal_confidence': latest_signal.confidence,
            'momentum_context': 1.0 if latest_signal.momentum_context == 'session_start' else 0.5,
            'pattern_confirmation': 1.0 if latest_signal.pattern else 0.0,
            'analysis_speed_ms': (self.total_analysis_time / self.analysis_count * 1000) 
                               if self.analysis_count > 0 else 0
        }
        
        # Add session-specific metrics
        if latest_signal.session in session_momentum:
            momentum_data = session_momentum[latest_signal.session]
            metrics['momentum_score'] = momentum_data.momentum_score
            metrics['momentum_acceleration'] = momentum_data.momentum_acceleration
            metrics['volume_momentum'] = momentum_data.volume_momentum
            metrics['price_momentum'] = momentum_data.price_momentum
        
        return metrics

    def get_session_info(self) -> Dict[str, Dict]:
        """Get information about all trading sessions"""
        return {name: {
            'start_hour_utc': session.start_hour,
            'end_hour_utc': session.end_hour,
            'typical_volume': session.typical_volume,
            'momentum_characteristics': session.momentum_characteristics
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
