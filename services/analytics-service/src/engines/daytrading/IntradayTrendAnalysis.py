"""
Intraday Trend Analysis Module
M15-H1 trend identification for day trading strategies.
Provides ultra-fast trend detection and momentum analysis for intraday trading.
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
class TrendData:
    """Trend analysis data structure"""
    timestamp: float
    direction: str  # 'uptrend', 'downtrend', 'sideways'
    strength: float  # 0-100
    duration: int  # Number of periods
    slope: float  # Trend line slope
    confidence: float  # 0-1


@dataclass
class TrendLine:
    """Trend line data"""
    start_time: float
    end_time: float
    start_price: float
    end_price: float
    slope: float
    r_squared: float  # Correlation coefficient
    type: str  # 'support', 'resistance', 'trend'


@dataclass
class MomentumIndicator:
    """Momentum indicator data"""
    name: str
    value: float
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float
    divergence: Optional[str]


@dataclass
class TrendSignal:
    """Trend-based trading signal"""
    timestamp: float
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    trend_direction: str
    momentum_confirmation: bool
    strength: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class IntradayTrendResult:
    """Complete intraday trend analysis result"""
    symbol: str
    timestamp: float
    current_trend: TrendData
    trend_lines: List[TrendLine]
    momentum_indicators: List[MomentumIndicator]
    signals: List[TrendSignal]
    trend_analysis: Dict[str, float]
    execution_metrics: Dict[str, float]


class IntradayTrendAnalysis:
    """
    Intraday Trend Analysis Engine for Day Trading
    Provides M15-H1 trend identification and momentum analysis
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for trend analysis
        self.trend_periods = [20, 50, 100]  # Different periods for trend analysis
        self.momentum_period = 14
        self.trend_threshold = 0.6  # Minimum correlation for trend confirmation
        self.sideways_threshold = 0.0005  # Range for sideways market (5 pips)
        
        # Data storage
        self.price_history: Dict[str, deque] = {}
        self.trend_history: Dict[str, deque] = {}
        self.momentum_history: Dict[str, deque] = {}
        self.signal_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0

    async def initialize(self) -> bool:
        """Initialize intraday trend analysis engine"""
        try:
            self.logger.info("Initializing Intraday Trend Analysis Engine...")
            
            # Test trend calculation
            test_prices = [1.1000, 1.1005, 1.1010, 1.1015, 1.1020, 1.1025, 1.1030]
            trend_slope = self._calculate_trend_slope(test_prices)
            
            if trend_slope is not None:
                self.ready = True
                self.logger.info("✅ Intraday Trend Analysis Engine initialized successfully")
                return True
            else:
                raise ValueError("Trend calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Intraday Trend Analysis Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_intraday_trend(self, symbol: str, price_data: List[Dict]) -> IntradayTrendResult:
        """
        Main intraday trend analysis function
        """
        if not self.ready:
            raise RuntimeError("Intraday Trend Analysis Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.price_history:
                self._initialize_symbol_buffers(symbol)
            
            # Extract price data
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
            
            # Analyze current trend
            current_trend = await self._analyze_current_trend(symbol, closes, timestamps)
            
            # Calculate trend lines
            trend_lines = await self._calculate_trend_lines(timestamps, highs, lows, closes)
            
            # Calculate momentum indicators
            momentum_indicators = await self._calculate_momentum_indicators(closes, highs, lows)
            
            # Generate trend signals
            signals = await self._generate_trend_signals(symbol, closes[-1], current_trend, 
                                                       momentum_indicators, trend_lines)
            
            # Perform comprehensive trend analysis
            trend_analysis = await self._perform_trend_analysis(current_trend, trend_lines, momentum_indicators)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(signals, current_trend)
            
            # Update performance tracking
            analysis_time = time.time() - start_time
            self.analysis_count += 1
            self.total_analysis_time += analysis_time
            
            return IntradayTrendResult(
                symbol=symbol,
                timestamp=time.time(),
                current_trend=current_trend,
                trend_lines=trend_lines,
                momentum_indicators=momentum_indicators,
                signals=signals,
                trend_analysis=trend_analysis,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Intraday trend analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        buffer_size = 500  # Keep enough data for trend analysis
        self.price_history[symbol] = deque(maxlen=buffer_size)
        self.trend_history[symbol] = deque(maxlen=buffer_size)
        self.momentum_history[symbol] = deque(maxlen=buffer_size)
        self.signal_history[symbol] = deque(maxlen=buffer_size)

    def _calculate_trend_slope(self, prices: List[float]) -> Optional[float]:
        """Calculate trend slope using linear regression"""
        if len(prices) < 2:
            return None
        
        n = len(prices)
        x = list(range(n))
        y = prices
        
        # Calculate linear regression slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _calculate_r_squared(self, prices: List[float]) -> float:
        """Calculate R-squared for trend strength"""
        if len(prices) < 3:
            return 0.0
        
        n = len(prices)
        x = list(range(n))
        y = prices
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate correlation coefficient
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if sum_sq_x == 0 or sum_sq_y == 0:
            return 0.0
        
        correlation = numerator / (sum_sq_x * sum_sq_y) ** 0.5
        return correlation ** 2

    async def _analyze_current_trend(self, symbol: str, closes: List[float], 
                                   timestamps: List[float]) -> TrendData:
        """Analyze current trend characteristics"""
        if len(closes) < 20:
            return TrendData(time.time(), 'sideways', 0.0, 0, 0.0, 0.0)
        
        # Calculate trend for different periods
        trend_results = {}
        for period in self.trend_periods:
            if len(closes) >= period:
                period_prices = closes[-period:]
                slope = self._calculate_trend_slope(period_prices)
                r_squared = self._calculate_r_squared(period_prices)
                
                trend_results[period] = {
                    'slope': slope or 0.0,
                    'r_squared': r_squared,
                    'strength': r_squared * 100
                }
        
        # Determine overall trend
        if not trend_results:
            return TrendData(time.time(), 'sideways', 0.0, 0, 0.0, 0.0)
        
        # Use shortest period for current trend (most responsive)
        current_period = min(self.trend_periods)
        if current_period in trend_results:
            result = trend_results[current_period]
            slope = result['slope']
            strength = result['strength']
            r_squared = result['r_squared']
            
            # Determine trend direction
            if abs(slope) < self.sideways_threshold and r_squared < 0.3:
                direction = 'sideways'
                confidence = 1 - r_squared  # Higher confidence for sideways when low correlation
            elif slope > 0:
                direction = 'uptrend'
                confidence = r_squared
            else:
                direction = 'downtrend'
                confidence = r_squared
            
            # Calculate trend duration (simplified)
            duration = current_period
            
            return TrendData(
                timestamp=time.time(),
                direction=direction,
                strength=strength,
                duration=duration,
                slope=slope,
                confidence=confidence
            )
        
        return TrendData(time.time(), 'sideways', 0.0, 0, 0.0, 0.0)

    async def _calculate_trend_lines(self, timestamps: List[float], highs: List[float], 
                                   lows: List[float], closes: List[float]) -> List[TrendLine]:
        """Calculate support and resistance trend lines"""
        trend_lines = []
        
        if len(closes) < 20:
            return trend_lines
        
        # Calculate support line (using lows)
        support_slope = self._calculate_trend_slope(lows[-20:])
        support_r_squared = self._calculate_r_squared(lows[-20:])
        
        if support_slope is not None and support_r_squared > 0.5:
            start_time = timestamps[-20] if len(timestamps) >= 20 else timestamps[0]
            end_time = timestamps[-1]
            start_price = lows[-20] if len(lows) >= 20 else lows[0]
            end_price = lows[-1]
            
            trend_lines.append(TrendLine(
                start_time=start_time,
                end_time=end_time,
                start_price=start_price,
                end_price=end_price,
                slope=support_slope,
                r_squared=support_r_squared,
                type='support'
            ))
        
        # Calculate resistance line (using highs)
        resistance_slope = self._calculate_trend_slope(highs[-20:])
        resistance_r_squared = self._calculate_r_squared(highs[-20:])
        
        if resistance_slope is not None and resistance_r_squared > 0.5:
            start_time = timestamps[-20] if len(timestamps) >= 20 else timestamps[0]
            end_time = timestamps[-1]
            start_price = highs[-20] if len(highs) >= 20 else highs[0]
            end_price = highs[-1]
            
            trend_lines.append(TrendLine(
                start_time=start_time,
                end_time=end_time,
                start_price=start_price,
                end_price=end_price,
                slope=resistance_slope,
                r_squared=resistance_r_squared,
                type='resistance'
            ))
        
        # Calculate main trend line (using closes)
        trend_slope = self._calculate_trend_slope(closes[-50:] if len(closes) >= 50 else closes)
        trend_r_squared = self._calculate_r_squared(closes[-50:] if len(closes) >= 50 else closes)
        
        if trend_slope is not None and trend_r_squared > 0.6:
            lookback = min(50, len(closes))
            start_time = timestamps[-lookback]
            end_time = timestamps[-1]
            start_price = closes[-lookback]
            end_price = closes[-1]
            
            trend_lines.append(TrendLine(
                start_time=start_time,
                end_time=end_time,
                start_price=start_price,
                end_price=end_price,
                slope=trend_slope,
                r_squared=trend_r_squared,
                type='trend'
            ))
        
        return trend_lines

    async def _calculate_momentum_indicators(self, closes: List[float], 
                                           highs: List[float], lows: List[float]) -> List[MomentumIndicator]:
        """Calculate momentum indicators for trend confirmation"""
        indicators = []
        
        if len(closes) < self.momentum_period + 1:
            return indicators
        
        # Rate of Change (ROC)
        if len(closes) >= self.momentum_period:
            current_price = closes[-1]
            past_price = closes[-self.momentum_period]
            
            if past_price != 0:
                roc = ((current_price - past_price) / past_price) * 100
                
                signal = 'neutral'
                strength = abs(roc)
                if roc > 1:  # 1% positive change
                    signal = 'bullish'
                elif roc < -1:  # 1% negative change
                    signal = 'bearish'
                
                indicators.append(MomentumIndicator(
                    name='ROC',
                    value=roc,
                    signal=signal,
                    strength=min(strength, 100),
                    divergence=None
                ))
        
        # Price Momentum (simple momentum)
        if len(closes) >= 10:
            momentum = closes[-1] - closes[-10]
            momentum_pct = (momentum / closes[-10]) * 100 if closes[-10] != 0 else 0
            
            signal = 'neutral'
            strength = abs(momentum_pct) * 10  # Scale for visibility
            if momentum > 0:
                signal = 'bullish'
            elif momentum < 0:
                signal = 'bearish'
            
            indicators.append(MomentumIndicator(
                name='Momentum',
                value=momentum,
                signal=signal,
                strength=min(strength, 100),
                divergence=None
            ))
        
        # Average Directional Index (ADX) - simplified
        if len(highs) >= 14 and len(lows) >= 14:
            # Calculate True Range and Directional Movement (simplified)
            tr_values = []
            dm_plus = []
            dm_minus = []
            
            for i in range(1, min(15, len(closes))):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i-1])
                low_close = abs(lows[i] - closes[i-1])
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
                
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                
                dm_plus.append(up_move if up_move > down_move and up_move > 0 else 0)
                dm_minus.append(down_move if down_move > up_move and down_move > 0 else 0)
            
            if tr_values and dm_plus and dm_minus:
                avg_tr = statistics.mean(tr_values)
                avg_dm_plus = statistics.mean(dm_plus)
                avg_dm_minus = statistics.mean(dm_minus)
                
                if avg_tr > 0:
                    di_plus = (avg_dm_plus / avg_tr) * 100
                    di_minus = (avg_dm_minus / avg_tr) * 100
                    
                    dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
                    
                    signal = 'neutral'
                    if dx > 25:  # Strong trend
                        signal = 'bullish' if di_plus > di_minus else 'bearish'
                    
                    indicators.append(MomentumIndicator(
                        name='ADX',
                        value=dx,
                        signal=signal,
                        strength=dx,
                        divergence=None
                    ))
        
        return indicators

    async def _generate_trend_signals(self, symbol: str, current_price: float, 
                                    current_trend: TrendData, momentum_indicators: List[MomentumIndicator],
                                    trend_lines: List[TrendLine]) -> List[TrendSignal]:
        """Generate trend-based trading signals"""
        signals = []
        
        # Check momentum confirmation
        bullish_momentum = sum(1 for ind in momentum_indicators if ind.signal == 'bullish')
        bearish_momentum = sum(1 for ind in momentum_indicators if ind.signal == 'bearish')
        total_indicators = len(momentum_indicators)
        
        momentum_confirmation = False
        if total_indicators > 0:
            momentum_confirmation = (bullish_momentum > bearish_momentum and current_trend.direction == 'uptrend') or \
                                  (bearish_momentum > bullish_momentum and current_trend.direction == 'downtrend')
        
        # Generate signal based on trend and momentum
        signal_type = 'hold'
        strength = current_trend.strength
        confidence = current_trend.confidence
        
        if current_trend.direction == 'uptrend' and current_trend.confidence > 0.6:
            signal_type = 'buy'
            if momentum_confirmation:
                strength = min(strength * 1.2, 100)
                confidence = min(confidence * 1.1, 1.0)
        elif current_trend.direction == 'downtrend' and current_trend.confidence > 0.6:
            signal_type = 'sell'
            if momentum_confirmation:
                strength = min(strength * 1.2, 100)
                confidence = min(confidence * 1.1, 1.0)
        
        # Calculate stop loss and take profit based on trend lines
        stop_loss, take_profit = self._calculate_trend_levels(current_price, signal_type, 
                                                            trend_lines, current_trend)
        
        signal = TrendSignal(
            timestamp=time.time(),
            symbol=symbol,
            signal_type=signal_type,
            trend_direction=current_trend.direction,
            momentum_confirmation=momentum_confirmation,
            strength=strength,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        signals.append(signal)
        return signals

    def _calculate_trend_levels(self, price: float, signal_type: str, 
                              trend_lines: List[TrendLine], current_trend: TrendData) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on trend analysis"""
        pip_value = 0.0001  # For major pairs
        
        # Default levels
        if signal_type == 'buy':
            stop_loss = price - (25 * pip_value)  # 25 pip stop
            take_profit = price + (50 * pip_value)  # 50 pip target (2:1 R/R)
        elif signal_type == 'sell':
            stop_loss = price + (25 * pip_value)  # 25 pip stop
            take_profit = price - (50 * pip_value)  # 50 pip target
        else:
            return price, price
        
        # Adjust based on trend lines
        for trend_line in trend_lines:
            if trend_line.type == 'support' and signal_type == 'buy':
                # Use support as stop loss level
                support_level = trend_line.end_price - (5 * pip_value)  # 5 pips below support
                if support_level < price:
                    stop_loss = max(stop_loss, support_level)
            elif trend_line.type == 'resistance' and signal_type == 'sell':
                # Use resistance as stop loss level
                resistance_level = trend_line.end_price + (5 * pip_value)  # 5 pips above resistance
                if resistance_level > price:
                    stop_loss = min(stop_loss, resistance_level)
        
        return stop_loss, take_profit

    async def _perform_trend_analysis(self, current_trend: TrendData, trend_lines: List[TrendLine],
                                    momentum_indicators: List[MomentumIndicator]) -> Dict[str, float]:
        """Perform comprehensive trend analysis"""
        analysis = {}
        
        # Basic trend metrics
        analysis['trend_strength'] = current_trend.strength
        analysis['trend_confidence'] = current_trend.confidence
        analysis['trend_duration'] = current_trend.duration
        analysis['trend_slope'] = current_trend.slope
        
        # Trend line analysis
        analysis['trend_lines_count'] = len(trend_lines)
        if trend_lines:
            avg_r_squared = statistics.mean([tl.r_squared for tl in trend_lines])
            analysis['trend_lines_quality'] = avg_r_squared
        else:
            analysis['trend_lines_quality'] = 0.0
        
        # Momentum analysis
        if momentum_indicators:
            bullish_count = sum(1 for ind in momentum_indicators if ind.signal == 'bullish')
            bearish_count = sum(1 for ind in momentum_indicators if ind.signal == 'bearish')
            total_count = len(momentum_indicators)
            
            analysis['momentum_bullish_ratio'] = bullish_count / total_count
            analysis['momentum_bearish_ratio'] = bearish_count / total_count
            analysis['momentum_consensus'] = max(bullish_count, bearish_count) / total_count
            
            avg_momentum_strength = statistics.mean([ind.strength for ind in momentum_indicators])
            analysis['momentum_strength'] = avg_momentum_strength
        else:
            analysis['momentum_bullish_ratio'] = 0.0
            analysis['momentum_bearish_ratio'] = 0.0
            analysis['momentum_consensus'] = 0.0
            analysis['momentum_strength'] = 0.0
        
        # Overall trend quality score
        trend_quality = (current_trend.confidence * 0.4 + 
                        analysis['trend_lines_quality'] * 0.3 + 
                        analysis['momentum_consensus'] * 0.3)
        analysis['overall_trend_quality'] = trend_quality
        
        return analysis

    async def _calculate_execution_metrics(self, signals: List[TrendSignal], 
                                         current_trend: TrendData) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        return {
            'signal_strength': latest_signal.strength,
            'signal_confidence': latest_signal.confidence,
            'momentum_confirmation': 1.0 if latest_signal.momentum_confirmation else 0.0,
            'trend_alignment': 1.0 if latest_signal.trend_direction != 'sideways' else 0.0,
            'risk_reward_ratio': abs(latest_signal.take_profit - latest_signal.entry_price) / 
                               abs(latest_signal.entry_price - latest_signal.stop_loss) 
                               if latest_signal.stop_loss != latest_signal.entry_price else 0,
            'trend_strength_score': current_trend.strength,
            'analysis_speed_ms': (self.total_analysis_time / self.analysis_count * 1000) 
                               if self.analysis_count > 0 else 0
        }

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'total_analyses': self.analysis_count,
            'average_analysis_time_ms': (self.total_analysis_time / self.analysis_count * 1000) 
                                      if self.analysis_count > 0 else 0,
            'analyses_per_second': self.analysis_count / self.total_analysis_time 
                                 if self.total_analysis_time > 0 else 0
        }
