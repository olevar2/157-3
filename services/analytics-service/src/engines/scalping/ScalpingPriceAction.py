"""
Scalping Price Action Analysis Module
Provides bid/ask spread analysis and real-time spread monitoring for M1-M5 scalping strategies.
Optimized for ultra-fast execution and sub-second signal generation.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import statistics
import functools
from concurrent.futures import ThreadPoolExecutor

# Performance optimization imports
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available. Some calculations may be slower.")

# Caching decorator for expensive calculations
def cache_result(ttl_seconds: int = 60):
    """Cache function results with TTL"""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
                else:
                    del cache[key]
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        
        return wrapper
    return decorator

# High-performance calculation functions
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def fast_spread_calculation(bid_prices: np.ndarray, ask_prices: np.ndarray) -> np.ndarray:
        """Ultra-fast spread calculation using Numba JIT compilation"""
        return ask_prices - bid_prices
    
    @numba.jit(nopython=True, cache=True)
    def fast_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
        """High-performance moving average calculation"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        result = np.empty(len(prices))
        result[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            result[i] = np.mean(prices[i-period+1:i+1])
        
        return result
    
    @numba.jit(nopython=True, cache=True)
    def fast_rsi_calculation(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Optimized RSI calculation for real-time analysis"""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gains = np.empty(len(gains))
        avg_losses = np.empty(len(losses))
        
        # Initial averages
        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])
        
        # Smoothed averages
        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        result = np.full(len(prices), np.nan)
        result[period:] = rsi[period-1:]
        
        return result

else:
    # Fallback implementations without Numba
    def fast_spread_calculation(bid_prices: np.ndarray, ask_prices: np.ndarray) -> np.ndarray:
        return ask_prices - bid_prices
    
    def fast_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(prices).rolling(window=period).mean().values
    
    def fast_rsi_calculation(prices: np.ndarray, period: int = 14) -> np.ndarray:
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result = np.full(len(prices), np.nan)
        result[1:] = rsi.values
        return result


@dataclass
class TickData:
    """Individual tick data structure for scalping analysis"""
    timestamp: float
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float


@dataclass
class SpreadAnalysis:
    """Bid/Ask spread analysis result"""
    current_spread: float
    spread_percentage: float
    average_spread: float
    spread_volatility: float
    spread_trend: str  # 'widening', 'tightening', 'stable'
    liquidity_score: float  # 0-100
    market_impact: float
    optimal_entry_side: str  # 'bid', 'ask', 'mid'


@dataclass
class PriceActionSignal:
    """Price action signal for scalping"""
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str  # M1, M5
    reason: str
    timestamp: float


@dataclass
class ScalpingPriceActionResult:
    """Complete scalping price action analysis result"""
    symbol: str
    timestamp: float
    spread_analysis: SpreadAnalysis
    signals: List[PriceActionSignal]
    market_microstructure: Dict
    execution_metrics: Dict


class ScalpingPriceAction:
    """
    Scalping Price Action Analysis Engine
    Optimized for M1-M5 timeframes with sub-second analysis capabilities
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for scalping
        self.max_spread_threshold = 0.0005  # 0.5 pips for major pairs
        self.min_liquidity_threshold = 50.0  # Minimum liquidity score
        self.tick_buffer_size = 1000  # Keep last 1000 ticks
        self.analysis_window = 60  # 60 seconds for spread analysis

        # Data storage
        self.tick_buffers: Dict[str, deque] = {}
        self.spread_history: Dict[str, deque] = {}
        self.last_analysis: Dict[str, float] = {}

        # Performance metrics
        self.analysis_count = 0
        self.total_analysis_time = 0.0

    async def initialize(self) -> None:
        """Initialize the scalping price action engine"""
        self.logger.info("Initializing Scalping Price Action Engine...")

        try:
            # Test calculations with sample data
            test_ticks = self._generate_test_ticks()
            test_result = await self._analyze_spread(test_ticks)

            if test_result.current_spread >= 0:
                self.ready = True
                self.logger.info("✅ Scalping Price Action Engine initialized")
            else:
                raise ValueError("Spread analysis test failed")

        except Exception as error:
            self.logger.error(f"❌ Scalping Price Action Engine initialization failed: {error}")
            raise error

    def is_ready(self) -> bool:
        """Check if engine is ready for analysis"""
        return self.ready

    async def analyze_real_time(self, symbol: str, tick_data: TickData) -> Optional[ScalpingPriceActionResult]:
        """
        Real-time tick analysis for scalping opportunities
        Optimized for sub-second execution
        """
        if not self.ready:
            raise RuntimeError("Scalping Price Action Engine not initialized")

        start_time = time.perf_counter()

        # Add tick to buffer
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = deque(maxlen=self.tick_buffer_size)
            self.spread_history[symbol] = deque(maxlen=self.tick_buffer_size)

        self.tick_buffers[symbol].append(tick_data)

        # Calculate current spread
        current_spread = tick_data.ask - tick_data.bid
        spread_percentage = (current_spread / tick_data.bid) * 100
        self.spread_history[symbol].append((tick_data.timestamp, current_spread, spread_percentage))

        # Perform analysis if enough data and time elapsed
        current_time = time.time()
        last_analysis_time = self.last_analysis.get(symbol, 0)

        if (len(self.tick_buffers[symbol]) >= 10 and
            current_time - last_analysis_time >= 1.0):  # Analyze every second

            result = await self._perform_scalping_analysis(symbol, tick_data)
            self.last_analysis[symbol] = current_time

            # Update performance metrics
            analysis_time = time.perf_counter() - start_time
            self.analysis_count += 1
            self.total_analysis_time += analysis_time

            if analysis_time > 0.1:  # Log if analysis takes more than 100ms
                self.logger.warning(f"Slow analysis for {symbol}: {analysis_time:.3f}s")

            return result

        return None

    async def analyze_batch(self, symbol: str, tick_data_list: List[TickData]) -> ScalpingPriceActionResult:
        """
        Batch analysis for historical data or bulk processing
        """
        if not self.ready:
            raise RuntimeError("Scalping Price Action Engine not initialized")

        if len(tick_data_list) < 10:
            raise ValueError("Insufficient tick data for analysis (minimum 10 ticks required)")

        self.logger.debug(f"Performing batch scalping analysis for {symbol} with {len(tick_data_list)} ticks")

        # Process all ticks
        for tick in tick_data_list:
            if symbol not in self.tick_buffers:
                self.tick_buffers[symbol] = deque(maxlen=self.tick_buffer_size)
                self.spread_history[symbol] = deque(maxlen=self.tick_buffer_size)

            self.tick_buffers[symbol].append(tick)
            current_spread = tick.ask - tick.bid
            spread_percentage = (current_spread / tick.bid) * 100
            self.spread_history[symbol].append((tick.timestamp, current_spread, spread_percentage))

        # Analyze with latest tick
        latest_tick = tick_data_list[-1]
        return await self._perform_scalping_analysis(symbol, latest_tick)

    async def _perform_scalping_analysis(self, symbol: str, current_tick: TickData) -> ScalpingPriceActionResult:
        """
        Core scalping analysis logic
        """
        # Get recent ticks for analysis
        recent_ticks = list(self.tick_buffers[symbol])[-100:]  # Last 100 ticks
        recent_spreads = list(self.spread_history[symbol])[-100:]

        # Analyze spread characteristics
        spread_analysis = await self._analyze_spread_detailed(recent_spreads, current_tick)

        # Generate scalping signals
        signals = await self._generate_scalping_signals(recent_ticks, spread_analysis)

        # Analyze market microstructure
        microstructure = await self._analyze_microstructure(recent_ticks)

        # Calculate execution metrics
        execution_metrics = await self._calculate_execution_metrics(recent_ticks, spread_analysis)

        return ScalpingPriceActionResult(
            symbol=symbol,
            timestamp=current_tick.timestamp,
            spread_analysis=spread_analysis,
            signals=signals,
            market_microstructure=microstructure,
            execution_metrics=execution_metrics
        )

    async def _analyze_spread(self, ticks: List[TickData]) -> SpreadAnalysis:
        """Basic spread analysis for testing"""
        if not ticks:
            return SpreadAnalysis(0, 0, 0, 0, 'stable', 0, 0, 'mid')

        spreads = [tick.ask - tick.bid for tick in ticks]
        current_spread = spreads[-1]
        avg_spread = statistics.mean(spreads)

        return SpreadAnalysis(
            current_spread=current_spread,
            spread_percentage=(current_spread / ticks[-1].bid) * 100,
            average_spread=avg_spread,
            spread_volatility=statistics.stdev(spreads) if len(spreads) > 1 else 0,
            spread_trend='stable',
            liquidity_score=75.0,
            market_impact=0.1,
            optimal_entry_side='mid'
        )

    def _generate_test_ticks(self) -> List[TickData]:
        """Generate test tick data for initialization"""
        test_ticks = []
        base_price = 1.1000

        for i in range(10):
            timestamp = time.time() + i
            bid = base_price + (i * 0.0001)
            ask = bid + 0.0002

            test_ticks.append(TickData(
                timestamp=timestamp,
                symbol="EURUSD",
                bid=bid,
                ask=ask,
                bid_size=100000,
                ask_size=100000,
                last_price=(bid + ask) / 2,
                volume=1000
            ))

        return test_ticks

    async def _analyze_spread_detailed(self, spread_history: List[Tuple], current_tick: TickData) -> SpreadAnalysis:
        """
        Detailed spread analysis for scalping optimization
        """
        if len(spread_history) < 5:
            return await self._analyze_spread([current_tick])

        # Extract spread data
        timestamps, spreads, spread_percentages = zip(*spread_history)

        current_spread = current_tick.ask - current_tick.bid
        current_spread_pct = (current_spread / current_tick.bid) * 100

        # Calculate statistics
        avg_spread = statistics.mean(spreads)
        spread_volatility = statistics.stdev(spreads) if len(spreads) > 1 else 0

        # Determine spread trend
        recent_spreads = spreads[-10:]  # Last 10 spreads
        if len(recent_spreads) >= 3:
            trend_slope = (recent_spreads[-1] - recent_spreads[0]) / len(recent_spreads)
            if trend_slope > avg_spread * 0.1:
                spread_trend = 'widening'
            elif trend_slope < -avg_spread * 0.1:
                spread_trend = 'tightening'
            else:
                spread_trend = 'stable'
        else:
            spread_trend = 'stable'

        # Calculate liquidity score (0-100)
        liquidity_score = self._calculate_liquidity_score(current_tick, avg_spread)

        # Calculate market impact
        market_impact = self._calculate_market_impact(current_tick, spread_volatility)

        # Determine optimal entry side
        optimal_entry_side = self._determine_optimal_entry_side(current_tick, spread_trend)

        return SpreadAnalysis(
            current_spread=current_spread,
            spread_percentage=current_spread_pct,
            average_spread=avg_spread,
            spread_volatility=spread_volatility,
            spread_trend=spread_trend,
            liquidity_score=liquidity_score,
            market_impact=market_impact,
            optimal_entry_side=optimal_entry_side
        )

    async def _generate_scalping_signals(self, recent_ticks: List[TickData], spread_analysis: SpreadAnalysis) -> List[PriceActionSignal]:
        """
        Generate scalping signals based on price action and spread analysis
        """
        signals = []

        if len(recent_ticks) < 5:
            return signals

        current_tick = recent_ticks[-1]

        # Signal 1: Tight spread scalping opportunity
        if (spread_analysis.current_spread <= self.max_spread_threshold and
            spread_analysis.liquidity_score >= self.min_liquidity_threshold):

            # Determine direction based on recent price movement
            price_direction = self._analyze_price_direction(recent_ticks[-5:])

            if price_direction == 'bullish':
                signals.append(PriceActionSignal(
                    signal_type='buy',
                    strength=0.8,
                    confidence=0.75,
                    entry_price=current_tick.ask,
                    stop_loss=current_tick.bid - (spread_analysis.current_spread * 2),
                    take_profit=current_tick.ask + (spread_analysis.current_spread * 3),
                    timeframe='M1',
                    reason='Tight spread + bullish momentum',
                    timestamp=current_tick.timestamp
                ))
            elif price_direction == 'bearish':
                signals.append(PriceActionSignal(
                    signal_type='sell',
                    strength=0.8,
                    confidence=0.75,
                    entry_price=current_tick.bid,
                    stop_loss=current_tick.ask + (spread_analysis.current_spread * 2),
                    take_profit=current_tick.bid - (spread_analysis.current_spread * 3),
                    timeframe='M1',
                    reason='Tight spread + bearish momentum',
                    timestamp=current_tick.timestamp
                ))

        # Signal 2: Spread reversion opportunity
        if spread_analysis.spread_trend == 'widening' and spread_analysis.current_spread > spread_analysis.average_spread * 1.5:
            signals.append(PriceActionSignal(
                signal_type='hold',
                strength=0.6,
                confidence=0.65,
                entry_price=(current_tick.bid + current_tick.ask) / 2,
                stop_loss=0,
                take_profit=0,
                timeframe='M1',
                reason='Wide spread - wait for tightening',
                timestamp=current_tick.timestamp
            ))

        return signals

    async def _analyze_microstructure(self, recent_ticks: List[TickData]) -> Dict:
        """
        Analyze market microstructure for scalping insights
        """
        if len(recent_ticks) < 10:
            return {}

        # Calculate order flow imbalance
        total_bid_size = sum(tick.bid_size for tick in recent_ticks)
        total_ask_size = sum(tick.ask_size for tick in recent_ticks)
        order_flow_imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)

        # Calculate price momentum
        price_changes = []
        for i in range(1, len(recent_ticks)):
            price_change = recent_ticks[i].last_price - recent_ticks[i-1].last_price
            price_changes.append(price_change)

        momentum = sum(price_changes) / len(price_changes) if price_changes else 0

        # Calculate volatility
        prices = [tick.last_price for tick in recent_ticks]
        volatility = statistics.stdev(prices) if len(prices) > 1 else 0

        return {
            'order_flow_imbalance': order_flow_imbalance,
            'price_momentum': momentum,
            'volatility': volatility,
            'tick_frequency': len(recent_ticks) / 60,  # ticks per minute
            'average_volume': statistics.mean([tick.volume for tick in recent_ticks])
        }

    async def _calculate_execution_metrics(self, recent_ticks: List[TickData], spread_analysis: SpreadAnalysis) -> Dict:
        """
        Calculate execution quality metrics for scalping
        """
        if len(recent_ticks) < 5:
            return {}

        current_tick = recent_ticks[-1]

        # Estimated slippage
        estimated_slippage = spread_analysis.current_spread / 2

        # Execution cost
        execution_cost = spread_analysis.current_spread + (spread_analysis.market_impact * 0.1)

        # Optimal execution timing score
        timing_score = self._calculate_timing_score(recent_ticks, spread_analysis)

        return {
            'estimated_slippage': estimated_slippage,
            'execution_cost': execution_cost,
            'timing_score': timing_score,
            'liquidity_available': min(current_tick.bid_size, current_tick.ask_size),
            'market_impact_estimate': spread_analysis.market_impact
        }

    def _calculate_liquidity_score(self, tick: TickData, avg_spread: float) -> float:
        """
        Calculate liquidity score based on bid/ask sizes and spread
        """
        # Base score from bid/ask sizes
        min_size = min(tick.bid_size, tick.ask_size)
        max_size = max(tick.bid_size, tick.ask_size)

        # Size score (0-50)
        size_score = min(50, (min_size / 100000) * 50)  # Normalize to 100k base

        # Spread score (0-50)
        current_spread = tick.ask - tick.bid
        spread_score = max(0, 50 - ((current_spread / avg_spread) * 25))

        return min(100, size_score + spread_score)

    def _calculate_market_impact(self, tick: TickData, spread_volatility: float) -> float:
        """
        Estimate market impact for order execution
        """
        # Base impact from spread volatility
        base_impact = spread_volatility * 0.5

        # Adjust for liquidity
        min_size = min(tick.bid_size, tick.ask_size)
        liquidity_factor = max(0.1, min(1.0, min_size / 100000))

        return base_impact / liquidity_factor

    def _determine_optimal_entry_side(self, tick: TickData, spread_trend: str) -> str:
        """
        Determine optimal entry side based on spread characteristics
        """
        if spread_trend == 'tightening':
            # Spread is tightening, prefer aggressive entry
            if tick.bid_size > tick.ask_size:
                return 'ask'  # More liquidity on bid, hit ask
            else:
                return 'bid'  # More liquidity on ask, hit bid
        elif spread_trend == 'widening':
            # Spread is widening, prefer passive entry
            return 'mid'
        else:
            # Stable spread, use mid-point
            return 'mid'

    def _analyze_price_direction(self, recent_ticks: List[TickData]) -> str:
        """
        Analyze recent price direction for signal generation
        """
        if len(recent_ticks) < 3:
            return 'neutral'

        prices = [tick.last_price for tick in recent_ticks]

        # Calculate price momentum
        price_changes = []
        for i in range(1, len(prices)):
            price_changes.append(prices[i] - prices[i-1])

        total_change = sum(price_changes)
        avg_change = total_change / len(price_changes)

        # Determine direction
        if avg_change > 0.0001:  # 0.1 pip threshold
            return 'bullish'
        elif avg_change < -0.0001:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_timing_score(self, recent_ticks: List[TickData], spread_analysis: SpreadAnalysis) -> float:
        """
        Calculate optimal timing score for execution (0-100)
        """
        score = 50  # Base score

        # Spread factor
        if spread_analysis.current_spread <= self.max_spread_threshold:
            score += 25
        elif spread_analysis.current_spread > spread_analysis.average_spread * 1.5:
            score -= 25

        # Liquidity factor
        if spread_analysis.liquidity_score >= self.min_liquidity_threshold:
            score += 15
        else:
            score -= 15

        # Volatility factor
        if len(recent_ticks) >= 5:
            prices = [tick.last_price for tick in recent_ticks[-5:]]
            volatility = statistics.stdev(prices) if len(prices) > 1 else 0

            if volatility < 0.0005:  # Low volatility is good for scalping
                score += 10
            elif volatility > 0.002:  # High volatility is risky
                score -= 10

        return max(0, min(100, score))

    def get_performance_metrics(self) -> Dict:
        """
        Get engine performance metrics
        """
        avg_analysis_time = (self.total_analysis_time / self.analysis_count
                           if self.analysis_count > 0 else 0)

        return {
            'total_analyses': self.analysis_count,
            'average_analysis_time_ms': avg_analysis_time * 1000,
            'analyses_per_second': self.analysis_count / max(1, self.total_analysis_time),
            'ready': self.ready,
            'active_symbols': len(self.tick_buffers)
        }
