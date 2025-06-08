"""
Live Trading Data Pipeline for Humanitarian Platform

Real-time market data processing pipeline optimized for generating
profits to fund medical aid, children's surgeries, and poverty relief.

This pipeline ensures continuous, clean market data flows to AI models
for maximum charitable impact through optimized trading.
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared', 'communication'))
from platform3_communication_framework import Platform3CommunicationFramework
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import json
import time
from dataclasses import dataclass, field
from collections import deque
import threading
from queue import Queue
import websockets

# Platform3 Communication Framework Integration
communication_framework = Platform3CommunicationFramework(
    service_name="live_trading_data",
    service_port=8000,  # Default port
    redis_url="redis://localhost:6379",
    consul_host="localhost",
    consul_port=8500
)

# Initialize the framework
try:
    communication_framework.initialize()
    print(f"Communication framework initialized for live_trading_data")
except Exception as e:
    print(f"Failed to initialize communication framework: {e}")

class MarketTick:
    """Single market tick data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float
    source: str = "live"

@dataclass
class ProcessedMarketData:
    """Processed market data with indicators"""
    symbol: str
    timestamp: datetime
    ohlc: Dict[str, float]  # Open, High, Low, Close
    volume: int
    indicators: Dict[str, float]
    volatility: float
    trend_strength: float
    quality_score: float

class DataQualityChecker:
    """Ensures data quality for charitable fund protection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            'max_spread_pips': 3.0,
            'min_volume': 1000,
            'max_price_gap': 0.001,  # 10 pips for majors
            'staleness_seconds': 5
        }
    
    def check_tick_quality(self, tick: MarketTick) -> Tuple[bool, float, str]:
        """
        Check individual tick quality
        Returns: (is_valid, quality_score, reason)
        """
        issues = []
        quality_score = 1.0
        
        # Check spread
        spread_pips = tick.spread * 10000
        if spread_pips > self.quality_thresholds['max_spread_pips']:
            quality_score -= 0.3
            issues.append(f"wide_spread({spread_pips:.1f})")
        
        # Check volume
        if tick.volume < self.quality_thresholds['min_volume']:
            quality_score -= 0.2
            issues.append("low_volume")
        
        # Check staleness
        age_seconds = (datetime.now() - tick.timestamp).total_seconds()
        if age_seconds > self.quality_thresholds['staleness_seconds']:
            quality_score -= 0.4
            issues.append(f"stale({age_seconds:.1f}s)")
        
        # Check price validity
        if tick.bid >= tick.ask:
            quality_score = 0.0
            issues.append("invalid_spread")
        
        is_valid = quality_score >= 0.6  # Minimum quality for trading
        reason = ", ".join(issues) if issues else "good"
        
        return is_valid, quality_score, reason

class TechnicalIndicators:
    """Fast technical indicator calculations for real-time processing"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_buffer = deque(maxlen=window_size)
        self.volume_buffer = deque(maxlen=window_size)
        
    def update(self, price: float, volume: int):
        """Update buffers with new data"""
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
    
    def calculate_indicators(self) -> Dict[str, float]:
        """Calculate all indicators from current buffer"""
        if len(self.price_buffer) < 2:
            return {}
        
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        
        indicators = {}
        
        try:
            # Simple Moving Average
            if len(prices) >= 5:
                indicators['sma_5'] = np.mean(prices[-5:])
            if len(prices) >= 10:
                indicators['sma_10'] = np.mean(prices[-10:])
            if len(prices) >= 20:
                indicators['sma_20'] = np.mean(prices[-20:])
            
            # RSI (simplified)
            if len(prices) >= 14:
                deltas = np.diff(prices[-14:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi'] = 100 - (100 / (1 + rs))
                else:
                    indicators['rsi'] = 100
            
            # Volatility
            if len(prices) >= 10:
                returns = np.diff(prices[-10:]) / prices[-11:-1]
                indicators['volatility'] = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized
            
            # Volume-weighted average price
            if len(prices) == len(volumes) and len(prices) >= 5:
                recent_prices = prices[-5:]
                recent_volumes = volumes[-5:]
                if np.sum(recent_volumes) > 0:
                    indicators['vwap'] = np.sum(recent_prices * recent_volumes) / np.sum(recent_volumes)
            
            # Price momentum
            if len(prices) >= 5:
                indicators['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5]
            
        except Exception as e:
            logging.warning(f"Indicator calculation error: {e}")
        
        return indicators

class LiveTradingDataPipeline:
    """
    Live trading data pipeline for humanitarian profit generation
    Processes real-time market feeds for AI model consumption
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_checker = DataQualityChecker()
        self.indicators = {}  # Per-symbol indicator calculators
        
        # Data streams
        self.tick_queue = Queue(maxsize=10000)
        self.processed_queue = Queue(maxsize=1000)
        
        # Performance tracking
        self.ticks_processed = 0
        self.quality_rejections = 0
        self.processing_times = deque(maxlen=1000)
        
        # Charitable mission tracking
        self.data_uptime = 0.0
        self.quality_score = 0.0
        
        # Control flags
        self.is_running = False
        self.processing_thread = None
        
        self.logger.info("🔄 Live Trading Data Pipeline initialized for humanitarian mission")

    def start_pipeline(self):
        """Start the data processing pipeline"""
        if self.is_running:
            self.logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_data_stream, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("🚀 Data pipeline started - processing live feeds for charitable trading")

    def stop_pipeline(self):
        """Stop the data processing pipeline"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        self.logger.info("⏹️ Data pipeline stopped")

    def add_market_tick(self, symbol: str, price: float, volume: int, bid: float, ask: float, source: str = "live"):
        """Add new market tick to processing queue"""
        tick = MarketTick(
            symbol=symbol,
            timestamp=datetime.now(),
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            spread=ask - bid,
            source=source
        )
        
        try:
            self.tick_queue.put_nowait(tick)
        except:
            self.logger.warning("Tick queue full - dropping old data")

    def _process_data_stream(self):
        """Main data processing loop"""
        self.logger.info("📊 Starting data stream processing")
        
        while self.is_running:
            try:
                # Get tick with timeout
                tick = self.tick_queue.get(timeout=1.0)
                
                start_time = time.perf_counter()
                
                # Process the tick
                processed_data = self._process_tick(tick)
                
                if processed_data:
                    # Add to output queue
                    try:
                        self.processed_queue.put_nowait(processed_data)
                    except:
                        self.logger.warning("Processed queue full")
                
                # Track performance
                processing_time = (time.perf_counter() - start_time) * 1000
                self.processing_times.append(processing_time)
                self.ticks_processed += 1
                
                # Log milestone
                if self.ticks_processed % 1000 == 0:
                    self._log_performance_milestone()
                
            except Exception as e:
                if self.is_running:  # Only log if not shutting down
                    self.logger.error(f"Data processing error: {e}")
                time.sleep(0.1)

    def _process_tick(self, tick: MarketTick) -> Optional[ProcessedMarketData]:
        """Process individual market tick"""
        
        # Check data quality first
        is_valid, quality_score, reason = self.quality_checker.check_tick_quality(tick)
        
        if not is_valid:
            self.quality_rejections += 1
            if self.quality_rejections % 100 == 0:
                self.logger.warning(f"🛡️ Data quality protection: rejected {self.quality_rejections} low-quality ticks")
            return None
        
        # Initialize indicators for new symbol
        if tick.symbol not in self.indicators:
            self.indicators[tick.symbol] = TechnicalIndicators()
        
        # Update indicators
        indicator_calc = self.indicators[tick.symbol]
        indicator_calc.update(tick.price, tick.volume)
        
        # Calculate indicators
        indicators = indicator_calc.calculate_indicators()
        
        # Calculate additional metrics
        volatility = indicators.get('volatility', 0.0)
        trend_strength = self._calculate_trend_strength(indicators)
        
        # Create OHLC from tick (simplified)
        ohlc = {
            'open': tick.price,
            'high': tick.price,
            'low': tick.price,
            'close': tick.price
        }
        
        return ProcessedMarketData(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            ohlc=ohlc,
            volume=tick.volume,
            indicators=indicators,
            volatility=volatility,
            trend_strength=trend_strength,
            quality_score=quality_score
        )

    def _calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """Calculate trend strength from indicators"""
        if not indicators:
            return 0.0
        
        trend_signals = []
        
        # SMA trend
        if 'sma_5' in indicators and 'sma_20' in indicators:
            sma_trend = (indicators['sma_5'] - indicators['sma_20']) / indicators['sma_20']
            trend_signals.append(sma_trend)
        
        # RSI trend
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70:
                trend_signals.append(0.5)  # Overbought
            elif rsi < 30:
                trend_signals.append(-0.5)  # Oversold
            else:
                trend_signals.append(0.0)  # Neutral
        
        # Momentum
        if 'momentum_5' in indicators:
            trend_signals.append(indicators['momentum_5'] * 100)
        
        if trend_signals:
            return np.mean(trend_signals)
        else:
            return 0.0

    def get_latest_data(self, symbol: str, timeout: float = 1.0) -> Optional[ProcessedMarketData]:
        """Get latest processed data for a symbol"""
        try:
            data = self.processed_queue.get(timeout=timeout)
            if data.symbol == symbol:
                return data
            else:
                # Put it back and return None
                self.processed_queue.put_nowait(data)
                return None
        except:
            return None

    async def get_data_stream(self) -> asyncio.AsyncGenerator[ProcessedMarketData, None]:
        """Async generator for processed data stream"""
        while self.is_running:
            try:
                data = self.processed_queue.get(timeout=0.1)
                yield data
            except:
                await asyncio.sleep(0.01)

    def _log_performance_milestone(self):
        """Log performance milestone for humanitarian tracking"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        quality_rate = ((self.ticks_processed - self.quality_rejections) / max(1, self.ticks_processed)) * 100
        
        self.logger.info(
            f"📈 Data Pipeline Milestone: {self.ticks_processed:,} ticks processed, "
            f"{avg_processing_time:.2f}ms avg, {quality_rate:.1f}% quality rate"
        )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        quality_rate = ((self.ticks_processed - self.quality_rejections) / max(1, self.ticks_processed)) * 100
        
        return {
            'is_running': self.is_running,
            'ticks_processed': self.ticks_processed,
            'quality_rejections': self.quality_rejections,
            'quality_rate_percent': quality_rate,
            'avg_processing_time_ms': avg_processing_time,
            'symbols_tracked': len(self.indicators),
            'queue_sizes': {
                'tick_queue': self.tick_queue.qsize(),
                'processed_queue': self.processed_queue.qsize()
            },
            'humanitarian_mission': 'active',
            'uptime_ready': self.is_running
        }

    async def simulate_market_feed(self, symbols: List[str], duration_minutes: int = 60):
        """Simulate market feed for testing (when live feeds unavailable)"""
        self.logger.info(f"🧪 Starting market simulation for {len(symbols)} symbols ({duration_minutes} min)")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        base_prices = {symbol: 1.0 + i * 0.1 for i, symbol in enumerate(symbols)}
        
        while datetime.now() < end_time and self.is_running:
            for symbol in symbols:
                # Generate realistic tick
                base_price = base_prices[symbol]
                price_change = np.random.normal(0, 0.0001)  # Small random walk
                new_price = base_price + price_change
                base_prices[symbol] = new_price
                
                # Generate bid/ask with spread
                spread = np.random.uniform(0.00005, 0.0002)
                bid = new_price - spread/2
                ask = new_price + spread/2
                
                volume = np.random.randint(1000, 50000)
                
                # Add to pipeline
                self.add_market_tick(symbol, new_price, volume, bid, ask, "simulation")
            
            # Wait for next tick
            await asyncio.sleep(0.1)  # 10 ticks per second
        
        self.logger.info("🧪 Market simulation completed")


# Initialize singleton pipeline
data_pipeline = LiveTradingDataPipeline()

def start_live_data_feed():
    """Start the live data processing pipeline"""
    data_pipeline.start_pipeline()

def stop_live_data_feed():
    """Stop the live data processing pipeline"""
    data_pipeline.stop_pipeline()

async def get_live_market_data(symbol: str) -> Optional[ProcessedMarketData]:
    """Get latest processed market data for symbol"""
    return data_pipeline.get_latest_data(symbol)

if __name__ == "__main__":
    # Test the data pipeline
    async def test_pipeline():
        print("🧪 Testing Live Trading Data Pipeline")
        
        # Start pipeline
        data_pipeline.start_pipeline()
        
        # Simulate some market data
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        # Start simulation
        simulation_task = asyncio.create_task(
            data_pipeline.simulate_market_feed(test_symbols, duration_minutes=1)
        )
        
        # Monitor processed data
        processed_count = 0
        async for data in data_pipeline.get_data_stream():
            processed_count += 1
            print(f"Processed: {data.symbol} @ {data.ohlc['close']:.5f}, "
                  f"Quality: {data.quality_score:.2f}, "
                  f"Indicators: {len(data.indicators)}")
            
            if processed_count >= 20:  # Sample 20 data points
                break
        
        # Wait for simulation to complete
        await simulation_task
        
        # Show final status
        status = data_pipeline.get_pipeline_status()
        print(f"\nPipeline Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Stop pipeline
        data_pipeline.stop_pipeline()
    
    # Run test
    asyncio.run(test_pipeline())
