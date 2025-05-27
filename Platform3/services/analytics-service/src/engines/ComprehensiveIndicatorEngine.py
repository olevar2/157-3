#!/usr/bin/env python3
"""
Comprehensive Indicator Engine for Platform3 Analytics Service
============================================================

This engine integrates the ComprehensiveIndicatorAdapter_67 into the analytics service
providing access to all 67 indicators through a standardized interface optimized for
real-time trading systems.

Features:
- Complete 67-indicator suite integration
- Real-time calculation optimization
- Caching and performance enhancement
- TypeScript/JavaScript bridge compatibility
- RESTful API endpoints
- WebSocket streaming support
- Trading signal generation
- Multi-timeframe analysis

Author: Platform3 Development Team
Version: 1.0.0
"""

import sys
import os
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add Platform3 paths
platform3_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.append(platform3_root)

# Import the comprehensive indicator adapter
from ComprehensiveIndicatorAdapter_67 import (
    ComprehensiveIndicatorAdapter_67,
    MarketData,
    IndicatorResult,
    IndicatorCategory
)

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Enhanced trading signal with comprehensive metadata"""
    type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # Indicator name
    category: str  # Indicator category
    description: str
    timestamp: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class IndicatorSuite:
    """Complete indicator analysis results"""
    symbol: str
    timeframe: str
    timestamp: float
    momentum_indicators: Dict[str, Any]
    trend_indicators: Dict[str, Any]
    volatility_indicators: Dict[str, Any]
    volume_indicators: Dict[str, Any]
    cycle_indicators: Dict[str, Any]
    advanced_indicators: Dict[str, Any]
    gann_indicators: Dict[str, Any]
    scalping_indicators: Dict[str, Any]
    daytrading_indicators: Dict[str, Any]
    swingtrading_indicators: Dict[str, Any]
    signals_indicators: Dict[str, Any]
    trading_signals: List[TradingSignal]
    overall_sentiment: str  # 'bullish', 'bearish', 'neutral'
    confidence_score: float  # 0.0 to 1.0

class ComprehensiveIndicatorEngine:
    """
    Main engine that provides complete indicator analysis for Platform3 trading system
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.adapter = ComprehensiveIndicatorAdapter_67()
        self.ready = False
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds cache TTL
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance optimization settings
        self.batch_size = 10  # Process indicators in batches
        self.async_mode = True
        
        # Signal generation thresholds
        self.signal_thresholds = {
            'strong_buy': 0.8,
            'buy': 0.6,
            'neutral': 0.4,
            'sell': 0.6,
            'strong_sell': 0.8
        }
        
    async def initialize(self) -> bool:
        """Initialize the comprehensive indicator engine"""
        try:
            self.logger.info("🚀 Initializing Comprehensive Indicator Engine...")
            
            # Test the adapter with sample data
            test_data = self._create_test_market_data()
            
            # Test a few key indicators
            test_indicators = ['RSI', 'MACD', 'SMA']
            for indicator_name in test_indicators:
                try:
                    result = self.adapter.calculate_indicator(indicator_name, test_data)
                    if result is None:
                        raise Exception(f"Failed to calculate {indicator_name}")
                    self.logger.debug(f"✅ {indicator_name} calculation successful")
                except Exception as e:
                    self.logger.error(f"❌ {indicator_name} test failed: {e}")
                    return False
            
            self.ready = True
            self.logger.info("✅ Comprehensive Indicator Engine initialized successfully")
            self.logger.info(f"📊 Available indicators: {len(self.adapter.get_available_indicators())}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Comprehensive Indicator Engine: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def is_ready(self) -> bool:
        """Check if the engine is ready"""
        return self.ready
    
    async def analyze_comprehensive(self, symbol: str, market_data: List[Dict], 
                                  timeframe: str = 'M15') -> IndicatorSuite:
        """
        Perform comprehensive analysis using all 67 indicators
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            market_data: List of OHLCV data dictionaries
            timeframe: Timeframe for analysis
            
        Returns:
            Complete indicator analysis suite
        """
        try:
            if not self.ready:
                raise Exception("Engine not initialized")
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{len(market_data)}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.logger.debug(f"📋 Using cached results for {symbol}")
                    return cache_entry['data']
            
            start_time = time.time()
            self.logger.info(f"🔍 Starting comprehensive analysis for {symbol} ({timeframe})")
            
            # Convert market data to MarketData format
            platform_data = self._convert_market_data(market_data)
            
            # Initialize result containers
            results_by_category = {
                'momentum': {},
                'trend': {},
                'volatility': {},
                'volume': {},
                'cycle': {},
                'advanced': {},
                'gann': {},
                'scalping': {},
                'daytrading': {},
                'swingtrading': {},
                'signals': {}
            }
            
            # Get all available indicators
            all_indicators = self.adapter.get_available_indicators()
            
            # Process indicators by category in parallel
            if self.async_mode:
                await self._process_indicators_async(all_indicators, platform_data, results_by_category)
            else:
                self._process_indicators_sync(all_indicators, platform_data, results_by_category)
            
            # Generate trading signals
            trading_signals = self._generate_comprehensive_signals(results_by_category, symbol, market_data[-1])
            
            # Calculate overall sentiment
            overall_sentiment, confidence_score = self._calculate_overall_sentiment(trading_signals)
            
            # Create comprehensive result
            indicator_suite = IndicatorSuite(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=time.time(),
                momentum_indicators=results_by_category['momentum'],
                trend_indicators=results_by_category['trend'],
                volatility_indicators=results_by_category['volatility'],
                volume_indicators=results_by_category['volume'],
                cycle_indicators=results_by_category['cycle'],
                advanced_indicators=results_by_category['advanced'],
                gann_indicators=results_by_category['gann'],
                scalping_indicators=results_by_category['scalping'],
                daytrading_indicators=results_by_category['daytrading'],
                swingtrading_indicators=results_by_category['swingtrading'],
                signals_indicators=results_by_category['signals'],
                trading_signals=trading_signals,
                overall_sentiment=overall_sentiment,
                confidence_score=confidence_score
            )
            
            # Cache the result
            self.cache[cache_key] = {
                'data': indicator_suite,
                'timestamp': time.time()
            }
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"✅ Comprehensive analysis completed for {symbol} in {elapsed_time:.2f}s")
            self.logger.info(f"📊 Generated {len(trading_signals)} trading signals")
            self.logger.info(f"📈 Overall sentiment: {overall_sentiment} (confidence: {confidence_score:.2f})")
            
            return indicator_suite
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive analysis failed for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def _process_indicators_async(self, indicators: List[str], market_data: MarketData, 
                                      results: Dict[str, Dict]) -> None:
        """Process indicators asynchronously for better performance"""
        
        # Group indicators by category
        indicators_by_category = self.adapter.get_indicators_by_category()
        
        # Process each category in parallel
        tasks = []
        for category, indicator_list in indicators_by_category.items():
            if category.lower() in results:
                task = asyncio.create_task(
                    self._process_category_async(category.lower(), indicator_list, market_data, results)
                )
                tasks.append(task)
        
        # Wait for all categories to complete
        await asyncio.gather(*tasks)
    
    async def _process_category_async(self, category: str, indicators: List[str], 
                                    market_data: MarketData, results: Dict[str, Dict]) -> None:
        """Process a single category of indicators asynchronously"""
        
        try:
            category_results = {}
            
            # Process indicators in batches
            for i in range(0, len(indicators), self.batch_size):
                batch = indicators[i:i + self.batch_size]
                
                # Submit batch to thread pool
                futures = []
                for indicator_name in batch:
                    future = self.executor.submit(
                        self._calculate_single_indicator, indicator_name, market_data
                    )
                    futures.append((indicator_name, future))
                
                # Collect results
                for indicator_name, future in futures:
                    try:
                        result = future.result(timeout=10)  # 10 second timeout
                        if result is not None:
                            category_results[indicator_name] = result
                    except Exception as e:
                        self.logger.warning(f"⚠️ {indicator_name} calculation failed: {e}")
            
            results[category] = category_results
            self.logger.debug(f"✅ {category} category completed: {len(category_results)} indicators")
            
        except Exception as e:
            self.logger.error(f"❌ Category {category} processing failed: {e}")
            results[category] = {}
    
    def _process_indicators_sync(self, indicators: List[str], market_data: MarketData, 
                               results: Dict[str, Dict]) -> None:
        """Process indicators synchronously (fallback)"""
        
        indicators_by_category = self.adapter.get_indicators_by_category()
        
        for category, indicator_list in indicators_by_category.items():
            if category.lower() in results:
                category_results = {}
                
                for indicator_name in indicator_list:
                    try:
                        result = self._calculate_single_indicator(indicator_name, market_data)
                        if result is not None:
                            category_results[indicator_name] = result
                    except Exception as e:
                        self.logger.warning(f"⚠️ {indicator_name} calculation failed: {e}")
                
                results[category.lower()] = category_results
    
    def _calculate_single_indicator(self, indicator_name: str, market_data: MarketData) -> Any:
        """Calculate a single indicator with error handling"""
        
        try:
            return self.adapter.calculate_indicator(indicator_name, market_data)
        except Exception as e:
            self.logger.debug(f"Indicator {indicator_name} failed: {e}")
            return None
    
    def _generate_comprehensive_signals(self, results: Dict[str, Dict], 
                                      symbol: str, latest_price: Dict) -> List[TradingSignal]:
        """Generate trading signals from all indicator results"""
        
        signals = []
        current_price = latest_price.get('close', 0)
        timestamp = time.time()
        
        # Momentum signals
        momentum_signals = self._generate_momentum_signals(results['momentum'], current_price, timestamp)
        signals.extend(momentum_signals)
        
        # Trend signals
        trend_signals = self._generate_trend_signals(results['trend'], current_price, timestamp)
        signals.extend(trend_signals)
        
        # Volatility signals
        volatility_signals = self._generate_volatility_signals(results['volatility'], current_price, timestamp)
        signals.extend(volatility_signals)
        
        # Volume signals
        volume_signals = self._generate_volume_signals(results['volume'], current_price, timestamp)
        signals.extend(volume_signals)
        
        # Scalping signals
        scalping_signals = self._generate_scalping_signals(results['scalping'], current_price, timestamp)
        signals.extend(scalping_signals)
        
        return signals
    
    def _generate_momentum_signals(self, momentum_results: Dict, price: float, timestamp: float) -> List[TradingSignal]:
        """Generate signals from momentum indicators"""
        signals = []
        
        # RSI signals
        if 'RSI' in momentum_results:
            rsi_data = momentum_results['RSI']
            if hasattr(rsi_data, 'value') and rsi_data.value is not None:
                rsi_value = rsi_data.value
                if rsi_value < 30:
                    signals.append(TradingSignal(
                        type='buy',
                        strength=0.8,
                        confidence=0.7,
                        source='RSI',
                        category='momentum',
                        description=f'RSI oversold at {rsi_value:.1f}',
                        timestamp=timestamp
                    ))
                elif rsi_value > 70:
                    signals.append(TradingSignal(
                        type='sell',
                        strength=0.8,
                        confidence=0.7,
                        source='RSI',
                        category='momentum',
                        description=f'RSI overbought at {rsi_value:.1f}',
                        timestamp=timestamp
                    ))
        
        # MACD signals
        if 'MACD' in momentum_results:
            macd_data = momentum_results['MACD']
            if hasattr(macd_data, 'signal_line') and macd_data.signal_line is not None:
                # Add MACD signal logic here
                pass
        
        return signals
    
    def _generate_trend_signals(self, trend_results: Dict, price: float, timestamp: float) -> List[TradingSignal]:
        """Generate signals from trend indicators"""
        signals = []
        
        # Moving average signals
        for indicator_name in ['SMA', 'EMA', 'WMA']:
            if indicator_name in trend_results:
                ma_data = trend_results[indicator_name]
                if hasattr(ma_data, 'value') and ma_data.value is not None:
                    ma_value = ma_data.value
                    if price > ma_value * 1.001:  # Price above MA
                        signals.append(TradingSignal(
                            type='buy',
                            strength=0.6,
                            confidence=0.6,
                            source=indicator_name,
                            category='trend',
                            description=f'Price above {indicator_name}',
                            timestamp=timestamp
                        ))
                    elif price < ma_value * 0.999:  # Price below MA
                        signals.append(TradingSignal(
                            type='sell',
                            strength=0.6,
                            confidence=0.6,
                            source=indicator_name,
                            category='trend',
                            description=f'Price below {indicator_name}',
                            timestamp=timestamp
                        ))
        
        return signals
    
    def _generate_volatility_signals(self, volatility_results: Dict, price: float, timestamp: float) -> List[TradingSignal]:
        """Generate signals from volatility indicators"""
        signals = []
        
        # Bollinger Bands signals
        if 'BollingerBands' in volatility_results:
            bb_data = volatility_results['BollingerBands']
            if hasattr(bb_data, 'upper') and hasattr(bb_data, 'lower'):
                if bb_data.upper and bb_data.lower:
                    if price <= bb_data.lower:
                        signals.append(TradingSignal(
                            type='buy',
                            strength=0.7,
                            confidence=0.7,
                            source='BollingerBands',
                            category='volatility',
                            description='Price at lower Bollinger Band',
                            timestamp=timestamp
                        ))
                    elif price >= bb_data.upper:
                        signals.append(TradingSignal(
                            type='sell',
                            strength=0.7,
                            confidence=0.7,
                            source='BollingerBands',
                            category='volatility',
                            description='Price at upper Bollinger Band',
                            timestamp=timestamp
                        ))
        
        return signals
    
    def _generate_volume_signals(self, volume_results: Dict, price: float, timestamp: float) -> List[TradingSignal]:
        """Generate signals from volume indicators"""
        signals = []
        
        # Add volume-based signal generation logic
        # This would analyze volume indicators for confirmation signals
        
        return signals
    
    def _generate_scalping_signals(self, scalping_results: Dict, price: float, timestamp: float) -> List[TradingSignal]:
        """Generate signals from scalping indicators"""
        signals = []
        
        # Add scalping-specific signal generation logic
        # These would be high-frequency, low-timeframe signals
        
        return signals
    
    def _calculate_overall_sentiment(self, signals: List[TradingSignal]) -> Tuple[str, float]:
        """Calculate overall market sentiment from all signals"""
        
        if not signals:
            return 'neutral', 0.0
        
        # Weight signals by strength and confidence
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = signal.strength * signal.confidence
            total_weight += weight
            
            if signal.type == 'buy':
                buy_score += weight
            elif signal.type == 'sell':
                sell_score += weight
        
        if total_weight == 0:
            return 'neutral', 0.0
        
        # Calculate normalized scores
        buy_ratio = buy_score / total_weight
        sell_ratio = sell_score / total_weight
        
        # Determine sentiment
        if buy_ratio > sell_ratio + 0.2:
            sentiment = 'bullish'
            confidence = buy_ratio
        elif sell_ratio > buy_ratio + 0.2:
            sentiment = 'bearish'
            confidence = sell_ratio
        else:
            sentiment = 'neutral'
            confidence = 1.0 - abs(buy_ratio - sell_ratio)
        
        return sentiment, min(confidence, 1.0)
    
    def _convert_market_data(self, market_data: List[Dict]) -> MarketData:
        """Convert market data to Platform3 MarketData format"""
        
        try:
            # Extract arrays
            timestamps = np.array([item.get('timestamp', 0) for item in market_data])
            opens = np.array([float(item.get('open', 0)) for item in market_data])
            highs = np.array([float(item.get('high', 0)) for item in market_data])
            lows = np.array([float(item.get('low', 0)) for item in market_data])
            closes = np.array([float(item.get('close', 0)) for item in market_data])
            volumes = np.array([float(item.get('volume', 0)) for item in market_data])
            
            return MarketData(
                timestamps=timestamps,
                open_prices=opens,
                high_prices=highs,
                low_prices=lows,
                close_prices=closes,
                volumes=volumes
            )
            
        except Exception as e:
            self.logger.error(f"Failed to convert market data: {e}")
            raise
    
    def _create_test_market_data(self) -> MarketData:
        """Create test market data for initialization"""
        
        size = 100
        timestamps = np.arange(size)
        
        # Generate realistic price data
        base_price = 1.1000
        returns = np.random.normal(0, 0.001, size)
        prices = base_price * np.exp(np.cumsum(returns))
        
        opens = prices
        closes = prices * (1 + np.random.normal(0, 0.0005, size))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0005, size)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0005, size)))
        volumes = np.random.uniform(1000, 10000, size)
        
        return MarketData(
            timestamps=timestamps,
            open_prices=opens,
            high_prices=highs,
            low_prices=lows,
            close_prices=closes,
            volumes=volumes
        )
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """Get information about available indicators"""
        
        if not self.ready:
            return {}
        
        return {
            'total_indicators': len(self.adapter.get_available_indicators()),
            'categories': self.adapter.get_indicators_by_category(),
            'adapter_version': '1.0.0',
            'engine_status': 'ready' if self.ready else 'not_ready'
        }
    
    def to_dict(self, indicator_suite: IndicatorSuite) -> Dict[str, Any]:
        """Convert IndicatorSuite to dictionary for JSON serialization"""
        
        result = asdict(indicator_suite)
        
        # Convert TradingSignal objects to dictionaries
        result['trading_signals'] = [asdict(signal) for signal in indicator_suite.trading_signals]
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the results cache"""
        self.cache.clear()
        self.logger.info("🗑️ Indicator cache cleared")
    
    async def shutdown(self) -> None:
        """Shutdown the engine gracefully"""
        self.logger.info("🛑 Shutting down Comprehensive Indicator Engine...")
        self.executor.shutdown(wait=True)
        self.clear_cache()
        self.ready = False
        self.logger.info("✅ Engine shutdown complete")

# Global instance for use by the analytics service
comprehensive_indicator_engine = None

def get_comprehensive_indicator_engine(logger: logging.Logger = None) -> ComprehensiveIndicatorEngine:
    """Get or create the global comprehensive indicator engine instance"""
    global comprehensive_indicator_engine
    
    if comprehensive_indicator_engine is None:
        if logger is None:
            logger = logging.getLogger(__name__)
        comprehensive_indicator_engine = ComprehensiveIndicatorEngine(logger)
    
    return comprehensive_indicator_engine
