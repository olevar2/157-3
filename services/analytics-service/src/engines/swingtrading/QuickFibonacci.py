"""
Quick Fibonacci Module
Fast retracements for H4 reversals in swing trading
Optimized for rapid Fibonacci level calculations and reversal detection.
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
class FibonacciLevel:
    """Fibonacci retracement level"""
    ratio: float
    price: float
    level_type: str  # 'retracement', 'extension', 'projection'
    strength: float  # 0-1 based on historical respect
    distance_from_current: float
    support_resistance: str  # 'support', 'resistance', 'neutral'


@dataclass
class FibonacciZone:
    """Fibonacci confluence zone"""
    price_range: Tuple[float, float]
    confluence_count: int
    strength: float
    zone_type: str  # 'reversal', 'continuation', 'breakout'
    timeframe_validity: List[str]


@dataclass
class ReversalSignal:
    """Fibonacci-based reversal signal"""
    signal_type: str  # 'buy', 'sell', 'wait'
    fibonacci_level: float
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    reversal_probability: float


@dataclass
class QuickFibonacciResult:
    """Quick Fibonacci analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    swing_high: float
    swing_low: float
    fibonacci_levels: List[FibonacciLevel]
    confluence_zones: List[FibonacciZone]
    reversal_signals: List[ReversalSignal]
    current_retracement: float
    trend_direction: str
    execution_metrics: Dict[str, float]


class QuickFibonacci:
    """
    Quick Fibonacci Engine for Swing Trading
    Provides fast Fibonacci retracement calculations for H4 reversal detection
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Fibonacci ratios for swing trading
        self.retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_ratios = [1.272, 1.414, 1.618, 2.0, 2.618]
        self.projection_ratios = [0.618, 1.0, 1.272, 1.618]
        
        # Confluence detection parameters
        self.confluence_threshold = 0.0015  # 15 pips for major pairs
        self.min_confluence_count = 2
        
        # Reversal detection parameters
        self.reversal_confirmation_periods = 3
        self.min_reversal_confidence = 0.65
        
        # Historical data cache for performance
        self.fibonacci_cache: Dict[str, deque] = {}
        self.level_strength_cache: Dict[str, Dict[float, float]] = {}

    async def initialize(self) -> bool:
        """Initialize the Quick Fibonacci engine"""
        try:
            self.logger.info("Initializing Quick Fibonacci Engine...")
            
            # Test Fibonacci calculations with sample data
            test_data = self._generate_test_data()
            test_result = await self._calculate_fibonacci_levels(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Quick Fibonacci Engine initialized")
                return True
            else:
                raise Exception("Fibonacci calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Quick Fibonacci Engine initialization failed: {e}")
            return False

    async def analyze_fibonacci_reversals(self, symbol: str, price_data: List[Dict], 
                                        timeframe: str = 'H4') -> QuickFibonacciResult:
        """
        Analyze Fibonacci retracements for quick reversal detection
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            timeframe: Chart timeframe (default H4)
            
        Returns:
            QuickFibonacciResult with Fibonacci analysis
        """
        if not self.ready:
            raise Exception("Quick Fibonacci Engine not initialized")
            
        if len(price_data) < 30:
            raise Exception("Insufficient data for Fibonacci analysis (minimum 30 periods)")
            
        try:
            start_time = time.time()
            
            # Extract price data
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
            
            # Identify swing high and low for Fibonacci calculation
            swing_high, swing_low = await self._identify_swing_points(highs, lows, closes)
            
            # Determine trend direction
            trend_direction = await self._determine_trend_direction(closes)
            
            # Calculate Fibonacci levels
            fibonacci_levels = await self._calculate_fibonacci_retracements(
                swing_high, swing_low, closes[-1], trend_direction
            )
            
            # Detect confluence zones
            confluence_zones = await self._detect_confluence_zones(fibonacci_levels, symbol)
            
            # Calculate current retracement percentage
            current_retracement = await self._calculate_current_retracement(
                swing_high, swing_low, closes[-1], trend_direction
            )
            
            # Generate reversal signals
            reversal_signals = await self._generate_reversal_signals(
                symbol, closes[-1], fibonacci_levels, confluence_zones, trend_direction
            )
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(
                reversal_signals, current_retracement
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Fibonacci analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return QuickFibonacciResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                swing_high=swing_high,
                swing_low=swing_low,
                fibonacci_levels=fibonacci_levels,
                confluence_zones=confluence_zones,
                reversal_signals=reversal_signals,
                current_retracement=current_retracement,
                trend_direction=trend_direction,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Fibonacci analysis failed for {symbol}: {e}")
            raise

    async def _identify_swing_points(self, highs: List[float], lows: List[float], 
                                   closes: List[float]) -> Tuple[float, float]:
        """Identify the most recent significant swing high and low"""
        lookback_periods = min(50, len(highs))  # Look back up to 50 periods
        
        # Find swing high (highest high in lookback period)
        recent_highs = highs[-lookback_periods:]
        swing_high = max(recent_highs)
        
        # Find swing low (lowest low in lookback period)
        recent_lows = lows[-lookback_periods:]
        swing_low = min(recent_lows)
        
        return swing_high, swing_low

    async def _determine_trend_direction(self, closes: List[float]) -> str:
        """Determine the current trend direction"""
        if len(closes) < 20:
            return 'neutral'
            
        # Use simple moving averages to determine trend
        short_ma = statistics.mean(closes[-10:])
        long_ma = statistics.mean(closes[-20:])
        
        if short_ma > long_ma * 1.001:  # 0.1% threshold
            return 'bullish'
        elif short_ma < long_ma * 0.999:
            return 'bearish'
        else:
            return 'neutral'

    async def _calculate_fibonacci_retracements(self, swing_high: float, swing_low: float,
                                              current_price: float, trend_direction: str) -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement levels"""
        fibonacci_levels = []
        swing_range = swing_high - swing_low
        
        # Calculate retracement levels
        for ratio in self.retracement_ratios:
            if trend_direction == 'bullish':
                # In uptrend, retracements are below swing high
                level_price = swing_high - (swing_range * ratio)
                sr_type = 'support' if current_price > level_price else 'resistance'
            else:
                # In downtrend, retracements are above swing low
                level_price = swing_low + (swing_range * ratio)
                sr_type = 'resistance' if current_price < level_price else 'support'
            
            # Calculate level strength based on historical respect
            strength = await self._calculate_level_strength(level_price, ratio)
            
            fibonacci_levels.append(FibonacciLevel(
                ratio=ratio,
                price=level_price,
                level_type='retracement',
                strength=strength,
                distance_from_current=abs(current_price - level_price),
                support_resistance=sr_type
            ))
        
        # Calculate extension levels
        for ratio in self.extension_ratios:
            if trend_direction == 'bullish':
                level_price = swing_high + (swing_range * (ratio - 1))
                sr_type = 'resistance'
            else:
                level_price = swing_low - (swing_range * (ratio - 1))
                sr_type = 'support'
            
            strength = await self._calculate_level_strength(level_price, ratio)
            
            fibonacci_levels.append(FibonacciLevel(
                ratio=ratio,
                price=level_price,
                level_type='extension',
                strength=strength,
                distance_from_current=abs(current_price - level_price),
                support_resistance=sr_type
            ))
        
        return fibonacci_levels

    async def _calculate_level_strength(self, level_price: float, ratio: float) -> float:
        """Calculate the strength of a Fibonacci level based on historical significance"""
        # Base strength based on common Fibonacci ratios
        strength_map = {
            0.382: 0.8,
            0.5: 0.7,
            0.618: 0.9,
            0.786: 0.6,
            1.272: 0.7,
            1.618: 0.85
        }
        
        base_strength = strength_map.get(ratio, 0.5)
        
        # Add randomness for demonstration (in real implementation, use historical data)
        historical_factor = np.random.uniform(0.8, 1.2)
        
        return min(base_strength * historical_factor, 1.0)

    async def _detect_confluence_zones(self, fibonacci_levels: List[FibonacciLevel], 
                                     symbol: str) -> List[FibonacciZone]:
        """Detect Fibonacci confluence zones where multiple levels cluster"""
        confluence_zones = []
        
        # Sort levels by price
        sorted_levels = sorted(fibonacci_levels, key=lambda x: x.price)
        
        i = 0
        while i < len(sorted_levels):
            zone_levels = [sorted_levels[i]]
            j = i + 1
            
            # Find levels within confluence threshold
            while j < len(sorted_levels):
                if abs(sorted_levels[j].price - sorted_levels[i].price) <= self.confluence_threshold:
                    zone_levels.append(sorted_levels[j])
                    j += 1
                else:
                    break
            
            # Create confluence zone if minimum count met
            if len(zone_levels) >= self.min_confluence_count:
                min_price = min(level.price for level in zone_levels)
                max_price = max(level.price for level in zone_levels)
                avg_strength = statistics.mean(level.strength for level in zone_levels)
                
                confluence_zones.append(FibonacciZone(
                    price_range=(min_price, max_price),
                    confluence_count=len(zone_levels),
                    strength=avg_strength,
                    zone_type='reversal' if avg_strength > 0.7 else 'continuation',
                    timeframe_validity=['H4', 'H1']
                ))
            
            i = j if j > i + 1 else i + 1
        
        return confluence_zones

    async def _calculate_current_retracement(self, swing_high: float, swing_low: float,
                                           current_price: float, trend_direction: str) -> float:
        """Calculate the current retracement percentage"""
        swing_range = swing_high - swing_low
        
        if trend_direction == 'bullish':
            # Retracement from swing high
            retracement = (swing_high - current_price) / swing_range
        else:
            # Retracement from swing low
            retracement = (current_price - swing_low) / swing_range
        
        return max(0.0, min(1.0, retracement))

    async def _generate_reversal_signals(self, symbol: str, current_price: float,
                                       fibonacci_levels: List[FibonacciLevel],
                                       confluence_zones: List[FibonacciZone],
                                       trend_direction: str) -> List[ReversalSignal]:
        """Generate Fibonacci-based reversal signals"""
        signals = []
        
        # Check for signals at Fibonacci levels
        for level in fibonacci_levels:
            if level.level_type == 'retracement' and level.strength > 0.7:
                distance_ratio = level.distance_from_current / current_price
                
                # Signal if price is near a strong Fibonacci level
                if distance_ratio < 0.002:  # Within 20 pips for major pairs
                    signal_type = self._determine_signal_type(level, trend_direction, current_price)
                    
                    if signal_type != 'wait':
                        confidence = level.strength * 0.8  # Base confidence from level strength
                        
                        # Increase confidence if in confluence zone
                        for zone in confluence_zones:
                            if zone.price_range[0] <= level.price <= zone.price_range[1]:
                                confidence = min(confidence * 1.2, 0.95)
                                break
                        
                        if confidence >= self.min_reversal_confidence:
                            target_price, stop_loss = self._calculate_signal_targets(
                                level, signal_type, current_price, trend_direction
                            )
                            
                            risk_reward = abs(target_price - current_price) / abs(stop_loss - current_price)
                            
                            signals.append(ReversalSignal(
                                signal_type=signal_type,
                                fibonacci_level=level.price,
                                confidence=confidence,
                                entry_price=current_price,
                                target_price=target_price,
                                stop_loss=stop_loss,
                                risk_reward_ratio=risk_reward,
                                reversal_probability=confidence * 0.9
                            ))
        
        return signals

    def _determine_signal_type(self, level: FibonacciLevel, trend_direction: str, 
                              current_price: float) -> str:
        """Determine the type of signal based on Fibonacci level and trend"""
        if level.support_resistance == 'support' and current_price <= level.price:
            return 'buy' if trend_direction in ['bullish', 'neutral'] else 'wait'
        elif level.support_resistance == 'resistance' and current_price >= level.price:
            return 'sell' if trend_direction in ['bearish', 'neutral'] else 'wait'
        else:
            return 'wait'

    def _calculate_signal_targets(self, level: FibonacciLevel, signal_type: str,
                                current_price: float, trend_direction: str) -> Tuple[float, float]:
        """Calculate target and stop loss for reversal signal"""
        level_distance = abs(current_price - level.price)
        
        if signal_type == 'buy':
            # Target: Next Fibonacci resistance or 1.5x risk
            target_price = current_price + (level_distance * 2.0)
            # Stop: Below Fibonacci support
            stop_loss = level.price - (level_distance * 0.5)
        else:  # sell
            # Target: Next Fibonacci support or 1.5x risk
            target_price = current_price - (level_distance * 2.0)
            # Stop: Above Fibonacci resistance
            stop_loss = level.price + (level_distance * 0.5)
        
        return target_price, stop_loss

    async def _calculate_execution_metrics(self, signals: List[ReversalSignal], 
                                         current_retracement: float) -> Dict[str, float]:
        """Calculate execution metrics for Fibonacci analysis"""
        metrics = {
            'signal_count': len(signals),
            'avg_confidence': statistics.mean([s.confidence for s in signals]) if signals else 0.0,
            'avg_risk_reward': statistics.mean([s.risk_reward_ratio for s in signals]) if signals else 0.0,
            'current_retracement_pct': current_retracement * 100,
            'reversal_probability': statistics.mean([s.reversal_probability for s in signals]) if signals else 0.0
        }
        
        return metrics

    def _generate_test_data(self) -> List[Dict]:
        """Generate test data for initialization"""
        test_data = []
        base_price = 1.1000
        
        for i in range(50):
            # Create a trending pattern with retracements
            trend = 0.0001 * i
            noise = (np.random.random() - 0.5) * 0.002
            price = base_price + trend + noise
            
            test_data.append({
                'timestamp': time.time() - (50 - i) * 3600,
                'open': price,
                'high': price + 0.001,
                'low': price - 0.001,
                'close': price,
                'volume': 1000
            })
            
        return test_data

    async def _calculate_fibonacci_levels(self, test_data: List[Dict]) -> List[FibonacciLevel]:
        """Test Fibonacci level calculation"""
        try:
            closes = [data['close'] for data in test_data]
            highs = [data['high'] for data in test_data]
            lows = [data['low'] for data in test_data]
            
            swing_high, swing_low = await self._identify_swing_points(highs, lows, closes)
            trend_direction = await self._determine_trend_direction(closes)
            
            return await self._calculate_fibonacci_retracements(
                swing_high, swing_low, closes[-1], trend_direction
            )
            
        except Exception:
            return []
