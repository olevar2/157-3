"""
Order Flow Imbalance Module
Bid/ask volume imbalances detection for scalping and day trading
Optimized for order flow imbalance alerts for quick profits.
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
class OrderFlowData:
    """Order flow data structure"""
    timestamp: float
    bid_volume: float
    ask_volume: float
    bid_price: float
    ask_price: float
    spread: float
    total_volume: float
    imbalance_ratio: float  # ask_volume / bid_volume


@dataclass
class ImbalanceSignal:
    """Order flow imbalance signal"""
    signal_type: str  # 'buy_imbalance', 'sell_imbalance', 'balanced'
    imbalance_strength: float  # 0-1
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    duration_periods: int
    volume_intensity: float
    price_impact_probability: float


@dataclass
class FlowPattern:
    """Order flow pattern"""
    pattern_type: str  # 'aggressive_buying', 'aggressive_selling', 'absorption', 'exhaustion'
    start_time: float
    end_time: float
    total_volume: float
    average_imbalance: float
    price_movement: float
    pattern_strength: float


@dataclass
class OrderFlowResult:
    """Order flow imbalance analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    current_imbalance: ImbalanceSignal
    imbalance_history: List[ImbalanceSignal]
    flow_patterns: List[FlowPattern]
    market_pressure: str  # 'buying_pressure', 'selling_pressure', 'balanced'
    scalping_alerts: List[Dict[str, Union[str, float]]]
    execution_metrics: Dict[str, float]


class OrderFlowImbalance:
    """
    Order Flow Imbalance Engine for Scalping
    Provides bid/ask volume imbalance detection for quick profit opportunities
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Imbalance detection parameters
        self.imbalance_threshold_strong = 2.0  # 2:1 ratio for strong imbalance
        self.imbalance_threshold_medium = 1.5  # 1.5:1 ratio for medium imbalance
        self.imbalance_duration_min = 3  # Minimum periods for sustained imbalance
        
        # Pattern detection parameters
        self.pattern_min_volume = 1000  # Minimum volume for pattern recognition
        self.pattern_min_duration = 5   # Minimum periods for pattern
        self.absorption_threshold = 0.8  # Threshold for absorption detection
        
        # Alert parameters
        self.alert_confidence_threshold = 0.7
        self.volume_spike_threshold = 1.8
        
        # Performance optimization
        self.flow_cache: Dict[str, deque] = {}
        self.pattern_cache: Dict[str, List[FlowPattern]] = {}

    async def initialize(self) -> bool:
        """Initialize the Order Flow Imbalance engine"""
        try:
            self.logger.info("Initializing Order Flow Imbalance Engine...")
            
            # Test order flow analysis with sample data
            test_data = self._generate_test_data()
            test_result = await self._analyze_order_flow(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Order Flow Imbalance Engine initialized")
                return True
            else:
                raise Exception("Order flow analysis test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Order Flow Imbalance Engine initialization failed: {e}")
            return False

    async def analyze_order_flow_imbalance(self, symbol: str, order_flow_data: List[Dict],
                                         timeframe: str = 'M1') -> OrderFlowResult:
        """
        Analyze order flow imbalances for scalping opportunities
        
        Args:
            symbol: Currency pair symbol
            order_flow_data: List of order flow data dictionaries
            timeframe: Chart timeframe (M1-M5)
            
        Returns:
            OrderFlowResult with imbalance analysis
        """
        if not self.ready:
            raise Exception("Order Flow Imbalance Engine not initialized")
            
        if len(order_flow_data) < 10:
            raise Exception("Insufficient data for order flow analysis (minimum 10 periods)")
            
        try:
            start_time = time.time()
            
            # Prepare order flow data
            flow_data = await self._prepare_order_flow_data(order_flow_data)
            
            # Detect current imbalance
            current_imbalance = await self._detect_current_imbalance(flow_data)
            
            # Analyze imbalance history
            imbalance_history = await self._analyze_imbalance_history(flow_data)
            
            # Detect flow patterns
            flow_patterns = await self._detect_flow_patterns(flow_data)
            
            # Determine market pressure
            market_pressure = await self._determine_market_pressure(flow_data, imbalance_history)
            
            # Generate scalping alerts
            scalping_alerts = await self._generate_scalping_alerts(
                symbol, flow_data, current_imbalance, flow_patterns, timeframe
            )
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(
                flow_data, imbalance_history, scalping_alerts
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Order flow analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return OrderFlowResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                current_imbalance=current_imbalance,
                imbalance_history=imbalance_history,
                flow_patterns=flow_patterns,
                market_pressure=market_pressure,
                scalping_alerts=scalping_alerts,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Order flow analysis failed for {symbol}: {e}")
            raise

    async def _prepare_order_flow_data(self, order_flow_data: List[Dict]) -> List[OrderFlowData]:
        """Prepare order flow data structures"""
        flow_data = []
        
        for data in order_flow_data:
            timestamp = float(data.get('timestamp', time.time()))
            bid_volume = float(data.get('bid_volume', 0))
            ask_volume = float(data.get('ask_volume', 0))
            bid_price = float(data.get('bid_price', 0))
            ask_price = float(data.get('ask_price', 0))
            
            spread = ask_price - bid_price
            total_volume = bid_volume + ask_volume
            
            # Calculate imbalance ratio (avoid division by zero)
            if bid_volume > 0:
                imbalance_ratio = ask_volume / bid_volume
            elif ask_volume > 0:
                imbalance_ratio = float('inf')  # Pure ask volume
            else:
                imbalance_ratio = 1.0  # No volume
            
            flow_data.append(OrderFlowData(
                timestamp=timestamp,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                bid_price=bid_price,
                ask_price=ask_price,
                spread=spread,
                total_volume=total_volume,
                imbalance_ratio=imbalance_ratio
            ))
        
        return flow_data

    async def _detect_current_imbalance(self, flow_data: List[OrderFlowData]) -> ImbalanceSignal:
        """Detect current order flow imbalance"""
        if not flow_data:
            return ImbalanceSignal(
                signal_type='balanced',
                imbalance_strength=0.5,
                direction='neutral',
                confidence=0.5,
                duration_periods=0,
                volume_intensity=0.5,
                price_impact_probability=0.5
            )
        
        current_flow = flow_data[-1]
        
        # Determine signal type and direction
        if current_flow.imbalance_ratio > self.imbalance_threshold_strong:
            signal_type = 'buy_imbalance'
            direction = 'bullish'
            imbalance_strength = min((current_flow.imbalance_ratio - 1) / (self.imbalance_threshold_strong - 1), 1.0)
        elif current_flow.imbalance_ratio < (1 / self.imbalance_threshold_strong):
            signal_type = 'sell_imbalance'
            direction = 'bearish'
            imbalance_strength = min((1 - current_flow.imbalance_ratio) / (1 - 1/self.imbalance_threshold_strong), 1.0)
        else:
            signal_type = 'balanced'
            direction = 'neutral'
            imbalance_strength = 0.5
        
        # Calculate duration of current imbalance
        duration_periods = await self._calculate_imbalance_duration(flow_data, signal_type)
        
        # Calculate volume intensity
        if len(flow_data) >= 10:
            avg_volume = statistics.mean([f.total_volume for f in flow_data[-10:]])
            volume_intensity = min(current_flow.total_volume / avg_volume, 2.0) / 2.0
        else:
            volume_intensity = 0.5
        
        # Calculate confidence
        confidence = (imbalance_strength + volume_intensity + min(duration_periods / 5, 1.0)) / 3
        
        # Calculate price impact probability
        price_impact_probability = confidence * imbalance_strength
        
        return ImbalanceSignal(
            signal_type=signal_type,
            imbalance_strength=imbalance_strength,
            direction=direction,
            confidence=confidence,
            duration_periods=duration_periods,
            volume_intensity=volume_intensity,
            price_impact_probability=price_impact_probability
        )

    async def _calculate_imbalance_duration(self, flow_data: List[OrderFlowData], 
                                          signal_type: str) -> int:
        """Calculate how long the current imbalance has persisted"""
        duration = 0
        
        # Look backwards from current data
        for i in range(len(flow_data) - 1, -1, -1):
            flow = flow_data[i]
            
            current_signal = await self._classify_imbalance(flow)
            
            if current_signal == signal_type:
                duration += 1
            else:
                break
        
        return duration

    async def _classify_imbalance(self, flow: OrderFlowData) -> str:
        """Classify a single flow data point"""
        if flow.imbalance_ratio > self.imbalance_threshold_medium:
            return 'buy_imbalance'
        elif flow.imbalance_ratio < (1 / self.imbalance_threshold_medium):
            return 'sell_imbalance'
        else:
            return 'balanced'

    async def _analyze_imbalance_history(self, flow_data: List[OrderFlowData]) -> List[ImbalanceSignal]:
        """Analyze historical imbalance patterns"""
        history = []
        
        # Analyze last 20 periods or available data
        analysis_periods = min(20, len(flow_data))
        
        for i in range(len(flow_data) - analysis_periods, len(flow_data)):
            if i < 0:
                continue
                
            # Create a subset for analysis
            subset = flow_data[max(0, i-5):i+1]
            
            if subset:
                imbalance = await self._detect_current_imbalance(subset)
                history.append(imbalance)
        
        return history

    async def _detect_flow_patterns(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect order flow patterns"""
        patterns = []
        
        if len(flow_data) < self.pattern_min_duration:
            return patterns
        
        # Detect aggressive buying patterns
        patterns.extend(await self._detect_aggressive_buying(flow_data))
        
        # Detect aggressive selling patterns
        patterns.extend(await self._detect_aggressive_selling(flow_data))
        
        # Detect absorption patterns
        patterns.extend(await self._detect_absorption_patterns(flow_data))
        
        # Detect exhaustion patterns
        patterns.extend(await self._detect_exhaustion_patterns(flow_data))
        
        return patterns

    async def _detect_aggressive_buying(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect aggressive buying patterns"""
        patterns = []
        
        # Look for sustained periods of buy imbalance with increasing volume
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            # Check for sustained buy imbalance
            buy_imbalance_count = sum(1 for f in window if f.imbalance_ratio > self.imbalance_threshold_medium)
            
            if buy_imbalance_count >= self.pattern_min_duration * 0.7:  # 70% of periods
                total_volume = sum(f.total_volume for f in window)
                avg_imbalance = statistics.mean([f.imbalance_ratio for f in window])
                
                if total_volume >= self.pattern_min_volume:
                    # Calculate price movement (simplified)
                    price_movement = (window[-1].ask_price - window[0].bid_price) / window[0].bid_price
                    
                    pattern_strength = min(avg_imbalance / self.imbalance_threshold_strong, 1.0)
                    
                    patterns.append(FlowPattern(
                        pattern_type='aggressive_buying',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_movement,
                        pattern_strength=pattern_strength
                    ))
        
        return patterns

    async def _detect_aggressive_selling(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect aggressive selling patterns"""
        patterns = []
        
        # Look for sustained periods of sell imbalance with increasing volume
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            # Check for sustained sell imbalance
            sell_imbalance_count = sum(1 for f in window if f.imbalance_ratio < (1 / self.imbalance_threshold_medium))
            
            if sell_imbalance_count >= self.pattern_min_duration * 0.7:  # 70% of periods
                total_volume = sum(f.total_volume for f in window)
                avg_imbalance = statistics.mean([1/f.imbalance_ratio if f.imbalance_ratio > 0 else 1 for f in window])
                
                if total_volume >= self.pattern_min_volume:
                    # Calculate price movement (simplified)
                    price_movement = (window[-1].bid_price - window[0].ask_price) / window[0].ask_price
                    
                    pattern_strength = min(avg_imbalance / self.imbalance_threshold_strong, 1.0)
                    
                    patterns.append(FlowPattern(
                        pattern_type='aggressive_selling',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_movement,
                        pattern_strength=pattern_strength
                    ))
        
        return patterns

    async def _detect_absorption_patterns(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect volume absorption patterns"""
        patterns = []
        
        # Look for high volume with minimal price movement
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            total_volume = sum(f.total_volume for f in window)
            avg_volume = statistics.mean([f.total_volume for f in flow_data[max(0, i-10):i]])
            
            if total_volume > avg_volume * self.volume_spike_threshold:
                # Check for minimal price movement
                price_range = max(f.ask_price for f in window) - min(f.bid_price for f in window)
                avg_spread = statistics.mean([f.spread for f in window])
                
                if price_range <= avg_spread * 2:  # Price movement within 2x average spread
                    avg_imbalance = statistics.mean([f.imbalance_ratio for f in window])
                    pattern_strength = min(total_volume / (avg_volume * self.volume_spike_threshold), 1.0)
                    
                    patterns.append(FlowPattern(
                        pattern_type='absorption',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_range,
                        pattern_strength=pattern_strength
                    ))
        
        return patterns

    async def _detect_exhaustion_patterns(self, flow_data: List[OrderFlowData]) -> List[FlowPattern]:
        """Detect exhaustion patterns"""
        patterns = []
        
        # Look for decreasing volume with continued imbalance
        for i in range(len(flow_data) - self.pattern_min_duration):
            window = flow_data[i:i + self.pattern_min_duration]
            
            # Check for volume decline
            volumes = [f.total_volume for f in window]
            volume_trend = await self._calculate_trend(volumes)
            
            if volume_trend < -0.1:  # Declining volume
                # Check for sustained imbalance
                imbalances = [f.imbalance_ratio for f in window]
                avg_imbalance = statistics.mean(imbalances)
                
                if avg_imbalance > self.imbalance_threshold_medium or avg_imbalance < (1 / self.imbalance_threshold_medium):
                    total_volume = sum(volumes)
                    price_movement = (window[-1].ask_price - window[0].bid_price) / window[0].bid_price
                    pattern_strength = abs(volume_trend) * 0.5 + (abs(avg_imbalance - 1) * 0.5)
                    
                    patterns.append(FlowPattern(
                        pattern_type='exhaustion',
                        start_time=window[0].timestamp,
                        end_time=window[-1].timestamp,
                        total_volume=total_volume,
                        average_imbalance=avg_imbalance,
                        price_movement=price_movement,
                        pattern_strength=min(pattern_strength, 1.0)
                    ))
        
        return patterns

    async def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        try:
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0

    async def _determine_market_pressure(self, flow_data: List[OrderFlowData], 
                                       history: List[ImbalanceSignal]) -> str:
        """Determine overall market pressure"""
        if not history:
            return 'balanced'
        
        # Count recent imbalance types
        recent_signals = history[-10:] if len(history) >= 10 else history
        
        buy_pressure = sum(1 for s in recent_signals if s.signal_type == 'buy_imbalance')
        sell_pressure = sum(1 for s in recent_signals if s.signal_type == 'sell_imbalance')
        
        if buy_pressure > sell_pressure * 1.5:
            return 'buying_pressure'
        elif sell_pressure > buy_pressure * 1.5:
            return 'selling_pressure'
        else:
            return 'balanced'

    async def _generate_scalping_alerts(self, symbol: str, flow_data: List[OrderFlowData],
                                      current_imbalance: ImbalanceSignal,
                                      patterns: List[FlowPattern],
                                      timeframe: str) -> List[Dict[str, Union[str, float]]]:
        """Generate scalping alerts based on order flow analysis"""
        alerts = []
        
        # Strong imbalance alerts
        if (current_imbalance.confidence >= self.alert_confidence_threshold and 
            current_imbalance.imbalance_strength > 0.7):
            
            alerts.append({
                'type': 'order_flow_imbalance',
                'signal': 'buy' if current_imbalance.direction == 'bullish' else 'sell',
                'confidence': current_imbalance.confidence,
                'imbalance_strength': current_imbalance.imbalance_strength,
                'duration': current_imbalance.duration_periods,
                'volume_intensity': current_imbalance.volume_intensity,
                'price_impact_probability': current_imbalance.price_impact_probability,
                'timeframe': timeframe,
                'alert_reason': 'strong_order_flow_imbalance'
            })
        
        # Pattern-based alerts
        for pattern in patterns:
            if pattern.pattern_strength > 0.7:
                if pattern.pattern_type == 'aggressive_buying':
                    alerts.append({
                        'type': 'aggressive_buying',
                        'signal': 'buy',
                        'confidence': pattern.pattern_strength,
                        'pattern_duration': (pattern.end_time - pattern.start_time) / 60,  # minutes
                        'total_volume': pattern.total_volume,
                        'timeframe': timeframe,
                        'alert_reason': 'aggressive_buying_detected'
                    })
                elif pattern.pattern_type == 'aggressive_selling':
                    alerts.append({
                        'type': 'aggressive_selling',
                        'signal': 'sell',
                        'confidence': pattern.pattern_strength,
                        'pattern_duration': (pattern.end_time - pattern.start_time) / 60,  # minutes
                        'total_volume': pattern.total_volume,
                        'timeframe': timeframe,
                        'alert_reason': 'aggressive_selling_detected'
                    })
                elif pattern.pattern_type == 'absorption':
                    alerts.append({
                        'type': 'volume_absorption',
                        'signal': 'reversal_pending',
                        'confidence': pattern.pattern_strength,
                        'absorption_volume': pattern.total_volume,
                        'timeframe': timeframe,
                        'alert_reason': 'volume_absorption_detected'
                    })
        
        return alerts

    async def _calculate_execution_metrics(self, flow_data: List[OrderFlowData],
                                         history: List[ImbalanceSignal],
                                         alerts: List[Dict]) -> Dict[str, float]:
        """Calculate execution metrics for order flow analysis"""
        metrics = {
            'total_volume': sum(f.total_volume for f in flow_data),
            'avg_imbalance_ratio': statistics.mean([f.imbalance_ratio for f in flow_data]) if flow_data else 1.0,
            'imbalance_periods': len([s for s in history if s.signal_type != 'balanced']),
            'alert_count': len(alerts),
            'avg_alert_confidence': statistics.mean([a['confidence'] for a in alerts]) if alerts else 0.0,
            'buying_pressure_ratio': len([s for s in history if s.direction == 'bullish']) / len(history) if history else 0.5,
            'selling_pressure_ratio': len([s for s in history if s.direction == 'bearish']) / len(history) if history else 0.5
        }
        
        return metrics

    def _generate_test_data(self) -> List[Dict]:
        """Generate test data for initialization"""
        test_data = []
        base_bid = 1.0998
        base_ask = 1.1002
        
        for i in range(20):
            # Create varying imbalances
            if i % 7 == 0:  # Occasional strong imbalance
                bid_volume = np.random.uniform(500, 1000)
                ask_volume = np.random.uniform(1500, 2500)  # Strong buy imbalance
            elif i % 5 == 0:
                bid_volume = np.random.uniform(1500, 2500)  # Strong sell imbalance
                ask_volume = np.random.uniform(500, 1000)
            else:
                bid_volume = np.random.uniform(800, 1200)
                ask_volume = np.random.uniform(800, 1200)
            
            # Small price movements
            price_change = (np.random.random() - 0.5) * 0.0002
            bid_price = base_bid + price_change
            ask_price = base_ask + price_change
            
            test_data.append({
                'timestamp': time.time() - (20 - i) * 60,  # M1 intervals
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'bid_price': bid_price,
                'ask_price': ask_price
            })
            
        return test_data

    async def _analyze_order_flow(self, test_data: List[Dict]) -> List[ImbalanceSignal]:
        """Test order flow analysis"""
        try:
            flow_data = await self._prepare_order_flow_data(test_data)
            return await self._analyze_imbalance_history(flow_data)
        except Exception:
            return []
