"""
Smart Money Indicators Module
Institutional flow detection for day trading
Optimized for smart money flow detection and institutional activity analysis.
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
class InstitutionalActivity:
    """Institutional trading activity detection"""
    activity_type: str  # 'accumulation', 'distribution', 'markup', 'markdown'
    intensity: float  # 0-1
    confidence: float
    volume_signature: str  # 'stealth', 'aggressive', 'iceberg'
    time_pattern: str  # 'session_start', 'session_end', 'overlap', 'off_hours'
    price_impact: float
    duration_minutes: float


@dataclass
class SmartMoneySignal:
    """Smart money trading signal"""
    signal_type: str  # 'follow_smart_money', 'fade_retail', 'breakout_confirmation'
    direction: str  # 'bullish', 'bearish'
    strength: float  # 0-1
    confidence: float
    institutional_evidence: List[str]
    retail_sentiment: str  # 'bullish', 'bearish', 'neutral'
    divergence_detected: bool


@dataclass
class VolumeFootprint:
    """Volume footprint analysis"""
    timestamp: float
    price_level: float
    bid_volume: float
    ask_volume: float
    delta: float
    cumulative_delta: float
    absorption_detected: bool
    iceberg_detected: bool


@dataclass
class MarketStructure:
    """Market structure from smart money perspective"""
    structure_type: str  # 'accumulation_phase', 'markup_phase', 'distribution_phase', 'markdown_phase'
    phase_strength: float
    institutional_participation: float
    retail_participation: float
    structure_break_probability: float
    next_phase_prediction: str


@dataclass
class SmartMoneyResult:
    """Smart money indicators analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    institutional_activity: InstitutionalActivity
    smart_money_signals: List[SmartMoneySignal]
    volume_footprint: List[VolumeFootprint]
    market_structure: MarketStructure
    flow_direction: str  # 'institutional_buying', 'institutional_selling', 'neutral'
    trading_recommendations: List[Dict[str, Union[str, float]]]


class SmartMoneyIndicators:
    """
    Smart Money Indicators Engine for Day Trading
    Provides institutional flow detection and smart money analysis
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Smart money detection parameters
        self.large_volume_threshold = 2.5  # 2.5x average volume
        self.stealth_volume_threshold = 0.8  # Below average for stealth
        self.iceberg_detection_periods = 5
        self.absorption_threshold = 0.7
        
        # Institutional timing patterns
        self.institutional_hours = {
            'london_open': (8, 10),    # UTC hours
            'ny_open': (13, 15),
            'overlap': (13, 17),
            'london_close': (16, 17),
            'ny_close': (21, 22)
        }
        
        # Market structure parameters
        self.structure_confirmation_periods = 10
        self.phase_transition_threshold = 0.75
        
        # Performance optimization
        self.smart_money_cache: Dict[str, deque] = {}
        self.structure_cache: Dict[str, MarketStructure] = {}

    async def initialize(self) -> bool:
        """Initialize the Smart Money Indicators engine"""
        try:
            self.logger.info("Initializing Smart Money Indicators Engine...")
            
            # Test smart money analysis with sample data
            test_data = self._generate_test_data()
            test_result = await self._detect_institutional_activity(test_data)
            
            if test_result:
                self.ready = True
                self.logger.info("✅ Smart Money Indicators Engine initialized")
                return True
            else:
                raise Exception("Smart money analysis test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Smart Money Indicators Engine initialization failed: {e}")
            return False

    async def analyze_smart_money_flow(self, symbol: str, price_data: List[Dict], 
                                     volume_data: List[Dict], order_flow_data: Optional[List[Dict]] = None,
                                     timeframe: str = 'M15') -> SmartMoneyResult:
        """
        Analyze smart money flow and institutional activity
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            volume_data: List of volume data dictionaries
            order_flow_data: Optional order flow data
            timeframe: Chart timeframe (M15-H1)
            
        Returns:
            SmartMoneyResult with smart money analysis
        """
        if not self.ready:
            raise Exception("Smart Money Indicators Engine not initialized")
            
        if len(price_data) < 20 or len(volume_data) < 20:
            raise Exception("Insufficient data for smart money analysis (minimum 20 periods)")
            
        try:
            start_time = time.time()
            
            # Detect institutional activity
            institutional_activity = await self._detect_institutional_activity(
                price_data, volume_data
            )
            
            # Generate smart money signals
            smart_money_signals = await self._generate_smart_money_signals(
                price_data, volume_data, institutional_activity
            )
            
            # Analyze volume footprint
            volume_footprint = await self._analyze_volume_footprint(
                price_data, volume_data, order_flow_data
            )
            
            # Analyze market structure
            market_structure = await self._analyze_market_structure(
                price_data, volume_data, institutional_activity
            )
            
            # Determine flow direction
            flow_direction = await self._determine_flow_direction(
                institutional_activity, smart_money_signals
            )
            
            # Generate trading recommendations
            trading_recommendations = await self._generate_trading_recommendations(
                symbol, institutional_activity, smart_money_signals, market_structure, timeframe
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Smart money analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return SmartMoneyResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                institutional_activity=institutional_activity,
                smart_money_signals=smart_money_signals,
                volume_footprint=volume_footprint,
                market_structure=market_structure,
                flow_direction=flow_direction,
                trading_recommendations=trading_recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Smart money analysis failed for {symbol}: {e}")
            raise

    async def _detect_institutional_activity(self, price_data: List[Dict], 
                                           volume_data: List[Dict]) -> InstitutionalActivity:
        """Detect institutional trading activity patterns"""
        if len(price_data) < 10:
            return InstitutionalActivity(
                activity_type='neutral',
                intensity=0.5,
                confidence=0.5,
                volume_signature='normal',
                time_pattern='regular',
                price_impact=0.0,
                duration_minutes=0.0
            )
        
        # Calculate volume characteristics
        volumes = [float(data.get('volume', 0)) for data in volume_data]
        avg_volume = statistics.mean(volumes)
        
        # Detect large volume periods
        large_volume_periods = []
        stealth_volume_periods = []
        
        for i, (price_bar, volume_bar) in enumerate(zip(price_data, volume_data)):
            volume = float(volume_bar.get('volume', 0))
            timestamp = float(price_bar.get('timestamp', time.time()))
            
            if volume > avg_volume * self.large_volume_threshold:
                large_volume_periods.append((i, timestamp, volume))
            elif volume < avg_volume * self.stealth_volume_threshold:
                stealth_volume_periods.append((i, timestamp, volume))
        
        # Analyze timing patterns
        time_pattern = await self._analyze_timing_patterns(price_data)
        
        # Detect volume signature
        volume_signature = await self._detect_volume_signature(volumes, price_data)
        
        # Calculate price impact
        price_impact = await self._calculate_price_impact(price_data, large_volume_periods)
        
        # Determine activity type
        activity_type = await self._determine_activity_type(
            price_data, large_volume_periods, stealth_volume_periods
        )
        
        # Calculate intensity and confidence
        intensity = len(large_volume_periods) / len(price_data)
        confidence = min(intensity * 2, 1.0) if large_volume_periods else 0.3
        
        # Calculate duration
        if large_volume_periods:
            duration_minutes = (large_volume_periods[-1][1] - large_volume_periods[0][1]) / 60
        else:
            duration_minutes = 0.0
        
        return InstitutionalActivity(
            activity_type=activity_type,
            intensity=intensity,
            confidence=confidence,
            volume_signature=volume_signature,
            time_pattern=time_pattern,
            price_impact=price_impact,
            duration_minutes=duration_minutes
        )

    async def _analyze_timing_patterns(self, price_data: List[Dict]) -> str:
        """Analyze timing patterns for institutional activity"""
        institutional_time_count = 0
        
        for price_bar in price_data:
            timestamp = float(price_bar.get('timestamp', time.time()))
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            
            # Check if timestamp falls within institutional hours
            for period_name, (start_hour, end_hour) in self.institutional_hours.items():
                if start_hour <= hour < end_hour:
                    institutional_time_count += 1
                    break
        
        institutional_ratio = institutional_time_count / len(price_data)
        
        if institutional_ratio > 0.7:
            return 'institutional_hours'
        elif institutional_ratio > 0.4:
            return 'mixed_hours'
        else:
            return 'off_hours'

    async def _detect_volume_signature(self, volumes: List[float], 
                                     price_data: List[Dict]) -> str:
        """Detect volume signature patterns"""
        avg_volume = statistics.mean(volumes)
        
        # Check for iceberg orders (consistent volume at similar levels)
        iceberg_count = 0
        for i in range(len(volumes) - self.iceberg_detection_periods):
            window = volumes[i:i + self.iceberg_detection_periods]
            volume_consistency = 1 - (statistics.stdev(window) / statistics.mean(window))
            
            if volume_consistency > 0.8 and all(v > avg_volume * 1.2 for v in window):
                iceberg_count += 1
        
        if iceberg_count > 2:
            return 'iceberg'
        
        # Check for stealth trading (below average volume with price movement)
        stealth_count = 0
        for i, (price_bar, volume) in enumerate(zip(price_data, volumes)):
            if i == 0:
                continue
                
            prev_close = float(price_data[i-1].get('close', 0))
            current_close = float(price_bar.get('close', 0))
            price_change = abs(current_close - prev_close) / prev_close
            
            if volume < avg_volume * 0.8 and price_change > 0.001:  # Low volume, significant price move
                stealth_count += 1
        
        if stealth_count > len(volumes) * 0.3:
            return 'stealth'
        
        # Check for aggressive trading (high volume spikes)
        aggressive_count = sum(1 for v in volumes if v > avg_volume * self.large_volume_threshold)
        
        if aggressive_count > len(volumes) * 0.2:
            return 'aggressive'
        
        return 'normal'

    async def _calculate_price_impact(self, price_data: List[Dict], 
                                    large_volume_periods: List[Tuple]) -> float:
        """Calculate price impact of large volume periods"""
        if not large_volume_periods:
            return 0.0
        
        total_impact = 0.0
        
        for i, timestamp, volume in large_volume_periods:
            if i == 0 or i >= len(price_data) - 1:
                continue
                
            before_price = float(price_data[i-1].get('close', 0))
            after_price = float(price_data[i+1].get('close', 0))
            
            if before_price > 0:
                impact = abs(after_price - before_price) / before_price
                total_impact += impact
        
        return total_impact / len(large_volume_periods) if large_volume_periods else 0.0

    async def _determine_activity_type(self, price_data: List[Dict], 
                                     large_volume_periods: List[Tuple],
                                     stealth_volume_periods: List[Tuple]) -> str:
        """Determine the type of institutional activity"""
        if not price_data:
            return 'neutral'
        
        # Calculate overall price trend
        start_price = float(price_data[0].get('close', 0))
        end_price = float(price_data[-1].get('close', 0))
        
        if start_price == 0:
            return 'neutral'
        
        price_change = (end_price - start_price) / start_price
        
        # Analyze volume patterns
        large_volume_ratio = len(large_volume_periods) / len(price_data)
        stealth_volume_ratio = len(stealth_volume_periods) / len(price_data)
        
        # Determine activity type based on price movement and volume patterns
        if price_change > 0.005:  # Significant upward movement
            if large_volume_ratio > 0.3:
                return 'markup'  # Aggressive buying
            elif stealth_volume_ratio > 0.4:
                return 'accumulation'  # Stealth accumulation
            else:
                return 'markup'
        elif price_change < -0.005:  # Significant downward movement
            if large_volume_ratio > 0.3:
                return 'markdown'  # Aggressive selling
            elif stealth_volume_ratio > 0.4:
                return 'distribution'  # Stealth distribution
            else:
                return 'markdown'
        else:  # Sideways movement
            if large_volume_ratio > 0.2:
                return 'accumulation'  # Accumulation phase
            else:
                return 'neutral'

    async def _generate_smart_money_signals(self, price_data: List[Dict], 
                                          volume_data: List[Dict],
                                          institutional_activity: InstitutionalActivity) -> List[SmartMoneySignal]:
        """Generate smart money trading signals"""
        signals = []
        
        # Follow smart money signal
        if institutional_activity.confidence > 0.7:
            if institutional_activity.activity_type in ['accumulation', 'markup']:
                direction = 'bullish'
                evidence = ['institutional_buying', 'volume_accumulation']
            elif institutional_activity.activity_type in ['distribution', 'markdown']:
                direction = 'bearish'
                evidence = ['institutional_selling', 'volume_distribution']
            else:
                direction = 'neutral'
                evidence = ['neutral_institutional_activity']
            
            if direction != 'neutral':
                signals.append(SmartMoneySignal(
                    signal_type='follow_smart_money',
                    direction=direction,
                    strength=institutional_activity.intensity,
                    confidence=institutional_activity.confidence,
                    institutional_evidence=evidence,
                    retail_sentiment='neutral',  # Simplified
                    divergence_detected=False
                ))
        
        # Stealth accumulation/distribution signals
        if institutional_activity.volume_signature == 'stealth':
            signals.append(SmartMoneySignal(
                signal_type='stealth_activity',
                direction='bullish' if institutional_activity.activity_type == 'accumulation' else 'bearish',
                strength=0.8,
                confidence=0.7,
                institutional_evidence=['stealth_volume', 'minimal_price_impact'],
                retail_sentiment='neutral',
                divergence_detected=True
            ))
        
        # Iceberg order signals
        if institutional_activity.volume_signature == 'iceberg':
            signals.append(SmartMoneySignal(
                signal_type='iceberg_orders',
                direction='bullish' if institutional_activity.activity_type in ['accumulation', 'markup'] else 'bearish',
                strength=0.9,
                confidence=0.8,
                institutional_evidence=['iceberg_orders', 'consistent_volume'],
                retail_sentiment='neutral',
                divergence_detected=False
            ))
        
        return signals

    async def _analyze_volume_footprint(self, price_data: List[Dict], 
                                      volume_data: List[Dict],
                                      order_flow_data: Optional[List[Dict]]) -> List[VolumeFootprint]:
        """Analyze volume footprint for smart money detection"""
        footprint = []
        cumulative_delta = 0.0
        
        for i, (price_bar, volume_bar) in enumerate(zip(price_data, volume_data)):
            timestamp = float(price_bar.get('timestamp', time.time()))
            close_price = float(price_bar.get('close', 0))
            volume = float(volume_bar.get('volume', 0))
            
            # Estimate bid/ask volume (simplified without order flow data)
            if order_flow_data and i < len(order_flow_data):
                bid_volume = float(order_flow_data[i].get('bid_volume', volume * 0.5))
                ask_volume = float(order_flow_data[i].get('ask_volume', volume * 0.5))
            else:
                # Estimate based on price action
                open_price = float(price_bar.get('open', close_price))
                if close_price > open_price:
                    ask_volume = volume * 0.6
                    bid_volume = volume * 0.4
                elif close_price < open_price:
                    ask_volume = volume * 0.4
                    bid_volume = volume * 0.6
                else:
                    ask_volume = volume * 0.5
                    bid_volume = volume * 0.5
            
            delta = ask_volume - bid_volume
            cumulative_delta += delta
            
            # Detect absorption
            absorption_detected = await self._detect_absorption(
                price_data[max(0, i-5):i+1], volume_data[max(0, i-5):i+1]
            )
            
            # Detect iceberg orders
            iceberg_detected = await self._detect_iceberg_orders(
                volume_data[max(0, i-5):i+1]
            )
            
            footprint.append(VolumeFootprint(
                timestamp=timestamp,
                price_level=close_price,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                delta=delta,
                cumulative_delta=cumulative_delta,
                absorption_detected=absorption_detected,
                iceberg_detected=iceberg_detected
            ))
        
        return footprint

    async def _detect_absorption(self, price_window: List[Dict], 
                               volume_window: List[Dict]) -> bool:
        """Detect volume absorption patterns"""
        if len(price_window) < 3 or len(volume_window) < 3:
            return False
        
        # Check for high volume with minimal price movement
        total_volume = sum(float(v.get('volume', 0)) for v in volume_window)
        
        prices = [float(p.get('close', 0)) for p in price_window]
        price_range = max(prices) - min(prices)
        avg_price = statistics.mean(prices)
        
        # Absorption: high volume, low price movement
        if total_volume > 0 and avg_price > 0:
            volume_intensity = total_volume / len(volume_window)
            price_movement_ratio = price_range / avg_price
            
            return volume_intensity > 1000 and price_movement_ratio < 0.002  # Simplified thresholds
        
        return False

    async def _detect_iceberg_orders(self, volume_window: List[Dict]) -> bool:
        """Detect iceberg order patterns"""
        if len(volume_window) < self.iceberg_detection_periods:
            return False
        
        volumes = [float(v.get('volume', 0)) for v in volume_window]
        
        # Check for consistent volume levels
        if len(volumes) > 1:
            volume_consistency = 1 - (statistics.stdev(volumes) / statistics.mean(volumes))
            avg_volume = statistics.mean(volumes)
            
            return volume_consistency > 0.8 and avg_volume > 1200  # Simplified thresholds
        
        return False

    async def _analyze_market_structure(self, price_data: List[Dict], 
                                      volume_data: List[Dict],
                                      institutional_activity: InstitutionalActivity) -> MarketStructure:
        """Analyze market structure from smart money perspective"""
        # Determine current phase
        structure_type = institutional_activity.activity_type + '_phase'
        
        # Calculate phase strength
        phase_strength = institutional_activity.intensity
        
        # Estimate participation levels
        institutional_participation = institutional_activity.confidence
        retail_participation = 1.0 - institutional_participation
        
        # Calculate structure break probability
        if institutional_activity.activity_type in ['markup', 'markdown']:
            structure_break_probability = 0.7
        else:
            structure_break_probability = 0.3
        
        # Predict next phase
        phase_transitions = {
            'accumulation': 'markup',
            'markup': 'distribution',
            'distribution': 'markdown',
            'markdown': 'accumulation'
        }
        
        next_phase_prediction = phase_transitions.get(institutional_activity.activity_type, 'neutral')
        
        return MarketStructure(
            structure_type=structure_type,
            phase_strength=phase_strength,
            institutional_participation=institutional_participation,
            retail_participation=retail_participation,
            structure_break_probability=structure_break_probability,
            next_phase_prediction=next_phase_prediction
        )

    async def _determine_flow_direction(self, institutional_activity: InstitutionalActivity,
                                      signals: List[SmartMoneySignal]) -> str:
        """Determine overall smart money flow direction"""
        if institutional_activity.activity_type in ['accumulation', 'markup']:
            return 'institutional_buying'
        elif institutional_activity.activity_type in ['distribution', 'markdown']:
            return 'institutional_selling'
        else:
            return 'neutral'

    async def _generate_trading_recommendations(self, symbol: str, 
                                              institutional_activity: InstitutionalActivity,
                                              signals: List[SmartMoneySignal],
                                              market_structure: MarketStructure,
                                              timeframe: str) -> List[Dict[str, Union[str, float]]]:
        """Generate trading recommendations based on smart money analysis"""
        recommendations = []
        
        # Follow institutional flow
        if institutional_activity.confidence > 0.7:
            if institutional_activity.activity_type == 'accumulation':
                recommendations.append({
                    'type': 'smart_money_follow',
                    'action': 'buy',
                    'confidence': institutional_activity.confidence,
                    'reasoning': 'institutional_accumulation_detected',
                    'timeframe': timeframe,
                    'risk_level': 'medium',
                    'expected_duration': 'medium_term'
                })
            elif institutional_activity.activity_type == 'distribution':
                recommendations.append({
                    'type': 'smart_money_follow',
                    'action': 'sell',
                    'confidence': institutional_activity.confidence,
                    'reasoning': 'institutional_distribution_detected',
                    'timeframe': timeframe,
                    'risk_level': 'medium',
                    'expected_duration': 'medium_term'
                })
        
        # Structure break recommendations
        if market_structure.structure_break_probability > 0.6:
            recommendations.append({
                'type': 'structure_break',
                'action': 'prepare_breakout',
                'confidence': market_structure.structure_break_probability,
                'reasoning': 'high_structure_break_probability',
                'timeframe': timeframe,
                'risk_level': 'high',
                'expected_phase': market_structure.next_phase_prediction
            })
        
        # Stealth activity recommendations
        stealth_signals = [s for s in signals if s.signal_type == 'stealth_activity']
        if stealth_signals:
            signal = stealth_signals[0]
            recommendations.append({
                'type': 'stealth_follow',
                'action': 'buy' if signal.direction == 'bullish' else 'sell',
                'confidence': signal.confidence,
                'reasoning': 'stealth_institutional_activity',
                'timeframe': timeframe,
                'risk_level': 'low',
                'stealth_strength': signal.strength
            })
        
        return recommendations

    def _generate_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate test data for initialization"""
        price_data = []
        volume_data = []
        base_price = 1.1000
        
        for i in range(30):
            # Create institutional-like patterns
            if i % 10 == 0:  # Occasional institutional activity
                volume = np.random.uniform(2000, 3000)  # High volume
                price_change = np.random.uniform(0.0005, 0.0015)  # Significant move
            else:
                volume = np.random.uniform(800, 1200)  # Normal volume
                price_change = np.random.uniform(-0.0003, 0.0003)  # Small move
            
            price = base_price + price_change
            
            price_data.append({
                'timestamp': time.time() - (30 - i) * 900,  # M15 intervals
                'open': price,
                'high': price + 0.0002,
                'low': price - 0.0002,
                'close': price
            })
            
            volume_data.append({
                'timestamp': time.time() - (30 - i) * 900,
                'volume': volume
            })
            
        return price_data, volume_data
