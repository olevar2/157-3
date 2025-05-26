"""
Order Book Analysis Module
Level 2 data analysis for scalping strategies with bid/ask depth analysis.
Optimized for M1-M5 scalping with real-time order flow detection.
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
class OrderBookLevel:
    """Individual order book level data"""
    price: float
    size: float
    orders: int
    side: str  # 'bid' or 'ask'


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: float
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    total_bid_volume: float
    total_ask_volume: float


@dataclass
class OrderFlowSignal:
    """Order flow-based scalping signal"""
    timestamp: float
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    imbalance_ratio: float
    depth_advantage: str  # 'bid', 'ask', 'neutral'
    liquidity_score: float
    execution_probability: float


@dataclass
class OrderBookAnalysisResult:
    """Complete order book analysis result"""
    symbol: str
    timestamp: float
    current_snapshot: OrderBookSnapshot
    imbalance_analysis: Dict[str, float]
    liquidity_metrics: Dict[str, float]
    signals: List[OrderFlowSignal]
    depth_analysis: Dict[str, float]
    execution_metrics: Dict[str, float]


class OrderBookAnalysis:
    """
    Order Book Analysis Engine for Scalping
    Provides level 2 data analysis and order flow detection for M1-M5 strategies
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False

        # Configuration for order book analysis
        self.depth_levels = 10  # Analyze top 10 levels
        self.imbalance_threshold = 0.3  # 30% imbalance for signals
        self.liquidity_threshold = 100.0  # Minimum liquidity requirement
        self.snapshot_buffer_size = 100  # Keep last 100 snapshots
        
        # Data storage
        self.order_book_history: Dict[str, deque] = {}
        self.imbalance_history: Dict[str, deque] = {}
        self.signals_history: Dict[str, deque] = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0

    async def initialize(self) -> bool:
        """Initialize order book analysis engine"""
        try:
            self.logger.info("Initializing Order Book Analysis Engine...")
            
            # Test order book calculations
            test_bids = [OrderBookLevel(1.1000, 100, 5, 'bid'), OrderBookLevel(1.0999, 150, 3, 'bid')]
            test_asks = [OrderBookLevel(1.1001, 120, 4, 'ask'), OrderBookLevel(1.1002, 80, 2, 'ask')]
            test_imbalance = self._calculate_order_imbalance(test_bids, test_asks)
            
            if test_imbalance is not None:
                self.ready = True
                self.logger.info("✅ Order Book Analysis Engine initialized successfully")
                return True
            else:
                raise ValueError("Order book calculation test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Order Book Analysis Engine initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if engine is ready"""
        return self.ready

    async def analyze_order_book(self, symbol: str, order_book_data: Dict) -> OrderBookAnalysisResult:
        """
        Main order book analysis function
        """
        if not self.ready:
            raise RuntimeError("Order Book Analysis Engine not initialized")

        start_time = time.time()
        
        try:
            # Initialize data buffers if needed
            if symbol not in self.order_book_history:
                self._initialize_symbol_buffers(symbol)
            
            # Parse order book snapshot
            snapshot = await self._parse_order_book_snapshot(symbol, order_book_data)
            
            # Store snapshot in history
            self.order_book_history[symbol].append(snapshot)
            
            # Analyze order imbalance
            imbalance_analysis = await self._analyze_order_imbalance(snapshot)
            
            # Calculate liquidity metrics
            liquidity_metrics = await self._calculate_liquidity_metrics(snapshot)
            
            # Perform depth analysis
            depth_analysis = await self._analyze_market_depth(snapshot)
            
            # Generate order flow signals
            signals = await self._generate_order_flow_signals(symbol, snapshot, imbalance_analysis)
            
            # Calculate execution metrics
            execution_metrics = await self._calculate_execution_metrics(snapshot, signals)
            
            # Update performance tracking
            analysis_time = time.time() - start_time
            self.analysis_count += 1
            self.total_analysis_time += analysis_time
            
            return OrderBookAnalysisResult(
                symbol=symbol,
                timestamp=time.time(),
                current_snapshot=snapshot,
                imbalance_analysis=imbalance_analysis,
                liquidity_metrics=liquidity_metrics,
                signals=signals,
                depth_analysis=depth_analysis,
                execution_metrics=execution_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Order book analysis failed for {symbol}: {e}")
            raise

    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize data buffers for a symbol"""
        self.order_book_history[symbol] = deque(maxlen=self.snapshot_buffer_size)
        self.imbalance_history[symbol] = deque(maxlen=self.snapshot_buffer_size)
        self.signals_history[symbol] = deque(maxlen=self.snapshot_buffer_size)

    async def _parse_order_book_snapshot(self, symbol: str, order_book_data: Dict) -> OrderBookSnapshot:
        """Parse raw order book data into structured snapshot"""
        timestamp = time.time()
        
        # Parse bids (buy orders)
        bids = []
        bid_data = order_book_data.get('bids', [])
        for i, bid in enumerate(bid_data[:self.depth_levels]):
            if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                price = float(bid[0])
                size = float(bid[1])
                orders = int(bid[2]) if len(bid) > 2 else 1
                bids.append(OrderBookLevel(price, size, orders, 'bid'))
        
        # Parse asks (sell orders)
        asks = []
        ask_data = order_book_data.get('asks', [])
        for i, ask in enumerate(ask_data[:self.depth_levels]):
            if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                price = float(ask[0])
                size = float(ask[1])
                orders = int(ask[2]) if len(ask) > 2 else 1
                asks.append(OrderBookLevel(price, size, orders, 'ask'))
        
        # Calculate derived metrics
        best_bid = bids[0].price if bids else 0.0
        best_ask = asks[0].price if asks else 0.0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
        
        total_bid_volume = sum(bid.size for bid in bids)
        total_ask_volume = sum(ask.size for ask in asks)
        
        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bids=bids,
            asks=asks,
            spread=spread,
            mid_price=mid_price,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume
        )

    def _calculate_order_imbalance(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel]) -> Optional[float]:
        """Calculate order imbalance ratio"""
        if not bids or not asks:
            return None
        
        bid_volume = sum(bid.size for bid in bids)
        ask_volume = sum(ask.size for ask in asks)
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        
        return None

    async def _analyze_order_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Analyze order imbalance for scalping signals"""
        imbalance = self._calculate_order_imbalance(snapshot.bids, snapshot.asks)
        
        # Calculate weighted imbalance (closer levels have more weight)
        weighted_bid_volume = 0.0
        weighted_ask_volume = 0.0
        
        for i, bid in enumerate(snapshot.bids[:5]):  # Top 5 levels
            weight = 1.0 / (i + 1)  # Decreasing weight
            weighted_bid_volume += bid.size * weight
        
        for i, ask in enumerate(snapshot.asks[:5]):  # Top 5 levels
            weight = 1.0 / (i + 1)  # Decreasing weight
            weighted_ask_volume += ask.size * weight
        
        total_weighted = weighted_bid_volume + weighted_ask_volume
        weighted_imbalance = ((weighted_bid_volume - weighted_ask_volume) / total_weighted) if total_weighted > 0 else 0.0
        
        return {
            'raw_imbalance': imbalance or 0.0,
            'weighted_imbalance': weighted_imbalance,
            'bid_volume': snapshot.total_bid_volume,
            'ask_volume': snapshot.total_ask_volume,
            'imbalance_strength': abs(weighted_imbalance) * 100,
            'dominant_side': 'bid' if weighted_imbalance > 0 else 'ask' if weighted_imbalance < 0 else 'neutral'
        }

    async def _calculate_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate liquidity metrics for execution quality"""
        # Calculate spread metrics
        spread_bps = (snapshot.spread / snapshot.mid_price * 10000) if snapshot.mid_price > 0 else 0
        
        # Calculate depth at different price levels
        depth_1_percent = 0.0
        depth_2_percent = 0.0
        
        if snapshot.mid_price > 0:
            price_1_percent = snapshot.mid_price * 0.01
            price_2_percent = snapshot.mid_price * 0.02
            
            # Calculate volume within price ranges
            for bid in snapshot.bids:
                if snapshot.mid_price - bid.price <= price_1_percent:
                    depth_1_percent += bid.size
                if snapshot.mid_price - bid.price <= price_2_percent:
                    depth_2_percent += bid.size
            
            for ask in snapshot.asks:
                if ask.price - snapshot.mid_price <= price_1_percent:
                    depth_1_percent += ask.size
                if ask.price - snapshot.mid_price <= price_2_percent:
                    depth_2_percent += ask.size
        
        return {
            'spread_bps': spread_bps,
            'total_liquidity': snapshot.total_bid_volume + snapshot.total_ask_volume,
            'depth_1_percent': depth_1_percent,
            'depth_2_percent': depth_2_percent,
            'liquidity_score': min((depth_1_percent / 1000) * 100, 100),  # Normalized score
            'market_impact_estimate': spread_bps / 2  # Estimated market impact
        }

    async def _analyze_market_depth(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Analyze market depth characteristics"""
        # Calculate average order size
        bid_sizes = [bid.size for bid in snapshot.bids]
        ask_sizes = [ask.size for ask in snapshot.asks]
        
        avg_bid_size = statistics.mean(bid_sizes) if bid_sizes else 0
        avg_ask_size = statistics.mean(ask_sizes) if ask_sizes else 0
        
        # Calculate order count metrics
        total_bid_orders = sum(bid.orders for bid in snapshot.bids)
        total_ask_orders = sum(ask.orders for ask in snapshot.asks)
        
        return {
            'avg_bid_size': avg_bid_size,
            'avg_ask_size': avg_ask_size,
            'total_bid_orders': total_bid_orders,
            'total_ask_orders': total_ask_orders,
            'depth_levels': len(snapshot.bids) + len(snapshot.asks),
            'size_concentration': max(bid_sizes + ask_sizes) / (avg_bid_size + avg_ask_size) if (avg_bid_size + avg_ask_size) > 0 else 0
        }

    async def _generate_order_flow_signals(self, symbol: str, snapshot: OrderBookSnapshot, 
                                         imbalance_analysis: Dict[str, float]) -> List[OrderFlowSignal]:
        """Generate order flow-based scalping signals"""
        signals = []
        
        imbalance = imbalance_analysis.get('weighted_imbalance', 0.0)
        imbalance_strength = imbalance_analysis.get('imbalance_strength', 0.0)
        
        # Generate signal based on order imbalance
        signal_type = 'hold'
        strength = 0.0
        execution_probability = 0.5
        
        if abs(imbalance) > self.imbalance_threshold:
            if imbalance > self.imbalance_threshold:  # Bid-heavy
                signal_type = 'buy'
                strength = min(imbalance_strength, 100)
                execution_probability = 0.7
            elif imbalance < -self.imbalance_threshold:  # Ask-heavy
                signal_type = 'sell'
                strength = min(imbalance_strength, 100)
                execution_probability = 0.7
        
        # Calculate liquidity score
        total_liquidity = snapshot.total_bid_volume + snapshot.total_ask_volume
        liquidity_score = min(total_liquidity / self.liquidity_threshold * 100, 100)
        
        signal = OrderFlowSignal(
            timestamp=time.time(),
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            imbalance_ratio=imbalance,
            depth_advantage=imbalance_analysis.get('dominant_side', 'neutral'),
            liquidity_score=liquidity_score,
            execution_probability=execution_probability
        )
        
        signals.append(signal)
        return signals

    async def _calculate_execution_metrics(self, snapshot: OrderBookSnapshot, 
                                         signals: List[OrderFlowSignal]) -> Dict[str, float]:
        """Calculate execution quality metrics"""
        if not signals:
            return {}
        
        latest_signal = signals[-1]
        
        # Estimate execution costs
        spread_cost = snapshot.spread / 2 if snapshot.spread > 0 else 0
        market_impact = spread_cost * 0.5  # Estimated market impact
        
        return {
            'spread_cost': spread_cost,
            'market_impact_estimate': market_impact,
            'liquidity_score': latest_signal.liquidity_score,
            'execution_probability': latest_signal.execution_probability,
            'signal_strength': latest_signal.strength,
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
