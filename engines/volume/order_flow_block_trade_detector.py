# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Order Flow Block Trade Detector - Institutional Trade Detection
Platform3 - Humanitarian Trading System

The Order Flow Block Trade Detector identifies large institutional trades by analyzing
trade tape data for significant volume spikes and block execution patterns. This indicator
helps identify when major market participants are active and their directional bias.

Key Features:
- Block trade detection and classification
- Iceberg order recognition
- Hidden liquidity discovery
- Institutional footprint analysis
- Trade clustering identification
- VWAP deviation tracking

Humanitarian Mission: Identify large institutional movements to leverage the advanced
information and strategy capabilities of institutional traders for humanitarian profit
maximization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from engines.indicator_base import IndicatorSignal, TechnicalIndicator, ServiceError
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class BlockTradeSignal(IndicatorSignal):
    """Block trade detection signal with institutional trade analysis"""
    block_trades_detected: int = 0
    block_volume_percent: float = 0.0
    avg_block_size: float = 0.0
    directional_bias: float = 0.0  # -1.0 to 1.0, positive means bullish
    iceberg_probability: float = 0.0
    significant_trades: List[Dict[str, Any]] = field(default_factory=list)


class OrderFlowBlockTradeDetector(TechnicalIndicator):
    """
    OrderFlowBlockTradeDetector analyzes trade data to identify large institutional orders,
    iceberg execution patterns, and block trades that may signal significant market moves
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, # Added config
                 block_threshold_percent: float = 2.0, 
                 iceberg_detection_window: int = 20,
                 directional_sensitivity: float = 1.0):
        """
        Initialize the OrderFlowBlockTradeDetector with configurable parameters
        
        Parameters:
        -----------
        config: Configuration dictionary.
        block_threshold_percent: float
            Percentage of avg daily volume to qualify as a block trade
        iceberg_detection_window: int
            Window of trades to analyze for iceberg detection
        directional_sensitivity: float
            Sensitivity multiplier for directional bias calculation
        """
        super().__init__(config=config) # Pass config to super
        self.logger.info(f"OrderFlowBlockTradeDetector initialized with block_threshold_percent={block_threshold_percent}")
        self.block_threshold_percent = block_threshold_percent
        self.iceberg_detection_window = iceberg_detection_window
        self.directional_sensitivity = directional_sensitivity
    
    def identify_block_trades(self, trades: List[Dict[str, Any]], 
                            avg_daily_volume: float) -> List[Dict[str, Any]]:
        """
        Identify block trades from trade data
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of trades with volume, price, etc.
        avg_daily_volume: float
            Average daily trading volume
            
        Returns:
        --------
        List[Dict[str, Any]]
            Identified block trades
        """
        try:
            if not trades or avg_daily_volume <= 0:
                return []
                
            # Calculate block threshold
            block_threshold = (avg_daily_volume * self.block_threshold_percent) / 100.0
            
            # Find all trades that exceed the threshold
            block_trades = []
            
            for trade in trades:
                if trade.get('volume', 0) >= block_threshold:
                    # Classify as block trade
                    block_trades.append({
                        'timestamp': trade.get('timestamp', datetime.now()),
                        'price': trade.get('price', 0.0),
                        'volume': trade.get('volume', 0.0),
                        'direction': trade.get('direction', 'neutral'),
                        'relative_size': trade.get('volume', 0.0) / block_threshold,
                        'price_impact': trade.get('price_impact', 0.0)
                    })
            
            # Sort by volume (descending)
            block_trades.sort(key=lambda x: x['volume'], reverse=True)
            
            return block_trades
            
        except Exception as e:
            self.logger.error(f"Error identifying block trades: {str(e)}")
            return []
    
    def detect_iceberg_orders(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Detect potential iceberg orders by analyzing trading patterns
        
        Parameters:
        -----------
        trades: List[Dict[str, Any]]
            List of trades to analyze
            
        Returns:
        --------
        Dict[str, float]
            Iceberg detection metrics
        """
        try:
            if len(trades) < self.iceberg_detection_window:
                return {'iceberg_probability': 0.0, 'side': 'neutral', 'avg_hidden_ratio': 0.0}
                
            # Look for repeated trades at same price level with similar size
            price_levels = {}
            
            for trade in trades:
                price = trade.get('price', 0.0)
                if price not in price_levels:
                    price_levels[price] = []
                    
                price_levels[price].append({
                    'volume': trade.get('volume', 0.0),
                    'timestamp': trade.get('timestamp', datetime.now()),
                    'direction': trade.get('direction', 'neutral')
                })
                
            # Find price levels with repeated trades
            iceberg_candidates = {}
            
            for price, price_trades in price_levels.items():
                if len(price_trades) >= 3:  # Need at least 3 trades at same level to consider
                    # Check time clustering (iceberg orders execute in sequence)
                    timestamps = [t.get('timestamp', datetime.now()) for t in price_trades 
                                if isinstance(t.get('timestamp'), datetime)]
                    
                    if len(timestamps) >= 3:
                        # Calculate time differences between consecutive trades
                        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                                    for i in range(len(timestamps)-1)]
                        
                        avg_time_diff = sum(time_diffs) / len(time_diffs)
                        
                        # Iceberg orders usually have consistent time gaps
                        time_consistency = np.std(time_diffs) / avg_time_diff if avg_time_diff > 0 else float('inf')
                        
                        # Check size consistency
                        volumes = [t.get('volume', 0.0) for t in price_trades]
                        avg_volume = sum(volumes) / len(volumes)
                        
                        # Calculate volume consistency (coefficient of variation)
                        vol_std = np.std(volumes)
                        vol_consistency = vol_std / avg_volume if avg_volume > 0 else float('inf')
                        
                        # Calculate direction consistency
                        directions = [1 if t.get('direction') == 'buy' else -1 if t.get('direction') == 'sell' else 0 
                                     for t in price_trades]
                        
                        # Calculate dominant direction
                        direction_sum = sum(directions)
                        
                        # Combine metrics into iceberg probability
                        if vol_consistency < 0.3 and time_consistency < 1.0 and abs(direction_sum) >= len(directions) * 0.7:
                            side = 'buy' if direction_sum > 0 else 'sell'
                            probability = min(1.0, (1.0 - vol_consistency) * (1.0 - min(1.0, time_consistency)) * 
                                           (abs(direction_sum) / len(directions)))
                            
                            # Estimate hidden portion
                            hidden_ratio = min(5.0, avg_volume * len(volumes) / sum(volumes))
                            
                            iceberg_candidates[price] = {
                                'probability': probability,
                                'side': side,
                                'trade_count': len(price_trades),
                                'avg_volume': avg_volume,
                                'total_volume': sum(volumes),
                                'hidden_ratio': hidden_ratio
                            }
            
            # If no candidates found
            if not iceberg_candidates:
                return {'iceberg_probability': 0.0, 'side': 'neutral', 'avg_hidden_ratio': 0.0}
                
            # Find most likely iceberg
            most_likely = max(iceberg_candidates.values(), key=lambda x: x['probability'])
            
            return {
                'iceberg_probability': most_likely['probability'],
                'side': most_likely['side'],
                'avg_hidden_ratio': most_likely['hidden_ratio']
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting iceberg orders: {str(e)}")
            return {'iceberg_probability': 0.0, 'side': 'neutral', 'avg_hidden_ratio': 0.0}
    
    def calculate_directional_bias(self, block_trades: List[Dict[str, Any]]) -> float:
        """
        Calculate directional bias from block trades
        
        Parameters:
        -----------
        block_trades: List[Dict[str, Any]]
            List of detected block trades
            
        Returns:
        --------
        float
            Directional bias (-1.0 to 1.0, positive means bullish)
        """
        try:
            if not block_trades:
                return 0.0
                
            # Sum up volume with direction
            buy_volume = 0.0
            sell_volume = 0.0
            
            for trade in block_trades:
                if trade['direction'] == 'buy':
                    buy_volume += trade['volume']
                elif trade['direction'] == 'sell':
                    sell_volume += trade['volume']
            
            total_volume = buy_volume + sell_volume
            
            # Calculate bias
            if total_volume > 0:
                bias = (buy_volume - sell_volume) / total_volume
                # Apply sensitivity factor
                bias = max(-1.0, min(1.0, bias * self.directional_sensitivity))
                return bias
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating directional bias: {str(e)}")
            return 0.0
    
    def calculate(self, data: Dict[str, Any]) -> BlockTradeSignal:
        """
        Calculate block trade metrics from trade data
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Market data containing trades and avg_daily_volume
            
        Returns:
        --------
        BlockTradeSignal
            Comprehensive block trade analysis
        """
        try:
            # Extract trade data
            trades = data.get('trades', [])
            avg_daily_volume = data.get('avg_daily_volume', 0.0)
            
            if not trades or avg_daily_volume <= 0:
                raise ValueError("Missing or invalid trade data")
                
            # Identify block trades
            block_trades = self.identify_block_trades(trades, avg_daily_volume)
            
            # Calculate block trade metrics
            block_count = len(block_trades)
            
            # Calculate block volume percentage
            total_volume = sum(trade.get('volume', 0.0) for trade in trades)
            block_volume = sum(trade['volume'] for trade in block_trades)
            
            block_volume_percent = (block_volume / total_volume * 100) if total_volume > 0 else 0.0
            avg_block_size = block_volume / block_count if block_count > 0 else 0.0
            
            # Calculate directional bias
            directional_bias = self.calculate_directional_bias(block_trades)
            
            # Detect iceberg orders
            iceberg_metrics = self.detect_iceberg_orders(trades)
            
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate signals based on block trade direction and iceberg detection
            if block_count > 0:
                if directional_bias > 0.3:
                    signal_direction = "buy"
                    signal_strength = min(1.0, directional_bias + (block_volume_percent / 100))
                    # Strengthen if iceberg detected on same side
                    if iceberg_metrics['side'] == 'buy' and iceberg_metrics['iceberg_probability'] > 0.6:
                        signal_strength = min(1.0, signal_strength * 1.2)
                        
                elif directional_bias < -0.3:
                    signal_direction = "sell"
                    signal_strength = min(1.0, abs(directional_bias) + (block_volume_percent / 100))
                    # Strengthen if iceberg detected on same side
                    if iceberg_metrics['side'] == 'sell' and iceberg_metrics['iceberg_probability'] > 0.6:
                        signal_strength = min(1.0, signal_strength * 1.2)
            
            # Create and return the block trade signal
            return BlockTradeSignal(
                block_trades_detected=block_count,
                block_volume_percent=block_volume_percent,
                avg_block_size=avg_block_size,
                directional_bias=directional_bias,
                iceberg_probability=iceberg_metrics['iceberg_probability'],
                significant_trades=block_trades[:5],  # Top 5 block trades
                signal=signal_direction,
                strength=signal_strength,
                timestamp=data.get('timestamp', datetime.now())
            )
        
        except Exception as e:
            self.logger.error(f"Error in OrderFlowBlockTradeDetector calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on block trade analysis
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Trade data for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Trading signal with direction, strength and analysis
        """
        signal = self.calculate(data)
        
        return {
            "direction": signal.signal,
            "strength": signal.strength,
            "timestamp": signal.timestamp,
            "metadata": {
                "block_trades_detected": signal.block_trades_detected,
                "block_volume_percent": signal.block_volume_percent,
                "avg_block_size": signal.avg_block_size,
                "directional_bias": signal.directional_bias,
                "iceberg_probability": signal.iceberg_probability,
                "significant_trades": [
                    {
                        "timestamp": trade['timestamp'],
                        "price": trade['price'],
                        "volume": trade['volume'],
                        "direction": trade['direction']
                    } for trade in signal.significant_trades
                ]
            }
        }