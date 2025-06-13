# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Volume Weighted Market Depth Indicator - Advanced Order Book Analysis
Platform3 - Humanitarian Trading System

The Volume Weighted Market Depth Indicator analyzes order book depth with volume weighting
to assess market liquidity structure, support/resistance concentration, and order
flow dynamics through a volumetric lens.

Key Features:
- Weighted market depth calculation
- Volume concentration analysis
- Support/resistance strength measurement
- Depth imbalance detection
- Liquidity hole identification
- Buying/selling pressure quantification

Humanitarian Mission: Enhance trade execution quality by identifying optimal
liquidity zones, reducing slippage, and maximizing capital efficiency for
humanitarian trading operations.
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
class MarketDepthSignal(IndicatorSignal):
    """Volume-weighted market depth signal with liquidity structure analysis"""
    weighted_bid_depth: float = 0.0
    weighted_ask_depth: float = 0.0
    depth_ratio: float = 1.0
    liquidity_balance: float = 0.0
    support_resistance_levels: Dict[str, List[Dict[str, float]]] = field(default_factory=lambda: {"support": [], "resistance": []})
    liquidity_holes: List[Dict[str, Any]] = field(default_factory=list)


class VolumeWeightedMarketDepthIndicator(TechnicalIndicator):
    """
    VolumeWeightedMarketDepthIndicator analyzes order book data with volume weighting
    to identify liquidity structure, support/resistance zones, and order flow dynamics
    """
    
    def __init__(self, config: dict = None, depth_levels: int = 10, 
                 weighting_factor: float = 0.85,
                 support_resistance_threshold: float = 2.0):
        """
        Initialize the VolumeWeightedMarketDepthIndicator with configurable parameters
        
        Parameters:
        -----------
        config: dict
            Configuration dictionary
        depth_levels: int
            Number of price levels to analyze in the order book
        weighting_factor: float
            Decay factor for weighting by distance from mid price (0-1)
        support_resistance_threshold: float
            Volume multiple to identify significant support/resistance levels
        """
        super().__init__(config)
        self.logger.info(f"VolumeWeightedMarketDepthIndicator initialized with depth_levels={depth_levels}")
        self.depth_levels = depth_levels
        self.weighting_factor = weighting_factor
        self.support_resistance_threshold = support_resistance_threshold
    
    def calculate_weighted_depth(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate volume-weighted market depth
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data with 'bids' and 'asks' lists
            
        Returns:
        --------
        Dict[str, float]
            Weighted depth metrics
        """
        try:
            # Validate input
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                raise ValueError("Invalid order book data format")
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return {
                    'weighted_bid_depth': 0.0,
                    'weighted_ask_depth': 0.0,
                    'depth_ratio': 1.0,
                    'liquidity_balance': 0.0
                }
                
            # Get mid price
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate weighted depths
            weighted_bid_depth = 0.0
            weighted_ask_depth = 0.0
            
            # Process bid side
            bid_depth = min(self.depth_levels, len(bids))
            for i in range(bid_depth):
                price_distance = (mid_price - bids[i]['price']) / mid_price
                # Weight decreases with distance from mid price
                weight = self.weighting_factor ** i  
                weighted_bid_depth += bids[i]['volume'] * weight
            
            # Process ask side
            ask_depth = min(self.depth_levels, len(asks))
            for i in range(ask_depth):
                price_distance = (asks[i]['price'] - mid_price) / mid_price
                # Weight decreases with distance from mid price
                weight = self.weighting_factor ** i  
                weighted_ask_depth += asks[i]['volume'] * weight
                
            # Calculate depth ratio and balance
            if weighted_ask_depth > 0:
                depth_ratio = weighted_bid_depth / weighted_ask_depth
            else:
                depth_ratio = 1.0 if weighted_bid_depth == 0 else 10.0
                
            # Calculate liquidity balance (-1 to 1, positive means more bid liquidity)
            if weighted_bid_depth + weighted_ask_depth > 0:
                liquidity_balance = (weighted_bid_depth - weighted_ask_depth) / (weighted_bid_depth + weighted_ask_depth)
            else:
                liquidity_balance = 0.0
                
            return {
                'weighted_bid_depth': weighted_bid_depth,
                'weighted_ask_depth': weighted_ask_depth,
                'depth_ratio': depth_ratio,
                'liquidity_balance': liquidity_balance
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted depth: {str(e)}")
            return {
                'weighted_bid_depth': 0.0,
                'weighted_ask_depth': 0.0,
                'depth_ratio': 1.0,
                'liquidity_balance': 0.0
            }
    
    def identify_support_resistance_levels(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, float]]]:
        """
        Identify significant support and resistance levels based on volume concentration
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data
            
        Returns:
        --------
        Dict[str, List[Dict[str, float]]]
            Identified support and resistance levels
        """
        try:
            levels = {"support": [], "resistance": []}
            
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return levels
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return levels
                
            # Calculate average volumes
            bid_volumes = [level['volume'] for level in bids]
            ask_volumes = [level['volume'] for level in asks]
            
            avg_bid_volume = np.mean(bid_volumes) if bid_volumes else 0
            avg_ask_volume = np.mean(ask_volumes) if ask_volumes else 0
            
            # Identify support levels (large bid volumes)
            for i, bid in enumerate(bids):
                if bid['volume'] > avg_bid_volume * self.support_resistance_threshold:
                    # Calculate strength based on volume and relative position
                    position_factor = 1.0 / (i + 1)  # Higher for levels closer to current price
                    strength = (bid['volume'] / avg_bid_volume) * position_factor
                    
                    levels["support"].append({
                        "price": bid['price'],
                        "volume": bid['volume'],
                        "strength": min(10.0, strength),  # Cap at 10x
                        "distance_pct": (bids[0]['price'] - bid['price']) / bids[0]['price'] * 100
                    })
            
            # Identify resistance levels (large ask volumes)
            for i, ask in enumerate(asks):
                if ask['volume'] > avg_ask_volume * self.support_resistance_threshold:
                    # Calculate strength based on volume and relative position
                    position_factor = 1.0 / (i + 1)  # Higher for levels closer to current price
                    strength = (ask['volume'] / avg_ask_volume) * position_factor
                    
                    levels["resistance"].append({
                        "price": ask['price'],
                        "volume": ask['volume'],
                        "strength": min(10.0, strength),  # Cap at 10x
                        "distance_pct": (ask['price'] - asks[0]['price']) / asks[0]['price'] * 100
                    })
                    
            # Sort by strength (descending)
            levels["support"].sort(key=lambda x: x["strength"], reverse=True)
            levels["resistance"].sort(key=lambda x: x["strength"], reverse=True)
            
            # Limit to top 5 levels
            levels["support"] = levels["support"][:5]
            levels["resistance"] = levels["resistance"][:5]
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {str(e)}")
            return {"support": [], "resistance": []}
    
    def detect_liquidity_holes(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Detect liquidity holes (price gaps with minimal volume)
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data
            
        Returns:
        --------
        List[Dict[str, Any]]
            Identified liquidity holes
        """
        try:
            holes = []
            
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return holes
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if len(bids) < 2 or len(asks) < 2:
                return holes
                
            # Process bid side (support holes)
            for i in range(1, min(self.depth_levels, len(bids))):
                if i < len(bids) - 1:
                    current_price = bids[i]['price']
                    next_price = bids[i+1]['price']
                    price_gap = current_price - next_price
                    
                    # Calculate average price gap as reference
                    avg_gap = np.mean([bids[j]['price'] - bids[j+1]['price'] for j in range(min(5, len(bids)-1))])
                    
                    # Detect significant price gap with low volume
                    if price_gap > avg_gap * 3 and bids[i+1]['volume'] < np.mean([b['volume'] for b in bids[:5]]) * 0.5:
                        holes.append({
                            "type": "support_hole",
                            "start_price": next_price,
                            "end_price": current_price,
                            "gap_size": price_gap,
                            "gap_ratio": price_gap / avg_gap,
                            "significance": min(10.0, price_gap / avg_gap * (1 - bids[i+1]['volume'] / bids[i]['volume']))
                        })
            
            # Process ask side (resistance holes)
            for i in range(1, min(self.depth_levels, len(asks))):
                if i < len(asks) - 1:
                    current_price = asks[i]['price']
                    next_price = asks[i+1]['price']
                    price_gap = next_price - current_price
                    
                    # Calculate average price gap as reference
                    avg_gap = np.mean([asks[j+1]['price'] - asks[j]['price'] for j in range(min(5, len(asks)-1))])
                    
                    # Detect significant price gap with low volume
                    if price_gap > avg_gap * 3 and asks[i+1]['volume'] < np.mean([a['volume'] for a in asks[:5]]) * 0.5:
                        holes.append({
                            "type": "resistance_hole",
                            "start_price": current_price,
                            "end_price": next_price,
                            "gap_size": price_gap,
                            "gap_ratio": price_gap / avg_gap,
                            "significance": min(10.0, price_gap / avg_gap * (1 - asks[i+1]['volume'] / asks[i]['volume']))
                        })
                        
            # Sort by significance
            holes.sort(key=lambda x: x["significance"], reverse=True)
            
            return holes[:3]  # Return top 3 most significant holes
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity holes: {str(e)}")
            return []
    
    def calculate(self, data: Dict[str, Any]) -> MarketDepthSignal:
        """
        Calculate volume-weighted market depth metrics from order book data
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Market data containing 'order_book' field with order book data
            
        Returns:
        --------
        MarketDepthSignal
            Comprehensive market depth analysis
        """
        try:
            # Extract order book data
            if not isinstance(data, dict) or 'order_book' not in data:
                raise ValueError("Missing order book data")
                
            order_book = data['order_book']
            
            # Calculate weighted depth metrics
            depth_metrics = self.calculate_weighted_depth(order_book)
            
            # Identify support and resistance levels
            levels = self.identify_support_resistance_levels(order_book)
            
            # Detect liquidity holes
            holes = self.detect_liquidity_holes(order_book)
            
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate signals based on liquidity balance
            liquidity_balance = depth_metrics['liquidity_balance']
            
            # Strong buy signal when significant buying pressure (high bid depth)
            if liquidity_balance > 0.3:
                signal_direction = "buy"
                signal_strength = min(1.0, liquidity_balance * 2)
            # Strong sell signal when significant selling pressure (high ask depth)
            elif liquidity_balance < -0.3:
                signal_direction = "sell"
                signal_strength = min(1.0, abs(liquidity_balance) * 2)
            
            # Adjust based on support/resistance
            if levels["support"] and signal_direction == "buy":
                # Stronger buy signal near strong support
                closest_support = min(levels["support"], key=lambda x: x["distance_pct"])
                if closest_support["distance_pct"] < 1.0:
                    signal_strength = min(1.0, signal_strength * (1 + closest_support["strength"] / 10))
            
            elif levels["resistance"] and signal_direction == "sell":
                # Stronger sell signal near strong resistance
                closest_resistance = min(levels["resistance"], key=lambda x: x["distance_pct"])
                if closest_resistance["distance_pct"] < 1.0:
                    signal_strength = min(1.0, signal_strength * (1 + closest_resistance["strength"] / 10))
            
            # Create and return the market depth signal
            return MarketDepthSignal(
                weighted_bid_depth=depth_metrics['weighted_bid_depth'],
                weighted_ask_depth=depth_metrics['weighted_ask_depth'],
                depth_ratio=depth_metrics['depth_ratio'],
                liquidity_balance=depth_metrics['liquidity_balance'],
                support_resistance_levels=levels,
                liquidity_holes=holes,
                signal=signal_direction,
                strength=signal_strength,
                timestamp=data.get('timestamp', datetime.now())
            )
        
        except Exception as e:
            self.logger.error(f"Error in VolumeWeightedMarketDepthIndicator calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on volume-weighted market depth analysis
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Order book data for analysis
            
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
                "weighted_bid_depth": signal.weighted_bid_depth,
                "weighted_ask_depth": signal.weighted_ask_depth,
                "depth_ratio": signal.depth_ratio,
                "liquidity_balance": signal.liquidity_balance,
                "support_levels": [{"price": level["price"], "strength": level["strength"]} 
                                for level in signal.support_resistance_levels["support"]],
                "resistance_levels": [{"price": level["price"], "strength": level["strength"]} 
                                    for level in signal.support_resistance_levels["resistance"]],
                "liquidity_holes": signal.liquidity_holes
            }
        }