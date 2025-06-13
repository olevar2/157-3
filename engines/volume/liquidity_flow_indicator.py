# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Liquidity Flow Indicator - Advanced Market Depth Analysis
Platform3 - Humanitarian Trading System

The Liquidity Flow Indicator analyzes the availability and movement of liquidity in the market
by tracking changes in market depth, identifying liquidity sweeps, and detecting significant
liquidity imbalances that often precede price movements.

Key Features:
- Market depth tracking
- Liquidity sweep detection
- Liquidity imbalance measurement
- Depth curve analysis
- Liquidity provision/removal alerts
- Order book change velocity tracking

Humanitarian Mission: Identify liquidity voids and imbalances to anticipate price movements
and capture favorable execution prices, enhancing profit potential for humanitarian causes.
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
class LiquidityFlowSignal(IndicatorSignal):
    """Liquidity flow-specific signal with detailed market depth analysis"""
    bid_liquidity: float = 0.0
    ask_liquidity: float = 0.0
    liquidity_imbalance: float = 0.0
    depth_ratio: float = 0.0
    liquidity_removed: float = 0.0
    sweep_direction: str = "none"  # "bid", "ask", "none"
    liquidity_quality: float = 0.0  # 0-1 scale


class LiquidityFlowIndicator(TechnicalIndicator):
    """
    LiquidityFlowIndicator analyzes market depth data to identify liquidity
    imbalances, sweeps, and significant changes in order book structure that
    often precede price movements.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None, # Added config
                 depth_levels: int = 10, 
                 imbalance_threshold: float = 0.25,
                 sweep_threshold: float = 0.5):
        """
        Initialize the LiquidityFlowIndicator with configurable parameters
        
        Parameters:
        -----------
        config: Configuration dictionary.
        depth_levels: int
            Number of price levels to analyze in the order book
        imbalance_threshold: float
            Threshold to determine significant liquidity imbalance (0.0 - 1.0)
        sweep_threshold: float
            Threshold to identify liquidity sweeps (0.0 - 1.0)
        """
        super().__init__(config=config) # Pass config to super
        self.logger.info(f"LiquidityFlowIndicator initialized with depth_levels={depth_levels}")
        self.depth_levels = depth_levels
        self.imbalance_threshold = imbalance_threshold
        self.sweep_threshold = sweep_threshold
        self.previous_book = None
    
    def calculate_liquidity_metrics(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate key liquidity metrics from order book data
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data with 'bids' and 'asks' lists
            
        Returns:
        --------
        Dict[str, float]
            Liquidity metrics
        """
        try:
            # Validate input
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                raise ValueError("Invalid order book data format")
                
            bids = order_book_data['bids']
            asks = order_book_data['asks']
            
            if not bids or not asks:
                return {
                    'bid_liquidity': 0.0,
                    'ask_liquidity': 0.0,
                    'liquidity_imbalance': 0.0,
                    'depth_ratio': 0.0
                }
                
            # Calculate total volume at each side, respecting depth_levels
            bid_depth = min(self.depth_levels, len(bids))
            ask_depth = min(self.depth_levels, len(asks))
            
            bid_liquidity = sum(level['volume'] for level in bids[:bid_depth])
            ask_liquidity = sum(level['volume'] for level in asks[:ask_depth])
            
            total_liquidity = bid_liquidity + ask_liquidity
            
            # Calculate liquidity imbalance (-1.0 to 1.0, positive means more bid liquidity)
            if total_liquidity > 0:
                liquidity_imbalance = (bid_liquidity - ask_liquidity) / total_liquidity
            else:
                liquidity_imbalance = 0.0
                
            # Calculate depth ratio (how liquidity is distributed across levels)
            top_bid_liquidity = bids[0]['volume'] if bids else 0
            top_ask_liquidity = asks[0]['volume'] if asks else 0
            
            if bid_liquidity > 0 and ask_liquidity > 0:
                bid_concentration = top_bid_liquidity / bid_liquidity
                ask_concentration = top_ask_liquidity / ask_liquidity
                depth_ratio = (bid_concentration + ask_concentration) / 2
            else:
                depth_ratio = 0.0
                
            return {
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'liquidity_imbalance': liquidity_imbalance,
                'depth_ratio': depth_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity metrics: {str(e)}")
            return {
                'bid_liquidity': 0.0,
                'ask_liquidity': 0.0,
                'liquidity_imbalance': 0.0,
                'depth_ratio': 0.0
            }
    
    def detect_liquidity_sweep(self, current_book: Dict[str, List[Dict[str, Any]]],
                             previous_book: Optional[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Detect liquidity sweeps (sudden removal of liquidity from one side)
        
        Parameters:
        -----------
        current_book: Dict[str, List[Dict[str, Any]]]
            Current order book data
        previous_book: Optional[Dict[str, List[Dict[str, Any]]]]
            Previous order book data for comparison
            
        Returns:
        --------
        Dict[str, Any]
            Sweep detection results
        """
        try:
            result = {
                'sweep_detected': False,
                'sweep_direction': "none",
                'liquidity_removed': 0.0,
                'percent_removed': 0.0
            }
            
            if not previous_book:
                return result
                
            # Calculate liquidity in previous book
            prev_bids = previous_book.get('bids', [])
            prev_asks = previous_book.get('asks', [])
            
            prev_bid_liquidity = sum(level['volume'] for level in prev_bids[:min(self.depth_levels, len(prev_bids))]) if prev_bids else 0
            prev_ask_liquidity = sum(level['volume'] for level in prev_asks[:min(self.depth_levels, len(prev_asks))]) if prev_asks else 0
            
            # Calculate liquidity in current book
            current_bids = current_book.get('bids', [])
            current_asks = current_book.get('asks', [])
            
            current_bid_liquidity = sum(level['volume'] for level in current_bids[:min(self.depth_levels, len(current_bids))]) if current_bids else 0
            current_ask_liquidity = sum(level['volume'] for level in current_asks[:min(self.depth_levels, len(current_asks))]) if current_asks else 0
            
            # Calculate liquidity changes
            bid_change = current_bid_liquidity - prev_bid_liquidity
            ask_change = current_ask_liquidity - prev_ask_liquidity
            
            # Check for significant removal
            if bid_change < 0 and abs(bid_change) > (prev_bid_liquidity * self.sweep_threshold):
                result['sweep_detected'] = True
                result['sweep_direction'] = "bid"
                result['liquidity_removed'] = abs(bid_change)
                result['percent_removed'] = abs(bid_change) / prev_bid_liquidity if prev_bid_liquidity > 0 else 1.0
            elif ask_change < 0 and abs(ask_change) > (prev_ask_liquidity * self.sweep_threshold):
                result['sweep_detected'] = True
                result['sweep_direction'] = "ask"
                result['liquidity_removed'] = abs(ask_change)
                result['percent_removed'] = abs(ask_change) / prev_ask_liquidity if prev_ask_liquidity > 0 else 1.0
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity sweep: {str(e)}")
            return {
                'sweep_detected': False,
                'sweep_direction': "none",
                'liquidity_removed': 0.0,
                'percent_removed': 0.0
            }
    
    def calculate_liquidity_quality(self, order_book_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Calculate liquidity quality score based on depth structure
        
        Parameters:
        -----------
        order_book_data: Dict[str, List[Dict[str, Any]]]
            Order book data
            
        Returns:
        --------
        float
            Liquidity quality score (0-1)
        """
        try:
            if not isinstance(order_book_data, dict) or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return 0.0
                
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])
            
            if not bids or not asks or len(bids) < 3 or len(asks) < 3:
                return 0.0
                
            # Extract prices and volumes
            bid_prices = [level['price'] for level in bids[:min(self.depth_levels, len(bids))]]
            ask_prices = [level['price'] for level in asks[:min(self.depth_levels, len(asks))]]
            bid_volumes = [level['volume'] for level in bids[:min(self.depth_levels, len(bids))]]
            ask_volumes = [level['volume'] for level in asks[:min(self.depth_levels, len(asks))]]
            
            # Calculate metrics
            
            # 1. Price spacing (regular = better quality)
            bid_spacing = [bid_prices[i] - bid_prices[i+1] for i in range(len(bid_prices)-1)]
            ask_spacing = [ask_prices[i+1] - ask_prices[i] for i in range(len(ask_prices)-1)]
            
            if bid_spacing and ask_spacing:
                spacing_regularity = 1.0 - (np.std(bid_spacing + ask_spacing) / np.mean(bid_spacing + ask_spacing))
                spacing_score = max(0.0, min(1.0, spacing_regularity))
            else:
                spacing_score = 0.0
                
            # 2. Volume distribution (should decrease with distance from mid)
            bid_decay = all(bid_volumes[i] >= bid_volumes[i+1] for i in range(len(bid_volumes)-1))
            ask_decay = all(ask_volumes[i] >= ask_volumes[i+1] for i in range(len(ask_volumes)-1))
            
            decay_score = (int(bid_decay) + int(ask_decay)) / 2
            
            # 3. Spread tightness
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price
            
            spread_score = max(0.0, min(1.0, 1.0 - (spread_pct * 100)))  # Normalize
            
            # 4. Depth consistency
            depth_consistency = 1.0 - abs(len(bid_prices) - len(ask_prices)) / max(len(bid_prices), len(ask_prices))
            
            # Combine into quality score (0-1)
            quality_weights = [0.3, 0.2, 0.3, 0.2]  # Weights for each component
            liquidity_quality = (
                spacing_score * quality_weights[0] +
                decay_score * quality_weights[1] +
                spread_score * quality_weights[2] +
                depth_consistency * quality_weights[3]
            )
            
            return max(0.0, min(1.0, liquidity_quality))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity quality: {str(e)}")
            return 0.0
    
    def calculate(self, data: Dict[str, Any]) -> LiquidityFlowSignal:
        """
        Calculate liquidity flow indicators from order book data
        
        Parameters:
        -----------
        data: Dict[str, Any]
            Market data containing 'order_book' field with order book data
            
        Returns:
        --------
        LiquidityFlowSignal
            Comprehensive liquidity flow analysis
        """
        try:
            # Extract order book data
            if not isinstance(data, dict) or 'order_book' not in data:
                raise ValueError("Missing order book data")
                
            current_book = data['order_book']
            
            # Calculate main liquidity metrics
            metrics = self.calculate_liquidity_metrics(current_book)
            
            # Detect liquidity sweep if we have previous data
            sweep_info = self.detect_liquidity_sweep(current_book, self.previous_book)
            
            # Calculate liquidity quality score
            quality = self.calculate_liquidity_quality(current_book)
            
            # Store current book for next comparison
            self.previous_book = current_book
            
            # Create signal with buy/sell recommendation
            signal_direction = "neutral"
            signal_strength = 0.0
            
            # Generate signals based on combined metrics
            imbalance = metrics['liquidity_imbalance']
            
            if sweep_info['sweep_detected']:
                if sweep_info['sweep_direction'] == "ask":
                    # Liquidity swept from ask side often precedes upward movement
                    signal_direction = "buy"
                    signal_strength = min(1.0, sweep_info['percent_removed'])
                elif sweep_info['sweep_direction'] == "bid":
                    # Liquidity swept from bid side often precedes downward movement
                    signal_direction = "sell"
                    signal_strength = min(1.0, sweep_info['percent_removed'])
            elif abs(imbalance) > self.imbalance_threshold:
                # Significant imbalance can indicate directional pressure
                signal_direction = "buy" if imbalance > 0 else "sell"
                signal_strength = min(1.0, abs(imbalance) * quality)
                
            # Create and return the liquidity flow signal
            return LiquidityFlowSignal(
                bid_liquidity=metrics['bid_liquidity'],
                ask_liquidity=metrics['ask_liquidity'],
                liquidity_imbalance=metrics['liquidity_imbalance'],
                depth_ratio=metrics['depth_ratio'],
                liquidity_removed=sweep_info['liquidity_removed'],
                sweep_direction=sweep_info['sweep_direction'],
                liquidity_quality=quality,
                signal=signal_direction,
                strength=signal_strength,
                timestamp=data.get('timestamp', datetime.now())
            )
        
        except Exception as e:
            self.logger.error(f"Error in LiquidityFlowIndicator calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on liquidity flow analysis
        
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
                "bid_liquidity": signal.bid_liquidity,
                "ask_liquidity": signal.ask_liquidity,
                "liquidity_imbalance": signal.liquidity_imbalance,
                "depth_ratio": signal.depth_ratio,
                "liquidity_removed": signal.liquidity_removed,
                "sweep_direction": signal.sweep_direction,
                "liquidity_quality": signal.liquidity_quality
            }
        }