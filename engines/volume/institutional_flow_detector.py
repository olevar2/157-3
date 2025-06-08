# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

"""
Institutional Flow Detector - Advanced Smart Money Analysis
Platform3 - Humanitarian Trading System

The Institutional Flow Detector identifies large block trades and institutional capital movement
by analyzing volume patterns, price action, and order flow. This indicator helps detect 
when smart money is entering or exiting positions.

Key Features:
- Large block trade detection
- Volume concentration analysis
- Smart money flow patterns
- Liquidity absorption detection
- Hidden accumulation/distribution identification
- Institutional buying/selling pressure measurement

Humanitarian Mission: Identify smart money flows to align with profitable institutional movements
while avoiding retail traps, enhancing capital efficiency for humanitarian profit maximization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from engines.indicator_base import IndicatorSignal, TechnicalIndicator, ServiceError
import logging

logger = logging.getLogger(__name__)

@dataclass
class InstitutionalFlowSignal(IndicatorSignal):
    """Institutional Flow-specific signal with detailed smart money analysis"""
    block_trade_ratio: float = 0.0
    volume_concentration: float = 0.0
    smart_money_flow_index: float = 0.0
    institutional_sentiment: str = "neutral"  # "accumulation", "distribution", "neutral"
    hidden_activity_score: float = 0.0
    confidence_level: float = 0.0


class InstitutionalFlowDetector(TechnicalIndicator):
    """
    InstitutionalFlowDetector analyzes market data to identify smart money and institutional activity
    by detecting large block trades, analyzing volume concentration patterns, and tracking liquidity absorption
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, # Added config
                 block_trade_threshold: int = 10000, 
                 volume_concentration_period: int = 20,
                 sensitivity: float = 1.0):
        """
        Initialize the InstitutionalFlowDetector with configurable parameters
        
        Parameters:
        -----------
        config: Optional[Dict[str, Any]]
            Configuration object for the indicator.
        block_trade_threshold: int
            Minimum volume to qualify as a block trade (institutional activity)
        volume_concentration_period: int
            Period for calculating volume concentration
        sensitivity: float
            Multiplier to adjust detection sensitivity (0.5 = less sensitive, 2.0 = more sensitive)
        """
        super().__init__(config=config) # Added super call with config
        self.logger.info(f"InstitutionalFlowDetector initialized with block_trade_threshold={block_trade_threshold}")
        self.block_trade_threshold = block_trade_threshold
        self.volume_concentration_period = volume_concentration_period
        self.sensitivity = sensitivity
        self.historical_flow = []
    
    def detect_block_trades(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect large block trades that likely represent institutional activity
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with required columns: datetime, volume, high, low, close
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of detected block trades with timestamp, volume, and price
        """
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Invalid data provided for block trade detection")
            
            # Identify large volume spikes (volume > threshold)
            mask = data['volume'] > self.block_trade_threshold
            block_trades = []
            
            if mask.any():
                for idx in data.index[mask]:
                    block_trades.append({
                        'timestamp': data.loc[idx, 'datetime'] if 'datetime' in data.columns else idx,
                        'volume': data.loc[idx, 'volume'],
                        'price': data.loc[idx, 'close'],
                        'imbalance': (data.loc[idx, 'close'] - data.loc[idx, 'low']) / 
                                     (data.loc[idx, 'high'] - data.loc[idx, 'low']) if 
                                     (data.loc[idx, 'high'] - data.loc[idx, 'low']) > 0 else 0.5
                    })
            
            return block_trades
        except Exception as e:
            self.logger.error(f"Error detecting block trades: {str(e)}")
            return []
    
    def calculate_concentration_index(self, data: pd.DataFrame) -> float:
        """
        Calculate volume concentration index to identify focused institutional activity
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with volume information
            
        Returns:
        --------
        float
            Volume concentration index (0-1 scale, higher means more concentrated)
        """
        try:
            if data.shape[0] < 3:
                return 0.0
            
            # Calculate normalized volume
            normalized_volume = data['volume'] / data['volume'].mean()
            
            # Calculate Gini coefficient for volume concentration
            normalized_volume = normalized_volume.sort_values()
            cum_volume = normalized_volume.cumsum()
            n = len(normalized_volume)
            
            # Calculate Gini coefficient directly
            if n > 1:
                idx = np.arange(1, n+1)
                gini = 2 * np.sum(idx * normalized_volume) / (n * np.sum(normalized_volume)) - (n+1)/n
                return max(0.0, min(1.0, gini))  # Ensure stays in [0,1]
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration index: {str(e)}")
            return 0.0
    
    def detect_smart_money_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect smart money patterns such as absorption, stopping volume, and testing
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with OHLCV information
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of pattern strengths (0-1 scale)
        """
        try:
            patterns = {
                'absorption': 0.0,
                'stopping_volume': 0.0,
                'testing': 0.0
            }
            
            if data.shape[0] < 5:
                return patterns
                
            # Calculate price ranges
            data['range'] = data['high'] - data['low']
            data['body'] = abs(data['close'] - data['open'])
            data['body_pct'] = data['body'] / data['range'].where(data['range'] > 0, 1)
            
            # Detect absorption (high volume with small price movement)
            vol_ma = data['volume'].rolling(5).mean()
            latest_vol_ratio = data['volume'].iloc[-1] / vol_ma.iloc[-1] if not np.isnan(vol_ma.iloc[-1]) and vol_ma.iloc[-1] > 0 else 1.0
            latest_body_pct = data['body_pct'].iloc[-1]
            
            if latest_vol_ratio > 1.5 and latest_body_pct < 0.3:
                # High volume with small candle body is absorption
                patterns['absorption'] = min(1.0, (latest_vol_ratio - 1) / 2)
            
            # Detect stopping volume (strong reversal on high volume)
            if data.shape[0] >= 3:
                down_trend = all(data['close'].iloc[-4:-1].pct_change().fillna(0) < 0)
                up_trend = all(data['close'].iloc[-4:-1].pct_change().fillna(0) > 0)
                
                latest_change = data['close'].iloc[-1] - data['close'].iloc[-2]
                prev_changes_mean = abs((data['close'].iloc[-3] - data['close'].iloc[-2]).mean())
                
                if down_trend and latest_change > 0 and data['volume'].iloc[-1] > vol_ma.iloc[-1] * 1.3:
                    # Potential stopping volume in downtrend
                    patterns['stopping_volume'] = min(1.0, latest_vol_ratio / 2)
                elif up_trend and latest_change < 0 and data['volume'].iloc[-1] > vol_ma.iloc[-1] * 1.3:
                    # Potential stopping volume in uptrend
                    patterns['stopping_volume'] = min(1.0, latest_vol_ratio / 2)
                    
            # Detect testing (price testing previous support/resistance with lower volume)
            if data.shape[0] >= 10:
                extremes = data['close'].rolling(5).apply(lambda x: (x.max() - x.min()) / x.mean(), raw=True)
                if not np.isnan(extremes.iloc[-1]):
                    recent_test = abs(data['close'].iloc[-1] - data['close'].iloc[-6:-2].min()) < data['range'].iloc[-1] * 0.3
                    vol_decline = data['volume'].iloc[-1] < data['volume'].iloc[-6:-2].mean()
                    
                    if recent_test and vol_decline:
                        patterns['testing'] = 0.7
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting smart money patterns: {str(e)}")
            return {'absorption': 0.0, 'stopping_volume': 0.0, 'testing': 0.0}
    
    def calculate(self, data: pd.DataFrame) -> InstitutionalFlowSignal:
        """
        Calculate institutional flow indicators from market data
        
        Parameters:
        -----------
        data: pd.DataFrame
            Market data with required columns (OHLCV)
            
        Returns:
        --------
        InstitutionalFlowSignal
            Comprehensive institutional flow analysis
        """
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                self.logger.warning("Input data is not a valid DataFrame or is empty.")
                return InstitutionalFlowSignal(confidence_level=0.0) # Return a default signal

            if data.shape[0] < self.volume_concentration_period:
                self.logger.warning(f"Insufficient data: need at least {self.volume_concentration_period} periods, got {data.shape[0]}")
                return InstitutionalFlowSignal(confidence_level=0.1) # Return a default signal with low confidence
            
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            if not all(field in data.columns for field in required_fields):
                raise ValueError(f"Missing required fields in data: {required_fields}")
                
            # Main analysis components
            large_block_trades = self.detect_block_trades(data)
            block_trade_ratio = len(large_block_trades) / len(data) if large_block_trades else 0.0
            
            volume_concentration = self.calculate_concentration_index(
                data.iloc[-self.volume_concentration_period:])
                
            smart_money_patterns = self.detect_smart_money_patterns(data.iloc[-10:])
            pattern_avg = np.mean(list(smart_money_patterns.values()))
            smart_money_flow = pattern_avg * self.sensitivity
            
            # Determine institutional sentiment
            close_ma5 = data['close'].rolling(5).mean().iloc[-1]
            close_ma20 = data['close'].rolling(20).mean().iloc[-1]
            price_trend = "up" if close_ma5 > close_ma20 else "down"
            
            # Combine volume and price indicators
            if volume_concentration > 0.6 and block_trade_ratio > 0.15:
                sentiment = "accumulation" if price_trend == "up" or smart_money_patterns['absorption'] > 0.5 else "distribution"
            else:
                sentiment = "neutral"
                
            # Calculate hidden activity score (0-1)
            vol_stdev = data['volume'].rolling(20).std().iloc[-1]
            vol_mean = data['volume'].rolling(20).mean().iloc[-1]
            price_volatility = data['close'].rolling(20).std().iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
            
            if vol_mean > 0 and not np.isnan(vol_stdev) and not np.isnan(price_volatility):
                hidden_score = min(1.0, (vol_stdev/vol_mean) * (1 - price_volatility*5))
                hidden_score = max(0.0, hidden_score)
            else:
                hidden_score = 0.0
            
            # Calculate confidence level based on available evidence
            confidence_factors = [
                0.7 if block_trade_ratio > 0.15 else 0.3 * block_trade_ratio / 0.15,
                0.6 if volume_concentration > 0.5 else 0.3 * volume_concentration / 0.5,
                0.5 if pattern_avg > 0.3 else 0.2 * pattern_avg / 0.3,
                0.3 if data.shape[0] > 50 else 0.1 * data.shape[0] / 50,
                0.4 if hidden_score > 0.5 else 0.2 * hidden_score / 0.5
            ]
            confidence_level = min(1.0, np.mean(confidence_factors) * 1.2)
            
            # Create and return the comprehensive signal
            return InstitutionalFlowSignal(
                block_trade_ratio=block_trade_ratio,
                volume_concentration=volume_concentration,
                smart_money_flow_index=smart_money_flow,
                institutional_sentiment=sentiment,
                hidden_activity_score=hidden_score,
                confidence_level=confidence_level,
                signal="buy" if sentiment == "accumulation" and confidence_level > 0.65 else
                       "sell" if sentiment == "distribution" and confidence_level > 0.65 else "neutral",
                strength=confidence_level * (0.8 if sentiment != "neutral" else 0.3),
                timestamp=data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
            )
        
        except Exception as e:
            self.logger.error(f"Error in InstitutionalFlowDetector calculation: {str(e)}")
            raise ServiceError(f"Calculation failed: {str(e)}")
            
    def generate_signal(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate trading signals based on institutional flow analysis
        
        Parameters:
        -----------
        data: Union[pd.DataFrame, Dict[str, Any]]
            Market data with required columns (OHLCV)
            
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
                "block_trade_ratio": signal.block_trade_ratio,
                "volume_concentration": signal.volume_concentration,
                "smart_money_flow": signal.smart_money_flow_index,
                "sentiment": signal.institutional_sentiment,
                "hidden_activity": signal.hidden_activity_score,
                "confidence": signal.confidence_level
            }
        }