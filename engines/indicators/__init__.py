"""
Technical indicators module for the trading platform
"""

from typing import Dict, List, Any, Optional
import numpy as np

class TechnicalIndicators:
    """Basic technical indicators implementation"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Default neutral value
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def calculate_macd(prices: List[float], 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow_period:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        # Simple implementation - would use exponential MA in production
        fast_ma = np.mean(prices[-fast_period:])
        slow_ma = np.mean(prices[-slow_period:])
        macd = fast_ma - slow_ma
        
        return {
            'macd': float(macd),
            'signal': float(macd * 0.9),  # Simplified
            'histogram': float(macd * 0.1)
        }
    
    @staticmethod
    def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high) < 2:
            return 0.0
        
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        if len(tr_list) >= period:
            return float(np.mean(tr_list[-period:]))
        elif tr_list:
            return float(np.mean(tr_list))
        return 0.0

# Export indicators
technical_indicators = TechnicalIndicators()
