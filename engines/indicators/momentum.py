"""
Momentum Indicators Module
Provides various momentum-based technical indicators
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class MomentumIndicators:
    """Collection of momentum-based technical indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MomentumIndicators")
    
    def rsi(self, prices: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.zeros_like(prices)
            avg_losses = np.zeros_like(prices)
            
            # Initial averages
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])
            
            # Smoothed averages
            for i in range(period + 1, len(prices)):
                avg_gains[i] = ((avg_gains[i-1] * (period - 1)) + gains[i-1]) / period
                avg_losses[i] = ((avg_losses[i-1] * (period - 1)) + losses[i-1]) / period
            
            rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.full(len(prices), 50.0)  # Neutral RSI on error
    
    def macd(self, prices: Union[pd.Series, np.ndarray], 
             fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            # Calculate EMAs
            ema_fast = self._ema(prices, fast_period)
            ema_slow = self._ema(prices, slow_period)
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = self._ema(macd_line, signal_period)
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            zeros = np.zeros(len(prices))
            return {'macd': zeros, 'signal': zeros, 'histogram': zeros}
    
    def stochastic(self, high: Union[pd.Series, np.ndarray], 
                   low: Union[pd.Series, np.ndarray], 
                   close: Union[pd.Series, np.ndarray], 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        try:
            if isinstance(high, pd.Series):
                high = high.values
            if isinstance(low, pd.Series):
                low = low.values
            if isinstance(close, pd.Series):
                close = close.values
            
            # Calculate %K
            lowest_low = np.zeros_like(close)
            highest_high = np.zeros_like(close)
            
            for i in range(k_period - 1, len(close)):
                lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
                highest_high[i] = np.max(high[i - k_period + 1:i + 1])
            
            k_percent = np.divide(
                (close - lowest_low) * 100,
                (highest_high - lowest_low),
                out=np.zeros_like(close),
                where=(highest_high - lowest_low) != 0
            )
            
            # Calculate %D (SMA of %K)
            d_percent = self._sma(k_percent, d_period)
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            zeros = np.zeros(len(close))
            return {'k_percent': zeros, 'd_percent': zeros}
    
    def momentum(self, prices: Union[pd.Series, np.ndarray], period: int = 10) -> np.ndarray:
        """Calculate Momentum indicator"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            momentum = np.zeros_like(prices)
            
            for i in range(period, len(prices)):
                momentum[i] = prices[i] - prices[i - period]
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating Momentum: {e}")
            return np.zeros(len(prices))
    
    def roc(self, prices: Union[pd.Series, np.ndarray], period: int = 10) -> np.ndarray:
        """Calculate Rate of Change (ROC)"""
        try:
            if isinstance(prices, pd.Series):
                prices = prices.values
            
            roc = np.zeros_like(prices)
            
            for i in range(period, len(prices)):
                if prices[i - period] != 0:
                    roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
            
            return roc
            
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {e}")
            return np.zeros(len(prices))
    
    def williams_r(self, high: Union[pd.Series, np.ndarray], 
                   low: Union[pd.Series, np.ndarray], 
                   close: Union[pd.Series, np.ndarray], 
                   period: int = 14) -> np.ndarray:
        """Calculate Williams %R"""
        try:
            if isinstance(high, pd.Series):
                high = high.values
            if isinstance(low, pd.Series):
                low = low.values
            if isinstance(close, pd.Series):
                close = close.values
            
            williams_r = np.zeros_like(close)
            
            for i in range(period - 1, len(close)):
                highest_high = np.max(high[i - period + 1:i + 1])
                lowest_low = np.min(low[i - period + 1:i + 1])
                
                if highest_high != lowest_low:
                    williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
            
            return williams_r
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return np.full(len(close), -50.0)
    
    def cci(self, high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], 
            period: int = 20) -> np.ndarray:
        """Calculate Commodity Channel Index (CCI)"""
        try:
            if isinstance(high, pd.Series):
                high = high.values
            if isinstance(low, pd.Series):
                low = low.values
            if isinstance(close, pd.Series):
                close = close.values
            
            # Typical Price
            tp = (high + low + close) / 3
            
            # Simple Moving Average of Typical Price
            sma_tp = self._sma(tp, period)
            
            # Mean Deviation
            mean_dev = np.zeros_like(tp)
            for i in range(period - 1, len(tp)):
                mean_dev[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - sma_tp[i]))
            
            # CCI
            cci = np.divide(
                (tp - sma_tp),
                (0.015 * mean_dev),
                out=np.zeros_like(tp),
                where=mean_dev != 0
            )
            
            return cci
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return np.zeros(len(close))
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.zeros_like(prices)
        
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return sma
