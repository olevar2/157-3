"""
Fractal Adaptive Moving Average (FRAMA)
=======================================

Adaptive trend following using fractal dimension for dynamic smoothing.
Developed by John Ehlers, FRAMA adjusts its smoothing factor based on market fractality.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..indicator_base import IndicatorBase


class FractalAdaptiveMovingAverage(IndicatorBase):
    """
    Fractal Adaptive Moving Average (FRAMA) indicator.
    
    FRAMA adjusts its smoothing constant based on the fractal dimension
    of the price series, providing faster response in trending markets
    and slower response in choppy markets.
    """
    
    def __init__(self, 
                 period: int = 20,
                 fast_limit: float = 0.67,
                 slow_limit: float = 0.03):
        """
        Initialize FRAMA indicator.
        
        Args:
            period: Period for fractal dimension calculation
            fast_limit: Fast smoothing limit (trending markets)
            slow_limit: Slow smoothing limit (choppy markets)
        """
        super().__init__()
        
        self.period = period
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        
        # Validation
        if period <= 0:
            raise ValueError("period must be positive")
        if not 0 < slow_limit < fast_limit < 1:
            raise ValueError("Must have 0 < slow_limit < fast_limit < 1")
        
        # Internal state
        self.prev_frama = None
        
    def _calculate_fractal_dimension(self, 
                                   high: np.ndarray, 
                                   low: np.ndarray, 
                                   close: np.ndarray) -> float:
        """
        Calculate fractal dimension using Ehlers' method.
        
        Args:
            high: High prices array
            low: Low prices array  
            close: Close prices array
            
        Returns:
            Fractal dimension value
        """
        n = len(high)
        if n < self.period:
            return 2.0  # Default dimension
        
        # Calculate N1, N2, N3 as per Ehlers' formula
        half_period = self.period // 2
        
        # N1: Length of price path over full period
        n1 = (np.max(high) - np.min(low)) / self.period
        
        # N2: Length of price path over first half
        n2 = (np.max(high[:half_period]) - np.min(low[:half_period])) / half_period
        
        # N3: Length of price path over second half  
        n3 = (np.max(high[half_period:]) - np.min(low[half_period:])) / half_period
        
        # Fractal dimension calculation
        if n1 > 0 and (n2 + n3) > 0:
            dimension = (np.log(n2 + n3) - np.log(n1)) / np.log(2)
        else:
            dimension = 2.0
            
        # Clamp dimension between 1 and 2
        return max(1.0, min(2.0, dimension))
    
    def _calculate_alpha(self, dimension: float) -> float:
        """
        Calculate smoothing factor (alpha) from fractal dimension.
        
        Args:
            dimension: Fractal dimension value
            
        Returns:
            Alpha smoothing factor
        """
        # Convert dimension to alpha using Ehlers' formula
        alpha = np.exp(-4.6 * (dimension - 1))
        
        # Constrain alpha within limits
        return max(self.slow_limit, min(self.fast_limit, alpha))
    
    def calculate(self, 
                 data: pd.DataFrame,
                 high_column: str = 'high',
                 low_column: str = 'low', 
                 close_column: str = 'close') -> pd.DataFrame:
        """
        Calculate FRAMA indicator.
        
        Args:
            data: DataFrame with OHLC data
            high_column: Column name for high prices
            low_column: Column name for low prices
            close_column: Column name for close prices
            
        Returns:
            DataFrame with FRAMA values and related metrics
        """
        if len(data) < self.period:
            raise ValueError(f"Insufficient data. Need at least {self.period} rows")
        
        high = data[high_column].values
        low = data[low_column].values
        close = data[close_column].values
        
        frama_values = []
        dimensions = []
        alphas = []
        
        for i in range(len(data)):
            if i < self.period - 1:
                # Insufficient data for calculation
                frama_values.append(np.nan)
                dimensions.append(np.nan)
                alphas.append(np.nan)
                continue
                
            # Get data window
            start_idx = i - self.period + 1
            window_high = high[start_idx:i + 1]
            window_low = low[start_idx:i + 1]
            window_close = close[start_idx:i + 1]
            
            # Calculate fractal dimension
            dimension = self._calculate_fractal_dimension(
                window_high, window_low, window_close
            )
            
            # Calculate alpha from dimension
            alpha = self._calculate_alpha(dimension)
            
            # Calculate FRAMA
            if i == self.period - 1:
                # Initialize with simple average
                frama = np.mean(window_close)
            else:
                # FRAMA formula: FRAMA = alpha * Close + (1 - alpha) * FRAMA[prev]
                prev_frama = frama_values[i - 1]
                frama = alpha * close[i] + (1 - alpha) * prev_frama
            
            frama_values.append(frama)
            dimensions.append(dimension)
            alphas.append(alpha)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'frama': frama_values,
            'fractal_dimension': dimensions,
            'alpha': alphas,
            'close': close
        })
        
        # Calculate additional metrics
        result_df['frama_slope'] = result_df['frama'].diff()
        result_df['price_vs_frama'] = result_df['close'] - result_df['frama']
        result_df['frama_trend'] = np.where(
            result_df['frama_slope'] > 0, 1,
            np.where(result_df['frama_slope'] < 0, -1, 0)
        )
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   trend_threshold: float = 0.001) -> pd.DataFrame:
        """
        Generate trading signals based on FRAMA.
        
        Args:
            indicator_data: DataFrame from calculate() method
            trend_threshold: Minimum slope for trend confirmation
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # Price crossover signals
        signals['long_entry'] = (
            (indicator_data['close'] > indicator_data['frama']) &
            (indicator_data['close'].shift(1) <= indicator_data['frama'].shift(1)) &
            (indicator_data['frama_trend'] == 1)
        ).astype(int)
        
        signals['short_entry'] = (
            (indicator_data['close'] < indicator_data['frama']) &
            (indicator_data['close'].shift(1) >= indicator_data['frama'].shift(1)) &
            (indicator_data['frama_trend'] == -1)
        ).astype(int)
        
        # Trend strength signals
        signals['trend_strength'] = np.abs(indicator_data['frama_slope'])
        
        # Adaptive signals based on fractal dimension
        signals['adaptive_long'] = (
            (indicator_data['close'] > indicator_data['frama']) &
            (indicator_data['fractal_dimension'] < 1.5) &  # Trending market
            (indicator_data['alpha'] > 0.3)  # Fast response
        ).astype(int)
        
        signals['adaptive_short'] = (
            (indicator_data['close'] < indicator_data['frama']) &
            (indicator_data['fractal_dimension'] < 1.5) &  # Trending market
            (indicator_data['alpha'] > 0.3)  # Fast response
        ).astype(int)
        
        # Market state signals
        signals['trending_market'] = (
            indicator_data['fractal_dimension'] < 1.5
        ).astype(int)
        
        signals['choppy_market'] = (
            indicator_data['fractal_dimension'] > 1.7
        ).astype(int)
        
        return signals
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current FRAMA state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        frama = latest_values.get('frama', 0)
        close = latest_values.get('close', 0)
        dimension = latest_values.get('fractal_dimension', 2.0)
        alpha = latest_values.get('alpha', 0.03)
        trend = latest_values.get('frama_trend', 0)
        
        # Market state interpretation
        if dimension < 1.3:
            market_state = "Strong trending market"
        elif dimension < 1.7:
            market_state = "Moderately trending market"
        else:
            market_state = "Choppy/sideways market"
        
        # Price position
        if close > frama:
            position = "above FRAMA (bullish)"
        elif close < frama:
            position = "below FRAMA (bearish)"
        else:
            position = "at FRAMA (neutral)"
        
        # Trend direction
        if trend == 1:
            trend_dir = "uptrend"
        elif trend == -1:
            trend_dir = "downtrend"
        else:
            trend_dir = "sideways"
        
        # Response speed
        if alpha > 0.5:
            speed = "fast response"
        elif alpha > 0.2:
            speed = "moderate response"
        else:
            speed = "slow response"
        
        return f"{market_state} (D={dimension:.2f}). Price {position}. " \
               f"FRAMA in {trend_dir} with {speed} (α={alpha:.3f})."


def create_frama_indicator(period: int = 20,
                          fast_limit: float = 0.67,
                          slow_limit: float = 0.03) -> FractalAdaptiveMovingAverage:
    """
    Factory function to create FRAMA indicator.
    
    Args:
        period: Period for fractal dimension calculation
        fast_limit: Fast smoothing limit (trending markets)
        slow_limit: Slow smoothing limit (choppy markets)
        
    Returns:
        Configured FractalAdaptiveMovingAverage instance
    """
    return FractalAdaptiveMovingAverage(
        period=period,
        fast_limit=fast_limit,
        slow_limit=slow_limit
    )
