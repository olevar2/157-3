"""
Stochastic Oscillator
====================

The Stochastic Oscillator is a momentum indicator that compares a particular 
closing price to a range of prices over a certain period of time.

Author: Platform3 AI System
Created: June 3, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class StochasticOscillator(IndicatorBase):
    """
    Stochastic Oscillator indicator.
    
    The Stochastic Oscillator consists of:
    - %K: Raw stochastic value
    - %D: Signal line (SMA of %K)
    - %D-Slow: Slower signal line (SMA of %D)
    """
    
    def __init__(self, 
                 k_period: int = 14,
                 d_period: int = 3,
                 d_slow_period: int = 3,
                 overbought_level: float = 80.0,
                 oversold_level: float = 20.0):
        """
        Initialize Stochastic Oscillator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            d_slow_period: Period for %D-Slow smoothing
            overbought_level: Overbought threshold
            oversold_level: Oversold threshold
        """
        super().__init__()
        
        self.k_period = k_period
        self.d_period = d_period
        self.d_slow_period = d_slow_period
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
        
        # Validation
        if k_period <= 0 or d_period <= 0 or d_slow_period <= 0:
            raise ValueError("All periods must be positive")
        if not 0 <= oversold_level < overbought_level <= 100:
            raise ValueError("Invalid overbought/oversold levels")
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Input data array
            period: SMA period
            
        Returns:
            SMA values array
        """
        sma = np.full_like(data, np.nan)
        
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        
        return sma
    
    def calculate(self, 
                 data: pd.DataFrame,
                 high_column: str = 'high',
                 low_column: str = 'low',
                 close_column: str = 'close') -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with OHLC data
            high_column: Column name for high prices
            low_column: Column name for low prices
            close_column: Column name for close prices
            
        Returns:
            DataFrame with Stochastic components and related metrics
        """
        if len(data) < self.k_period + self.d_period + self.d_slow_period:
            raise ValueError(f"Insufficient data. Need at least {self.k_period + self.d_period + self.d_slow_period} rows")
        
        high = data[high_column].values
        low = data[low_column].values
        close = data[close_column].values
        
        # Calculate %K (Fast Stochastic)
        k_values = np.full_like(close, np.nan)
        
        for i in range(self.k_period - 1, len(data)):
            # Find highest high and lowest low in the period
            period_high = np.max(high[i - self.k_period + 1:i + 1])
            period_low = np.min(low[i - self.k_period + 1:i + 1])
            
            # Calculate %K
            if period_high != period_low:
                k_values[i] = 100 * (close[i] - period_low) / (period_high - period_low)
            else:
                k_values[i] = 50  # Neutral value when no range
        
        # Calculate %D (Signal line - SMA of %K)
        d_values = self._calculate_sma(k_values, self.d_period)
        
        # Calculate %D-Slow (SMA of %D)
        d_slow_values = self._calculate_sma(d_values, self.d_slow_period)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'stoch_k': k_values,
            'stoch_d': d_values,
            'stoch_d_slow': d_slow_values,
            'high': high,
            'low': low,
            'close': close
        })
        
        # Calculate additional metrics
        result_df['k_momentum'] = result_df['stoch_k'].diff()
        result_df['d_momentum'] = result_df['stoch_d'].diff()
        result_df['k_d_spread'] = result_df['stoch_k'] - result_df['stoch_d']
        result_df['d_dslow_spread'] = result_df['stoch_d'] - result_df['stoch_d_slow']
        
        # Position indicators
        result_df['k_above_d'] = (result_df['stoch_k'] > result_df['stoch_d']).astype(int)
        result_df['d_above_dslow'] = (result_df['stoch_d'] > result_df['stoch_d_slow']).astype(int)
        
        # Overbought/Oversold conditions
        result_df['k_overbought'] = (result_df['stoch_k'] > self.overbought_level).astype(int)
        result_df['k_oversold'] = (result_df['stoch_k'] < self.oversold_level).astype(int)
        result_df['d_overbought'] = (result_df['stoch_d'] > self.overbought_level).astype(int)
        result_df['d_oversold'] = (result_df['stoch_d'] < self.oversold_level).astype(int)
        
        # Both %K and %D in extreme zones
        result_df['both_overbought'] = (
            result_df['k_overbought'] & result_df['d_overbought']
        ).astype(int)
        result_df['both_oversold'] = (
            result_df['k_oversold'] & result_df['d_oversold']
        ).astype(int)
        
        # Distance from extreme levels
        result_df['distance_to_overbought'] = self.overbought_level - result_df['stoch_k']
        result_df['distance_to_oversold'] = result_df['stoch_k'] - self.oversold_level
        
        return result_df
    
    def get_signals(self, 
                   indicator_data: pd.DataFrame,
                   confirm_with_d: bool = True) -> pd.DataFrame:
        """
        Generate trading signals based on Stochastic Oscillator.
        
        Args:
            indicator_data: DataFrame from calculate() method
            confirm_with_d: Whether to confirm signals with %D line
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=indicator_data.index)
        
        # Basic %K crossing %D signals
        signals['stoch_bullish_crossover'] = (
            (indicator_data['stoch_k'] > indicator_data['stoch_d']) &
            (indicator_data['stoch_k'].shift(1) <= indicator_data['stoch_d'].shift(1))
        ).astype(int)
        
        signals['stoch_bearish_crossover'] = (
            (indicator_data['stoch_k'] < indicator_data['stoch_d']) &
            (indicator_data['stoch_k'].shift(1) >= indicator_data['stoch_d'].shift(1))
        ).astype(int)
        
        # Overbought/Oversold reversal signals
        signals['oversold_reversal'] = (
            (indicator_data['stoch_k'] > self.oversold_level) &
            (indicator_data['stoch_k'].shift(1) <= self.oversold_level) &
            (indicator_data['k_momentum'] > 0)
        ).astype(int)
        
        signals['overbought_reversal'] = (
            (indicator_data['stoch_k'] < self.overbought_level) &
            (indicator_data['stoch_k'].shift(1) >= self.overbought_level) &
            (indicator_data['k_momentum'] < 0)
        ).astype(int)
        
        # Confirmed signals (both %K and %D must agree)
        if confirm_with_d:
            signals['confirmed_bullish'] = (
                signals['stoch_bullish_crossover'] &
                (indicator_data['stoch_d'] < 50)  # %D below midline
            ).astype(int)
            
            signals['confirmed_bearish'] = (
                signals['stoch_bearish_crossover'] &
                (indicator_data['stoch_d'] > 50)  # %D above midline
            ).astype(int)
        else:
            signals['confirmed_bullish'] = signals['stoch_bullish_crossover']
            signals['confirmed_bearish'] = signals['stoch_bearish_crossover']
        
        # Extreme zone signals
        signals['extreme_oversold'] = (
            (indicator_data['stoch_k'] < 10) &
            (indicator_data['stoch_d'] < 10)
        ).astype(int)
        
        signals['extreme_overbought'] = (
            (indicator_data['stoch_k'] > 90) &
            (indicator_data['stoch_d'] > 90)
        ).astype(int)
        
        # Momentum signals
        signals['k_momentum_up'] = (
            indicator_data['k_momentum'] > 0
        ).astype(int)
        
        signals['k_momentum_down'] = (
            indicator_data['k_momentum'] < 0
        ).astype(int)
        
        signals['d_momentum_up'] = (
            indicator_data['d_momentum'] > 0
        ).astype(int)
        
        signals['d_momentum_down'] = (
            indicator_data['d_momentum'] < 0
        ).astype(int)
        
        # Both lines moving in same direction
        signals['both_rising'] = (
            (indicator_data['k_momentum'] > 0) &
            (indicator_data['d_momentum'] > 0)
        ).astype(int)
        
        signals['both_falling'] = (
            (indicator_data['k_momentum'] < 0) &
            (indicator_data['d_momentum'] < 0)
        ).astype(int)
        
        # Fast Stochastic signals (based on raw %K)
        signals['fast_stoch_bullish'] = (
            (indicator_data['stoch_k'] > 20) &
            (indicator_data['stoch_k'].shift(1) <= 20) &
            (indicator_data['k_momentum'] > 1)
        ).astype(int)
        
        signals['fast_stoch_bearish'] = (
            (indicator_data['stoch_k'] < 80) &
            (indicator_data['stoch_k'].shift(1) >= 80) &
            (indicator_data['k_momentum'] < -1)
        ).astype(int)
        
        # Triple line confirmation (K, D, D-Slow all agree)
        signals['triple_line_bullish'] = (
            (indicator_data['stoch_k'] > indicator_data['stoch_d']) &
            (indicator_data['stoch_d'] > indicator_data['stoch_d_slow']) &
            (indicator_data['k_momentum'] > 0) &
            (indicator_data['d_momentum'] > 0)
        ).astype(int)
        
        signals['triple_line_bearish'] = (
            (indicator_data['stoch_k'] < indicator_data['stoch_d']) &
            (indicator_data['stoch_d'] < indicator_data['stoch_d_slow']) &
            (indicator_data['k_momentum'] < 0) &
            (indicator_data['d_momentum'] < 0)
        ).astype(int)
        
        return signals
    
    def detect_divergence(self, 
                         price_data: np.ndarray,
                         stoch_data: np.ndarray,
                         lookback: int = 10) -> Dict[str, bool]:
        """
        Detect bullish and bearish divergences.
        
        Args:
            price_data: Price data array
            stoch_data: Stochastic %K data array
            lookback: Lookback period for divergence detection
            
        Returns:
            Dictionary with divergence detection results
        """
        if len(price_data) < lookback * 2 or len(stoch_data) < lookback * 2:
            return {'bullish_divergence': False, 'bearish_divergence': False}
        
        # Find recent highs and lows
        recent_price = price_data[-lookback:]
        recent_stoch = stoch_data[-lookback:]
        prev_price = price_data[-lookback*2:-lookback]
        prev_stoch = stoch_data[-lookback*2:-lookback]
        
        # Find peaks and troughs
        recent_price_high = np.max(recent_price)
        recent_stoch_high = np.max(recent_stoch)
        prev_price_high = np.max(prev_price)
        prev_stoch_high = np.max(prev_stoch)
        
        recent_price_low = np.min(recent_price)
        recent_stoch_low = np.min(recent_stoch)
        prev_price_low = np.min(prev_price)
        prev_stoch_low = np.min(prev_stoch)
        
        # Bearish divergence: higher highs in price, lower highs in Stochastic
        bearish_divergence = (
            recent_price_high > prev_price_high and
            recent_stoch_high < prev_stoch_high and
            recent_stoch_high > 70  # Must be in overbought area
        )
        
        # Bullish divergence: lower lows in price, higher lows in Stochastic
        bullish_divergence = (
            recent_price_low < prev_price_low and
            recent_stoch_low > prev_stoch_low and
            recent_stoch_low < 30  # Must be in oversold area
        )
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def get_interpretation(self, latest_values: Dict) -> str:
        """
        Provide interpretation of current Stochastic state.
        
        Args:
            latest_values: Dictionary with latest indicator values
            
        Returns:
            String interpretation
        """
        k = latest_values.get('stoch_k', 50)
        d = latest_values.get('stoch_d', 50)
        d_slow = latest_values.get('stoch_d_slow', 50)
        k_momentum = latest_values.get('k_momentum', 0)
        d_momentum = latest_values.get('d_momentum', 0)
        
        # Position interpretation
        if k > self.overbought_level and d > self.overbought_level:
            position_desc = "deeply overbought"
        elif k > self.overbought_level:
            position_desc = "overbought"
        elif k > 70:
            position_desc = "bullish zone"
        elif k > 50:
            position_desc = "slightly bullish"
        elif k > 30:
            position_desc = "slightly bearish"
        elif k > self.oversold_level:
            position_desc = "bearish zone"
        elif k < self.oversold_level and d < self.oversold_level:
            position_desc = "deeply oversold"
        else:
            position_desc = "oversold"
        
        # Momentum interpretation
        if k_momentum > 2 and d_momentum > 0:
            momentum_desc = "strong upward momentum"
        elif k_momentum > 0 and d_momentum > 0:
            momentum_desc = "upward momentum"
        elif k_momentum < -2 and d_momentum < 0:
            momentum_desc = "strong downward momentum"
        elif k_momentum < 0 and d_momentum < 0:
            momentum_desc = "downward momentum"
        else:
            momentum_desc = "mixed momentum"
        
        # Line relationship
        if k > d and d > d_slow:
            relationship = "all lines rising"
        elif k < d and d < d_slow:
            relationship = "all lines falling"
        elif k > d:
            relationship = "%K above %D"
        elif k < d:
            relationship = "%K below %D"
        else:
            relationship = "lines converging"
        
        return f"Stochastic in {position_desc} territory (%K: {k:.1f}, %D: {d:.1f}). " \
               f"Shows {momentum_desc} with {relationship}."


def create_stochastic_indicator(k_period: int = 14,
                               d_period: int = 3,
                               d_slow_period: int = 3,
                               overbought_level: float = 80.0,
                               oversold_level: float = 20.0) -> StochasticOscillator:
    """
    Factory function to create Stochastic Oscillator.
    
    Args:
        k_period: Period for %K calculation
        d_period: Period for %D smoothing
        d_slow_period: Period for %D-Slow smoothing
        overbought_level: Overbought threshold
        oversold_level: Oversold threshold
        
    Returns:
        Configured StochasticOscillator instance
    """
    return StochasticOscillator(
        k_period=k_period,
        d_period=d_period,
        d_slow_period=d_slow_period,
        overbought_level=overbought_level,
        oversold_level=oversold_level
    )
