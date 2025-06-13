#!/usr/bin/env python3
"""
VPTTrendState - Volume Price Trend State Analysis Indicator

A comprehensive volume-price trend state analyzer that determines the current
market trend state based on volume and price action patterns.

Author: Platform3 AI Enhancement Engine
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # For direct script execution, create a minimal base class
    class BaseIndicator:
        def __init__(self, name: str):
            self.name = name


class VPTTrendState(BaseIndicator):
    """
    Volume Price Trend State Indicator
    
    Analyzes the relationship between volume and price trends to determine
    the current market state. Uses volume flow analysis combined with
    price momentum to classify market conditions.
    
    The indicator provides multiple signals:
    - Trend strength classification
    - Volume confirmation levels
    - Market state transitions
    - Momentum persistence
    """
    
    def __init__(self, 
                 vpt_period: int = 14,
                 trend_period: int = 10,
                 volume_threshold: float = 1.2,
                 strength_levels: List[float] = None):
        """
        Initialize VPTTrendState indicator.
        
        Args:
            vpt_period: Period for VPT calculation (default: 14)
            trend_period: Period for trend analysis (default: 10)
            volume_threshold: Volume threshold multiplier (default: 1.2)
            strength_levels: Trend strength classification levels
        """
        super().__init__("VPTTrendState")
        self.vpt_period = max(1, vpt_period)
        self.trend_period = max(1, trend_period)
        self.volume_threshold = max(0.1, volume_threshold)
        
        if strength_levels is None:
            self.strength_levels = [0.2, 0.4, 0.6, 0.8]
        else:
            self.strength_levels = sorted(strength_levels)
        
        # State variables
        self.vpt_values = []
        self.trend_states = []
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate VPT trend state signals.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing:
            - vpt: Volume Price Trend values
            - vpt_trend: VPT trend direction
            - volume_confirmation: Volume confirmation level
            - trend_strength: Trend strength classification
            - market_state: Market state classification
            - signal: Main trading signal (-1 to 1)
        """
        try:
            if len(data) < max(self.vpt_period, self.trend_period):
                return self._empty_result(len(data))
            
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate Volume Price Trend
            vpt = self._calculate_vpt(close, volume)
            
            # Calculate VPT trend
            vpt_trend = self._calculate_vpt_trend(vpt)
            
            # Calculate volume confirmation
            volume_confirmation = self._calculate_volume_confirmation(volume, close)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(vpt_trend, volume_confirmation)
            
            # Determine market state
            market_state = self._determine_market_state(vpt_trend, trend_strength, volume_confirmation)
            
            # Generate main signal
            signal = self._generate_signal(market_state, trend_strength, volume_confirmation)
            
            return {
                'vpt': vpt,
                'vpt_trend': vpt_trend,
                'volume_confirmation': volume_confirmation,
                'trend_strength': trend_strength,
                'market_state': market_state,
                'signal': signal
            }
            
        except Exception as e:
            warnings.warn(f"Error in VPTTrendState calculation: {str(e)}")
            return self._empty_result(len(data))
    
    def _calculate_vpt(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate Volume Price Trend."""
        vpt = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if close[i-1] != 0:
                price_change_pct = (close[i] - close[i-1]) / close[i-1]
                vpt[i] = vpt[i-1] + (volume[i] * price_change_pct)
            else:
                vpt[i] = vpt[i-1]
        
        return vpt
    
    def _calculate_vpt_trend(self, vpt: np.ndarray) -> np.ndarray:
        """Calculate VPT trend direction."""
        if len(vpt) < self.trend_period:
            return np.zeros_like(vpt)
        
        trend = np.zeros_like(vpt)
        
        for i in range(self.trend_period, len(vpt)):
            # Calculate slope over trend period
            x = np.arange(self.trend_period)
            y = vpt[i-self.trend_period+1:i+1]
            
            if len(y) == self.trend_period:
                # Simple linear regression slope
                slope = np.polyfit(x, y, 1)[0]
                trend[i] = np.tanh(slope / (np.std(y) + 1e-8))  # Normalize and bound
        
        return trend
    
    def _calculate_volume_confirmation(self, volume: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate volume confirmation level."""
        if len(volume) < self.vpt_period:
            return np.zeros_like(volume)
        
        confirmation = np.zeros_like(volume)
        
        # Calculate rolling volume average
        volume_ma = pd.Series(volume).rolling(
            window=self.vpt_period, min_periods=1).mean().values
        
        # Calculate price momentum
        price_momentum = np.zeros_like(close)
        for i in range(1, len(close)):
            if close[i-1] != 0:
                price_momentum[i] = (close[i] - close[i-1]) / close[i-1]
        
        # Volume confirmation based on volume relative to average and price movement
        for i in range(len(volume)):
            if volume_ma[i] > 0:
                volume_ratio = volume[i] / volume_ma[i]
                
                # Higher volume during price moves = confirmation
                momentum_factor = abs(price_momentum[i]) * 10  # Scale up
                confirmation[i] = np.tanh(volume_ratio * momentum_factor)
            
        return confirmation
    
    def _calculate_trend_strength(self, vpt_trend: np.ndarray, 
                                volume_confirmation: np.ndarray) -> np.ndarray:
        """Calculate trend strength classification."""
        # Combine VPT trend and volume confirmation
        raw_strength = np.abs(vpt_trend) * (1 + volume_confirmation)
        
        # Classify into strength levels
        strength = np.zeros_like(raw_strength)
        
        for i, level in enumerate(self.strength_levels):
            mask = raw_strength >= level
            strength[mask] = i + 1
        
        # Normalize to 0-1 range
        if len(self.strength_levels) > 0:
            strength = strength / len(self.strength_levels)
        
        return strength
    
    def _determine_market_state(self, vpt_trend: np.ndarray, trend_strength: np.ndarray,
                              volume_confirmation: np.ndarray) -> np.ndarray:
        """Determine market state classification."""
        market_state = np.zeros_like(vpt_trend)
        
        for i in range(len(vpt_trend)):
            # State classification based on trend and strength
            if trend_strength[i] > 0.8:
                if vpt_trend[i] > 0.3:
                    market_state[i] = 1.0  # Strong Uptrend
                elif vpt_trend[i] < -0.3:
                    market_state[i] = -1.0  # Strong Downtrend
                else:
                    market_state[i] = 0.0  # Neutral/Consolidation
            elif trend_strength[i] > 0.4:
                if vpt_trend[i] > 0.1:
                    market_state[i] = 0.5  # Weak Uptrend
                elif vpt_trend[i] < -0.1:
                    market_state[i] = -0.5  # Weak Downtrend
                else:
                    market_state[i] = 0.0  # Neutral
            else:
                market_state[i] = 0.0  # Weak/No trend
        
        return market_state
    
    def _generate_signal(self, market_state: np.ndarray, trend_strength: np.ndarray,
                        volume_confirmation: np.ndarray) -> np.ndarray:
        """Generate main trading signal."""
        # Combine market state with strength and volume confirmation
        signal = market_state * trend_strength * (1 + volume_confirmation * 0.5)
        
        # Apply smoothing
        if len(signal) >= 3:
            signal_smooth = pd.Series(signal).rolling(
                window=3, min_periods=1).mean().values
        else:
            signal_smooth = signal
        
        return np.clip(signal_smooth, -1.0, 1.0)
    
    def _empty_result(self, length: int) -> Dict[str, np.ndarray]:
        """Return empty result arrays."""
        return {
            'vpt': np.full(length, np.nan),
            'vpt_trend': np.zeros(length),
            'volume_confirmation': np.zeros(length),
            'trend_strength': np.zeros(length),
            'market_state': np.zeros(length),
            'signal': np.zeros(length)
        }
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get trading signals as a DataFrame.
        
        Args:
            data: Input OHLCV data
            
        Returns:
            DataFrame with signal columns
        """
        result = self.calculate(data)
        
        signals_df = pd.DataFrame(index=data.index)
        for key, value in result.items():
            signals_df[f'vpt_trend_{key}'] = value
        
        return signals_df
    
    def get_market_state_description(self, state_value: float) -> str:
        """Get human-readable market state description."""
        if state_value >= 0.8:
            return "Strong Uptrend"
        elif state_value >= 0.3:
            return "Weak Uptrend"
        elif state_value <= -0.8:
            return "Strong Downtrend"
        elif state_value <= -0.3:
            return "Weak Downtrend"
        else:
            return "Neutral/Consolidation"


def demonstrate_vpt_trend_state():
    """Demonstrate VPTTrendState indicator usage."""
    print("=" * 50)
    print("VPTTrendState Indicator Demonstration")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create realistic OHLCV data with trend patterns
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Add trend and noise
        if i < 30:
            trend = 0.02  # Uptrend
        elif i < 60:
            trend = -0.015  # Downtrend
        else:
            trend = 0.005  # Weak uptrend
        
        noise = np.random.normal(0, 0.01)
        price_change = trend + noise
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + price_change)
        
        # Create OHLC from base price
        volatility = abs(price_change) * 3
        high = price * (1 + volatility * np.random.uniform(0, 1))
        low = price * (1 - volatility * np.random.uniform(0, 1))
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        close = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        prices.append(close)
        
        # Volume correlated with price movement and trend strength
        base_volume = 1000
        movement_volume = abs(price_change) * 10000
        trend_volume = abs(trend) * 5000
        random_volume = np.random.exponential(1000)
        
        volume = int(base_volume + movement_volume + trend_volume + random_volume)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'open': [prices[0]] + [p * 0.999 for p in prices[:-1]],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Test different parameter sets
    configs = [
        {"vpt_period": 10, "trend_period": 7, "name": "Fast"},
        {"vpt_period": 14, "trend_period": 10, "name": "Standard"},
        {"vpt_period": 20, "trend_period": 15, "name": "Slow"}
    ]
    
    for config in configs:
        print(f"\n{config['name']} VPTTrendState Configuration:")
        print(f"VPT Period: {config['vpt_period']}, Trend Period: {config['trend_period']}")
        
        # Create and calculate indicator
        indicator = VPTTrendState(
            vpt_period=config['vpt_period'],
            trend_period=config['trend_period']
        )
        
        result = indicator.calculate(data)
        
        # Display statistics
        print(f"\nResults Summary:")
        for key, values in result.items():
            if len(values) > 0 and not np.all(np.isnan(values)):
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    print(f"{key}:")
                    print(f"  Range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
                    print(f"  Mean: {valid_values.mean():.4f}")
                    print(f"  Std: {valid_values.std():.4f}")
        
        # Show recent signals and market states
        print(f"\nRecent Market States (last 10 periods):")
        recent_states = result['market_state'][-10:]
        recent_signals = result['signal'][-10:]
        
        for i in range(len(recent_states)):
            date = data.index[-10 + i].strftime('%Y-%m-%d')
            state = recent_states[i]
            signal = recent_signals[i]
            state_desc = indicator.get_market_state_description(state)
            print(f"  {date}: State={state:.3f} ({state_desc}), Signal={signal:.3f}")
        
        # Current state analysis
        current_state = result['market_state'][-1]
        current_signal = result['signal'][-1]
        current_strength = result['trend_strength'][-1]
        current_confirmation = result['volume_confirmation'][-1]
        
        print(f"\nCurrent Analysis:")
        print(f"  Market State: {indicator.get_market_state_description(current_state)}")
        print(f"  Signal Strength: {current_signal:.4f}")
        print(f"  Trend Strength: {current_strength:.4f}")
        print(f"  Volume Confirmation: {current_confirmation:.4f}")
        
        if current_signal > 0.5:
            print("  -> Strong Buy Signal")
        elif current_signal > 0.2:
            print("  -> Buy Signal")
        elif current_signal < -0.5:
            print("  -> Strong Sell Signal")
        elif current_signal < -0.2:
            print("  -> Sell Signal")
        else:
            print("  -> Neutral/Hold")
    
    # Test edge cases
    print(f"\n" + "="*50)
    print("Edge Case Testing:")
    print("="*50)
    
    # Test with minimal data
    small_data = data.head(5)
    indicator = VPTTrendState(vpt_period=20, trend_period=15)
    result = indicator.calculate(small_data)
    print(f"Small dataset test - Signal range: [{result['signal'].min():.3f}, {result['signal'].max():.3f}]")
    
    # Test with zero volume
    zero_vol_data = data.copy()
    zero_vol_data['volume'] = 0
    result = indicator.calculate(zero_vol_data)
    print(f"Zero volume test - Signal range: [{result['signal'].min():.3f}, {result['signal'].max():.3f}]")
    
    # Test with constant prices
    const_data = data.copy()
    const_data['high'] = const_data['low'] = const_data['close'] = 100
    result = indicator.calculate(const_data)
    print(f"Constant price test - Signal range: [{result['signal'].min():.3f}, {result['signal'].max():.3f}]")
    
    print(f"\n" + "="*50)
    print("VPTTrendState demonstration completed!")
    print("="*50)


if __name__ == "__main__":
    demonstrate_vpt_trend_state()