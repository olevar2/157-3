#!/usr/bin/env python3
"""
TickVolumeSignal - Professional Tick Volume Analysis Indicator

A comprehensive tick volume analysis indicator that tracks and analyzes
price movement per tick to identify market activity patterns.

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

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # For direct script execution, create a minimal base class
    class BaseIndicator:
        def __init__(self, name: str):
            self.name = name


class TickVolumeSignal(BaseIndicator):
    """
    Tick Volume Signal Indicator
    
    Analyzes tick volume patterns to identify market participation and
    price movement efficiency. Uses tick-by-tick volume data to determine
    market strength and potential reversal points.
    
    The indicator provides multiple signals:
    - Tick volume divergence with price
    - Volume per tick efficiency
    - Market participation levels
    - Tick distribution patterns
    """
    
    def __init__(self, 
                 period: int = 20,
                 smooth_period: int = 5,
                 volume_threshold: float = 1.5,
                 divergence_periods: int = 10):
        """
        Initialize TickVolumeSignal indicator.
        
        Args:
            period: Period for volume analysis (default: 20)
            smooth_period: Smoothing period for signals (default: 5)
            volume_threshold: Volume threshold multiplier (default: 1.5)
            divergence_periods: Periods to check for divergence (default: 10)
        """
        super().__init__("TickVolumeSignal")
        self.period = max(1, period)
        self.smooth_period = max(1, smooth_period)
        self.volume_threshold = max(0.1, volume_threshold)
        self.divergence_periods = max(1, divergence_periods)
        
        # State variables
        self.tick_volumes = []
        self.tick_prices = []
        self.volume_per_tick = []
        self.price_changes = []
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate tick volume signals.
        
        Args:
            data: DataFrame with OHLCV columns and optional tick data
            
        Returns:
            Dictionary containing:
            - tick_volume_ratio: Volume per tick ratio
            - volume_efficiency: Price movement per tick volume
            - participation_level: Market participation indicator
            - tick_divergence: Price-volume divergence signal
            - signal: Main trading signal (-1 to 1)
        """
        try:
            if len(data) < self.period:
                return self._empty_result(len(data))
            
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate tick approximations if tick data not available
            tick_count = self._estimate_tick_count(high, low, close, volume)
            
            # Core calculations
            volume_per_tick = self._calculate_volume_per_tick(volume, tick_count)
            price_movement = self._calculate_price_movement(high, low, close)
            efficiency = self._calculate_efficiency(price_movement, volume_per_tick)
            participation = self._calculate_participation(volume, tick_count)
            divergence = self._calculate_divergence(close, volume_per_tick)
            
            # Generate main signal
            signal = self._generate_signal(efficiency, participation, divergence)
            
            return {
                'tick_volume_ratio': volume_per_tick,
                'volume_efficiency': efficiency,
                'participation_level': participation,
                'tick_divergence': divergence,
                'signal': signal
            }
            
        except Exception as e:
            warnings.warn(f"Error in TickVolumeSignal calculation: {str(e)}")
            return self._empty_result(len(data))
    
    def _estimate_tick_count(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Estimate tick count from OHLC data."""
        # Use range and volume to estimate tick activity
        price_range = high - low
        avg_range = np.full_like(price_range, np.mean(price_range[price_range > 0]))
        avg_range[avg_range == 0] = 0.01  # Avoid division by zero
        
        # Estimate ticks based on range and volume
        tick_estimate = (price_range / avg_range) * np.log1p(volume)
        tick_estimate = np.maximum(tick_estimate, 1.0)  # Minimum 1 tick
        
        return tick_estimate
    
    def _calculate_volume_per_tick(self, volume: np.ndarray, 
                                 tick_count: np.ndarray) -> np.ndarray:
        """Calculate volume per tick ratio."""
        # Avoid division by zero
        tick_count_safe = np.maximum(tick_count, 1.0)
        volume_per_tick = volume / tick_count_safe
        
        # Normalize by rolling average
        window = min(self.period, len(volume_per_tick))
        if window > 1:
            rolling_avg = pd.Series(volume_per_tick).rolling(
                window=window, min_periods=1).mean().values
            rolling_avg = np.maximum(rolling_avg, 1.0)  # Avoid division by zero
            normalized_ratio = volume_per_tick / rolling_avg
        else:
            normalized_ratio = np.ones_like(volume_per_tick)
        
        return normalized_ratio
    
    def _calculate_price_movement(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray) -> np.ndarray:
        """Calculate price movement efficiency."""
        # True range style calculation
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Price change efficiency
        price_change = np.abs(close - prev_close)
        
        # Avoid division by zero
        true_range_safe = np.maximum(true_range, 0.0001)
        efficiency = price_change / true_range_safe
        
        return efficiency
    
    def _calculate_efficiency(self, price_movement: np.ndarray, 
                            volume_per_tick: np.ndarray) -> np.ndarray:
        """Calculate price movement efficiency per volume."""
        # Avoid division by zero
        volume_safe = np.maximum(volume_per_tick, 0.0001)
        efficiency = price_movement / volume_safe
        
        # Smooth the efficiency
        if len(efficiency) >= self.smooth_period:
            efficiency_smooth = pd.Series(efficiency).rolling(
                window=self.smooth_period, min_periods=1).mean().values
        else:
            efficiency_smooth = efficiency
        
        return efficiency_smooth
    
    def _calculate_participation(self, volume: np.ndarray, 
                               tick_count: np.ndarray) -> np.ndarray:
        """Calculate market participation level."""
        # Combine volume and tick activity
        participation = np.log1p(volume) * np.log1p(tick_count)
        
        # Normalize by rolling percentile
        window = min(self.period, len(participation))
        if window > 1:
            rolling_rank = pd.Series(participation).rolling(
                window=window, min_periods=1).rank(pct=True).values
        else:
            rolling_rank = np.full_like(participation, 0.5)
        
        return rolling_rank
    
    def _calculate_divergence(self, price: np.ndarray, 
                            volume_per_tick: np.ndarray) -> np.ndarray:
        """Calculate price-volume divergence."""
        if len(price) < self.divergence_periods:
            return np.zeros_like(price)
        
        # Calculate trends
        price_trend = self._calculate_trend(price, self.divergence_periods)
        volume_trend = self._calculate_trend(volume_per_tick, self.divergence_periods)
        
        # Divergence occurs when trends oppose
        divergence = price_trend * volume_trend
        
        # Convert to signal: negative values indicate divergence
        divergence_signal = np.tanh(-divergence)  # Invert and normalize
        
        return divergence_signal
    
    def _calculate_trend(self, data: np.ndarray, periods: int) -> np.ndarray:
        """Calculate trend direction over specified periods."""
        trend = np.zeros_like(data)
        
        for i in range(periods, len(data)):
            recent = data[i-periods:i+1]
            if len(recent) >= 2:
                # Simple linear trend
                x = np.arange(len(recent))
                slope = np.corrcoef(x, recent)[0, 1] if len(recent) > 1 else 0
                trend[i] = slope
        
        return trend
    
    def _generate_signal(self, efficiency: np.ndarray, participation: np.ndarray, 
                        divergence: np.ndarray) -> np.ndarray:
        """Generate main trading signal."""
        # Combine signals
        signal = np.zeros_like(efficiency)
        
        # Efficiency component (higher is better)
        eff_norm = (efficiency - np.mean(efficiency)) / (np.std(efficiency) + 1e-8)
        
        # Participation component (higher is better)
        participation_signal = (participation - 0.5) * 2  # Convert to -1 to 1
        
        # Divergence component (negative divergence is bearish)
        divergence_signal = divergence
        
        # Combine signals with weights
        signal = (0.4 * np.tanh(eff_norm) + 
                 0.3 * participation_signal + 
                 0.3 * divergence_signal)
        
        # Apply smoothing
        if len(signal) >= self.smooth_period:
            signal = pd.Series(signal).rolling(
                window=self.smooth_period, min_periods=1).mean().values
        
        return np.clip(signal, -1.0, 1.0)
    
    def _empty_result(self, length: int) -> Dict[str, np.ndarray]:
        """Return empty result arrays."""
        return {
            'tick_volume_ratio': np.full(length, np.nan),
            'volume_efficiency': np.full(length, np.nan),
            'participation_level': np.full(length, np.nan),
            'tick_divergence': np.full(length, np.nan),
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
            signals_df[f'tick_volume_{key}'] = value
        
        return signals_df


def demonstrate_tick_volume_signal():
    """Demonstrate TickVolumeSignal indicator usage."""
    print("=" * 50)
    print("TickVolumeSignal Indicator Demonstration")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create realistic OHLCV data with tick patterns
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Add some trend and noise
        trend = 0.01 * np.sin(i * 0.1)
        noise = np.random.normal(0, 0.02)
        price_change = trend + noise
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + price_change)
        
        # Create OHLC from base price
        volatility = abs(price_change) * 2
        high = price * (1 + volatility * np.random.uniform(0, 1))
        low = price * (1 - volatility * np.random.uniform(0, 1))
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        close = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        prices.append(close)
        
        # Volume correlated with volatility and some random patterns
        volume = int(1000 + volatility * 5000 + np.random.exponential(2000))
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
        {"period": 14, "smooth_period": 3, "name": "Fast"},
        {"period": 20, "smooth_period": 5, "name": "Standard"},
        {"period": 30, "smooth_period": 7, "name": "Slow"}
    ]
    
    for config in configs:
        print(f"\n{config['name']} TickVolumeSignal Configuration:")
        print(f"Period: {config['period']}, Smooth: {config['smooth_period']}")
        
        # Create and calculate indicator
        indicator = TickVolumeSignal(
            period=config['period'],
            smooth_period=config['smooth_period']
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
        
        # Show recent signals
        print(f"\nRecent Signals (last 10 periods):")
        recent_signals = result['signal'][-10:]
        for i, signal in enumerate(recent_signals):
            date = data.index[-10 + i].strftime('%Y-%m-%d')
            print(f"  {date}: {signal:.4f}")
        
        # Trading signal interpretation
        final_signal = result['signal'][-1]
        print(f"\nCurrent Signal: {final_signal:.4f}")
        if final_signal > 0.3:
            print("  -> Strong Bullish (High efficiency with strong participation)")
        elif final_signal > 0.1:
            print("  -> Bullish (Positive volume efficiency)")
        elif final_signal < -0.3:
            print("  -> Strong Bearish (Volume divergence detected)")
        elif final_signal < -0.1:
            print("  -> Bearish (Weak volume participation)")
        else:
            print("  -> Neutral (Mixed signals)")
    
    # Test edge cases
    print(f"\n" + "="*50)
    print("Edge Case Testing:")
    print("="*50)
    
    # Test with minimal data
    small_data = data.head(5)
    indicator = TickVolumeSignal(period=20)
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
    print("TickVolumeSignal demonstration completed!")
    print("="*50)


if __name__ == "__main__":
    demonstrate_tick_volume_signal()