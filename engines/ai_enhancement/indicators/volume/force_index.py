#!/usr/bin/env python3
"""
Force Index Technical Indicator

The Force Index was developed by Dr. Alexander Elder to measure the use of power 
by bulls or bears to move prices. It combines price movement and volume to assess 
the amount of force used to move the price. This indicator helps confirm price 
movements and identify potential reversals.

Author: Platform3 AI Enhancement Engine
Date: 2024
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings
from dataclasses import dataclass

# Import base class
try:
    from ..base_indicator import BaseIndicator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from base_indicator import BaseIndicator
    except ImportError:
        # Fallback to creating a minimal base class
        class BaseIndicator:
            def __init__(self):
                pass


@dataclass
class ForceIndexConfig:
    """Configuration for Force Index parameters."""
    period: int = 13              # Smoothing period for Force Index
    use_smoothing: bool = True    # Whether to apply EMA smoothing
    signal_threshold: float = 0.0 # Signal threshold for trend changes
    min_periods: int = 13         # Minimum periods for calculation


class ForceIndex(BaseIndicator):
    """
    Force Index Technical Indicator Implementation
    
    The Force Index uses price and volume to assess the power behind a move or 
    identify possible turning points. It measures the amount of force used to 
    move the price of an asset.
    
    Formula:
    Raw Force Index = (Close - Previous Close) * Volume
    
    Optional smoothing:
    Smoothed Force Index = EMA(Raw Force Index, period)
    
    Interpretation:
    - Positive Force Index: Bulls are in control (buying pressure)
    - Negative Force Index: Bears are in control (selling pressure)
    - Large Force Index values: Strong conviction behind the move
    - Small Force Index values: Weak conviction behind the move
    - Zero line crossovers: Potential trend changes
    - Divergences: Early warning of potential reversals
    
    Usage variations:
    - 2-period FI: Very sensitive, good for day trading
    - 13-period FI: Medium-term analysis
    - 100-period FI: Long-term trend analysis
    """
    
    def __init__(self, config: Optional[ForceIndexConfig] = None):
        """
        Initialize Force Index indicator.
        
        Args:
            config: Configuration object with indicator parameters
        """
        self.config = config or ForceIndexConfig()
        self.name = "Force Index"
        self.category = "volume"
        
        # Validation
        if self.config.period < 1:
            raise ValueError("Period must be positive")
        if self.config.min_periods < 1:
            raise ValueError("Min periods must be positive")
        
        # Internal state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset internal calculation state."""
        self.raw_force_values: List[float] = []
        self.smoothed_force_values: List[float] = []
        self.price_changes: List[float] = []
        self.volumes: List[float] = []
        self.prev_close: Optional[float] = None
        self.signals: List[str] = []
        self.initialized = False
    
    def _calculate_ema(self, value: float, prev_ema: Optional[float], period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            value: Current value
            prev_ema: Previous EMA value
            period: EMA period
            
        Returns:
            Current EMA value
        """
        alpha = 2.0 / (period + 1)
        
        if prev_ema is None:
            return value
        
        return alpha * value + (1 - alpha) * prev_ema
    
    def update(self, high: float, low: float, close: float, volume: float, 
               timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Update indicator with new market data.
        
        Args:
            high: High price (not used directly but kept for interface consistency)
            low: Low price (not used directly but kept for interface consistency)
            close: Close price
            volume: Volume
            timestamp: Data timestamp
            
        Returns:
            Dictionary containing indicator values and signals
        """
        try:
            # Calculate price change
            if self.prev_close is not None:
                price_change = close - self.prev_close
            else:
                price_change = 0.0
            
            self.price_changes.append(price_change)
            self.volumes.append(volume)
            
            # Calculate raw Force Index
            raw_force = price_change * volume
            self.raw_force_values.append(raw_force)
            
            # Calculate smoothed Force Index if enabled
            if self.config.use_smoothing:
                smoothed_force = self._calculate_ema(
                    raw_force,
                    self.smoothed_force_values[-1] if self.smoothed_force_values else None,
                    self.config.period
                )
            else:
                smoothed_force = raw_force
            
            self.smoothed_force_values.append(smoothed_force)
            
            # Generate signals
            signal = self._generate_signals(raw_force, smoothed_force)
            self.signals.append(signal)
            
            # Update previous close
            self.prev_close = close
            
            # Mark as initialized after minimum periods
            if len(self.raw_force_values) >= self.config.min_periods:
                self.initialized = True
            
            return {
                'force_index': smoothed_force,
                'raw_force_index': raw_force,
                'price_change': price_change,
                'volume': volume,
                'signal': signal,
                'timestamp': timestamp,
                'initialized': self.initialized
            }
            
        except Exception as e:
            warnings.warn(f"Error updating Force Index: {str(e)}")
            return {
                'force_index': 0.0,
                'raw_force_index': 0.0,
                'price_change': 0.0,
                'volume': 0.0,
                'signal': 'NEUTRAL',
                'timestamp': timestamp,
                'initialized': False
            }
    
    def _generate_signals(self, raw_force: float, smoothed_force: float) -> str:
        """
        Generate trading signals based on Force Index.
        
        Args:
            raw_force: Current raw Force Index value
            smoothed_force: Current smoothed Force Index value
            
        Returns:
            Signal string
        """
        if not self.initialized or len(self.smoothed_force_values) < 2:
            return 'NEUTRAL'
        
        prev_smoothed = self.smoothed_force_values[-2]
        current_smoothed = smoothed_force
        
        # Zero line crossovers (primary signals)
        if prev_smoothed <= 0 < current_smoothed:
            return 'BUY'  # Force Index crosses above zero
        elif prev_smoothed >= 0 > current_smoothed:
            return 'SELL'  # Force Index crosses below zero
        
        # Trend strength signals
        if current_smoothed > self.config.signal_threshold:
            # Check momentum
            if len(self.smoothed_force_values) >= 3:
                momentum = current_smoothed - self.smoothed_force_values[-3]
                if momentum > 0:
                    return 'STRONG_BULLISH'
                else:
                    return 'BULLISH'
            return 'BULLISH'
        elif current_smoothed < -self.config.signal_threshold:
            # Check momentum
            if len(self.smoothed_force_values) >= 3:
                momentum = current_smoothed - self.smoothed_force_values[-3]
                if momentum < 0:
                    return 'STRONG_BEARISH'
                else:
                    return 'BEARISH'
            return 'BEARISH'
        
        return 'NEUTRAL'
    
    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator for entire dataset.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator values
        """
        required_columns = ['close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Reset state
        self._reset_state()
        
        results = []
        for idx, row in data.iterrows():
            # Use close price for high/low if not available
            high = row.get('high', row['close'])
            low = row.get('low', row['close'])
            
            result = self.update(
                high=high,
                low=low,
                close=row['close'],
                volume=row['volume'],
                timestamp=row.get('timestamp', idx)
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_signals(self) -> List[str]:
        """Get all generated signals."""
        return self.signals.copy()
    
    def get_force_values(self) -> List[float]:
        """Get all smoothed Force Index values."""
        return self.smoothed_force_values.copy()
    
    def get_raw_force_values(self) -> List[float]:
        """Get all raw Force Index values."""
        return self.raw_force_values.copy()
    
    def detect_divergence(self, price_data: List[float], lookback: int = 14) -> Optional[str]:
        """
        Detect bullish/bearish divergences between price and Force Index.
        
        Args:
            price_data: List of price values (typically close prices)
            lookback: Periods to look back for divergence
            
        Returns:
            Divergence type or None
        """
        if len(self.smoothed_force_values) < lookback or len(price_data) < lookback:
            return None
        
        # Get recent data
        recent_force = self.smoothed_force_values[-lookback:]
        recent_prices = price_data[-lookback:]
        
        # Find peaks and troughs
        force_max_idx = np.argmax(recent_force)
        force_min_idx = np.argmin(recent_force)
        price_max_idx = np.argmax(recent_prices)
        price_min_idx = np.argmin(recent_prices)
        
        # Bullish divergence: Lower price low, higher Force Index low
        if (price_min_idx > force_min_idx and 
            recent_prices[price_min_idx] < recent_prices[force_min_idx] and
            recent_force[price_min_idx] > recent_force[force_min_idx]):
            return 'BULLISH_DIVERGENCE'
        
        # Bearish divergence: Higher price high, lower Force Index high
        if (price_max_idx > force_max_idx and 
            recent_prices[price_max_idx] > recent_prices[force_max_idx] and
            recent_force[price_max_idx] < recent_force[force_max_idx]):
            return 'BEARISH_DIVERGENCE'
        
        return None
    
    def get_force_strength(self) -> Optional[str]:
        """
        Assess the strength of current force based on recent values.
        
        Returns:
            Force strength assessment
        """
        if len(self.smoothed_force_values) < self.config.period:
            return None
        
        recent_values = self.smoothed_force_values[-self.config.period:]
        current_force = recent_values[-1]
        avg_force = np.mean([abs(x) for x in recent_values])
        std_force = np.std(recent_values)
        
        # Normalize current force
        if std_force > 0:
            z_score = abs(current_force) / std_force
        else:
            z_score = 0
        
        # Assess strength
        if z_score > 2:
            if current_force > 0:
                return 'VERY_STRONG_BULLISH_FORCE'
            else:
                return 'VERY_STRONG_BEARISH_FORCE'
        elif z_score > 1:
            if current_force > 0:
                return 'STRONG_BULLISH_FORCE'
            else:
                return 'STRONG_BEARISH_FORCE'
        elif abs(current_force) > avg_force:
            if current_force > 0:
                return 'MODERATE_BULLISH_FORCE'
            else:
                return 'MODERATE_BEARISH_FORCE'
        else:
            return 'WEAK_FORCE'
    
    def get_volume_price_relationship(self) -> Dict[str, Union[float, str]]:
        """
        Analyze the relationship between volume and price changes.
        
        Returns:
            Dictionary with volume-price analysis
        """
        if len(self.price_changes) < 10:
            return {}
        
        # Get recent data
        recent_price_changes = self.price_changes[-10:]
        recent_volumes = self.volumes[-10:]
        recent_force = self.raw_force_values[-10:]
        
        # Separate positive and negative price moves
        pos_moves = [(pc, v, f) for pc, v, f in zip(recent_price_changes, recent_volumes, recent_force) if pc > 0]
        neg_moves = [(pc, v, f) for pc, v, f in zip(recent_price_changes, recent_volumes, recent_force) if pc < 0]
        
        # Calculate averages
        avg_pos_volume = np.mean([v for _, v, _ in pos_moves]) if pos_moves else 0
        avg_neg_volume = np.mean([v for _, v, _ in neg_moves]) if neg_moves else 0
        avg_pos_force = np.mean([f for _, _, f in pos_moves]) if pos_moves else 0
        avg_neg_force = np.mean([abs(f) for _, _, f in neg_moves]) if neg_moves else 0
        
        # Determine dominant force
        if avg_pos_force > avg_neg_force * 1.2:
            dominant_force = 'BULLISH_DOMINANCE'
        elif avg_neg_force > avg_pos_force * 1.2:
            dominant_force = 'BEARISH_DOMINANCE'
        else:
            dominant_force = 'BALANCED'
        
        # Volume pattern
        if avg_pos_volume > avg_neg_volume * 1.2:
            volume_pattern = 'HIGHER_VOLUME_ON_UPTICKS'
        elif avg_neg_volume > avg_pos_volume * 1.2:
            volume_pattern = 'HIGHER_VOLUME_ON_DOWNTICKS'
        else:
            volume_pattern = 'BALANCED_VOLUME'
        
        return {
            'dominant_force': dominant_force,
            'volume_pattern': volume_pattern,
            'avg_positive_force': avg_pos_force,
            'avg_negative_force': avg_neg_force,
            'avg_positive_volume': avg_pos_volume,
            'avg_negative_volume': avg_neg_volume,
            'force_strength': self.get_force_strength()
        }


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate realistic price data with trend and reversals
    base_price = 100.0
    prices = [base_price]
    volumes = []
    
    for i in range(1, periods):
        # Create trend phases
        if i < periods // 3:
            trend = 0.02  # Uptrend
        elif i < 2 * periods // 3:
            trend = -0.015  # Downtrend
        else:
            trend = 0.01  # Recovery
        
        noise = np.random.normal(0, 1.0)
        price_change = trend + noise
        new_price = max(prices[-1] + price_change, 50.0)  # Ensure reasonable prices
        prices.append(new_price)
        
        # Volume correlated with price change magnitude
        base_vol = 100000
        price_change_magnitude = abs(price_change)
        volume_multiplier = 1 + price_change_magnitude * 2
        volume = base_vol * volume_multiplier * np.random.uniform(0.5, 1.5)
        volumes.append(max(volume, 10000))
    
    # Add initial volume
    volumes.insert(0, 100000)
    
    data = []
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        # Create OHLC around close price
        high = close + np.random.uniform(0, 2)
        low = close - np.random.uniform(0, 2)
        
        data.append({
            'timestamp': date,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def main():
    """Test the Force Index indicator."""
    print("=== Force Index Indicator Test ===")
    
    # Create sample data
    data = create_sample_data(50)
    print(f"Created sample data with {len(data)} periods")
    
    # Test with default configuration (13-period smoothing)
    print("\n1. Testing with default configuration (13-period EMA)...")
    config = ForceIndexConfig()
    fi = ForceIndex(config)
    
    # Calculate batch
    results = fi.calculate_batch(data)
    
    print(f"Calculated Force Index for {len(results)} periods")
    print(f"Indicator initialized: {fi.initialized}")
    
    # Display recent results
    print("\nRecent Force Index values:")
    recent = results.tail(10)
    for idx, row in recent.iterrows():
        print(f"FI: {row['force_index']:.0f}, "
              f"Raw FI: {row['raw_force_index']:.0f}, "
              f"Price Change: {row['price_change']:.2f}, "
              f"Signal: {row['signal']}")
    
    # Test force strength assessment
    print("\n2. Testing force strength assessment...")
    force_strength = fi.get_force_strength()
    print(f"Current force strength: {force_strength}")
    
    # Test volume-price relationship analysis
    print("\n3. Testing volume-price relationship analysis...")
    vp_analysis = fi.get_volume_price_relationship()
    for key, value in vp_analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test divergence detection
    print("\n4. Testing divergence detection...")
    close_prices = data['close'].tolist()
    divergence = fi.detect_divergence(close_prices)
    print(f"Divergence detected: {divergence}")
    
    # Test raw Force Index (2-period for sensitivity)
    print("\n5. Testing raw Force Index (no smoothing)...")
    raw_config = ForceIndexConfig(
        period=2,
        use_smoothing=False
    )
    fi_raw = ForceIndex(raw_config)
    
    # Test single updates
    print("\n6. Testing single updates (raw FI)...")
    for i in range(5):
        row = data.iloc[i]
        result = fi_raw.update(
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        print(f"Update {i+1}: Force={result['force_index']:.0f}, "
              f"Price Change={result['price_change']:.2f}, "
              f"Volume={result['volume']:.0f}, "
              f"Signal={result['signal']}")
    
    # Test long-term Force Index (100-period)
    print("\n7. Testing long-term Force Index (100-period EMA)...")
    long_config = ForceIndexConfig(
        period=100,
        use_smoothing=True
    )
    fi_long = ForceIndex(long_config)
    
    # Calculate for subset of data
    long_results = fi_long.calculate_batch(data)
    print(f"100-period FI final value: {long_results.iloc[-1]['force_index']:.0f}")
    
    # Test error handling
    print("\n8. Testing error handling...")
    try:
        invalid_config = ForceIndexConfig(period=0)
        ForceIndex(invalid_config)
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Display signal distribution
    print("\n9. Signal distribution:")
    signal_counts = {}
    for signal in fi.get_signals():
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} times")
    
    # Force Index statistics
    print("\n10. Force Index statistics:")
    force_values = fi.get_force_values()
    if force_values:
        print(f"Max Force Index: {max(force_values):.0f}")
        print(f"Min Force Index: {min(force_values):.0f}")
        print(f"Average absolute Force: {np.mean([abs(x) for x in force_values]):.0f}")
        
        # Count positive vs negative periods
        positive_periods = sum(1 for x in force_values if x > 0)
        negative_periods = sum(1 for x in force_values if x < 0)
        print(f"Positive periods: {positive_periods}, Negative periods: {negative_periods}")
    
    print("\n=== Force Index Test Complete ===")


if __name__ == "__main__":
    main()