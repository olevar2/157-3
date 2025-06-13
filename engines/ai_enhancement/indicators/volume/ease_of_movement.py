#!/usr/bin/env python3
"""
Ease of Movement (EOM) Technical Indicator

The Ease of Movement indicator was developed by Richard Arms Jr. to combine price and volume 
information to assess how easily a price can move. It attempts to quantify the relationship 
between price change and volume, and identify periods when price moves with relatively 
little volume (easy movement) versus periods requiring heavy volume.

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
class EaseOfMovementConfig:
    """Configuration for Ease of Movement parameters."""
    period: int = 14              # Smoothing period for EOM
    scale_factor: int = 100000000 # Scale factor for readability (typically 100M)
    signal_threshold: float = 0.0 # Signal threshold
    min_periods: int = 14         # Minimum periods for calculation


class EaseOfMovement(BaseIndicator):
    """
    Ease of Movement (EOM) Technical Indicator Implementation
    
    The Ease of Movement indicator shows the relationship between price change and volume,
    indicating how much volume is required to move prices. Positive values suggest that
    prices are advancing with relative ease (light volume), while negative values suggest
    prices are declining with relative ease.
    
    Formula:
    1. Distance Moved = ((High + Low) / 2) - ((Previous High + Previous Low) / 2)
    2. Box Height = (Volume / Scale Factor) / (High - Low)
    3. 1-Period EMV = Distance Moved / Box Height (when Box Height != 0)
    4. N-Period EOM = SMA(1-Period EMV, period)
    
    Interpretation:
    - Positive EOM: Price moving up with ease (light volume)
    - Negative EOM: Price moving down with ease (light volume)
    - Values near zero: Price movement requires heavy volume
    - Divergences with price: Potential reversal signals
    """
    
    def __init__(self, config: Optional[EaseOfMovementConfig] = None):
        """
        Initialize Ease of Movement indicator.
        
        Args:
            config: Configuration object with indicator parameters
        """
        self.config = config or EaseOfMovementConfig()
        self.name = "Ease of Movement"
        self.category = "volume"
        
        # Validation
        if self.config.period < 1:
            raise ValueError("Period must be a positive integer")
        if self.config.scale_factor <= 0:
            raise ValueError("Scale factor must be positive")
        
        # Internal state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset internal calculation state."""
        self.raw_emv_values: List[float] = []
        self.eom_values: List[float] = []
        self.distance_moved_values: List[float] = []
        self.box_height_values: List[float] = []
        self.signals: List[str] = []
        self.prev_mid_point: Optional[float] = None
        self.initialized = False
    
    def _calculate_distance_moved(self, high: float, low: float) -> float:
        """
        Calculate distance moved (change in midpoint).
        
        Args:
            high: Current high price
            low: Current low price
            
        Returns:
            Distance moved value
        """
        current_mid_point = (high + low) / 2
        
        if self.prev_mid_point is None:
            distance_moved = 0.0
        else:
            distance_moved = current_mid_point - self.prev_mid_point
        
        self.prev_mid_point = current_mid_point
        return distance_moved
    
    def _calculate_box_height(self, high: float, low: float, volume: float) -> float:
        """
        Calculate box height (scaled volume divided by price range).
        
        Args:
            high: High price
            low: Low price
            volume: Volume
            
        Returns:
            Box height value
        """
        price_range = high - low
        
        if price_range == 0:
            return float('inf')  # Infinite box height when no price range
        
        scaled_volume = volume / self.config.scale_factor
        box_height = scaled_volume / price_range
        
        return box_height
    
    def _calculate_raw_emv(self, distance_moved: float, box_height: float) -> float:
        """
        Calculate raw EMV value for one period.
        
        Args:
            distance_moved: Distance moved value
            box_height: Box height value
            
        Returns:
            Raw EMV value
        """
        if box_height == 0 or box_height == float('inf'):
            return 0.0
        
        return distance_moved / box_height
    
    def _calculate_sma(self, values: List[float], period: int) -> float:
        """
        Calculate Simple Moving Average.
        
        Args:
            values: List of values
            period: Period for SMA
            
        Returns:
            SMA value
        """
        if len(values) < period:
            return np.mean(values)
        
        return np.mean(values[-period:])
    
    def update(self, high: float, low: float, close: float, volume: float, 
               timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Update indicator with new market data.
        
        Args:
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            timestamp: Data timestamp
            
        Returns:
            Dictionary containing indicator values and signals
        """
        try:
            # Calculate distance moved
            distance_moved = self._calculate_distance_moved(high, low)
            self.distance_moved_values.append(distance_moved)
            
            # Calculate box height
            box_height = self._calculate_box_height(high, low, volume)
            self.box_height_values.append(box_height)
            
            # Calculate raw EMV
            raw_emv = self._calculate_raw_emv(distance_moved, box_height)
            self.raw_emv_values.append(raw_emv)
            
            # Calculate smoothed EOM
            eom_value = self._calculate_sma(self.raw_emv_values, self.config.period)
            self.eom_values.append(eom_value)
            
            # Generate signals
            signal = self._generate_signals(eom_value)
            self.signals.append(signal)
            
            # Mark as initialized after minimum periods
            if len(self.eom_values) >= self.config.min_periods:
                self.initialized = True
            
            return {
                'ease_of_movement': eom_value,
                'raw_emv': raw_emv,
                'distance_moved': distance_moved,
                'box_height': box_height,
                'signal': signal,
                'timestamp': timestamp,
                'initialized': self.initialized
            }
            
        except Exception as e:
            warnings.warn(f"Error updating Ease of Movement: {str(e)}")
            return {
                'ease_of_movement': 0.0,
                'raw_emv': 0.0,
                'distance_moved': 0.0,
                'box_height': 0.0,
                'signal': 'NEUTRAL',
                'timestamp': timestamp,
                'initialized': False
            }
    
    def _generate_signals(self, eom_value: float) -> str:
        """
        Generate trading signals based on Ease of Movement.
        
        Args:
            eom_value: Current EOM value
            
        Returns:
            Signal string
        """
        if not self.initialized or len(self.eom_values) < 2:
            return 'NEUTRAL'
        
        prev_eom = self.eom_values[-2]
        
        # Zero line crossovers
        if prev_eom <= 0 < eom_value:
            return 'BUY'  # Bullish crossover
        elif prev_eom >= 0 > eom_value:
            return 'SELL'  # Bearish crossover
        
        # Momentum signals
        if eom_value > self.config.signal_threshold:
            return 'BULLISH'
        elif eom_value < -self.config.signal_threshold:
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
        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Reset state
        self._reset_state()
        
        results = []
        for idx, row in data.iterrows():
            result = self.update(
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=row.get('timestamp', idx)
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_signals(self) -> List[str]:
        """Get all generated signals."""
        return self.signals.copy()
    
    def get_eom_values(self) -> List[float]:
        """Get all EOM values."""
        return self.eom_values.copy()
    
    def get_raw_emv_values(self) -> List[float]:
        """Get all raw EMV values."""
        return self.raw_emv_values.copy()
    
    def detect_divergence(self, price_data: List[float], lookback: int = 14) -> Optional[str]:
        """
        Detect bullish/bearish divergences between price and EOM.
        
        Args:
            price_data: List of price values (typically close prices)
            lookback: Periods to look back for divergence
            
        Returns:
            Divergence type or None
        """
        if len(self.eom_values) < lookback or len(price_data) < lookback:
            return None
        
        # Get recent data
        recent_eom = self.eom_values[-lookback:]
        recent_prices = price_data[-lookback:]
        
        # Find peaks and troughs
        eom_max_idx = np.argmax(recent_eom)
        eom_min_idx = np.argmin(recent_eom)
        price_max_idx = np.argmax(recent_prices)
        price_min_idx = np.argmin(recent_prices)
        
        # Bullish divergence: Lower price low, higher EOM low
        if (price_min_idx > eom_min_idx and 
            recent_prices[price_min_idx] < recent_prices[eom_min_idx] and
            recent_eom[price_min_idx] > recent_eom[eom_min_idx]):
            return 'BULLISH_DIVERGENCE'
        
        # Bearish divergence: Higher price high, lower EOM high
        if (price_max_idx > eom_max_idx and 
            recent_prices[price_max_idx] > recent_prices[eom_max_idx] and
            recent_eom[price_max_idx] < recent_eom[eom_max_idx]):
            return 'BEARISH_DIVERGENCE'
        
        return None
    
    def get_trend_strength(self) -> Optional[str]:
        """
        Assess trend strength based on EOM values.
        
        Returns:
            Trend strength assessment
        """
        if len(self.eom_values) < self.config.period:
            return None
        
        recent_values = self.eom_values[-self.config.period:]
        avg_eom = np.mean(recent_values)
        std_eom = np.std(recent_values)
        
        if abs(avg_eom) > 2 * std_eom:
            if avg_eom > 0:
                return 'STRONG_UPTREND'
            else:
                return 'STRONG_DOWNTREND'
        elif abs(avg_eom) > std_eom:
            if avg_eom > 0:
                return 'MODERATE_UPTREND'
            else:
                return 'MODERATE_DOWNTREND'
        else:
            return 'SIDEWAYS'


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate realistic price data with varying volume
    base_price = 100.0
    price_changes = np.random.normal(0, 2, periods)
    prices = base_price + np.cumsum(price_changes)
    
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Create OHLC with some randomness
        high = price + np.random.uniform(0.5, 3.0)
        low = price - np.random.uniform(0.5, 3.0)
        close = price + np.random.uniform(-1.5, 1.5)
        
        # Volume inversely correlated with price changes for more realistic EOM signals
        volume_base = 50000
        price_change = abs(close - price) if i > 0 else 1
        volume = volume_base + np.random.uniform(10000, 100000) / max(price_change, 0.1)
        
        data.append({
            'timestamp': date,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def main():
    """Test the Ease of Movement indicator."""
    print("=== Ease of Movement Indicator Test ===")
    
    # Create sample data
    data = create_sample_data(50)
    print(f"Created sample data with {len(data)} periods")
    
    # Test with default configuration
    print("\n1. Testing with default configuration...")
    config = EaseOfMovementConfig()
    eom = EaseOfMovement(config)
    
    # Calculate batch
    results = eom.calculate_batch(data)
    
    print(f"Calculated EOM for {len(results)} periods")
    print(f"Indicator initialized: {eom.initialized}")
    
    # Display recent results
    print("\nRecent EOM values:")
    recent = results.tail(10)
    for idx, row in recent.iterrows():
        print(f"EOM: {row['ease_of_movement']:.4f}, "
              f"Raw EMV: {row['raw_emv']:.4f}, "
              f"Signal: {row['signal']}")
    
    # Test trend strength
    print("\n2. Testing trend strength assessment...")
    trend_strength = eom.get_trend_strength()
    print(f"Current trend strength: {trend_strength}")
    
    # Test divergence detection
    print("\n3. Testing divergence detection...")
    close_prices = data['close'].tolist()
    divergence = eom.detect_divergence(close_prices)
    print(f"Divergence detected: {divergence}")
    
    # Test different parameters
    print("\n4. Testing with custom parameters...")
    custom_config = EaseOfMovementConfig(
        period=21,
        scale_factor=1000000,  # Different scale factor
        signal_threshold=0.1
    )
    eom_custom = EaseOfMovement(custom_config)
    
    # Test single updates
    print("\n5. Testing single updates...")
    for i in range(5):
        row = data.iloc[i]
        result = eom_custom.update(
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        print(f"Update {i+1}: EOM={result['ease_of_movement']:.4f}, "
              f"Box Height={result['box_height']:.4f}, "
              f"Signal={result['signal']}")
    
    # Test error handling
    print("\n6. Testing error handling...")
    try:
        invalid_config = EaseOfMovementConfig(period=-5)
        EaseOfMovement(invalid_config)
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Test zero price range scenario
    print("\n7. Testing zero price range scenario...")
    result = eom.update(high=100, low=100, close=100, volume=10000)
    print(f"Zero range result: EOM={result['ease_of_movement']:.4f}")
    
    print("\n=== Ease of Movement Test Complete ===")


if __name__ == "__main__":
    main()