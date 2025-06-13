#!/usr/bin/env python3
"""
Price Volume Trend (PVT) Technical Indicator

The Price Volume Trend indicator was developed as an alternative to On-Balance Volume (OBV).
Unlike OBV which adds or subtracts the entire volume based on price direction, PVT adds 
or subtracts a percentage of the volume based on the percentage change in price.

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
        try:
        from base_indicator import BaseIndicator
    except ImportError:
        # Fallback to creating a minimal base class
        class BaseIndicator:
            def __init__(self):
                pass


@dataclass
class PriceVolumeTrendConfig:
    """Configuration for Price Volume Trend parameters."""
    min_periods: int = 10         # Minimum periods for calculation
    smoothing_period: int = 21    # Optional smoothing period
    use_smoothing: bool = False   # Whether to apply smoothing
    signal_threshold: float = 0.0 # Signal threshold for trend changes


class PriceVolumeTrend(BaseIndicator):
    """
    Price Volume Trend (PVT) Technical Indicator Implementation
    
    The Price Volume Trend indicator is a momentum-based indicator that uses volume
    and price to confirm trends and identify potential reversal points. It's similar
    to OBV but more sensitive to the magnitude of price changes.
    
    Formula:
    PVT = Previous PVT + (Volume * ((Close - Previous Close) / Previous Close))
    
    Where:
    - If current close > previous close: positive volume contribution
    - If current close < previous close: negative volume contribution  
    - If current close = previous close: no volume contribution
    
    Optional smoothing:
    Smoothed PVT = SMA(PVT, smoothing_period)
    
    Interpretation:
    - Rising PVT: Buying pressure, uptrend confirmation
    - Falling PVT: Selling pressure, downtrend confirmation
    - PVT divergence from price: Potential reversal signal
    - PVT breakouts: Trend acceleration signals
    """
    
    def __init__(self, config: Optional[PriceVolumeTrendConfig] = None):
        """
        Initialize Price Volume Trend indicator.
        
        Args:
            config: Configuration object with indicator parameters
        """
        self.config = config or PriceVolumeTrendConfig()
        self.name = "Price Volume Trend"
        self.category = "volume"
        
        # Validation
        if self.config.min_periods < 1:
            raise ValueError("Min periods must be positive")
        if self.config.smoothing_period < 1:
            raise ValueError("Smoothing period must be positive")
        
        # Internal state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset internal calculation state."""
        self.pvt_values: List[float] = []
        self.smoothed_pvt_values: List[float] = []
        self.volume_contributions: List[float] = []
        self.price_changes: List[float] = []
        self.close_prices: List[float] = []
        self.signals: List[str] = []
        self.initialized = False
    
    def _calculate_price_change_ratio(self, current_close: float, prev_close: float) -> float:
        """
        Calculate price change ratio.
        
        Args:
            current_close: Current close price
            prev_close: Previous close price
            
        Returns:
            Price change ratio
        """
        if prev_close == 0:
            return 0.0
        
        return (current_close - prev_close) / prev_close
    
    def _calculate_volume_contribution(self, volume: float, price_change_ratio: float) -> float:
        """
        Calculate volume contribution to PVT.
        
        Args:
            volume: Volume for the period
            price_change_ratio: Price change ratio
            
        Returns:
            Volume contribution
        """
        return volume * price_change_ratio
    
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
            return np.mean(values) if values else 0.0
        
        return np.mean(values[-period:])
    
    def update(self, high: float, low: float, close: float, volume: float, 
               timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Update indicator with new market data.
        
        Args:
            high: High price (not used in PVT calculation but kept for interface consistency)
            low: Low price (not used in PVT calculation but kept for interface consistency)
            close: Close price
            volume: Volume
            timestamp: Data timestamp
            
        Returns:
            Dictionary containing indicator values and signals
        """
        try:
            self.close_prices.append(close)
            
            # Calculate price change ratio
            if len(self.close_prices) >= 2:
                prev_close = self.close_prices[-2]
                price_change_ratio = self._calculate_price_change_ratio(close, prev_close)
            else:
                price_change_ratio = 0.0
            
            self.price_changes.append(price_change_ratio)
            
            # Calculate volume contribution
            volume_contribution = self._calculate_volume_contribution(volume, price_change_ratio)
            self.volume_contributions.append(volume_contribution)
            
            # Calculate cumulative PVT
            if self.pvt_values:
                pvt_value = self.pvt_values[-1] + volume_contribution
            else:
                pvt_value = volume_contribution
            
            self.pvt_values.append(pvt_value)
            
            # Calculate smoothed PVT if enabled
            if self.config.use_smoothing:
                smoothed_pvt = self._calculate_sma(self.pvt_values, self.config.smoothing_period)
                self.smoothed_pvt_values.append(smoothed_pvt)
            else:
                smoothed_pvt = pvt_value
                self.smoothed_pvt_values.append(smoothed_pvt)
            
            # Generate signals
            signal = self._generate_signals(pvt_value, smoothed_pvt)
            self.signals.append(signal)
            
            # Mark as initialized after minimum periods
            if len(self.pvt_values) >= self.config.min_periods:
                self.initialized = True
            
            return {
                'pvt': pvt_value,
                'smoothed_pvt': smoothed_pvt,
                'volume_contribution': volume_contribution,
                'price_change_ratio': price_change_ratio,
                'signal': signal,
                'timestamp': timestamp,
                'initialized': self.initialized
            }
            
        except Exception as e:
            warnings.warn(f"Error updating Price Volume Trend: {str(e)}")
            return {
                'pvt': 0.0,
                'smoothed_pvt': 0.0,
                'volume_contribution': 0.0,
                'price_change_ratio': 0.0,
                'signal': 'NEUTRAL',
                'timestamp': timestamp,
                'initialized': False
            }
    
    def _generate_signals(self, pvt_value: float, smoothed_pvt: float) -> str:
        """
        Generate trading signals based on PVT.
        
        Args:
            pvt_value: Current PVT value
            smoothed_pvt: Current smoothed PVT value
            
        Returns:
            Signal string
        """
        if not self.initialized or len(self.pvt_values) < 2:
            return 'NEUTRAL'
        
        prev_pvt = self.pvt_values[-2]
        current_pvt = pvt_value
        
        # Trend signals based on PVT direction
        if current_pvt > prev_pvt:
            if current_pvt > self.config.signal_threshold:
                return 'BULLISH'
            else:
                return 'BUY_PRESSURE'
        elif current_pvt < prev_pvt:
            if current_pvt < -self.config.signal_threshold:
                return 'BEARISH'
            else:
                return 'SELL_PRESSURE'
        
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
    
    def get_pvt_values(self) -> List[float]:
        """Get all PVT values."""
        return self.pvt_values.copy()
    
    def get_smoothed_pvt_values(self) -> List[float]:
        """Get all smoothed PVT values."""
        return self.smoothed_pvt_values.copy()
    
    def detect_divergence(self, price_data: List[float], lookback: int = 14) -> Optional[str]:
        """
        Detect bullish/bearish divergences between price and PVT.
        
        Args:
            price_data: List of price values (typically close prices)
            lookback: Periods to look back for divergence
            
        Returns:
            Divergence type or None
        """
        if len(self.pvt_values) < lookback or len(price_data) < lookback:
            return None
        
        # Get recent data
        recent_pvt = self.pvt_values[-lookback:]
        recent_prices = price_data[-lookback:]
        
        # Find peaks and troughs
        pvt_max_idx = np.argmax(recent_pvt)
        pvt_min_idx = np.argmin(recent_pvt)
        price_max_idx = np.argmax(recent_prices)
        price_min_idx = np.argmin(recent_prices)
        
        # Bullish divergence: Lower price low, higher PVT low
        if (price_min_idx > pvt_min_idx and 
            recent_prices[price_min_idx] < recent_prices[pvt_min_idx] and
            recent_pvt[price_min_idx] > recent_pvt[pvt_min_idx]):
            return 'BULLISH_DIVERGENCE'
        
        # Bearish divergence: Higher price high, lower PVT high
        if (price_max_idx > pvt_max_idx and 
            recent_prices[price_max_idx] > recent_prices[pvt_max_idx] and
            recent_pvt[price_max_idx] < recent_pvt[pvt_max_idx]):
            return 'BEARISH_DIVERGENCE'
        
        return None
    
    def get_trend_strength(self) -> Optional[str]:
        """
        Assess trend strength based on PVT momentum.
        
        Returns:
            Trend strength assessment
        """
        if len(self.pvt_values) < self.config.min_periods:
            return None
        
        # Calculate PVT rate of change over different periods
        short_roc = self._calculate_roc(5)
        medium_roc = self._calculate_roc(10)
        long_roc = self._calculate_roc(self.config.min_periods)
        
        # Assess trend strength based on consistency across timeframes
        if short_roc > 0 and medium_roc > 0 and long_roc > 0:
            if short_roc > medium_roc > long_roc:
                return 'STRONG_ACCELERATING_BULLISH'
            else:
                return 'CONSISTENT_BULLISH'
        elif short_roc < 0 and medium_roc < 0 and long_roc < 0:
            if short_roc < medium_roc < long_roc:
                return 'STRONG_ACCELERATING_BEARISH'
            else:
                return 'CONSISTENT_BEARISH'
        elif abs(short_roc) < 0.001 and abs(medium_roc) < 0.001:
            return 'SIDEWAYS'
        else:
            return 'MIXED_SIGNALS'
    
    def _calculate_roc(self, periods: int) -> float:
        """
        Calculate Rate of Change for PVT over specified periods.
        
        Args:
            periods: Number of periods for ROC calculation
            
        Returns:
            Rate of change
        """
        if len(self.pvt_values) < periods + 1:
            return 0.0
        
        current_pvt = self.pvt_values[-1]
        past_pvt = self.pvt_values[-(periods + 1)]
        
        if past_pvt == 0:
            return 0.0
        
        return (current_pvt - past_pvt) / abs(past_pvt)
    
    def get_momentum_analysis(self) -> Dict[str, float]:
        """
        Get detailed momentum analysis based on PVT.
        
        Returns:
            Dictionary with momentum metrics
        """
        if len(self.pvt_values) < self.config.min_periods:
            return {}
        
        # Calculate various momentum metrics
        current_pvt = self.pvt_values[-1]
        pvt_mean = np.mean(self.pvt_values[-self.config.min_periods:])
        pvt_std = np.std(self.pvt_values[-self.config.min_periods:])
        
        # Z-score (how many standard deviations from mean)
        z_score = (current_pvt - pvt_mean) / pvt_std if pvt_std > 0 else 0
        
        # Recent momentum (5-period average change)
        recent_changes = []
        for i in range(1, min(6, len(self.pvt_values))):
            if len(self.pvt_values) > i:
                change = self.pvt_values[-i] - self.pvt_values[-(i+1)]
                recent_changes.append(change)
        
        avg_recent_change = np.mean(recent_changes) if recent_changes else 0
        
        return {
            'current_pvt': current_pvt,
            'pvt_mean': pvt_mean,
            'pvt_std': pvt_std,
            'z_score': z_score,
            'recent_momentum': avg_recent_change,
            'trend_strength': self.get_trend_strength()
        }


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate realistic price data with trends
    base_price = 100.0
    prices = [base_price]
    volumes = []
    
    for i in range(1, periods):
        # Add trend and noise
        trend = 0.02 * np.sin(i / 15)  # Cyclical trend
        noise = np.random.normal(0, 0.5)
        change = trend + noise
        new_price = max(prices[-1] + change, 1.0)  # Ensure positive prices
        prices.append(new_price)
    
    # Generate volumes with some correlation to price changes
    for i in range(periods):
        base_volume = 100000
        if i > 0:
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            volume_multiplier = 1 + price_change * 2  # Higher volume on bigger moves
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
        else:
            volume = base_volume
        volumes.append(volume)
    
    data = []
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        # Create OHLC around close price
        high = close + np.random.uniform(0, 1)
        low = close - np.random.uniform(0, 1)
        
        data.append({
            'timestamp': date,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def main():
    """Test the Price Volume Trend indicator."""
    print("=== Price Volume Trend Indicator Test ===")
    
    # Create sample data
    data = create_sample_data(50)
    print(f"Created sample data with {len(data)} periods")
    
    # Test with default configuration
    print("\n1. Testing with default configuration...")
    config = PriceVolumeTrendConfig()
    pvt = PriceVolumeTrend(config)
    
    # Calculate batch
    results = pvt.calculate_batch(data)
    
    print(f"Calculated PVT for {len(results)} periods")
    print(f"Indicator initialized: {pvt.initialized}")
    
    # Display recent results
    print("\nRecent PVT values:")
    recent = results.tail(10)
    for idx, row in recent.iterrows():
        print(f"PVT: {row['pvt']:.2f}, "
              f"Volume Contrib: {row['volume_contribution']:.2f}, "
              f"Signal: {row['signal']}")
    
    # Test trend strength
    print("\n2. Testing trend strength assessment...")
    trend_strength = pvt.get_trend_strength()
    print(f"Current trend strength: {trend_strength}")
    
    # Test momentum analysis
    print("\n3. Testing momentum analysis...")
    momentum = pvt.get_momentum_analysis()
    for key, value in momentum.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test divergence detection
    print("\n4. Testing divergence detection...")
    close_prices = data['close'].tolist()
    divergence = pvt.detect_divergence(close_prices)
    print(f"Divergence detected: {divergence}")
    
    # Test with smoothing enabled
    print("\n5. Testing with smoothing enabled...")
    smooth_config = PriceVolumeTrendConfig(
        use_smoothing=True,
        smoothing_period=10
    )
    pvt_smooth = PriceVolumeTrend(smooth_config)
    
    # Test single updates
    print("\n6. Testing single updates...")
    for i in range(5):
        row = data.iloc[i]
        result = pvt_smooth.update(
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        print(f"Update {i+1}: PVT={result['pvt']:.2f}, "
              f"Smoothed={result['smoothed_pvt']:.2f}, "
              f"Signal={result['signal']}")
    
    # Test error handling
    print("\n7. Testing error handling...")
    try:
        invalid_config = PriceVolumeTrendConfig(min_periods=-1)
        PriceVolumeTrend(invalid_config)
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Display signal distribution
    print("\n8. Signal distribution:")
    signal_counts = {}
    for signal in pvt.get_signals():
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} times")
    
    print("\n=== Price Volume Trend Test Complete ===")


if __name__ == "__main__":
    main()