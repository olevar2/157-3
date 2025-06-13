#!/usr/bin/env python3
"""
Volume Rate of Change (VROC) Technical Indicator

The Volume Rate of Change indicator measures the rate of change in volume over a 
specified period. It helps identify when volume is increasing or decreasing relative 
to recent history, which can signal changing market participation and potential 
trend confirmations or reversals.

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
class VolumeRateOfChangeConfig:
    """Configuration for Volume Rate of Change parameters."""
    period: int = 14              # Period for ROC calculation
    smoothing_period: int = 21    # Optional smoothing period
    use_smoothing: bool = False   # Whether to apply smoothing
    signal_threshold: float = 20.0 # Signal threshold (percentage)
    min_periods: int = 14         # Minimum periods for calculation


class VolumeRateOfChange(BaseIndicator):
    """
    Volume Rate of Change (VROC) Technical Indicator Implementation
    
    The Volume Rate of Change measures the percentage change in volume over a 
    specified period. It helps identify periods of unusual volume activity that 
    may precede or confirm price movements.
    
    Formula:
    VROC = ((Current Volume - Volume N periods ago) / Volume N periods ago) * 100
    
    Optional smoothing:
    Smoothed VROC = SMA(VROC, smoothing_period)
    
    Interpretation:
    - Positive VROC: Volume increasing (higher participation)
    - Negative VROC: Volume decreasing (lower participation)
    - High positive VROC: Surge in volume (potential breakout/breakdown)
    - High negative VROC: Volume drying up (potential consolidation)
    - VROC divergences: Early warning of trend changes
    
    Usage:
    - Confirm price breakouts with volume surges
    - Identify potential reversals when volume diverges from price
    - Spot accumulation/distribution phases
    - Filter false signals in other indicators
    """
    
    def __init__(self, config: Optional[VolumeRateOfChangeConfig] = None):
        """
        Initialize Volume Rate of Change indicator.
        
        Args:
            config: Configuration object with indicator parameters
        """
        self.config = config or VolumeRateOfChangeConfig()
        self.name = "Volume Rate of Change"
        self.category = "volume"
        
        # Validation
        if self.config.period < 1:
            raise ValueError("Period must be positive")
        if self.config.smoothing_period < 1:
            raise ValueError("Smoothing period must be positive")
        if self.config.min_periods < 1:
            raise ValueError("Min periods must be positive")
        
        # Internal state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset internal calculation state."""
        self.volumes: List[float] = []
        self.vroc_values: List[float] = []
        self.smoothed_vroc_values: List[float] = []
        self.signals: List[str] = []
        self.initialized = False
    
    def _calculate_vroc(self, current_volume: float, past_volume: float) -> float:
        """
        Calculate Volume Rate of Change.
        
        Args:
            current_volume: Current volume
            past_volume: Volume N periods ago
            
        Returns:
            VROC percentage
        """
        if past_volume == 0:
            return 0.0
        
        return ((current_volume - past_volume) / past_volume) * 100
    
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
            high: High price (not used directly but kept for interface consistency)
            low: Low price (not used directly but kept for interface consistency)
            close: Close price (not used directly but kept for interface consistency)
            volume: Volume
            timestamp: Data timestamp
            
        Returns:
            Dictionary containing indicator values and signals
        """
        try:
            self.volumes.append(volume)
            
            # Calculate VROC if we have enough data
            if len(self.volumes) > self.config.period:
                past_volume = self.volumes[-(self.config.period + 1)]
                vroc_value = self._calculate_vroc(volume, past_volume)
            else:
                vroc_value = 0.0
            
            self.vroc_values.append(vroc_value)
            
            # Calculate smoothed VROC if enabled
            if self.config.use_smoothing:
                smoothed_vroc = self._calculate_sma(self.vroc_values, self.config.smoothing_period)
                self.smoothed_vroc_values.append(smoothed_vroc)
            else:
                smoothed_vroc = vroc_value
                self.smoothed_vroc_values.append(smoothed_vroc)
            
            # Generate signals
            signal = self._generate_signals(vroc_value, smoothed_vroc)
            self.signals.append(signal)
            
            # Mark as initialized after minimum periods
            if len(self.vroc_values) >= self.config.min_periods:
                self.initialized = True
            
            return {
                'vroc': vroc_value,
                'smoothed_vroc': smoothed_vroc,
                'volume': volume,
                'signal': signal,
                'timestamp': timestamp,
                'initialized': self.initialized
            }
            
        except Exception as e:
            warnings.warn(f"Error updating Volume Rate of Change: {str(e)}")
            return {
                'vroc': 0.0,
                'smoothed_vroc': 0.0,
                'volume': 0.0,
                'signal': 'NEUTRAL',
                'timestamp': timestamp,
                'initialized': False
            }
    
    def _generate_signals(self, vroc_value: float, smoothed_vroc: float) -> str:
        """
        Generate trading signals based on VROC.
        
        Args:
            vroc_value: Current VROC value
            smoothed_vroc: Current smoothed VROC value
            
        Returns:
            Signal string
        """
        if not self.initialized or len(self.vroc_values) < 2:
            return 'NEUTRAL'
        
        # Use smoothed value if available, otherwise raw value
        current_vroc = smoothed_vroc if self.config.use_smoothing else vroc_value
        
        # Volume surge signals
        if current_vroc > self.config.signal_threshold * 2:
            return 'VOLUME_SURGE'
        elif current_vroc > self.config.signal_threshold:
            return 'HIGH_VOLUME'
        elif current_vroc < -self.config.signal_threshold * 2:
            return 'VOLUME_COLLAPSE'
        elif current_vroc < -self.config.signal_threshold:
            return 'LOW_VOLUME'
        
        # Trend signals based on VROC direction
        if len(self.smoothed_vroc_values) >= 2:
            prev_vroc = self.smoothed_vroc_values[-2]
            
            # Zero line crossovers
            if prev_vroc <= 0 < current_vroc:
                return 'VOLUME_INCREASING'
            elif prev_vroc >= 0 > current_vroc:
                return 'VOLUME_DECREASING'
        
        return 'NEUTRAL'
    
    def calculate_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator for entire dataset.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator values
        """
        required_columns = ['volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Reset state
        self._reset_state()
        
        results = []
        for idx, row in data.iterrows():
            # Use default values for missing OHLC data
            high = row.get('high', 100.0)
            low = row.get('low', 100.0)
            close = row.get('close', 100.0)
            
            result = self.update(
                high=high,
                low=low,
                close=close,
                volume=row['volume'],
                timestamp=row.get('timestamp', idx)
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_signals(self) -> List[str]:
        """Get all generated signals."""
        return self.signals.copy()
    
    def get_vroc_values(self) -> List[float]:
        """Get all VROC values."""
        return self.vroc_values.copy()
    
    def get_smoothed_vroc_values(self) -> List[float]:
        """Get all smoothed VROC values."""
        return self.smoothed_vroc_values.copy()
    
    def get_volumes(self) -> List[float]:
        """Get all volume values."""
        return self.volumes.copy()
    
    def detect_volume_patterns(self, lookback: int = 20) -> Dict[str, Union[str, float]]:
        """
        Detect volume patterns based on VROC behavior.
        
        Args:
            lookback: Periods to analyze for patterns
            
        Returns:
            Dictionary with pattern analysis
        """
        if len(self.vroc_values) < lookback:
            return {}
        
        recent_vroc = self.vroc_values[-lookback:]
        recent_volumes = self.volumes[-lookback:]
        
        # Calculate statistics
        avg_vroc = np.mean(recent_vroc)
        std_vroc = np.std(recent_vroc)
        max_vroc = max(recent_vroc)
        min_vroc = min(recent_vroc)
        
        # Volume trend
        volume_trend_slope = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Pattern identification
        surge_count = sum(1 for x in recent_vroc if x > self.config.signal_threshold)
        collapse_count = sum(1 for x in recent_vroc if x < -self.config.signal_threshold)
        
        # Determine pattern
        if surge_count >= lookback * 0.3:
            pattern = 'PERSISTENT_HIGH_VOLUME'
        elif collapse_count >= lookback * 0.3:
            pattern = 'PERSISTENT_LOW_VOLUME'
        elif std_vroc > 50:
            pattern = 'VOLATILE_VOLUME'
        elif std_vroc < 10:
            pattern = 'STABLE_VOLUME'
        else:
            pattern = 'NORMAL_VOLUME'
        
        # Volume cycle detection
        zero_crossings = 0
        for i in range(1, len(recent_vroc)):
            if (recent_vroc[i-1] >= 0) != (recent_vroc[i] >= 0):
                zero_crossings += 1
        
        if zero_crossings >= 6:
            cycle_pattern = 'CYCLICAL'
        elif zero_crossings <= 2:
            cycle_pattern = 'TRENDING'
        else:
            cycle_pattern = 'MIXED'
        
        return {
            'pattern': pattern,
            'cycle_pattern': cycle_pattern,
            'avg_vroc': avg_vroc,
            'vroc_volatility': std_vroc,
            'max_vroc': max_vroc,
            'min_vroc': min_vroc,
            'volume_trend_slope': volume_trend_slope,
            'surge_periods': surge_count,
            'collapse_periods': collapse_count
        }
    
    def get_volume_strength(self) -> Optional[str]:
        """
        Assess current volume strength relative to recent history.
        
        Returns:
            Volume strength assessment
        """
        if len(self.vroc_values) < self.config.period:
            return None
        
        current_vroc = self.vroc_values[-1]
        recent_vroc = self.vroc_values[-self.config.period:]
        
        # Calculate percentile of current VROC
        percentile = (sum(1 for x in recent_vroc if x <= current_vroc) / len(recent_vroc)) * 100
        
        if percentile >= 95:
            return 'EXTREMELY_HIGH_VOLUME'
        elif percentile >= 80:
            return 'HIGH_VOLUME'
        elif percentile >= 60:
            return 'ABOVE_AVERAGE_VOLUME'
        elif percentile >= 40:
            return 'AVERAGE_VOLUME'
        elif percentile >= 20:
            return 'BELOW_AVERAGE_VOLUME'
        elif percentile >= 5:
            return 'LOW_VOLUME'
        else:
            return 'EXTREMELY_LOW_VOLUME'
    
    def detect_volume_divergence(self, price_data: List[float], lookback: int = 14) -> Optional[str]:
        """
        Detect divergences between volume changes and price movements.
        
        Args:
            price_data: List of price values
            lookback: Periods to look back for divergence
            
        Returns:
            Divergence type or None
        """
        if len(self.vroc_values) < lookback or len(price_data) < lookback:
            return None
        
        # Get recent data
        recent_vroc = self.vroc_values[-lookback:]
        recent_prices = price_data[-lookback:]
        
        # Calculate price rate of change
        price_roc = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] != 0:
                roc = ((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) * 100
                price_roc.append(roc)
            else:
                price_roc.append(0.0)
        
        if len(price_roc) < lookback - 1:
            return None
        
        # Calculate correlation between price ROC and volume ROC
        # Align the arrays (price_roc is one element shorter)
        aligned_vroc = recent_vroc[1:]
        
        if len(aligned_vroc) != len(price_roc):
            return None
        
        correlation = np.corrcoef(price_roc, aligned_vroc)[0, 1]
        
        # Find recent peaks
        vroc_peak_idx = np.argmax(aligned_vroc[-lookback//2:])
        price_peak_idx = np.argmax(price_roc[-lookback//2:])
        
        # Detect divergences
        if correlation < -0.3:  # Negative correlation
            if price_roc[-1] > 0 and aligned_vroc[-1] < 0:
                return 'BEARISH_VOLUME_DIVERGENCE'  # Price up, volume down
            elif price_roc[-1] < 0 and aligned_vroc[-1] > 0:
                return 'BULLISH_VOLUME_DIVERGENCE'  # Price down, volume up
        
        return None
    
    def get_volume_momentum(self) -> Dict[str, float]:
        """
        Calculate volume momentum metrics.
        
        Returns:
            Dictionary with momentum metrics
        """
        if len(self.vroc_values) < 10:
            return {}
        
        # Short-term momentum (3-5 periods)
        short_term = np.mean(self.vroc_values[-3:])
        
        # Medium-term momentum (5-10 periods)
        medium_term = np.mean(self.vroc_values[-10:-5]) if len(self.vroc_values) >= 10 else 0
        
        # Calculate acceleration (change in momentum)
        if len(self.vroc_values) >= 6:
            recent_momentum = np.mean(self.vroc_values[-3:])
            past_momentum = np.mean(self.vroc_values[-6:-3])
            acceleration = recent_momentum - past_momentum
        else:
            acceleration = 0
        
        return {
            'short_term_momentum': short_term,
            'medium_term_momentum': medium_term,
            'momentum_acceleration': acceleration,
            'current_vroc': self.vroc_values[-1] if self.vroc_values else 0
        }


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate realistic volume data with patterns
    base_volume = 100000
    volumes = []
    
    for i in range(periods):
        # Create volume cycles and trends
        cycle = 0.3 * np.sin(i / 10)  # Regular cycle
        trend = 0.001 * i  # Slight upward trend
        
        # Add volume spikes and drops
        if i % 15 == 0:  # Periodic volume spike
            spike = 1.5
        elif i % 23 == 0:  # Periodic volume drop
            spike = -0.4
        else:
            spike = 0
        
        noise = np.random.normal(0, 0.2)
        volume_multiplier = 1 + cycle + trend + spike + noise
        volume = base_volume * max(volume_multiplier, 0.1)  # Ensure positive volume
        volumes.append(volume)
    
    # Generate price data (not used in VROC calculation but needed for interface)
    prices = [100.0]
    for i in range(1, periods):
        change = np.random.normal(0, 1)
        prices.append(max(prices[-1] + change, 50.0))
    
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
    """Test the Volume Rate of Change indicator."""
    print("=== Volume Rate of Change Indicator Test ===")
    
    # Create sample data
    data = create_sample_data(60)
    print(f"Created sample data with {len(data)} periods")
    
    # Test with default configuration
    print("\n1. Testing with default configuration (14-period VROC)...")
    config = VolumeRateOfChangeConfig()
    vroc = VolumeRateOfChange(config)
    
    # Calculate batch
    results = vroc.calculate_batch(data)
    
    print(f"Calculated VROC for {len(results)} periods")
    print(f"Indicator initialized: {vroc.initialized}")
    
    # Display recent results
    print("\nRecent VROC values:")
    recent = results.tail(10)
    for idx, row in recent.iterrows():
        print(f"VROC: {row['vroc']:.1f}%, "
              f"Volume: {row['volume']:.0f}, "
              f"Signal: {row['signal']}")
    
    # Test volume strength assessment
    print("\n2. Testing volume strength assessment...")
    volume_strength = vroc.get_volume_strength()
    print(f"Current volume strength: {volume_strength}")
    
    # Test volume patterns
    print("\n3. Testing volume pattern detection...")
    patterns = vroc.detect_volume_patterns()
    for key, value in patterns.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test volume momentum
    print("\n4. Testing volume momentum analysis...")
    momentum = vroc.get_volume_momentum()
    for key, value in momentum.items():
        print(f"{key}: {value:.2f}")
    
    # Test divergence detection
    print("\n5. Testing volume divergence detection...")
    close_prices = data['close'].tolist()
    divergence = vroc.detect_volume_divergence(close_prices)
    print(f"Volume divergence detected: {divergence}")
    
    # Test with smoothing enabled
    print("\n6. Testing with smoothing enabled...")
    smooth_config = VolumeRateOfChangeConfig(
        period=10,
        use_smoothing=True,
        smoothing_period=5
    )
    vroc_smooth = VolumeRateOfChange(smooth_config)
    
    # Test single updates
    print("\n7. Testing single updates (smoothed)...")
    for i in range(min(8, len(data))):
        row = data.iloc[i]
        result = vroc_smooth.update(
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        print(f"Update {i+1}: VROC={result['vroc']:.1f}%, "
              f"Smoothed={result['smoothed_vroc']:.1f}%, "
              f"Signal={result['signal']}")
    
    # Test error handling
    print("\n8. Testing error handling...")
    try:
        invalid_config = VolumeRateOfChangeConfig(period=0)
        VolumeRateOfChange(invalid_config)
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Display signal distribution
    print("\n9. Signal distribution:")
    signal_counts = {}
    for signal in vroc.get_signals():
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} times")
    
    # VROC statistics
    print("\n10. VROC statistics:")
    vroc_values = vroc.get_vroc_values()
    if vroc_values:
        print(f"Max VROC: {max(vroc_values):.1f}%")
        print(f"Min VROC: {min(vroc_values):.1f}%")
        print(f"Average VROC: {np.mean(vroc_values):.1f}%")
        print(f"VROC Std Dev: {np.std(vroc_values):.1f}%")
        
        # Count extreme values
        high_volume_periods = sum(1 for x in vroc_values if x > 20)
        low_volume_periods = sum(1 for x in vroc_values if x < -20)
        print(f"High volume periods (>20%): {high_volume_periods}")
        print(f"Low volume periods (<-20%): {low_volume_periods}")
    
    print("\n=== Volume Rate of Change Test Complete ===")


if __name__ == "__main__":
    main()