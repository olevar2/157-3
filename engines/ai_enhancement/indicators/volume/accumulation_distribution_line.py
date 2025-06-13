#!/usr/bin/env python3
"""
Accumulation/Distribution Line (A/D Line) Technical Indicator

The Accumulation/Distribution Line was developed by Marc Chaikin to measure the cumulative 
flow of money into and out of a security. It uses price and volume to determine whether 
a stock is being accumulated (bought) or distributed (sold). The A/D Line helps confirm 
price trends and can provide early warnings of potential reversals through divergences.

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
class AccumulationDistributionConfig:
    """Configuration for Accumulation/Distribution Line parameters."""
    smoothing_period: int = 21    # Optional smoothing period
    use_smoothing: bool = False   # Whether to apply smoothing
    signal_threshold: float = 0.0 # Signal threshold for trend changes
    min_periods: int = 10         # Minimum periods for calculation


class AccumulationDistribution(BaseIndicator):
    """
    Accumulation/Distribution Line Technical Indicator Implementation
    
    The A/D Line is a cumulative indicator that uses volume and price to assess 
    whether a stock is being accumulated or distributed. It's based on the premise 
    that volume precedes price, so changes in the A/D Line can help predict future 
    price movements.
    
    Formula:
    1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    2. Money Flow Volume = Money Flow Multiplier * Volume
    3. A/D Line = Previous A/D Line + Money Flow Volume
    
    Money Flow Multiplier ranges from -1 to +1:
    - +1: Close = High (maximum buying pressure)
    - 0: Close = (High + Low) / 2 (neutral)
    - -1: Close = Low (maximum selling pressure)
    
    Interpretation:
    - Rising A/D Line: Accumulation (buying pressure)
    - Falling A/D Line: Distribution (selling pressure)
    - A/D Line confirming price trend: Trend strength
    - A/D Line diverging from price: Potential reversal signal
    - Flat A/D Line: Balanced buying/selling pressure
    """
    
    def __init__(self, config: Optional[AccumulationDistributionConfig] = None):
        """
        Initialize Accumulation/Distribution indicator.
        
        Args:
            config: Configuration object with indicator parameters
        """
        self.config = config or AccumulationDistributionConfig()
        self.name = "Accumulation/Distribution Line"
        self.category = "volume"
        
        # Validation
        if self.config.smoothing_period < 1:
            raise ValueError("Smoothing period must be positive")
        if self.config.min_periods < 1:
            raise ValueError("Min periods must be positive")
        
        # Internal state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset internal calculation state."""
        self.ad_line_values: List[float] = []
        self.smoothed_ad_values: List[float] = []
        self.money_flow_multipliers: List[float] = []
        self.money_flow_volumes: List[float] = []
        self.signals: List[str] = []
        self.initialized = False
    
    def _calculate_money_flow_multiplier(self, high: float, low: float, close: float) -> float:
        """
        Calculate Money Flow Multiplier.
        
        Args:
            high: High price
            low: Low price
            close: Close price
            
        Returns:
            Money Flow Multiplier (-1 to +1)
        """
        if high == low:
            # Avoid division by zero - if no price range, assume neutral
            return 0.0
        
        return ((close - low) - (high - close)) / (high - low)
    
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
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            timestamp: Data timestamp
            
        Returns:
            Dictionary containing indicator values and signals
        """
        try:
            # Calculate Money Flow Multiplier
            mf_multiplier = self._calculate_money_flow_multiplier(high, low, close)
            self.money_flow_multipliers.append(mf_multiplier)
            
            # Calculate Money Flow Volume
            mf_volume = mf_multiplier * volume
            self.money_flow_volumes.append(mf_volume)
            
            # Calculate cumulative A/D Line
            if self.ad_line_values:
                ad_value = self.ad_line_values[-1] + mf_volume
            else:
                ad_value = mf_volume
            
            self.ad_line_values.append(ad_value)
            
            # Calculate smoothed A/D Line if enabled
            if self.config.use_smoothing:
                smoothed_ad = self._calculate_sma(self.ad_line_values, self.config.smoothing_period)
                self.smoothed_ad_values.append(smoothed_ad)
            else:
                smoothed_ad = ad_value
                self.smoothed_ad_values.append(smoothed_ad)
            
            # Generate signals
            signal = self._generate_signals(ad_value, smoothed_ad, mf_multiplier)
            self.signals.append(signal)
            
            # Mark as initialized after minimum periods
            if len(self.ad_line_values) >= self.config.min_periods:
                self.initialized = True
            
            return {
                'ad_line': ad_value,
                'smoothed_ad_line': smoothed_ad,
                'money_flow_multiplier': mf_multiplier,
                'money_flow_volume': mf_volume,
                'signal': signal,
                'timestamp': timestamp,
                'initialized': self.initialized
            }
            
        except Exception as e:
            warnings.warn(f"Error updating Accumulation/Distribution: {str(e)}")
            return {
                'ad_line': 0.0,
                'smoothed_ad_line': 0.0,
                'money_flow_multiplier': 0.0,
                'money_flow_volume': 0.0,
                'signal': 'NEUTRAL',
                'timestamp': timestamp,
                'initialized': False
            }
    
    def _generate_signals(self, ad_value: float, smoothed_ad: float, mf_multiplier: float) -> str:
        """
        Generate trading signals based on A/D Line.
        
        Args:
            ad_value: Current A/D Line value
            smoothed_ad: Current smoothed A/D Line value
            mf_multiplier: Current Money Flow Multiplier
            
        Returns:
            Signal string
        """
        if not self.initialized or len(self.ad_line_values) < 2:
            return 'NEUTRAL'
        
        prev_ad = self.ad_line_values[-2]
        current_ad = ad_value
        
        # Trend signals based on A/D Line direction
        ad_change = current_ad - prev_ad
        
        # Strong signals based on money flow multiplier
        if mf_multiplier > 0.7:
            return 'STRONG_ACCUMULATION'
        elif mf_multiplier < -0.7:
            return 'STRONG_DISTRIBUTION'
        
        # Trend continuation signals
        if ad_change > self.config.signal_threshold:
            if mf_multiplier > 0:
                return 'ACCUMULATION'
            else:
                return 'WEAK_ACCUMULATION'
        elif ad_change < -self.config.signal_threshold:
            if mf_multiplier < 0:
                return 'DISTRIBUTION'
            else:
                return 'WEAK_DISTRIBUTION'
        
        # Momentum signals
        if len(self.ad_line_values) >= 3:
            recent_momentum = current_ad - self.ad_line_values[-3]
            if recent_momentum > 0:
                return 'BULLISH_MOMENTUM'
            elif recent_momentum < 0:
                return 'BEARISH_MOMENTUM'
        
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
    
    def get_ad_line_values(self) -> List[float]:
        """Get all A/D Line values."""
        return self.ad_line_values.copy()
    
    def get_smoothed_ad_values(self) -> List[float]:
        """Get all smoothed A/D Line values."""
        return self.smoothed_ad_values.copy()
    
    def get_money_flow_multipliers(self) -> List[float]:
        """Get all Money Flow Multiplier values."""
        return self.money_flow_multipliers.copy()
    
    def detect_divergence(self, price_data: List[float], lookback: int = 14) -> Optional[str]:
        """
        Detect bullish/bearish divergences between price and A/D Line.
        
        Args:
            price_data: List of price values (typically close prices)
            lookback: Periods to look back for divergence
            
        Returns:
            Divergence type or None
        """
        if len(self.ad_line_values) < lookback or len(price_data) < lookback:
            return None
        
        # Get recent data
        recent_ad = self.ad_line_values[-lookback:]
        recent_prices = price_data[-lookback:]
        
        # Find peaks and troughs
        ad_max_idx = np.argmax(recent_ad)
        ad_min_idx = np.argmin(recent_ad)
        price_max_idx = np.argmax(recent_prices)
        price_min_idx = np.argmin(recent_prices)
        
        # Bullish divergence: Lower price low, higher A/D low
        if (price_min_idx > ad_min_idx and 
            recent_prices[price_min_idx] < recent_prices[ad_min_idx] and
            recent_ad[price_min_idx] > recent_ad[ad_min_idx]):
            return 'BULLISH_DIVERGENCE'
        
        # Bearish divergence: Higher price high, lower A/D high
        if (price_max_idx > ad_max_idx and 
            recent_prices[price_max_idx] > recent_prices[ad_max_idx] and
            recent_ad[price_max_idx] < recent_ad[ad_max_idx]):
            return 'BEARISH_DIVERGENCE'
        
        return None
    
    def get_accumulation_distribution_pressure(self) -> Dict[str, Union[float, str]]:
        """
        Analyze current accumulation/distribution pressure.
        
        Returns:
            Dictionary with pressure analysis
        """
        if len(self.money_flow_multipliers) < 10:
            return {}
        
        # Get recent data
        recent_mf = self.money_flow_multipliers[-10:]
        recent_ad_changes = []
        
        for i in range(1, min(11, len(self.ad_line_values))):
            change = self.ad_line_values[-i] - self.ad_line_values[-(i+1)]
            recent_ad_changes.append(change)
        
        # Calculate statistics
        avg_mf = np.mean(recent_mf)
        positive_mf_count = sum(1 for x in recent_mf if x > 0)
        strong_accumulation_count = sum(1 for x in recent_mf if x > 0.5)
        strong_distribution_count = sum(1 for x in recent_mf if x < -0.5)
        
        # Determine pressure type
        if avg_mf > 0.3:
            pressure_type = 'STRONG_ACCUMULATION_PRESSURE'
        elif avg_mf > 0.1:
            pressure_type = 'MODERATE_ACCUMULATION_PRESSURE'
        elif avg_mf < -0.3:
            pressure_type = 'STRONG_DISTRIBUTION_PRESSURE'
        elif avg_mf < -0.1:
            pressure_type = 'MODERATE_DISTRIBUTION_PRESSURE'
        else:
            pressure_type = 'BALANCED_PRESSURE'
        
        # Calculate consistency
        consistency_score = positive_mf_count / len(recent_mf) * 100
        
        return {
            'pressure_type': pressure_type,
            'avg_money_flow_multiplier': avg_mf,
            'consistency_score': consistency_score,
            'strong_accumulation_periods': strong_accumulation_count,
            'strong_distribution_periods': strong_distribution_count,
            'recent_ad_trend': 'UP' if np.mean(recent_ad_changes) > 0 else 'DOWN'
        }
    
    def get_volume_price_efficiency(self) -> Optional[float]:
        """
        Calculate how efficiently volume is moving price (A/D Line slope vs volume).
        
        Returns:
            Efficiency ratio or None
        """
        if len(self.ad_line_values) < 10:
            return None
        
        # Calculate A/D Line trend
        recent_ad = self.ad_line_values[-10:]
        x = np.arange(len(recent_ad))
        ad_slope = np.polyfit(x, recent_ad, 1)[0]
        
        # Get recent volume data
        recent_volumes = self.money_flow_volumes[-10:]
        avg_volume = np.mean([abs(x) for x in recent_volumes])
        
        # Calculate efficiency (how much A/D Line moves per unit of volume)
        if avg_volume > 0:
            efficiency = abs(ad_slope) / avg_volume * 100
        else:
            efficiency = 0
        
        return efficiency
    
    def get_trend_strength(self) -> Optional[str]:
        """
        Assess trend strength based on A/D Line behavior.
        
        Returns:
            Trend strength assessment
        """
        if len(self.ad_line_values) < self.config.min_periods:
            return None
        
        # Calculate multiple timeframe trends
        short_term = self.ad_line_values[-1] - self.ad_line_values[-5] if len(self.ad_line_values) >= 5 else 0
        medium_term = self.ad_line_values[-1] - self.ad_line_values[-10] if len(self.ad_line_values) >= 10 else 0
        long_term = self.ad_line_values[-1] - self.ad_line_values[-20] if len(self.ad_line_values) >= 20 else 0
        
        # Assess consistency across timeframes
        trends = [short_term, medium_term, long_term]
        positive_trends = sum(1 for x in trends if x > 0)
        negative_trends = sum(1 for x in trends if x < 0)
        
        if positive_trends == 3:
            # Check if accelerating
            if short_term > medium_term > long_term:
                return 'STRONG_ACCELERATING_UPTREND'
            else:
                return 'CONSISTENT_UPTREND'
        elif negative_trends == 3:
            # Check if accelerating
            if short_term < medium_term < long_term:
                return 'STRONG_ACCELERATING_DOWNTREND'
            else:
                return 'CONSISTENT_DOWNTREND'
        elif positive_trends > negative_trends:
            return 'MIXED_BULLISH'
        elif negative_trends > positive_trends:
            return 'MIXED_BEARISH'
        else:
            return 'SIDEWAYS'


def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='D')
    
    # Generate realistic price data with accumulation/distribution phases
    base_price = 100.0
    prices = [base_price]
    volumes = []
    
    for i in range(1, periods):
        # Create phases: accumulation, markup, distribution, markdown
        phase = (i // 20) % 4
        
        if phase == 0:  # Accumulation phase
            trend = 0.005  # Slight upward bias
            vol_multiplier = 1.2  # Higher volume
        elif phase == 1:  # Markup phase
            trend = 0.03   # Strong uptrend
            vol_multiplier = 1.5  # High volume
        elif phase == 2:  # Distribution phase
            trend = -0.005 # Slight downward bias
            vol_multiplier = 1.3  # Higher volume
        else:  # Markdown phase
            trend = -0.025 # Strong downtrend
            vol_multiplier = 1.1  # Moderate volume
        
        noise = np.random.normal(0, 1.2)
        price_change = trend + noise
        new_price = max(prices[-1] + price_change, 50.0)
        prices.append(new_price)
        
        # Volume correlated with price movement and phase
        base_vol = 100000
        price_change_factor = 1 + abs(price_change) * 0.5
        volume = base_vol * vol_multiplier * price_change_factor * np.random.uniform(0.7, 1.3)
        volumes.append(max(volume, 10000))
    
    # Add initial volume
    volumes.insert(0, 100000)
    
    data = []
    for i, (date, close, volume) in enumerate(zip(dates, prices, volumes)):
        # Create OHLC with realistic intraday movement
        daily_range = abs(np.random.normal(0, 1.5))
        high = close + daily_range * 0.6
        low = close - daily_range * 0.4
        
        # Ensure OHLC consistency
        high = max(high, close)
        low = min(low, close)
        
        data.append({
            'timestamp': date,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def main():
    """Test the Accumulation/Distribution Line indicator."""
    print("=== Accumulation/Distribution Line Indicator Test ===")
    
    # Create sample data
    data = create_sample_data(60)
    print(f"Created sample data with {len(data)} periods")
    
    # Test with default configuration
    print("\n1. Testing with default configuration...")
    config = AccumulationDistributionConfig()
    ad = AccumulationDistribution(config)
    
    # Calculate batch
    results = ad.calculate_batch(data)
    
    print(f"Calculated A/D Line for {len(results)} periods")
    print(f"Indicator initialized: {ad.initialized}")
    
    # Display recent results
    print("\nRecent A/D Line values:")
    recent = results.tail(10)
    for idx, row in recent.iterrows():
        print(f"A/D: {row['ad_line']:.0f}, "
              f"MF Mult: {row['money_flow_multiplier']:.3f}, "
              f"MF Vol: {row['money_flow_volume']:.0f}, "
              f"Signal: {row['signal']}")
    
    # Test accumulation/distribution pressure analysis
    print("\n2. Testing accumulation/distribution pressure analysis...")
    pressure = ad.get_accumulation_distribution_pressure()
    for key, value in pressure.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test trend strength assessment
    print("\n3. Testing trend strength assessment...")
    trend_strength = ad.get_trend_strength()
    print(f"Current trend strength: {trend_strength}")
    
    # Test volume-price efficiency
    print("\n4. Testing volume-price efficiency...")
    efficiency = ad.get_volume_price_efficiency()
    print(f"Volume-price efficiency: {efficiency:.4f}" if efficiency else "Not enough data")
    
    # Test divergence detection
    print("\n5. Testing divergence detection...")
    close_prices = data['close'].tolist()
    divergence = ad.detect_divergence(close_prices)
    print(f"Divergence detected: {divergence}")
    
    # Test with smoothing enabled
    print("\n6. Testing with smoothing enabled...")
    smooth_config = AccumulationDistributionConfig(
        use_smoothing=True,
        smoothing_period=10
    )
    ad_smooth = AccumulationDistribution(smooth_config)
    
    # Test single updates
    print("\n7. Testing single updates (smoothed)...")
    for i in range(5):
        row = data.iloc[i]
        result = ad_smooth.update(
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        print(f"Update {i+1}: A/D={result['ad_line']:.0f}, "
              f"Smoothed={result['smoothed_ad_line']:.0f}, "
              f"MF Mult={result['money_flow_multiplier']:.3f}, "
              f"Signal={result['signal']}")
    
    # Test error handling
    print("\n8. Testing error handling...")
    try:
        invalid_config = AccumulationDistributionConfig(min_periods=0)
        AccumulationDistribution(invalid_config)
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Display signal distribution
    print("\n9. Signal distribution:")
    signal_counts = {}
    for signal in ad.get_signals():
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} times")
    
    # A/D Line statistics
    print("\n10. A/D Line statistics:")
    ad_values = ad.get_ad_line_values()
    mf_multipliers = ad.get_money_flow_multipliers()
    
    if ad_values and mf_multipliers:
        print(f"Final A/D Line value: {ad_values[-1]:.0f}")
        print(f"A/D Line range: {min(ad_values):.0f} to {max(ad_values):.0f}")
        print(f"Average Money Flow Multiplier: {np.mean(mf_multipliers):.3f}")
        print(f"MF Multiplier range: {min(mf_multipliers):.3f} to {max(mf_multipliers):.3f}")
        
        # Count accumulation vs distribution periods
        accumulation_periods = sum(1 for x in mf_multipliers if x > 0)
        distribution_periods = sum(1 for x in mf_multipliers if x < 0)
        neutral_periods = sum(1 for x in mf_multipliers if x == 0)
        
        print(f"Accumulation periods: {accumulation_periods}")
        print(f"Distribution periods: {distribution_periods}")
        print(f"Neutral periods: {neutral_periods}")
    
    print("\n=== Accumulation/Distribution Line Test Complete ===")


if __name__ == "__main__":
    main()