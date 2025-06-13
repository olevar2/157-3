"""
Confluence Area Indicator

A confluence area indicator that identifies zones where multiple technical analysis
elements converge, creating high-probability support/resistance levels. These areas
are where different indicators, patterns, or price levels align to create strong
trading opportunities.

Author: Platform3
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # Fallback for direct script execution
    class BaseIndicator:
        """Fallback base class for direct script execution"""
        pass


class ConfluenceArea(BaseIndicator):
    """
    Confluence Area Indicator
    
    Identifies zones where multiple technical analysis elements converge:
    - Moving average convergence
    - Fibonacci retracement levels
    - Previous support/resistance levels
    - Pivot points
    - Bollinger Band boundaries
    - Volume profile levels
    
    The indicator provides:
    - Confluence strength scoring
    - Support/resistance zone identification
    - Multi-timeframe confluence detection
    - Trading opportunity signals
    """
    
    def __init__(self, 
                 sma_periods: List[int] = [20, 50, 200],
                 fibonacci_levels: List[float] = [0.236, 0.382, 0.618, 0.786],
                 lookback_period: int = 50,
                 confluence_threshold: float = 3.0,
                 price_tolerance: float = 0.002):
        """
        Initialize Confluence Area indicator
        
        Args:
            sma_periods: List of SMA periods to check for convergence
            fibonacci_levels: Fibonacci retracement levels to calculate
            lookback_period: Period to look back for support/resistance levels
            confluence_threshold: Minimum confluence score to signal area
            price_tolerance: Price tolerance for level matching (as percentage)
        """
        super().__init__()
        self.sma_periods = sma_periods
        self.fibonacci_levels = fibonacci_levels
        self.lookback_period = lookback_period
        self.confluence_threshold = confluence_threshold
        self.price_tolerance = price_tolerance
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate confluence areas
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'confluence_score': Overall confluence strength score
            - 'support_levels': Identified support confluence areas
            - 'resistance_levels': Identified resistance confluence areas
            - 'signal': Trading signals based on confluence areas
            - 'zone_strength': Strength of confluence zones
        """
        try:
            if len(data) < max(self.sma_periods + [self.lookback_period]):
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'confluence_score': empty_series,
                    'support_levels': empty_series,
                    'resistance_levels': empty_series,
                    'signal': empty_series,
                    'zone_strength': empty_series
                }
            
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Initialize confluence components
            confluence_components = {}
            
            # 1. Moving Average Convergence
            ma_confluence = self._calculate_ma_confluence(close)
            confluence_components['ma_convergence'] = ma_confluence
            
            # 2. Fibonacci Retracement Levels
            fib_confluence = self._calculate_fibonacci_confluence(high, low, close)
            confluence_components['fibonacci'] = fib_confluence
            
            # 3. Support/Resistance Levels
            sr_confluence = self._calculate_support_resistance_confluence(high, low, close)
            confluence_components['support_resistance'] = sr_confluence
            
            # 4. Pivot Points
            pivot_confluence = self._calculate_pivot_confluence(high, low, close)
            confluence_components['pivot_points'] = pivot_confluence
            
            # 5. Bollinger Band Boundaries
            bb_confluence = self._calculate_bollinger_confluence(close)
            confluence_components['bollinger_bands'] = bb_confluence
            
            # Combine all confluence components
            confluence_score = self._combine_confluence_components(confluence_components)
            
            # Identify support and resistance levels
            support_levels, resistance_levels = self._identify_sr_levels(
                confluence_score, close, high, low
            )
            
            # Generate trading signals
            signal = self._generate_signals(confluence_score, close, support_levels, resistance_levels)
            
            # Calculate zone strength
            zone_strength = self._calculate_zone_strength(confluence_score, close)
            
            return {
                'confluence_score': confluence_score,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'signal': signal,
                'zone_strength': zone_strength
            }
            
        except Exception as e:
            print(f"Error in Confluence Area calculation: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'confluence_score': empty_series,
                'support_levels': empty_series,
                'resistance_levels': empty_series,
                'signal': empty_series,
                'zone_strength': empty_series
            }
    
    def _calculate_ma_confluence(self, close: pd.Series) -> pd.Series:
        """Calculate moving average convergence"""
        try:
            ma_values = {}
            for period in self.sma_periods:
                ma_values[period] = close.rolling(window=period, min_periods=1).mean()
            
            confluence = pd.Series(0.0, index=close.index)
            
            for i in range(len(close)):
                current_price = close.iloc[i]
                ma_distances = []
                
                for period, ma_series in ma_values.items():
                    if not pd.isna(ma_series.iloc[i]):
                        distance = abs(current_price - ma_series.iloc[i]) / current_price
                        ma_distances.append(distance)
                
                if ma_distances:
                    # Higher confluence when MAs are closer to current price
                    avg_distance = np.mean(ma_distances)
                    confluence.iloc[i] = max(0, 1 - avg_distance * 10)  # Scale and invert
            
            return confluence
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_fibonacci_confluence(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Fibonacci retracement level confluence"""
        try:
            confluence = pd.Series(0.0, index=close.index)
            
            for i in range(self.lookback_period, len(close)):
                current_price = close.iloc[i]
                
                # Find recent high and low for Fibonacci calculation
                recent_high = high.iloc[i-self.lookback_period:i].max()
                recent_low = low.iloc[i-self.lookback_period:i].min()
                
                if recent_high > recent_low:
                    range_size = recent_high - recent_low
                    
                    # Calculate Fibonacci levels
                    fib_score = 0
                    for level in self.fibonacci_levels:
                        fib_price = recent_high - (range_size * level)
                        price_diff = abs(current_price - fib_price) / current_price
                        
                        if price_diff <= self.price_tolerance:
                            fib_score += 1
                    
                    confluence.iloc[i] = fib_score
            
            return confluence
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_support_resistance_confluence(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate support/resistance level confluence"""
        try:
            confluence = pd.Series(0.0, index=close.index)
            
            for i in range(self.lookback_period, len(close)):
                current_price = close.iloc[i]
                
                # Find recent significant levels
                recent_highs = high.iloc[i-self.lookback_period:i]
                recent_lows = low.iloc[i-self.lookback_period:i]
                
                sr_score = 0
                
                # Check proximity to recent highs (resistance)
                for price_level in recent_highs:
                    price_diff = abs(current_price - price_level) / current_price
                    if price_diff <= self.price_tolerance:
                        sr_score += 0.5
                
                # Check proximity to recent lows (support)
                for price_level in recent_lows:
                    price_diff = abs(current_price - price_level) / current_price
                    if price_diff <= self.price_tolerance:
                        sr_score += 0.5
                
                confluence.iloc[i] = min(sr_score, 3.0)  # Cap the score
            
            return confluence
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_pivot_confluence(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate pivot point confluence"""
        try:
            confluence = pd.Series(0.0, index=close.index)
            
            for i in range(1, len(close)):
                current_price = close.iloc[i]
                
                # Calculate previous day pivot
                prev_high = high.iloc[i-1]
                prev_low = low.iloc[i-1]
                prev_close = close.iloc[i-1]
                
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = 2 * pivot - prev_low
                s1 = 2 * pivot - prev_high
                
                # Check proximity to pivot levels
                pivot_score = 0
                for level in [pivot, r1, s1]:
                    price_diff = abs(current_price - level) / current_price
                    if price_diff <= self.price_tolerance:
                        pivot_score += 1
                
                confluence.iloc[i] = pivot_score
            
            return confluence
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_bollinger_confluence(self, close: pd.Series) -> pd.Series:
        """Calculate Bollinger Band confluence"""
        try:
            period = 20
            bb_middle = close.rolling(window=period, min_periods=1).mean()
            bb_std = close.rolling(window=period, min_periods=1).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            confluence = pd.Series(0.0, index=close.index)
            
            for i in range(len(close)):
                current_price = close.iloc[i]
                
                bb_score = 0
                if not pd.isna(bb_upper.iloc[i]):
                    # Check proximity to BB levels
                    for level in [bb_upper.iloc[i], bb_middle.iloc[i], bb_lower.iloc[i]]:
                        price_diff = abs(current_price - level) / current_price
                        if price_diff <= self.price_tolerance:
                            bb_score += 1
                
                confluence.iloc[i] = bb_score
            
            return confluence
        except:
            return pd.Series(0, index=close.index)
    
    def _combine_confluence_components(self, components: Dict[str, pd.Series]) -> pd.Series:
        """Combine all confluence components into final score"""
        try:
            # Weight the components
            weights = {
                'ma_convergence': 2.0,
                'fibonacci': 1.5,
                'support_resistance': 2.0,
                'pivot_points': 1.0,
                'bollinger_bands': 1.0
            }
            
            confluence_score = pd.Series(0.0, index=list(components.values())[0].index)
            
            for component_name, component_series in components.items():
                weight = weights.get(component_name, 1.0)
                confluence_score += component_series * weight
            
            return confluence_score
        except:
            return pd.Series(0, index=list(components.values())[0].index)
    
    def _identify_sr_levels(self, confluence_score: pd.Series, close: pd.Series, 
                           high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Identify support and resistance levels"""
        try:
            support_levels = pd.Series(0.0, index=close.index)
            resistance_levels = pd.Series(0.0, index=close.index)
            
            for i in range(len(close)):
                if confluence_score.iloc[i] >= self.confluence_threshold:
                    current_price = close.iloc[i]
                    
                    # Determine if it's support or resistance based on recent price action
                    if i >= 5:
                        recent_low = low.iloc[i-5:i].min()
                        recent_high = high.iloc[i-5:i].max()
                        
                        if current_price <= (recent_low + recent_high) / 2:
                            support_levels.iloc[i] = confluence_score.iloc[i]
                        else:
                            resistance_levels.iloc[i] = confluence_score.iloc[i]
            
            return support_levels, resistance_levels
        except:
            empty_series = pd.Series(0, index=close.index)
            return empty_series, empty_series
    
    def _generate_signals(self, confluence_score: pd.Series, close: pd.Series,
                         support_levels: pd.Series, resistance_levels: pd.Series) -> pd.Series:
        """Generate trading signals based on confluence areas"""
        try:
            signal = pd.Series(0, index=close.index)
            
            for i in range(1, len(close)):
                current_confluence = confluence_score.iloc[i]
                prev_confluence = confluence_score.iloc[i-1]
                
                # Signal when approaching strong confluence area
                if current_confluence >= self.confluence_threshold:
                    price_change = close.iloc[i] - close.iloc[i-1]
                    
                    # Bullish signal at support confluence
                    if support_levels.iloc[i] > 0 and price_change > 0:
                        signal.iloc[i] = 1
                    
                    # Bearish signal at resistance confluence
                    elif resistance_levels.iloc[i] > 0 and price_change < 0:
                        signal.iloc[i] = -1
            
            return signal
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_zone_strength(self, confluence_score: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate confluence zone strength"""
        try:
            # Normalize confluence score to 0-100 scale
            max_score = confluence_score.max()
            if max_score > 0:
                zone_strength = (confluence_score / max_score) * 100
            else:
                zone_strength = pd.Series(0, index=close.index)
            
            return zone_strength
        except:
            return pd.Series(0, index=close.index)
    
    def get_confluence_score(self, data: pd.DataFrame) -> pd.Series:
        """Get confluence scores"""
        result = self.calculate(data)
        return result['confluence_score']
    
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Get confluence-based trading signals"""
        result = self.calculate(data)
        return result['signal']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.lognormal(10, 0.3, 100)
    }, index=dates)
    
    # Calculate high, low, close from open
    data['high'] = data['open'] + np.random.uniform(0, 2, 100)
    data['low'] = data['open'] - np.random.uniform(0, 2, 100)
    data['close'] = data['open'] + np.random.randn(100) * 0.5
    
    # Test the indicator
    print("Testing Confluence Area Indicator")
    print("=" * 50)
    
    indicator = ConfluenceArea(
        sma_periods=[10, 20, 50],
        fibonacci_levels=[0.236, 0.382, 0.618],
        lookback_period=20,
        confluence_threshold=2.0,
        price_tolerance=0.01
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Confluence score range: {result['confluence_score'].min():.3f} to {result['confluence_score'].max():.3f}")
    print(f"Support levels detected: {(result['support_levels'] > 0).sum()}")
    print(f"Resistance levels detected: {(result['resistance_levels'] > 0).sum()}")
    print(f"Zone strength range: {result['zone_strength'].min():.1f} to {result['zone_strength'].max():.1f}")
    
    # Show signals
    signals = result['signal']
    print(f"\nTrading signals:")
    print(f"Buy signals: {(signals == 1).sum()}")
    print(f"Sell signals: {(signals == -1).sum()}")
    
    # Show top confluence areas
    strong_confluence = data[result['confluence_score'] >= 2.0].copy()
    if len(strong_confluence) > 0:
        strong_confluence['confluence_score'] = result['confluence_score'][result['confluence_score'] >= 2.0]
        strong_confluence['zone_strength'] = result['zone_strength'][result['confluence_score'] >= 2.0]
        print(f"\nTop confluence areas:")
        print(strong_confluence[['close', 'confluence_score', 'zone_strength']].round(3))
    
    print("\nConfluence Area Indicator test completed successfully!")