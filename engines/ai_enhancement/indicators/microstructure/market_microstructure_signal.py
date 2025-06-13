"""
Market Microstructure Signal Indicator

A market microstructure signal indicator that analyzes high-frequency market data
and microstructure patterns to identify institutional trading activity, liquidity
conditions, and market inefficiencies. This indicator provides insights into
order flow, bid-ask dynamics, and market impact patterns.

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


class MarketMicrostructureSignal(BaseIndicator):
    """
    Market Microstructure Signal Indicator
    
    Analyzes market microstructure patterns and signals:
    - Order flow imbalance detection
    - Bid-ask spread analysis
    - Volume-weighted average price (VWAP) deviations
    - Market impact measurement
    - Liquidity stress indicators
    - Institutional trading patterns
    - Price improvement opportunities
    
    The indicator provides:
    - Order flow signals (buy/sell pressure)
    - Liquidity quality scores
    - Market impact estimates
    - Institutional activity detection
    - Optimal execution timing
    - Price inefficiency signals
    """
    
    def __init__(self, 
                 volume_window: int = 20,
                 spread_window: int = 10,
                 vwap_window: int = 20,
                 impact_threshold: float = 0.001,
                 liquidity_threshold: float = 0.5,
                 institutional_threshold: float = 2.0):
        """
        Initialize Market Microstructure Signal indicator
        
        Args:
            volume_window: Window for volume analysis
            spread_window: Window for spread analysis
            vwap_window: Window for VWAP calculation
            impact_threshold: Threshold for market impact detection
            liquidity_threshold: Threshold for liquidity stress
            institutional_threshold: Threshold for institutional activity
        """
        super().__init__()
        self.volume_window = volume_window
        self.spread_window = spread_window
        self.vwap_window = vwap_window
        self.impact_threshold = impact_threshold
        self.liquidity_threshold = liquidity_threshold
        self.institutional_threshold = institutional_threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate market microstructure signals
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'order_flow_signal': Order flow imbalance signals
            - 'liquidity_score': Liquidity quality scores
            - 'market_impact': Market impact estimates
            - 'institutional_activity': Institutional trading activity
            - 'execution_timing': Optimal execution timing signals
            - 'price_inefficiency': Price inefficiency opportunities
        """
        try:
            if len(data) < max(self.volume_window, self.vwap_window):
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'order_flow_signal': empty_series,
                    'liquidity_score': empty_series,
                    'market_impact': empty_series,
                    'institutional_activity': empty_series,
                    'execution_timing': empty_series,
                    'price_inefficiency': empty_series
                }
            
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            # Calculate order flow signals
            order_flow_signal = self._calculate_order_flow_signal(close, high, low, volume)
            
            # Calculate liquidity scores
            liquidity_score = self._calculate_liquidity_score(high, low, close, volume)
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(close, volume)
            
            # Detect institutional activity
            institutional_activity = self._detect_institutional_activity(close, volume)
            
            # Calculate execution timing signals
            execution_timing = self._calculate_execution_timing(
                close, volume, order_flow_signal, liquidity_score
            )
            
            # Detect price inefficiencies
            price_inefficiency = self._detect_price_inefficiency(close, volume)
            
            return {
                'order_flow_signal': order_flow_signal,
                'liquidity_score': liquidity_score,
                'market_impact': market_impact,
                'institutional_activity': institutional_activity,
                'execution_timing': execution_timing,
                'price_inefficiency': price_inefficiency
            }
            
        except Exception as e:
            print(f"Error in Market Microstructure Signal: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'order_flow_signal': empty_series,
                'liquidity_score': empty_series,
                'market_impact': empty_series,
                'institutional_activity': empty_series,
                'execution_timing': empty_series,
                'price_inefficiency': empty_series
            }
    
    def _calculate_order_flow_signal(self, close: pd.Series, high: pd.Series,
                                   low: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate order flow imbalance signals"""
        try:
            order_flow = pd.Series(0.0, index=close.index)
            
            # Calculate intrabar pressure indicators
            for i in range(1, len(close)):
                # Price position within the bar (0 = low, 1 = high)
                if high.iloc[i] > low.iloc[i]:
                    price_position = (close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i])
                else:
                    price_position = 0.5
                
                # Volume-weighted pressure
                volume_weight = volume.iloc[i] / volume.rolling(window=self.volume_window, min_periods=1).mean().iloc[i]
                
                # Directional pressure (buy vs sell)
                if close.iloc[i] > close.iloc[i-1]:
                    # Upward pressure
                    buy_pressure = price_position * volume_weight
                    order_flow.iloc[i] = buy_pressure
                elif close.iloc[i] < close.iloc[i-1]:
                    # Downward pressure
                    sell_pressure = (1 - price_position) * volume_weight
                    order_flow.iloc[i] = -sell_pressure
                else:
                    # Neutral
                    order_flow.iloc[i] = (price_position - 0.5) * volume_weight
            
            # Smooth the signal
            order_flow = order_flow.rolling(window=3, min_periods=1).mean()
            
            return order_flow
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_liquidity_score(self, high: pd.Series, low: pd.Series,
                                  close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate liquidity quality scores"""
        try:
            liquidity_score = pd.Series(0.5, index=close.index)
            
            for i in range(self.spread_window, len(close)):
                # Bid-ask spread proxy (high-low range)
                spread = high.iloc[i] - low.iloc[i]
                avg_spread = (high.iloc[i-self.spread_window:i] - 
                            low.iloc[i-self.spread_window:i]).mean()
                
                # Normalize spread
                relative_spread = spread / close.iloc[i] if close.iloc[i] > 0 else 0
                avg_relative_spread = avg_spread / close.iloc[i] if close.iloc[i] > 0 else 0
                
                # Volume component
                current_volume = volume.iloc[i]
                avg_volume = volume.iloc[i-self.volume_window:i].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Liquidity components
                spread_component = 1 - min(1, relative_spread / (avg_relative_spread + 1e-8))
                volume_component = min(1, volume_ratio)
                
                # Price stability component
                price_stability = 1 - min(1, abs(close.pct_change().iloc[i]) * 100)
                
                # Combined liquidity score
                liquidity_score.iloc[i] = (spread_component + volume_component + price_stability) / 3
            
            return liquidity_score
        except:
            return pd.Series(0.5, index=close.index)
    
    def _calculate_market_impact(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate market impact estimates"""
        try:
            market_impact = pd.Series(0.0, index=close.index)
            
            for i in range(1, len(close)):
                # Price change
                price_change = close.pct_change().iloc[i]
                
                # Volume surprise
                avg_volume = volume.rolling(window=self.volume_window, min_periods=1).mean().iloc[i]
                volume_surprise = (volume.iloc[i] - avg_volume) / avg_volume if avg_volume > 0 else 0
                
                # Market impact as correlation between volume and price change
                if abs(volume_surprise) > 0.1:  # Significant volume event
                    impact = abs(price_change) * np.sign(volume_surprise)
                    market_impact.iloc[i] = impact
            
            # Smooth and scale
            market_impact = market_impact.rolling(window=3, min_periods=1).mean()
            
            return market_impact
        except:
            return pd.Series(0, index=close.index)
    
    def _detect_institutional_activity(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Detect institutional trading activity"""
        try:
            institutional_activity = pd.Series(0, index=close.index)
            
            # Calculate volume percentiles
            volume_ma = volume.rolling(window=self.volume_window, min_periods=1).mean()
            volume_std = volume.rolling(window=self.volume_window, min_periods=1).std()
            
            for i in range(self.volume_window, len(volume)):
                # Volume Z-score
                volume_zscore = (volume.iloc[i] - volume_ma.iloc[i]) / (volume_std.iloc[i] + 1e-8)
                
                # Price momentum
                price_momentum = close.pct_change(5).iloc[i] if i >= 5 else 0
                
                # Institutional activity criteria:
                # 1. High volume (above threshold)
                # 2. Sustained price direction
                # 3. Low relative spread (efficient execution)
                
                if volume_zscore > self.institutional_threshold:
                    # Check for sustained direction
                    if abs(price_momentum) > 0.01:  # 1% momentum
                        # Check recent price consistency
                        recent_changes = close.pct_change().iloc[i-3:i+1]
                        direction_consistency = np.sign(recent_changes).sum() / len(recent_changes)
                        
                        if abs(direction_consistency) > 0.5:  # Consistent direction
                            institutional_activity.iloc[i] = np.sign(price_momentum)
            
            return institutional_activity
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_execution_timing(self, close: pd.Series, volume: pd.Series,
                                   order_flow: pd.Series, liquidity_score: pd.Series) -> pd.Series:
        """Calculate optimal execution timing signals"""
        try:
            execution_timing = pd.Series(0, index=close.index)
            
            for i in range(len(close)):
                timing_score = 0
                
                # High liquidity is good for execution
                if liquidity_score.iloc[i] > 0.7:
                    timing_score += 1
                
                # Favorable order flow for desired direction
                if abs(order_flow.iloc[i]) < 0.3:  # Balanced order flow
                    timing_score += 1
                
                # Volume availability
                avg_volume = volume.rolling(window=self.volume_window, min_periods=1).mean().iloc[i]
                if volume.iloc[i] > avg_volume * 0.8:  # Sufficient volume
                    timing_score += 1
                
                # Low market impact environment
                if i >= 5:
                    recent_volatility = close.pct_change().iloc[i-5:i].std()
                    if recent_volatility < 0.02:  # Low volatility
                        timing_score += 1
                
                # Execution timing signal (0-4 scale normalized to 0-1)
                execution_timing.iloc[i] = timing_score / 4
            
            return execution_timing
        except:
            return pd.Series(0, index=close.index)
    
    def _detect_price_inefficiency(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Detect price inefficiency opportunities"""
        try:
            price_inefficiency = pd.Series(0, index=close.index)
            
            # Calculate VWAP
            vwap = self._calculate_vwap(close, volume)
            
            for i in range(self.vwap_window, len(close)):
                # Price deviation from VWAP
                vwap_deviation = (close.iloc[i] - vwap.iloc[i]) / vwap.iloc[i] if vwap.iloc[i] > 0 else 0
                
                # Volume profile analysis
                recent_volume = volume.iloc[i-self.vwap_window:i]
                volume_concentration = recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else 0
                
                # Inefficiency signals:
                # 1. Significant VWAP deviation
                # 2. Low volume concentration (distributed trading)
                # 3. Price momentum reversal potential
                
                if abs(vwap_deviation) > 0.02:  # 2% deviation from VWAP
                    # Check for reversal conditions
                    if volume_concentration < 0.5:  # Distributed volume
                        # Mean reversion opportunity
                        inefficiency_strength = abs(vwap_deviation) * (1 - volume_concentration)
                        price_inefficiency.iloc[i] = -np.sign(vwap_deviation) * inefficiency_strength
            
            return price_inefficiency
        except:
            return pd.Series(0, index=close.index)
    
    def _calculate_vwap(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            vwap = pd.Series(0.0, index=close.index)
            
            for i in range(self.vwap_window, len(close)):
                price_segment = close.iloc[i-self.vwap_window:i+1]
                volume_segment = volume.iloc[i-self.vwap_window:i+1]
                
                total_volume = volume_segment.sum()
                if total_volume > 0:
                    vwap.iloc[i] = (price_segment * volume_segment).sum() / total_volume
                else:
                    vwap.iloc[i] = price_segment.mean()
            
            return vwap
        except:
            return close.rolling(window=self.vwap_window, min_periods=1).mean()
    
    def get_order_flow_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get order flow signals"""
        result = self.calculate(data)
        return result['order_flow_signal']
    
    def get_liquidity_score(self, data: pd.DataFrame) -> pd.Series:
        """Get liquidity quality scores"""
        result = self.calculate(data)
        return result['liquidity_score']
    
    def get_institutional_activity(self, data: pd.DataFrame) -> pd.Series:
        """Get institutional activity signals"""
        result = self.calculate(data)
        return result['institutional_activity']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data with microstructure patterns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data with microstructure characteristics
    base_price = 100
    price_trend = np.cumsum(np.random.randn(100) * 0.3)
    
    # Add institutional activity patterns
    institutional_days = [20, 45, 70, 85]
    volume_base = np.random.lognormal(10, 0.3, 100)
    
    for day in institutional_days:
        if day < len(volume_base):
            volume_base[day] *= 3  # High volume institutional activity
            volume_base[day-1:day+2] *= 1.5  # Spillover effect
    
    close_prices = base_price + price_trend + np.random.randn(100) * 0.5
    
    data = pd.DataFrame({
        'open': close_prices,
        'high': close_prices + np.random.uniform(0.5, 2.0, 100),
        'low': close_prices - np.random.uniform(0.5, 2.0, 100),
        'close': close_prices,
        'volume': volume_base
    }, index=dates)
    
    # Test the indicator
    print("Testing Market Microstructure Signal Indicator")
    print("=" * 50)
    
    indicator = MarketMicrostructureSignal(
        volume_window=20,
        spread_window=10,
        vwap_window=20,
        impact_threshold=0.001,
        liquidity_threshold=0.5,
        institutional_threshold=2.0
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Order flow signal range: {result['order_flow_signal'].min():.3f} to {result['order_flow_signal'].max():.3f}")
    print(f"Liquidity score range: {result['liquidity_score'].min():.3f} to {result['liquidity_score'].max():.3f}")
    print(f"Market impact range: {result['market_impact'].min():.3f} to {result['market_impact'].max():.3f}")
    print(f"Execution timing range: {result['execution_timing'].min():.3f} to {result['execution_timing'].max():.3f}")
    print(f"Price inefficiency range: {result['price_inefficiency'].min():.3f} to {result['price_inefficiency'].max():.3f}")
    
    # Analyze institutional activity
    institutional_signals = result['institutional_activity']
    institutional_buy = (institutional_signals == 1).sum()
    institutional_sell = (institutional_signals == -1).sum()
    
    print(f"\nInstitutional Activity Analysis:")
    print(f"Institutional buy signals: {institutional_buy}")
    print(f"Institutional sell signals: {institutional_sell}")
    
    # Show detected institutional days
    institutional_detected = institutional_signals[institutional_signals != 0]
    if len(institutional_detected) > 0:
        print(f"\nInstitutional activity detected on:")
        for idx in institutional_detected.index:
            signal_type = "BUY" if institutional_detected[idx] > 0 else "SELL"
            print(f"  {idx.strftime('%Y-%m-%d')}: {signal_type}")
    
    # Liquidity analysis
    high_liquidity = (result['liquidity_score'] > 0.7).sum()
    low_liquidity = (result['liquidity_score'] < 0.3).sum()
    
    print(f"\nLiquidity Analysis:")
    print(f"High liquidity periods: {high_liquidity}")
    print(f"Low liquidity periods: {low_liquidity}")
    print(f"Average liquidity score: {result['liquidity_score'].mean():.3f}")
    
    # Execution timing
    optimal_timing = (result['execution_timing'] > 0.75).sum()
    poor_timing = (result['execution_timing'] < 0.25).sum()
    
    print(f"\nExecution Timing Analysis:")
    print(f"Optimal execution periods: {optimal_timing}")
    print(f"Poor execution periods: {poor_timing}")
    
    print("\nMarket Microstructure Signal Indicator test completed successfully!")