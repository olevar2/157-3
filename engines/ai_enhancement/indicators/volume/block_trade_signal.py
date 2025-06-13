"""
Block Trade Signal Indicator

A block trade signal indicator that detects large volume transactions that may indicate
institutional trading activity. Block trades are typically characterized by significantly
higher volume than average and can signal important market movements.

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


class BlockTradeSignal(BaseIndicator):
    """
    Block Trade Signal Indicator
    
    Detects large volume transactions that may indicate institutional trading activity.
    Block trades are typically characterized by:
    - Volume significantly above average (threshold-based detection)
    - Price impact analysis
    - Timing concentration (multiple large trades in short timeframe)
    
    The indicator provides:
    - Block trade detection signals
    - Volume anomaly scoring
    - Institutional flow estimation
    - Market impact assessment
    """
    
    def __init__(self, 
                 volume_threshold_multiplier: float = 3.0,
                 rolling_window: int = 20,
                 min_block_volume: Optional[float] = None,
                 price_impact_threshold: float = 0.001):
        """
        Initialize Block Trade Signal indicator
        
        Args:
            volume_threshold_multiplier: Multiplier for average volume to detect blocks
            rolling_window: Period for calculating average volume baseline
            min_block_volume: Absolute minimum volume for block trade detection
            price_impact_threshold: Minimum price impact to consider as block trade
        """
        super().__init__()
        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.rolling_window = rolling_window
        self.min_block_volume = min_block_volume
        self.price_impact_threshold = price_impact_threshold
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, float, Dict]]:
        """
        Calculate block trade signals
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing:
            - 'signal': Block trade detection signals (1=block buy, -1=block sell, 0=none)
            - 'volume_anomaly': Volume anomaly score
            - 'price_impact': Price impact measurement
            - 'institutional_flow': Estimated institutional flow
            - 'block_strength': Strength of block trade signal
        """
        try:
            if len(data) < self.rolling_window:
                # Return empty series for insufficient data
                empty_series = pd.Series(0, index=data.index)
                return {
                    'signal': empty_series,
                    'volume_anomaly': empty_series,
                    'price_impact': empty_series,
                    'institutional_flow': empty_series,
                    'block_strength': empty_series
                }
            
            # Calculate volume statistics
            volume = data['volume']
            avg_volume = volume.rolling(window=self.rolling_window, min_periods=1).mean()
            volume_std = volume.rolling(window=self.rolling_window, min_periods=1).std()
            
            # Calculate volume anomaly score
            volume_anomaly = (volume - avg_volume) / (volume_std + 1e-8)
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(data)
            
            # Detect block trades based on volume threshold
            volume_threshold = avg_volume * self.volume_threshold_multiplier
            
            # Apply minimum block volume if specified
            if self.min_block_volume is not None:
                volume_threshold = np.maximum(volume_threshold, self.min_block_volume)
            
            # Block trade conditions
            is_high_volume = volume > volume_threshold
            is_significant_impact = np.abs(price_impact) > self.price_impact_threshold
            
            # Combine conditions for block trade detection
            block_condition = is_high_volume & is_significant_impact
            
            # Determine signal direction based on price movement and volume
            price_change = data['close'].pct_change()
            signal = pd.Series(0, index=data.index)
            
            # Block buy signal (high volume + positive price impact)
            signal.loc[block_condition & (price_change > 0)] = 1
            
            # Block sell signal (high volume + negative price impact)
            signal.loc[block_condition & (price_change < 0)] = -1
            
            # Calculate institutional flow estimate
            institutional_flow = self._calculate_institutional_flow(data, block_condition, signal)
            
            # Calculate block strength
            block_strength = np.where(
                block_condition,
                volume_anomaly * np.abs(price_impact) * 100,
                0
            )
            block_strength = pd.Series(block_strength, index=data.index)
            
            return {
                'signal': signal,
                'volume_anomaly': volume_anomaly,
                'price_impact': price_impact,
                'institutional_flow': institutional_flow,
                'block_strength': block_strength
            }
            
        except Exception as e:
            print(f"Error in Block Trade Signal calculation: {e}")
            empty_series = pd.Series(0, index=data.index)
            return {
                'signal': empty_series,
                'volume_anomaly': empty_series,
                'price_impact': empty_series,
                'institutional_flow': empty_series,
                'block_strength': empty_series
            }
    
    def _calculate_price_impact(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price impact measurement"""
        try:
            # Simple price impact: percentage change from open to close
            if 'open' in data.columns:
                price_impact = (data['close'] - data['open']) / data['open']
            else:
                # Fallback: use high-low range relative to close
                price_impact = (data['high'] - data['low']) / data['close']
            
            return price_impact.fillna(0)
        except:
            return pd.Series(0, index=data.index)
    
    def _calculate_institutional_flow(self, 
                                    data: pd.DataFrame, 
                                    block_condition: pd.Series, 
                                    signal: pd.Series) -> pd.Series:
        """Calculate estimated institutional flow"""
        try:
            volume = data['volume']
            
            # Estimate institutional flow based on block trades
            institutional_volume = np.where(block_condition, volume, 0)
            
            # Apply directional weighting
            institutional_flow = institutional_volume * signal
            
            # Smooth the flow with rolling sum
            flow_window = min(5, len(data))
            institutional_flow = pd.Series(institutional_flow, index=data.index)
            institutional_flow = institutional_flow.rolling(window=flow_window, min_periods=1).sum()
            
            return institutional_flow
        except:
            return pd.Series(0, index=data.index)
    
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Get block trade signals"""
        result = self.calculate(data)
        return result['signal']
    
    def get_volume_anomaly(self, data: pd.DataFrame) -> pd.Series:
        """Get volume anomaly scores"""
        result = self.calculate(data)
        return result['volume_anomaly']


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data with some block trades
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.lognormal(10, 0.5, 100)  # Log-normal volume distribution
    }, index=dates)
    
    # Calculate high, low, close from open
    data['high'] = data['open'] + np.random.uniform(0, 2, 100)
    data['low'] = data['open'] - np.random.uniform(0, 2, 100)
    data['close'] = data['open'] + np.random.randn(100) * 0.5
    
    # Add some artificial block trades
    block_indices = [20, 35, 60, 80]
    for idx in block_indices:
        data.loc[data.index[idx], 'volume'] *= 5  # 5x normal volume
        # Add price impact
        if idx < len(data) - 1:
            impact = 0.02 if np.random.random() > 0.5 else -0.02
            data.loc[data.index[idx], 'close'] = data.loc[data.index[idx], 'open'] * (1 + impact)
    
    # Test the indicator
    print("Testing Block Trade Signal Indicator")
    print("=" * 50)
    
    indicator = BlockTradeSignal(
        volume_threshold_multiplier=3.0,
        rolling_window=20,
        price_impact_threshold=0.01
    )
    
    result = indicator.calculate(data)
    
    print(f"Data shape: {data.shape}")
    print(f"Signal range: {result['signal'].min():.3f} to {result['signal'].max():.3f}")
    print(f"Volume anomaly range: {result['volume_anomaly'].min():.3f} to {result['volume_anomaly'].max():.3f}")
    print(f"Price impact range: {result['price_impact'].min():.3f} to {result['price_impact'].max():.3f}")
    
    # Show some statistics
    print(f"\nBlock trade signals detected: {(result['signal'] != 0).sum()}")
    print(f"Block buy signals: {(result['signal'] == 1).sum()}")
    print(f"Block sell signals: {(result['signal'] == -1).sum()}")
    
    # Show top block trades
    block_trades = data[result['signal'] != 0].copy()
    if len(block_trades) > 0:
        block_trades['signal'] = result['signal'][result['signal'] != 0]
        block_trades['block_strength'] = result['block_strength'][result['signal'] != 0]
        print(f"\nTop block trades:")
        print(block_trades[['volume', 'signal', 'block_strength']].round(3))
    
    print("\nBlock Trade Signal Indicator test completed successfully!")