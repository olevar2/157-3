"""
Force Index (FI) - Volume Indicator

The Force Index is a technical indicator that measures the pressure behind price movements
by combining price change and volume. It helps identify the strength of price movements
and potential reversal points.

Formula:
FI = (Close - Previous Close) * Volume
FI_MA = Moving Average of FI (typically 2 or 13 periods)

Interpretation:
- Positive values indicate buying pressure (bullish)
- Negative values indicate selling pressure (bearish)
- Divergences between price and Force Index can signal reversals
- Higher volume amplifies the Force Index signal

Author: Platform3 Team
Date: June 2025
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType

class ForceIndex(TechnicalIndicator):
    """
    Force Index (FI) Calculator    
    The Force Index measures the pressure behind price movements by combining
    price change and volume data.
    """
    
    def __init__(self, period: int = 13, config=None):
        """
        Initialize Force Index calculator
        
        Parameters:
        -----------
        period : int, default 13
            Period for smoothing the Force Index (moving average)
        config : dict, optional
            Configuration dictionary containing parameters
        """
        # Handle config parameter
        if config is not None:
            if isinstance(config, dict):
                self.period = config.get('period', period)
            else:
                self.period = period
        else:
            self.period = period
            
        self.name = f"Force Index ({self.period})"
        
    def calculate(self, data=None, high=None, low=None, close=None, volume=None) -> pd.Series:
        """
        Calculate Force Index
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame with OHLCV data
        high : pd.Series or np.ndarray, optional
            High prices
        low : pd.Series or np.ndarray, optional
            Low prices  
        close : pd.Series or np.ndarray, optional
            Close prices
        volume : pd.Series or np.ndarray, optional
            Volume data
            
        Returns:
        --------
        pd.Series
            Force Index values
        """
        # Handle case where data is a DataFrame
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns and 'volume' in data.columns:
                close = data['close']
                volume = data['volume']
            elif len(data.columns) >= 5:  # Assume standard OHLCV format
                close = data.iloc[:, 3]  # Assuming Close is the 4th column
                volume = data.iloc[:, 4]  # Assuming Volume is the 5th column
        
        # Validate inputs
        if close is None or volume is None:
            raise ValueError("Missing required inputs: close prices and volume data required")
            
        # Convert to pandas Series if needed
        close = pd.Series(close) if not isinstance(close, pd.Series) else close
        volume = pd.Series(volume) if not isinstance(volume, pd.Series) else volume
        
        # Calculate raw Force Index
        price_change = close.diff()
        raw_fi = price_change * volume
        
        # Apply moving average smoothing
        if self.period == 1:
            fi = raw_fi
        else:
            fi = raw_fi.rolling(window=self.period, min_periods=1).mean()
        
        return fi
    
    def calculate_with_signals(self,
                             high: Union[pd.Series, np.ndarray],
                             low: Union[pd.Series, np.ndarray],
                             close: Union[pd.Series, np.ndarray], 
                             volume: Union[pd.Series, np.ndarray],
                             threshold: float = 0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Force Index with buy/sell signals
        
        Parameters:
        -----------
        high : pd.Series or np.ndarray
            High prices
        low : pd.Series or np.ndarray
            Low prices
        close : pd.Series or np.ndarray
            Close prices
        volume : pd.Series or np.ndarray
            Volume data
        threshold : float, default 0
            Threshold for generating signals
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            Force Index values and signals (1=buy, -1=sell, 0=hold)
        """
        fi = self.calculate(high, low, close, volume)
        
        # Generate signals based on zero line and threshold
        signals = pd.Series(0, index=fi.index)
        
        # Buy signal: Force Index crosses above threshold (bullish pressure)
        buy_condition = (fi > threshold) & (fi.shift(1) <= threshold)
        signals[buy_condition] = 1
        
        # Sell signal: Force Index crosses below -threshold (bearish pressure)  
        sell_condition = (fi < -threshold) & (fi.shift(1) >= -threshold)
        signals[sell_condition] = -1
        
        return fi, signals
    
    def calculate_divergence(self,
                           high: Union[pd.Series, np.ndarray],
                           low: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           volume: Union[pd.Series, np.ndarray],
                           lookback: int = 20) -> pd.Series:
        """
        Detect divergences between price and Force Index
        
        Parameters:
        -----------
        high : pd.Series or np.ndarray
            High prices
        low : pd.Series or np.ndarray
            Low prices
        close : pd.Series or np.ndarray
            Close prices
        volume : pd.Series or np.ndarray
            Volume data
        lookback : int, default 20
            Period to look back for divergence detection
            
        Returns:
        --------
        pd.Series
            Divergence signals (1=bullish divergence, -1=bearish divergence, 0=none)
        """
        close = pd.Series(close) if not isinstance(close, pd.Series) else close
        fi = self.calculate(high, low, close, volume)
        
        divergence = pd.Series(0, index=close.index)
        
        for i in range(lookback, len(close)):
            # Get recent period
            recent_close = close.iloc[i-lookback:i+1]
            recent_fi = fi.iloc[i-lookback:i+1]
            
            # Find local peaks and troughs
            close_max_idx = recent_close.idxmax()
            close_min_idx = recent_close.idxmin()
            fi_max_idx = recent_fi.idxmax()
            fi_min_idx = recent_fi.idxmin()
            
            # Bullish divergence: price makes lower low, FI makes higher low
            if (close_min_idx == recent_close.index[-1] and 
                fi_min_idx != recent_fi.index[-1] and
                recent_fi.iloc[-1] > recent_fi.min()):
                divergence.iloc[i] = 1
            
            # Bearish divergence: price makes higher high, FI makes lower high  
            elif (close_max_idx == recent_close.index[-1] and
                  fi_max_idx != recent_fi.index[-1] and
                  recent_fi.iloc[-1] < recent_fi.max()):
                divergence.iloc[i] = -1
        
        return divergence
    
    def generate_signal(self, data: List[MarketData]) -> Optional[IndicatorSignal]:
        """
        Generate trading signal based on Force Index values
        
        Args:
            data: List of MarketData objects
            
        Returns:
            Optional[IndicatorSignal]: Trading signal or None
        """
        if len(data) < self.period + 1:
            return None
            
        # Extract data
        highs = pd.Series([d.high for d in data])
        lows = pd.Series([d.low for d in data])
        closes = pd.Series([d.close for d in data])
        volumes = pd.Series([d.volume for d in data])
        
        # Calculate Force Index
        fi_values = self.calculate(highs, lows, closes, volumes)
        
        if fi_values.isna().iloc[-1]:
            return None
            
        latest_fi = fi_values.iloc[-1]
        prev_fi = fi_values.iloc[-2] if len(fi_values) > 1 else 0
        
        # Generate signal based on Force Index crossover and magnitude
        signal_type = SignalType.HOLD
        strength = 0.0
        confidence = 0.5
        
        # Normalize strength based on typical FI values
        fi_std = fi_values.std() if len(fi_values) > 10 else abs(latest_fi)
        
        if latest_fi > 0 and prev_fi <= 0:
            signal_type = SignalType.BUY
            strength = min(abs(latest_fi) / (fi_std + 1e-8), 1.0) if fi_std > 0 else 0.5
            confidence = 0.7
        elif latest_fi < 0 and prev_fi >= 0:
            signal_type = SignalType.SELL
            strength = min(abs(latest_fi) / (fi_std + 1e-8), 1.0) if fi_std > 0 else 0.5
            confidence = 0.7
        
        return IndicatorSignal(
            timestamp=data[-1].timestamp,
            indicator_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            metadata={'force_index_value': latest_fi}
        )

    def generate_signal(self, data: MarketData, period_start=None, period_end=None) -> List[IndicatorSignal]:
        """
        Generate trading signals based on Force Index.
        
        Parameters:
        -----------
        data : MarketData
            Market data containing OHLCV information
        period_start : datetime, optional
            Start of the period for signal generation
        period_end : datetime, optional
            End of the period for signal generation
            
        Returns:
        --------
        List[IndicatorSignal]
            List of generated trading signals
        """
        # Calculate Force Index
        fi = self.calculate(data=data.df)
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_fi = fi.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_fi = fi
        
        # Only analyze where we have sufficient data
        for i in range(self.period + 1, len(working_fi)):
            current_idx = working_df.index[i]
            current_price = working_df['close'].iloc[i]
            current_fi = working_fi.iloc[i]
            prev_fi = working_fi.iloc[i-1]
            
            # Buy signal: Force Index crosses above zero (bullish pressure)
            if prev_fi <= 0 and current_fi > 0:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_fi,
                        price=current_price,
                        strength=min(abs(current_fi) / 100, 1.0),  # Normalize strength
                        indicator_name=self.name
                    )
                )
            
            # Sell signal: Force Index crosses below zero (bearish pressure)
            elif prev_fi >= 0 and current_fi < 0:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.SELL,
                        indicator_value=current_fi,
                        price=current_price,
                        strength=min(abs(current_fi) / 100, 1.0),  # Normalize strength
                        indicator_name=self.name
                    )
                )
        
        return signals

def calculate_force_index(high: Union[pd.Series, np.ndarray],
                         low: Union[pd.Series, np.ndarray],
                         close: Union[pd.Series, np.ndarray],
                         volume: Union[pd.Series, np.ndarray], 
                         period: int = 13) -> pd.Series:
    """
    Convenience function to calculate Force Index
    
    Parameters:
    -----------
    high : pd.Series or np.ndarray
        High prices
    low : pd.Series or np.ndarray
        Low prices
    close : pd.Series or np.ndarray
        Close prices
    volume : pd.Series or np.ndarray
        Volume data
    period : int, default 13
        Smoothing period
        
    Returns:
    --------
    pd.Series
        Force Index values
    """
    calculator = ForceIndex(period=period)
    return calculator.calculate(high, low, close, volume)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    volumes = np.random.randint(1000, 10000, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Test Force Index calculation
    fi_calc = ForceIndex(period=13)
    
    # Calculate Force Index
    fi = fi_calc.calculate(df['high'], df['low'], df['close'], df['volume'])
    print("Force Index (13-period):")
    print(fi.tail())
    
    # Calculate with signals
    fi_signals, signals = fi_calc.calculate_with_signals(
        df['high'], df['low'], df['close'], df['volume'], threshold=1000
    )
    print(f"\nSignals generated: {signals.sum()}")
    print("Recent signals:")
    print(signals.tail(10))
    
    # Test divergence detection
    divergence = fi_calc.calculate_divergence(
        df['high'], df['low'], df['close'], df['volume']
    )
    print(f"\nDivergences detected: {divergence.abs().sum()}")
    
    # Test convenience function
    fi_simple = calculate_force_index(df['high'], df['low'], df['close'], df['volume'])
    print(f"\nConvenience function test - matches: {fi.equals(fi_simple)}")
    
    print("\n=== Force Index Analysis Complete ===")
