"""
Relative Volatility Index (RVI)
A technical momentum indicator that measures the direction of volatility.
Developed by Donald Dorsey.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


class RelativeVolatilityIndex:
    """
    Relative Volatility Index (RVI)
    
    The RVI measures the direction of volatility by applying the RSI 
    formula to standard deviation instead of price changes. It oscillates 
    between 0 and 100, with readings above 50 indicating that higher 
    closes are more volatile than lower closes.
    
    Formula:
    1. Calculate standard deviation of closing prices over a period
    2. Separate up and down periods based on price direction
    3. Calculate average of standard deviations for up and down periods
    4. RVI = 100 * (Up StdDev Average) / (Up StdDev Average + Down StdDev Average)
    """
    
    def __init__(self, period: int = 14, std_period: int = 10):
        """
        Initialize Relative Volatility Index
        
        Args:
            period: Period for RVI calculation (default: 14)
            std_period: Period for standard deviation calculation (default: 10)
        """
        self.period = period
        self.std_period = std_period
        self.name = "Relative Volatility Index"
        
    def calculate(self, close: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate Relative Volatility Index
        
        Args:
            close: Closing prices
            
        Returns:
            pd.Series: RVI values
        """
        # Convert to pandas Series if numpy array
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        # Calculate standard deviation over std_period
        std_dev = close.rolling(window=self.std_period).std()
        
        # Determine up and down periods
        price_change = close.diff()
        up_periods = price_change > 0
        down_periods = price_change <= 0
        
        # Create series for up and down standard deviations
        up_std = std_dev.where(up_periods, 0)
        down_std = std_dev.where(down_periods, 0)
        
        # Calculate smoothed averages using Wilder's smoothing (similar to RSI)
        alpha = 1.0 / self.period
        
        # Initialize first values
        up_avg = pd.Series(index=close.index, dtype=float)
        down_avg = pd.Series(index=close.index, dtype=float)
        
        # Calculate initial averages
        first_valid_idx = std_dev.first_valid_index()
        if first_valid_idx is not None:
            start_idx = close.index.get_loc(first_valid_idx) + self.period - 1
            
            if start_idx < len(close):
                # Initial averages
                initial_up = up_std.iloc[:start_idx + 1].mean()
                initial_down = down_std.iloc[:start_idx + 1].mean()
                
                up_avg.iloc[start_idx] = initial_up
                down_avg.iloc[start_idx] = initial_down
                
                # Smooth the rest using Wilder's method
                for i in range(start_idx + 1, len(close)):
                    up_avg.iloc[i] = ((up_avg.iloc[i-1] * (self.period - 1)) + up_std.iloc[i]) / self.period
                    down_avg.iloc[i] = ((down_avg.iloc[i-1] * (self.period - 1)) + down_std.iloc[i]) / self.period
        
        # Calculate RVI
        rvi = 100 * up_avg / (up_avg + down_avg)
        
        return rvi
    
    def calculate_with_signals(self, 
                              close: Union[pd.Series, np.ndarray],
                              overbought: float = 70.0,
                              oversold: float = 30.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate RVI with overbought/oversold signals
        
        Args:
            close: Closing prices
            overbought: Overbought threshold (default: 70.0)
            oversold: Oversold threshold (default: 30.0)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (rvi, signals)
                signals: 1 for overbought, -1 for oversold, 0 for neutral
        """
        rvi = self.calculate(close)
        
        # Generate signals
        signals = pd.Series(0, index=rvi.index, dtype=int)
        signals[rvi > overbought] = 1   # Overbought
        signals[rvi < oversold] = -1    # Oversold
        
        return rvi, signals
    
    def calculate_divergence(self, 
                           close: Union[pd.Series, np.ndarray],
                           lookback: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Identify bullish and bearish divergences
        
        Args:
            close: Closing prices
            lookback: Lookback period for divergence detection
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (rvi, bullish_divergence, bearish_divergence)
        """
        rvi = self.calculate(close)
        
        bullish_div = pd.Series(False, index=close.index)
        bearish_div = pd.Series(False, index=close.index)
        
        # Look for divergences over lookback period
        for i in range(lookback, len(close)):
            current_idx = i
            lookback_start = i - lookback
            
            # Price and RVI extremes in lookback period
            price_segment = close.iloc[lookback_start:current_idx + 1]
            rvi_segment = rvi.iloc[lookback_start:current_idx + 1]
            
            if len(price_segment) < 2 or len(rvi_segment) < 2:
                continue
                
            # Find price lows and highs
            price_min_idx = price_segment.idxmin()
            price_max_idx = price_segment.idxmax()
            rvi_min_idx = rvi_segment.idxmin()
            rvi_max_idx = rvi_segment.idxmax()
            
            # Bullish divergence: price makes lower low, RVI makes higher low
            if (close.loc[close.index[current_idx]] > close.loc[price_min_idx] and
                rvi.loc[rvi.index[current_idx]] > rvi.loc[rvi_min_idx] and
                close.loc[price_min_idx] < close.iloc[lookback_start:current_idx].min()):
                bullish_div.iloc[current_idx] = True
            
            # Bearish divergence: price makes higher high, RVI makes lower high
            if (close.loc[close.index[current_idx]] < close.loc[price_max_idx] and
                rvi.loc[rvi.index[current_idx]] < rvi.loc[rvi_max_idx] and
                close.loc[price_max_idx] > close.iloc[lookback_start:current_idx].max()):
                bearish_div.iloc[current_idx] = True
        
        return rvi, bullish_div, bearish_div
    
    def get_volatility_trend(self, 
                           close: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        Determine volatility trend based on RVI
        
        Args:
            close: Closing prices
            
        Returns:
            Tuple[pd.Series, pd.Series]: (rvi, volatility_trend)
                volatility_trend: 1 for increasing volatility, -1 for decreasing, 0 for neutral
        """
        rvi = self.calculate(close)
        
        # Calculate trend based on RVI direction
        rvi_change = rvi.diff()
        volatility_trend = pd.Series(0, index=rvi.index, dtype=int)
        
        # Smooth the trend to avoid noise
        rvi_ma = rvi.rolling(window=3).mean()
        rvi_ma_change = rvi_ma.diff()
        
        volatility_trend[rvi_ma_change > 0] = 1   # Increasing volatility
        volatility_trend[rvi_ma_change < 0] = -1  # Decreasing volatility
        
        return rvi, volatility_trend
    
    @staticmethod
    def validate_data(close: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate input data
        
        Args:
            close: Closing prices
            
        Returns:
            bool: True if data is valid
        """
        if len(close) < 20:  # Need sufficient data
            return False
        return True


def relative_volatility_index(close: Union[pd.Series, np.ndarray],
                             period: int = 14,
                             std_period: int = 10) -> pd.Series:
    """
    Calculate Relative Volatility Index (functional interface)
    
    Args:
        close: Closing prices
        period: Period for RVI calculation (default: 14)
        std_period: Period for standard deviation calculation (default: 10)
        
    Returns:
        pd.Series: RVI values
    """
    indicator = RelativeVolatilityIndex(period=period, std_period=std_period)
    return indicator.calculate(close)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data with varying volatility
    base_price = 100
    volatilities = [0.01] * 30 + [0.03] * 20 + [0.015] * 30 + [0.025] * 20
    
    prices = [base_price]
    for i, vol in enumerate(volatilities[1:]):
        change = np.random.normal(0, vol)
        prices.append(prices[-1] * (1 + change))
    
    close_prices = pd.Series(prices, index=dates)
    
    # Test the indicator
    rvi = RelativeVolatilityIndex(period=14, std_period=10)
    
    print("Testing Relative Volatility Index")
    print("=" * 40)
    
    # Basic calculation
    rvi_values = rvi.calculate(close_prices)
    print(f"Last 10 RVI values:")
    print(rvi_values.tail(10).round(4))
    
    # Statistical summary
    print(f"\nRVI Statistics:")
    print(f"Mean: {rvi_values.mean():.4f}")
    print(f"Max: {rvi_values.max():.4f}")
    print(f"Min: {rvi_values.min():.4f}")
    print(f"Std: {rvi_values.std():.4f}")
    
    # Overbought/oversold signals
    rvi_values, signals = rvi.calculate_with_signals(close_prices)
    print(f"\nSignals summary:")
    print(f"Overbought periods: {(signals == 1).sum()}")
    print(f"Oversold periods: {(signals == -1).sum()}")
    print(f"Neutral periods: {(signals == 0).sum()}")
    
    # Volatility trend
    rvi_values, vol_trend = rvi.get_volatility_trend(close_prices)
    print(f"\nVolatility trend summary:")
    print(f"Increasing volatility: {(vol_trend == 1).sum()}")
    print(f"Decreasing volatility: {(vol_trend == -1).sum()}")
    print(f"Neutral volatility: {(vol_trend == 0).sum()}")
    
    # Divergence analysis
    rvi_values, bullish_div, bearish_div = rvi.calculate_divergence(close_prices)
    print(f"\nDivergence signals:")
    print(f"Bullish divergences: {bullish_div.sum()}")
    print(f"Bearish divergences: {bearish_div.sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"RVI Period: {rvi.period}")
    print(f"Standard Deviation Period: {rvi.std_period}")
    print(f"Indicator Name: {rvi.name}")
