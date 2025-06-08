"""
Platform3 Market Data Models

This module provides standardized data structures for market data representation
used across all indicators and trading components.
"""

from typing import NamedTuple, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

class OHLCV(NamedTuple):
    """
    Standard OHLCV (Open, High, Low, Close, Volume) data structure
    
    This represents a single candle/bar of market data with all essential price and volume information.
    Used consistently across all Platform3 indicators and trading components.
    """
    open: float      # Opening price
    high: float      # Highest price  
    low: float       # Lowest price
    close: float     # Closing price
    volume: float    # Trading volume
    timestamp: Optional[datetime] = None  # Optional timestamp

@dataclass
class MarketData:
    """
    Comprehensive market data container for multiple time periods
    
    Contains historical OHLCV data with utility methods for indicator calculations.
    """
    data: List[OHLCV]
    symbol: str
    timeframe: str
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.data:
            raise ValueError("Market data cannot be empty")
        
        # Ensure all OHLCV entries are valid
        for i, candle in enumerate(self.data):
            if not isinstance(candle, OHLCV):
                raise ValueError(f"Data at index {i} is not OHLCV type")
            
            # Basic price validation
            if candle.high < candle.low:
                raise ValueError(f"Invalid candle at index {i}: high ({candle.high}) < low ({candle.low})")
            if candle.open < 0 or candle.close < 0:
                raise ValueError(f"Invalid candle at index {i}: negative prices not allowed")
            if candle.volume < 0:
                raise ValueError(f"Invalid candle at index {i}: negative volume not allowed")
    
    @property
    def length(self) -> int:
        """Number of candles in the dataset"""
        return len(self.data)
    
    @property
    def opens(self) -> np.ndarray:
        """Array of opening prices"""
        return np.array([candle.open for candle in self.data])
    
    @property
    def highs(self) -> np.ndarray:
        """Array of high prices"""
        return np.array([candle.high for candle in self.data])
    
    @property
    def lows(self) -> np.ndarray:
        """Array of low prices"""
        return np.array([candle.low for candle in self.data])
    
    @property
    def closes(self) -> np.ndarray:
        """Array of closing prices"""
        return np.array([candle.close for candle in self.data])
    
    @property
    def volumes(self) -> np.ndarray:
        """Array of volumes"""
        return np.array([candle.volume for candle in self.data])
    
    @property
    def timestamps(self) -> List[Optional[datetime]]:
        """Array of timestamps"""
        return [candle.timestamp for candle in self.data]
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert market data to pandas DataFrame
        
        Returns:
            DataFrame with OHLCV columns and optional timestamp index
        """
        df_data = {
            'open': self.opens,
            'high': self.highs,
            'low': self.lows,
            'close': self.closes,
            'volume': self.volumes
        }
        
        # Add timestamp as index if available
        timestamps = self.timestamps
        if timestamps and timestamps[0] is not None:
            df = pd.DataFrame(df_data, index=timestamps)
        else:
            df = pd.DataFrame(df_data)
        
        return df
    
    def slice(self, start: int, end: Optional[int] = None) -> 'MarketData':
        """
        Create a slice of the market data
        
        Args:
            start: Starting index
            end: Ending index (exclusive), if None uses all data from start
            
        Returns:
            New MarketData instance with sliced data
        """
        if end is None:
            end = len(self.data)
        
        return MarketData(
            data=self.data[start:end],
            symbol=self.symbol,
            timeframe=self.timeframe
        )
    
    def latest(self, count: int = 1) -> 'MarketData':
        """
        Get the latest N candles
        
        Args:
            count: Number of latest candles to retrieve
            
        Returns:
            New MarketData instance with latest candles
        """
        return self.slice(-count)
    
    def get_candle(self, index: int) -> OHLCV:
        """
        Get a specific candle by index
        
        Args:
            index: Index of the candle (supports negative indexing)
            
        Returns:
            OHLCV candle data
        """
        return self.data[index]

# Type aliases for convenience
PriceData = Union[List[float], np.ndarray]
OHLCVList = List[OHLCV]

# Utility functions
def create_ohlcv(open_price: float, high_price: float, low_price: float, 
                 close_price: float, volume: float, timestamp: Optional[datetime] = None) -> OHLCV:
    """
    Create an OHLCV instance with validation
    
    Args:
        open_price: Opening price
        high_price: Highest price
        low_price: Lowest price  
        close_price: Closing price
        volume: Trading volume
        timestamp: Optional timestamp
        
    Returns:
        Validated OHLCV instance
        
    Raises:
        ValueError: If price data is invalid
    """
    # Validate price relationships
    if high_price < low_price:
        raise ValueError(f"High price ({high_price}) cannot be less than low price ({low_price})")
    
    if open_price < 0 or close_price < 0:
        raise ValueError("Prices cannot be negative")
    
    if volume < 0:
        raise ValueError("Volume cannot be negative")
    
    return OHLCV(
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        timestamp=timestamp
    )

def generate_test_data(length: int = 100, base_price: float = 100.0, 
                      volatility: float = 0.02) -> MarketData:
    """
    Generate realistic test market data for indicator testing
    
    Args:
        length: Number of candles to generate
        base_price: Starting price
        volatility: Price volatility (standard deviation as fraction)
        
    Returns:
        MarketData with generated OHLCV data
    """
    import random
    from datetime import timedelta
    
    data = []
    current_price = base_price
    current_time = datetime.now()
    
    for i in range(length):
        # Generate realistic OHLC based on random walk
        price_change = random.gauss(0, volatility * current_price)
        new_close = max(0.01, current_price + price_change)  # Ensure positive prices
        
        # Generate realistic high/low around open/close
        high_extra = random.uniform(0, volatility * current_price * 0.5)
        low_reduction = random.uniform(0, volatility * current_price * 0.5)
        
        high = max(current_price, new_close) + high_extra
        low = min(current_price, new_close) - low_reduction
        low = max(0.01, low)  # Ensure positive
        
        # Generate realistic volume
        base_volume = 1000000
        volume_variation = random.uniform(0.5, 2.0)
        volume = base_volume * volume_variation
        
        candle = OHLCV(
            open=current_price,
            high=high,
            low=low,
            close=new_close,
            volume=volume,
            timestamp=current_time + timedelta(minutes=i)
        )
        
        data.append(candle)
        current_price = new_close
    
    return MarketData(
        data=data,
        symbol="TEST/USD",
        timeframe="1m"
    )

__all__ = [
    'OHLCV',
    'MarketData', 
    'PriceData',
    'OHLCVList',
    'create_ohlcv',
    'generate_test_data'
]
