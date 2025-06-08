"""
Mass Index Indicator
A technical analysis indicator used to identify trend reversals.
Developed by Donald Dorsey.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


class MassIndex:
    """
    Mass Index Indicator
    
    The Mass Index uses the high-low range to identify trend reversals 
    based on the expansion and contraction of the trading range. It is 
    designed to identify when the range widens enough to suggest a 
    trend reversal.
    
    Formula:
    1. Single EMA = EMA(High - Low, period)
    2. Double EMA = EMA(Single EMA, period)
    3. EMA Ratio = Single EMA / Double EMA
    4. Mass Index = Sum of EMA Ratio over sum_period
    """
    
    def __init__(self, period: int = 9, sum_period: int = 25):
        """
        Initialize Mass Index
        
        Args:
            period: Period for EMA calculations (default: 9)
            sum_period: Period for summing the ratios (default: 25)
        """
        self.period = period
        self.sum_period = sum_period
        self.name = "Mass Index"
        
    def calculate(self, 
                 high: Union[pd.Series, np.ndarray], 
                 low: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate Mass Index
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            pd.Series: Mass Index values
        """
        # Convert to pandas Series if numpy arrays
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
            
        # Calculate high-low range
        hl_range = high - low
        
        # Calculate single EMA of the range
        single_ema = hl_range.ewm(span=self.period, adjust=False).mean()
        
        # Calculate double EMA (EMA of the single EMA)
        double_ema = single_ema.ewm(span=self.period, adjust=False).mean()
        
        # Calculate EMA ratio
        ema_ratio = single_ema / double_ema
        
        # Calculate Mass Index (sum of EMA ratios over sum_period)
        mass_index = ema_ratio.rolling(window=self.sum_period).sum()
        
        return mass_index
    
    def calculate_with_signals(self, 
                              high: Union[pd.Series, np.ndarray], 
                              low: Union[pd.Series, np.ndarray],
                              reversal_threshold: float = 27.0,
                              setup_threshold: float = 26.5) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Mass Index with reversal signals
        
        Args:
            high: High prices
            low: Low prices
            reversal_threshold: Threshold above which reversal is likely (default: 27.0)
            setup_threshold: Threshold for setup signal (default: 26.5)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (mass_index, signals)
                signals: 1 for potential reversal, 0 for neutral
        """
        mass_index = self.calculate(high, low)
        
        # Generate reversal signals
        signals = pd.Series(0, index=mass_index.index, dtype=int)
        
        # Signal when Mass Index crosses above reversal threshold and then falls back below
        above_threshold = mass_index > reversal_threshold
        setup_condition = mass_index > setup_threshold
        
        # Look for peaks above threshold followed by decline
        for i in range(1, len(mass_index)):
            if (above_threshold.iloc[i-1] and 
                mass_index.iloc[i] < reversal_threshold and 
                mass_index.iloc[i-1] >= reversal_threshold):
                signals.iloc[i] = 1
        
        return mass_index, signals
    
    def get_reversal_zones(self, 
                          high: Union[pd.Series, np.ndarray], 
                          low: Union[pd.Series, np.ndarray],
                          threshold_low: float = 26.5,
                          threshold_high: float = 27.0) -> Tuple[pd.Series, pd.Series]:
        """
        Identify reversal zones
        
        Args:
            high: High prices
            low: Low prices
            threshold_low: Lower threshold for reversal zone
            threshold_high: Upper threshold for reversal zone
            
        Returns:
            Tuple[pd.Series, pd.Series]: (mass_index, in_reversal_zone)
        """
        mass_index = self.calculate(high, low)
        
        # Identify when Mass Index is in reversal zone
        in_reversal_zone = (mass_index >= threshold_low) & (mass_index <= threshold_high)
        
        return mass_index, in_reversal_zone
    
    def get_bulge_signals(self, 
                         high: Union[pd.Series, np.ndarray], 
                         low: Union[pd.Series, np.ndarray],
                         bulge_threshold: float = 27.0,
                         min_duration: int = 2) -> Tuple[pd.Series, pd.Series]:
        """
        Identify bulge patterns (sustained high Mass Index values)
        
        Args:
            high: High prices
            low: Low prices
            bulge_threshold: Threshold for bulge identification
            min_duration: Minimum duration for bulge pattern
            
        Returns:
            Tuple[pd.Series, pd.Series]: (mass_index, bulge_signals)
        """
        mass_index = self.calculate(high, low)
        
        # Identify bulge patterns
        above_threshold = mass_index > bulge_threshold
        bulge_signals = pd.Series(0, index=mass_index.index, dtype=int)
        
        # Count consecutive periods above threshold
        consecutive_count = 0
        for i in range(len(above_threshold)):
            if above_threshold.iloc[i]:
                consecutive_count += 1
                if consecutive_count >= min_duration:
                    bulge_signals.iloc[i] = 1
            else:
                consecutive_count = 0
        
        return mass_index, bulge_signals
    
    @staticmethod
    def validate_data(high: Union[pd.Series, np.ndarray], 
                     low: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate input data
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            bool: True if data is valid
        """
        if len(high) != len(low):
            return False
        if len(high) < 10:  # Need sufficient data for calculations
            return False
        return True


def mass_index(high: Union[pd.Series, np.ndarray], 
               low: Union[pd.Series, np.ndarray],
               period: int = 9,
               sum_period: int = 25) -> pd.Series:
    """
    Calculate Mass Index (functional interface)
    
    Args:
        high: High prices
        low: Low prices
        period: Period for EMA calculations (default: 9)
        sum_period: Period for summing the ratios (default: 25)
        
    Returns:
        pd.Series: Mass Index values
    """
    indicator = MassIndex(period=period, sum_period=sum_period)
    return indicator.calculate(high, low)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data with trend changes
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    
    # Add some trend changes and volatility spikes
    for i in range(20, 25):
        returns[i] = np.random.normal(0, 0.05)  # Volatility spike
    for i in range(50, 55):
        returns[i] = np.random.normal(0, 0.04)  # Another volatility spike
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create high/low with varying spreads
    close_prices = pd.Series(prices, index=dates)
    spreads = np.random.uniform(0.01, 0.03, 100)
    spreads[20:25] = np.random.uniform(0.03, 0.06, 5)  # Wider spreads during volatility
    spreads[50:55] = np.random.uniform(0.025, 0.05, 5)
    
    high_prices = close_prices * (1 + spreads/2)
    low_prices = close_prices * (1 - spreads/2)
    
    # Test the indicator
    mi = MassIndex(period=9, sum_period=25)
    
    print("Testing Mass Index Indicator")
    print("=" * 40)
    
    # Basic calculation
    mass_idx = mi.calculate(high_prices, low_prices)
    print(f"Last 10 Mass Index values:")
    print(mass_idx.tail(10).round(4))
    
    # Statistical summary
    print(f"\nMass Index Statistics:")
    print(f"Mean: {mass_idx.mean():.4f}")
    print(f"Max: {mass_idx.max():.4f}")
    print(f"Min: {mass_idx.min():.4f}")
    print(f"Std: {mass_idx.std():.4f}")
    
    # Reversal signals
    mass_idx, signals = mi.calculate_with_signals(high_prices, low_prices)
    print(f"\nReversal signals: {signals.sum()}")
    
    # Reversal zones
    mass_idx, in_zone = mi.get_reversal_zones(high_prices, low_prices)
    print(f"Periods in reversal zone: {in_zone.sum()}")
    
    # Bulge patterns
    mass_idx, bulge_signals = mi.get_bulge_signals(high_prices, low_prices)
    print(f"Bulge pattern periods: {bulge_signals.sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"EMA Period: {mi.period}")
    print(f"Sum Period: {mi.sum_period}")
    print(f"Indicator Name: {mi.name}")
