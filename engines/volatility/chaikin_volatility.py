"""
Chaikin Volatility Indicator
Measures the volatility of a security by comparing high-low spreads over time.
Developed by Marc Chaikin.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


class ChaikinVolatility:
    """
    Chaikin Volatility Indicator
    
    The Chaikin Volatility indicator measures volatility by comparing 
    the difference between high and low prices. It calculates the 
    percentage change in a moving average of the high-low spread.
    
    Formula:
    1. High-Low Spread = High - Low
    2. EMA of High-Low Spread
    3. Chaikin Volatility = ((Current EMA - Previous EMA) / Previous EMA) * 100
    """
    
    def __init__(self, period: int = 10, rate_of_change_period: int = 10):
        """
        Initialize Chaikin Volatility
        
        Args:
            period: Period for EMA calculation (default: 10)
            rate_of_change_period: Period for rate of change calculation (default: 10)
        """
        self.period = period
        self.rate_of_change_period = rate_of_change_period
        self.name = "Chaikin Volatility"
        
    def calculate(self, 
                 high: Union[pd.Series, np.ndarray], 
                 low: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate Chaikin Volatility
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            pd.Series: Chaikin Volatility values
        """
        # Convert to pandas Series if numpy arrays
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
            
        # Calculate high-low spread
        hl_spread = high - low
        
        # Calculate EMA of high-low spread
        ema_spread = hl_spread.ewm(span=self.period, adjust=False).mean()
        
        # Calculate rate of change
        chaikin_volatility = ((ema_spread - ema_spread.shift(self.rate_of_change_period)) / 
                             ema_spread.shift(self.rate_of_change_period)) * 100
        
        return chaikin_volatility
    
    def calculate_with_signals(self, 
                              high: Union[pd.Series, np.ndarray], 
                              low: Union[pd.Series, np.ndarray],
                              threshold: float = 10.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Chaikin Volatility with buy/sell signals
        
        Args:
            high: High prices
            low: Low prices
            threshold: Threshold for high/low volatility signals
            
        Returns:
            Tuple[pd.Series, pd.Series]: (chaikin_volatility, signals)
                signals: 1 for high volatility, -1 for low volatility, 0 for neutral
        """
        chaikin_vol = self.calculate(high, low)
        
        # Generate signals based on volatility threshold
        signals = pd.Series(0, index=chaikin_vol.index, dtype=int)
        signals[chaikin_vol > threshold] = 1  # High volatility
        signals[chaikin_vol < -threshold] = -1  # Low volatility
        
        return chaikin_vol, signals
    
    def get_overbought_oversold(self, 
                               high: Union[pd.Series, np.ndarray], 
                               low: Union[pd.Series, np.ndarray],
                               high_threshold: float = 20.0,
                               low_threshold: float = -20.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Get overbought/oversold conditions
        
        Args:
            high: High prices
            low: Low prices
            high_threshold: Threshold for high volatility (default: 20.0)
            low_threshold: Threshold for low volatility (default: -20.0)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (chaikin_volatility, overbought, oversold)
        """
        chaikin_vol = self.calculate(high, low)
        
        overbought = chaikin_vol > high_threshold
        oversold = chaikin_vol < low_threshold
        
        return chaikin_vol, overbought, oversold
    
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
        if len(high) < 2:
            return False
        return True


def chaikin_volatility(high: Union[pd.Series, np.ndarray], 
                      low: Union[pd.Series, np.ndarray],
                      period: int = 10,
                      rate_of_change_period: int = 10) -> pd.Series:
    """
    Calculate Chaikin Volatility (functional interface)
    
    Args:
        high: High prices
        low: Low prices
        period: Period for EMA calculation (default: 10)
        rate_of_change_period: Period for rate of change calculation (default: 10)
        
    Returns:
        pd.Series: Chaikin Volatility values
    """
    indicator = ChaikinVolatility(period=period, rate_of_change_period=rate_of_change_period)
    return indicator.calculate(high, low)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create high/low with some spread
    close_prices = pd.Series(prices, index=dates)
    high_prices = close_prices * (1 + np.random.uniform(0.001, 0.02, 100))
    low_prices = close_prices * (1 - np.random.uniform(0.001, 0.02, 100))
    
    # Test the indicator
    cv = ChaikinVolatility(period=10, rate_of_change_period=10)
    
    print("Testing Chaikin Volatility Indicator")
    print("=" * 40)
    
    # Basic calculation
    volatility = cv.calculate(high_prices, low_prices)
    print(f"Last 10 Chaikin Volatility values:")
    print(volatility.tail(10).round(4))
    
    # With signals
    volatility, signals = cv.calculate_with_signals(high_prices, low_prices, threshold=5.0)
    print(f"\nSignals summary:")
    print(f"High volatility periods: {(signals == 1).sum()}")
    print(f"Low volatility periods: {(signals == -1).sum()}")
    print(f"Neutral periods: {(signals == 0).sum()}")
    
    # Overbought/oversold
    volatility, overbought, oversold = cv.get_overbought_oversold(
        high_prices, low_prices, high_threshold=10.0, low_threshold=-10.0
    )
    print(f"\nOverbought periods: {overbought.sum()}")
    print(f"Oversold periods: {oversold.sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"Period: {cv.period}")
    print(f"Rate of Change Period: {cv.rate_of_change_period}")
    print(f"Indicator Name: {cv.name}")
