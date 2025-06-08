"""
Ease of Movement (EMV) Indicator
A volume-based oscillator that is designed to quantify the "ease" of price movement.
Developed by Richard Arms Jr.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType


class EaseOfMovement(TechnicalIndicator):
    """
    Ease of Movement (EMV) Indicator
    
    The Ease of Movement indicator combines price and volume to assess how 
    easily a price can move. It is based on the assumption that prices 
    advance with relative ease when volume is light and the high-low 
    spread (range) is large.
    
    Formula:
    1. Distance Moved = ((High + Low) / 2) - ((Prior High + Prior Low) / 2)
    2. Box Height = Volume / (High - Low)    3. 1-Period EMV = Distance Moved / Box Height
    4. EMV = N-Period Simple Moving Average of 1-Period EMV
    """
    
    def __init__(self, period: int = 14, config=None):
        """
        Initialize Ease of Movement
        
        Args:
            period: Period for moving average smoothing (default: 14)
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
            
        self.name = "Ease of Movement"
        
    def calculate(self, data=None, high=None, low=None, close=None, volume=None) -> pd.Series:
        """
        Calculate Ease of Movement
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame with OHLCV data
        high : pd.Series or np.ndarray, optional
            High prices
        low : pd.Series or np.ndarray, optional
            Low prices
        close : pd.Series or np.ndarray, optional
            Close prices (not used in calculation but included for API consistency)
        volume : pd.Series or np.ndarray, optional
            Volume data
            
        Returns:
        --------
        pd.Series
            EMV values
        """
        # Handle case where data is a DataFrame
        if isinstance(data, pd.DataFrame):
            if 'high' in data.columns and 'low' in data.columns and 'volume' in data.columns:
                high = data['high']
                low = data['low']
                volume = data['volume']
            elif len(data.columns) >= 5:  # Assume standard OHLCV format
                high = data.iloc[:, 1]  # Assuming High is the 2nd column
                low = data.iloc[:, 2]  # Assuming Low is the 3rd column
                volume = data.iloc[:, 4]  # Assuming Volume is the 5th column
        
        # Validate inputs
        if high is None or low is None or volume is None:
            raise ValueError("Missing required inputs: high, low, and volume data required")
            
        # Convert to pandas Series if numpy arrays
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
            
        # Calculate distance moved (change in mid-point)
        mid_point = (high + low) / 2
        distance_moved = mid_point.diff()
        
        # Calculate box height (volume divided by range)
        price_range = high - low
        # Avoid division by zero
        price_range = price_range.replace(0, np.nan)
        box_height = volume / price_range
        
        # Calculate 1-period EMV
        # Avoid division by zero or infinity
        box_height = box_height.replace([np.inf, -np.inf], np.nan)
        raw_emv = distance_moved / box_height
        
        # Handle extreme values
        raw_emv = raw_emv.replace([np.inf, -np.inf], np.nan)
        
        # Calculate smoothed EMV using simple moving average
        emv = raw_emv.rolling(window=self.period).mean()
        
        return emv
    
    def calculate_with_signals(self, 
                              high: Union[pd.Series, np.ndarray], 
                              low: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              signal_threshold: float = 0.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate EMV with buy/sell signals
        
        Args:
            high: High prices
            low: Low prices
            volume: Volume
            signal_threshold: Threshold for signal generation (default: 0.0)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (emv, signals)
                signals: 1 for bullish (easy upward movement), -1 for bearish, 0 for neutral
        """
        emv = self.calculate(high, low, volume)
        
        # Generate signals based on EMV crossing threshold
        signals = pd.Series(0, index=emv.index, dtype=int)
        signals[emv > signal_threshold] = 1   # Bullish (easy upward movement)
        signals[emv < -signal_threshold] = -1  # Bearish (easy downward movement)
        
        return emv, signals
    
    def calculate_divergence(self, 
                           high: Union[pd.Series, np.ndarray], 
                           low: Union[pd.Series, np.ndarray],
                           volume: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           lookback: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Identify bullish and bearish divergences between EMV and price
        
        Args:
            high: High prices
            low: Low prices
            volume: Volume
            close: Closing prices
            lookback: Lookback period for divergence detection
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (emv, bullish_divergence, bearish_divergence)
        """
        emv = self.calculate(high, low, volume)
        
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        bullish_div = pd.Series(False, index=emv.index)
        bearish_div = pd.Series(False, index=emv.index)
        
        # Look for divergences over lookback period
        for i in range(lookback, len(close)):
            # Get segments for analysis
            price_segment = close.iloc[i-lookback:i+1]
            emv_segment = emv.iloc[i-lookback:i+1]
            
            if emv_segment.isna().all() or price_segment.isna().all():
                continue
                
            # Find recent lows and highs
            recent_price_low = price_segment.min()
            recent_price_high = price_segment.max()
            recent_emv_low = emv_segment.min()
            recent_emv_high = emv_segment.max()
            
            current_price = close.iloc[i]
            current_emv = emv.iloc[i]
            
            # Bullish divergence: price makes lower low, EMV makes higher low
            if (current_price <= recent_price_low and 
                current_emv > recent_emv_low and
                not pd.isna(current_emv)):
                bullish_div.iloc[i] = True
                
            # Bearish divergence: price makes higher high, EMV makes lower high
            if (current_price >= recent_price_high and 
                current_emv < recent_emv_high and
                not pd.isna(current_emv)):
                bearish_div.iloc[i] = True
        
        return emv, bullish_div, bearish_div
    
    def get_trend_confirmation(self, 
                              high: Union[pd.Series, np.ndarray], 
                              low: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              close: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        Get trend confirmation signals
        
        Args:
            high: High prices
            low: Low prices
            volume: Volume
            close: Closing prices
            
        Returns:
            Tuple[pd.Series, pd.Series]: (emv, trend_confirmation)
                trend_confirmation: 1 for confirmed uptrend, -1 for confirmed downtrend, 0 for no confirmation
        """
        emv = self.calculate(high, low, volume)
        
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        # Calculate price trend
        price_change = close.diff()
        
        # Generate trend confirmation
        trend_confirmation = pd.Series(0, index=emv.index, dtype=int)
        
        # Confirmed uptrend: positive price change and positive EMV
        uptrend_confirmed = (price_change > 0) & (emv > 0)
        trend_confirmation[uptrend_confirmed] = 1
        
        # Confirmed downtrend: negative price change and negative EMV
        downtrend_confirmed = (price_change < 0) & (emv < 0)
        trend_confirmation[downtrend_confirmed] = -1
        
        return emv, trend_confirmation
    
    @staticmethod
    def validate_data(high: Union[pd.Series, np.ndarray], 
                     low: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate input data
        
        Args:
            high: High prices
            low: Low prices
            volume: Volume
            
        Returns:
            bool: True if data is valid
        """
        if not (len(high) == len(low) == len(volume)):
            return False
        if len(high) < 2:
            return False
        return True
    
    def generate_signal(self, data: MarketData, period_start=None, period_end=None) -> List[IndicatorSignal]:
        """
        Generate trading signals based on Ease of Movement values.
        
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
        # Calculate Ease of Movement
        df = data.df
        emv = self.calculate(
            high=df['high'],
            low=df['low'],
            volume=df['volume']
        )
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_emv = emv.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_emv = emv
        
        # Generate signals only where we have sufficient data
        for i in range(self.period + 1, len(working_emv)):
            current_idx = working_df.index[i]
            current_price = working_df['close'].iloc[i]
            current_emv = working_emv.iloc[i]
            prev_emv = working_emv.iloc[i-1]
            
            # Buy signal: EMV crosses above zero
            if prev_emv <= 0 and current_emv > 0:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_emv,
                        price=current_price,
                        strength=min(abs(current_emv) * 10, 1.0),  # Normalize strength
                        indicator_name=f"{self.name} ({self.period})"
                    )
                )
            
            # Sell signal: EMV crosses below zero
            elif prev_emv >= 0 and current_emv < 0:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.SELL,
                        indicator_value=current_emv,
                        price=current_price,
                        strength=min(abs(current_emv) * 10, 1.0),  # Normalize strength
                        indicator_name=f"{self.name} ({self.period})"
                    )
                )
            
            # Strong movement signal: EMV makes a large move in either direction
            elif abs(current_emv) > 0.1 and abs(current_emv - prev_emv) > 0.05:
                signal_type = SignalType.BUY if current_emv > 0 else SignalType.SELL
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=signal_type,
                        indicator_value=current_emv,
                        price=current_price,
                        strength=min(abs(current_emv) * 5, 1.0),  # Normalize strength
                        indicator_name=f"{self.name} Strong ({self.period})"
                    )
                )
        
        return signals


def ease_of_movement(high: Union[pd.Series, np.ndarray], 
                    low: Union[pd.Series, np.ndarray],
                    volume: Union[pd.Series, np.ndarray],
                    period: int = 14) -> pd.Series:
    """
    Calculate Ease of Movement (functional interface)
    
    Args:
        high: High prices
        low: Low prices
        volume: Volume
        period: Period for moving average smoothing (default: 14)
        
    Returns:
        pd.Series: EMV values
    """
    indicator = EaseOfMovement(period=period)
    return indicator.calculate(high, low, volume)


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
    
    close_prices = pd.Series(prices, index=dates)
    
    # Create high/low with realistic spreads
    spreads = np.random.uniform(0.005, 0.02, 100)
    high_prices = close_prices * (1 + spreads)
    low_prices = close_prices * (1 - spreads)
    
    # Generate volume with some correlation to price movements
    base_volume = 1000000
    volume_multipliers = 1 + np.abs(returns) * 2  # Higher volume on bigger moves
    volumes = pd.Series(base_volume * volume_multipliers, index=dates)
    
    # Test the indicator
    emv = EaseOfMovement(period=14)
    
    print("Testing Ease of Movement Indicator")
    print("=" * 40)
    
    # Basic calculation
    emv_values = emv.calculate(high_prices, low_prices, volumes)
    print(f"Last 10 EMV values:")
    print(emv_values.tail(10).round(6))
    
    # Statistical summary
    valid_emv = emv_values.dropna()
    if len(valid_emv) > 0:
        print(f"\nEMV Statistics:")
        print(f"Mean: {valid_emv.mean():.6f}")
        print(f"Max: {valid_emv.max():.6f}")
        print(f"Min: {valid_emv.min():.6f}")
        print(f"Std: {valid_emv.std():.6f}")
    
    # Signals
    emv_values, signals = emv.calculate_with_signals(high_prices, low_prices, volumes)
    print(f"\nSignals summary:")
    print(f"Bullish signals: {(signals == 1).sum()}")
    print(f"Bearish signals: {(signals == -1).sum()}")
    print(f"Neutral signals: {(signals == 0).sum()}")
    
    # Trend confirmation
    emv_values, trend_conf = emv.get_trend_confirmation(high_prices, low_prices, volumes, close_prices)
    print(f"\nTrend confirmation:")
    print(f"Confirmed uptrends: {(trend_conf == 1).sum()}")
    print(f"Confirmed downtrends: {(trend_conf == -1).sum()}")
    print(f"No confirmation: {(trend_conf == 0).sum()}")
    
    # Divergence analysis
    emv_values, bullish_div, bearish_div = emv.calculate_divergence(
        high_prices, low_prices, volumes, close_prices
    )
    print(f"\nDivergence signals:")
    print(f"Bullish divergences: {bullish_div.sum()}")
    print(f"Bearish divergences: {bearish_div.sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"Period: {emv.period}")
    print(f"Indicator Name: {emv.name}")
