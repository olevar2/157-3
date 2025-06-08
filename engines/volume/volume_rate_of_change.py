"""
Volume Rate of Change (VROC)
Measures the rate of change in volume over a specified period.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType


class VolumeRateOfChange(TechnicalIndicator):
    """
    Volume Rate of Change (VROC)
    
    The Volume Rate of Change measures the percentage change in volume 
    compared to volume n periods ago. It helps identify unusual volume 
    activity that may precede significant price movements.
    
    Formula:
    VROC = ((Current Volume - Volume n periods ago) / Volume n periods ago) * 100
    """
    
    def __init__(self, period: int = 12, config=None):
        """
        Initialize Volume Rate of Change
        
        Args:
            period: Lookback period for rate of change calculation (default: 12)
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
            
        self.name = "Volume Rate of Change"
        
    def calculate(self, volume: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate Volume Rate of Change
        
        Args:
            volume: Volume data
            
        Returns:
            pd.Series: VROC values
        """
        # Convert to pandas Series if numpy array
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
            
        # Calculate rate of change
        vroc = volume.pct_change(periods=self.period) * 100
        
        return vroc
    
    def calculate_with_signals(self, 
                              volume: Union[pd.Series, np.ndarray],
                              threshold: float = 25.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate VROC with volume expansion/contraction signals
        
        Args:
            volume: Volume data
            threshold: Threshold for significant volume changes (default: 25.0%)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (vroc, signals)
                signals: 1 for volume expansion, -1 for volume contraction, 0 for normal
        """
        vroc = self.calculate(volume)
        
        # Generate signals based on threshold
        signals = pd.Series(0, index=vroc.index, dtype=int)
        signals[vroc > threshold] = 1    # Volume expansion
        signals[vroc < -threshold] = -1  # Volume contraction
        
        return vroc, signals
    
    def calculate_with_ma(self, 
                         volume: Union[pd.Series, np.ndarray],
                         ma_period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate VROC with its moving average for smoothing
        
        Args:
            volume: Volume data
            ma_period: Moving average period (default: 14)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (vroc, vroc_ma)
        """
        vroc = self.calculate(volume)
        vroc_ma = vroc.rolling(window=ma_period).mean()
        
        return vroc, vroc_ma
    
    def get_volume_momentum(self, 
                           volume: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        Get volume momentum classification
        
        Args:
            volume: Volume data
            
        Returns:
            Tuple[pd.Series, pd.Series]: (vroc, momentum_class)
                momentum_class: 2 for very high, 1 for high, 0 for normal, -1 for low, -2 for very low
        """
        vroc = self.calculate(volume)
        
        # Calculate momentum classes based on VROC quartiles
        momentum_class = pd.Series(0, index=vroc.index, dtype=int)
        
        # Use rolling statistics for dynamic thresholds
        vroc_std = vroc.rolling(window=50).std()
        vroc_mean = vroc.rolling(window=50).mean()
        
        # Define thresholds based on standard deviations from mean
        upper_2 = vroc_mean + 2 * vroc_std
        upper_1 = vroc_mean + 1 * vroc_std
        lower_1 = vroc_mean - 1 * vroc_std
        lower_2 = vroc_mean - 2 * vroc_std
        
        momentum_class[vroc > upper_2] = 2   # Very high volume momentum
        momentum_class[(vroc > upper_1) & (vroc <= upper_2)] = 1  # High volume momentum
        momentum_class[(vroc < lower_1) & (vroc >= lower_2)] = -1  # Low volume momentum
        momentum_class[vroc < lower_2] = -2  # Very low volume momentum
        
        return vroc, momentum_class
    
    def calculate_with_price_confirmation(self, 
                                        volume: Union[pd.Series, np.ndarray],
                                        close: Union[pd.Series, np.ndarray],
                                        price_period: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate VROC with price confirmation analysis
        
        Args:
            volume: Volume data
            close: Closing prices
            price_period: Period for price rate of change (default: same as volume period)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (vroc, price_roc, confirmation)
                confirmation: 1 for bullish confirmation, -1 for bearish confirmation, 0 for no confirmation
        """
        if price_period is None:
            price_period = self.period
            
        vroc = self.calculate(volume)
        
        # Convert close to pandas Series if numpy array
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        # Calculate price rate of change
        price_roc = close.pct_change(periods=price_period) * 100
        
        # Generate confirmation signals
        confirmation = pd.Series(0, index=vroc.index, dtype=int)
        
        # Bullish confirmation: Rising volume with rising prices
        bullish = (vroc > 0) & (price_roc > 0)
        confirmation[bullish] = 1
        
        # Bearish confirmation: Rising volume with falling prices
        bearish = (vroc > 0) & (price_roc < 0)
        confirmation[bearish] = -1
        
        return vroc, price_roc, confirmation
    
    def get_volume_spikes(self, 
                         volume: Union[pd.Series, np.ndarray],
                         spike_threshold: float = 100.0) -> Tuple[pd.Series, pd.Series]:
        """
        Identify volume spikes
        
        Args:
            volume: Volume data
            spike_threshold: Threshold for volume spike identification (default: 100.0%)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (vroc, volume_spikes)
        """
        vroc = self.calculate(volume)
        
        # Identify volume spikes
        volume_spikes = vroc > spike_threshold
        
        return vroc, volume_spikes
    
    def calculate_volume_trend(self, 
                              volume: Union[pd.Series, np.ndarray],
                              trend_period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate volume trend based on VROC moving average
        
        Args:
            volume: Volume data
            trend_period: Period for trend calculation (default: 20)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (vroc, volume_trend)
                volume_trend: 1 for increasing volume trend, -1 for decreasing, 0 for neutral
        """
        vroc = self.calculate(volume)
        
        # Calculate moving average of VROC
        vroc_ma = vroc.rolling(window=trend_period).mean()
        
        # Determine trend direction
        volume_trend = pd.Series(0, index=vroc.index, dtype=int)
        volume_trend[vroc_ma > 0] = 1   # Increasing volume trend
        volume_trend[vroc_ma < 0] = -1  # Decreasing volume trend
        
        return vroc, volume_trend
    
    @staticmethod
    def validate_data(volume: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate input data
        
        Args:
            volume: Volume data
            
        Returns:
            bool: True if data is valid
        """
        if len(volume) < 2:
            return False
        return True
    
    def generate_signal(self, data: MarketData, period_start=None, period_end=None) -> List[IndicatorSignal]:
        """
        Generate trading signals based on Volume Rate of Change.
        
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
        # Calculate Volume Rate of Change
        vroc = self.calculate(data.df['volume'])
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_vroc = vroc.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_vroc = vroc
        
        # Define thresholds for unusual volume
        high_volume_threshold = 50  # 50% increase
        extreme_volume_threshold = 100  # 100% increase
        
        # Generate signals only where we have sufficient data
        for i in range(self.period + 5, len(working_vroc)):
            current_idx = working_df.index[i]
            current_price = working_df['close'].iloc[i]
            current_vroc = working_vroc.iloc[i]
            prev_price = working_df['close'].iloc[i-1]
            price_change = (current_price - prev_price) / prev_price * 100
            
            # Significant volume increase with price up movement
            if current_vroc > high_volume_threshold and price_change > 0:
                signal_strength = min(current_vroc / 200, 1.0)  # Normalize strength
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_vroc,
                        price=current_price,
                        strength=signal_strength,
                        indicator_name=f"{self.name} ({self.period})"
                    )
                )
            
            # Extreme volume increase (potential reversal or acceleration)
            elif current_vroc > extreme_volume_threshold:
                # Strong volume surge often precedes price movements
                # Use with confirmation from price action
                signal_type = SignalType.BUY if price_change > 0 else SignalType.SELL
                signal_strength = min(current_vroc / 200, 1.0)  # Normalize strength
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=signal_type,
                        indicator_value=current_vroc,
                        price=current_price,
                        strength=signal_strength,
                        indicator_name=f"{self.name} Surge ({self.period})"
                    )
                )
            
            # Significant volume decrease (potential exhaustion)
            elif current_vroc < -50 and abs(price_change) < 0.5:
                # Volume drying up with minimal price movement might indicate exhaustion
                # Direction depends on preceding trend
                signal_type = SignalType.SELL if working_df['close'].iloc[i-5:i].pct_change().mean() > 0 else SignalType.BUY
                signal_strength = min(abs(current_vroc) / 100, 1.0)  # Normalize strength
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=signal_type,
                        indicator_value=current_vroc,
                        price=current_price,
                        strength=signal_strength,
                        indicator_name=f"{self.name} Exhaustion ({self.period})"
                    )
                )
        
        return signals
    
    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on Volume Rate of Change for compatibility with indicator registry
        
        Parameters:
        -----------
        data : MarketData
            Market data containing OHLCV information
            
        Returns:
        --------
        IndicatorSignal
            Signal object with buy/sell/neutral recommendation
        """
        if "volume" not in data.data:
            return IndicatorSignal(self.name, SignalType.NEUTRAL, None, None)
            
        volume = data.data["volume"]
        
        # Calculate Volume Rate of Change
        vroc = self.calculate(volume)
        
        # Get the latest value
        last_vroc = vroc.iloc[-1]
        
        # Define thresholds
        high_threshold = 50  # 50% increase
        low_threshold = -25  # 25% decrease
        
        if last_vroc > high_threshold:
            signal_type = SignalType.BUY
            message = f"Volume increased significantly (VROC: {last_vroc:.2f}%), suggesting potential upward movement"
        elif last_vroc < low_threshold:
            signal_type = SignalType.SELL
            message = f"Volume decreased significantly (VROC: {last_vroc:.2f}%), suggesting potential downward movement"
        else:
            signal_type = SignalType.NEUTRAL
            message = f"Volume change is normal (VROC: {last_vroc:.2f}%), no clear signal"
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_type=signal_type,
            message=message,
            plot_data={
                "vroc": vroc.tolist()
            }
        )


def volume_rate_of_change(volume: Union[pd.Series, np.ndarray],
                         period: int = 12) -> pd.Series:
    """
    Calculate Volume Rate of Change (functional interface)
    
    Args:
        volume: Volume data
        period: Lookback period for rate of change calculation (default: 12)
        
    Returns:
        pd.Series: VROC values
    """
    indicator = VolumeRateOfChange(period=period)
    return indicator.calculate(volume)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Generate realistic volume data with patterns
    base_volume = 1000000
    
    # Create volume with trends and spikes
    volume_data = []
    trend = 0
    for i in range(200):
        # Add some trend changes
        if i == 50:
            trend = 0.002  # Increasing volume trend
        elif i == 120:
            trend = -0.001  # Decreasing volume trend
        elif i == 170:
            trend = 0  # Neutral trend
        
        # Base change with trend
        change = np.random.normal(trend, 0.15)
        
        # Add occasional volume spikes
        if i in [30, 80, 150]:
            change += 1.0  # Volume spike
        
        if i == 0:
            volume_data.append(base_volume)
        else:
            new_volume = max(volume_data[-1] * (1 + change), 50000)  # Minimum volume
            volume_data.append(new_volume)
    
    volumes = pd.Series(volume_data, index=dates)
    
    # Generate corresponding price data
    price_returns = np.random.normal(0, 0.02, 200)
    # Add some correlation with volume spikes
    price_returns[30] = 0.05   # Price jump with volume spike
    price_returns[80] = -0.04  # Price drop with volume spike
    price_returns[150] = 0.03  # Price rise with volume spike
    
    prices = [100]
    for ret in price_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    close_prices = pd.Series(prices, index=dates)
    
    # Test the indicator
    vroc = VolumeRateOfChange(period=12)
    
    print("Testing Volume Rate of Change")
    print("=" * 40)
    
    # Basic calculation
    vroc_values = vroc.calculate(volumes)
    print(f"Last 10 VROC values:")
    print(vroc_values.tail(10).round(2))
    
    # Statistical summary
    valid_vroc = vroc_values.dropna()
    if len(valid_vroc) > 0:
        print(f"\nVROC Statistics:")
        print(f"Mean: {valid_vroc.mean():.2f}%")
        print(f"Max: {valid_vroc.max():.2f}%")
        print(f"Min: {valid_vroc.min():.2f}%")
        print(f"Std: {valid_vroc.std():.2f}%")
    
    # Volume expansion/contraction signals
    vroc_values, signals = vroc.calculate_with_signals(volumes, threshold=25.0)
    print(f"\nVolume signals:")
    print(f"Volume expansion periods: {(signals == 1).sum()}")
    print(f"Volume contraction periods: {(signals == -1).sum()}")
    print(f"Normal volume periods: {(signals == 0).sum()}")
    
    # Volume momentum
    vroc_values, momentum_class = vroc.get_volume_momentum(volumes)
    print(f"\nVolume momentum distribution:")
    print(f"Very high momentum: {(momentum_class == 2).sum()}")
    print(f"High momentum: {(momentum_class == 1).sum()}")
    print(f"Normal momentum: {(momentum_class == 0).sum()}")
    print(f"Low momentum: {(momentum_class == -1).sum()}")
    print(f"Very low momentum: {(momentum_class == -2).sum()}")
    
    # Price confirmation
    vroc_values, price_roc, confirmation = vroc.calculate_with_price_confirmation(volumes, close_prices)
    print(f"\nPrice confirmation signals:")
    print(f"Bullish confirmations: {(confirmation == 1).sum()}")
    print(f"Bearish confirmations: {(confirmation == -1).sum()}")
    print(f"No confirmation: {(confirmation == 0).sum()}")
    
    # Volume spikes
    vroc_values, spikes = vroc.get_volume_spikes(volumes, spike_threshold=50.0)
    print(f"\nVolume spikes detected: {spikes.sum()}")
    
    # Volume trend
    vroc_values, vol_trend = vroc.calculate_volume_trend(volumes)
    print(f"\nVolume trend summary:")
    print(f"Increasing volume trend: {(vol_trend == 1).sum()}")
    print(f"Decreasing volume trend: {(vol_trend == -1).sum()}")
    print(f"Neutral volume trend: {(vol_trend == 0).sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"Period: {vroc.period}")
    print(f"Indicator Name: {vroc.name}")
