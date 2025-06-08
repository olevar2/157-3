"""
Negative Volume Index (NVI)
A cumulative indicator that uses the change in volume to decide when to update cumulative values.
Developed by Paul Dysart and refined by Norman Fosback.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType


class NegativeVolumeIndex(TechnicalIndicator):
    """
    Negative Volume Index (NVI)
    
    The NVI assumes that uninformed investors are active on days when volume 
    increases, while informed investors are active when volume decreases. 
    The NVI only changes on days when volume decreases from the previous day.
    
    Formula:
    If Volume[today] < Volume[yesterday]:
        NVI[today] = NVI[yesterday] + ((Close[today] - Close[yesterday]) / Close[yesterday]) * NVI[yesterday]
    Else:
        NVI[today] = NVI[yesterday]
    """
    
    def __init__(self, base_value: float = 1000.0, config=None):
        """
        Initialize Negative Volume Index
        
        Args:
            base_value: Starting value for the NVI (default: 1000.0)
            config : dict, optional
                Configuration dictionary containing parameters
        """
        # Handle config parameter
        if config is not None:
            if isinstance(config, dict):
                self.base_value = config.get('base_value', base_value)
            else:
                self.base_value = base_value
        else:
            self.base_value = base_value
            
        self.name = "Negative Volume Index"
        
    def calculate(self, data=None, high=None, low=None, close=None, volume=None) -> pd.Series:
        """
        Calculate Negative Volume Index
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame with OHLCV data
        high : pd.Series or np.ndarray, optional
            High prices (not used in calculation but included for API consistency)
        low : pd.Series or np.ndarray, optional
            Low prices (not used in calculation but included for API consistency)
        close : pd.Series or np.ndarray, optional
            Close prices
        volume : pd.Series or np.ndarray, optional
            Volume data
            
        Returns:
        --------
        pd.Series
            NVI values
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
            
        # Convert to pandas Series if numpy arrays
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        # Convert to pandas Series if numpy arrays
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
            
        # Initialize NVI series
        nvi = pd.Series(index=close.index, dtype=float)
        nvi.iloc[0] = self.base_value
        
        # Calculate NVI
        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i-1]:
                # Volume decreased - update NVI
                price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change_pct)
            else:
                # Volume increased or stayed same - keep previous NVI
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi
    
    def calculate_with_ma(self, 
                         close: Union[pd.Series, np.ndarray],
                         volume: Union[pd.Series, np.ndarray],
                         ma_period: int = 255) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate NVI with its moving average (typically 255 days for yearly average)
        
        Args:
            close: Closing prices
            volume: Volume
            ma_period: Moving average period (default: 255)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (nvi, nvi_ma)
        """
        nvi = self.calculate(close, volume)
        nvi_ma = nvi.rolling(window=ma_period).mean()
        
        return nvi, nvi_ma
    
    def calculate_with_signals(self, 
                              close: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              ma_period: int = 255) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate NVI with buy/sell signals based on moving average crossover
        
        Args:
            close: Closing prices
            volume: Volume
            ma_period: Moving average period for signal generation (default: 255)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (nvi, nvi_ma, signals)
                signals: 1 for bullish (NVI > MA), -1 for bearish (NVI < MA), 0 for neutral
        """
        nvi, nvi_ma = self.calculate_with_ma(close, volume, ma_period)
        
        # Generate signals based on NVI vs its moving average
        signals = pd.Series(0, index=nvi.index, dtype=int)
        signals[nvi > nvi_ma] = 1   # Bullish
        signals[nvi < nvi_ma] = -1  # Bearish
        
        return nvi, nvi_ma, signals
    
    def calculate_divergence(self, 
                           close: Union[pd.Series, np.ndarray],
                           volume: Union[pd.Series, np.ndarray],
                           lookback: int = 50) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Identify bullish and bearish divergences between NVI and price
        
        Args:
            close: Closing prices
            volume: Volume
            lookback: Lookback period for divergence detection
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (nvi, bullish_divergence, bearish_divergence)
        """
        nvi = self.calculate(close, volume)
        
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        bullish_div = pd.Series(False, index=nvi.index)
        bearish_div = pd.Series(False, index=nvi.index)
        
        # Look for divergences over lookback period
        for i in range(lookback, len(close)):
            # Get segments for analysis
            price_segment = close.iloc[i-lookback:i+1]
            nvi_segment = nvi.iloc[i-lookback:i+1]
            
            # Find recent extremes
            price_low_idx = price_segment.idxmin()
            price_high_idx = price_segment.idxmax()
            nvi_low_idx = nvi_segment.idxmin()
            nvi_high_idx = nvi_segment.idxmax()
            
            current_price = close.iloc[i]
            current_nvi = nvi.iloc[i]
            
            # Bullish divergence: price makes lower low, NVI makes higher low
            if (current_price <= price_segment.loc[price_low_idx] and 
                current_nvi > nvi_segment.loc[nvi_low_idx]):
                bullish_div.iloc[i] = True
                
            # Bearish divergence: price makes higher high, NVI makes lower high
            if (current_price >= price_segment.loc[price_high_idx] and 
                current_nvi < nvi_segment.loc[nvi_high_idx]):
                bearish_div.iloc[i] = True
        
        return nvi, bullish_div, bearish_div
    
    def get_bear_market_probability(self, 
                                   close: Union[pd.Series, np.ndarray],
                                   volume: Union[pd.Series, np.ndarray],
                                   ma_period: int = 255) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate bear market probability based on NVI position relative to MA
        According to Fosback's research, when NVI is below its MA, there's a 95% chance of a bear market
        
        Args:
            close: Closing prices
            volume: Volume
            ma_period: Moving average period (default: 255)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (nvi, nvi_ma, bear_market_probability)
        """
        nvi, nvi_ma = self.calculate_with_ma(close, volume, ma_period)
        
        # Calculate bear market probability
        bear_market_prob = pd.Series(0.05, index=nvi.index)  # 5% when above MA
        bear_market_prob[nvi < nvi_ma] = 0.95  # 95% when below MA
        
        return nvi, nvi_ma, bear_market_prob
    
    def get_trend_strength(self, 
                          close: Union[pd.Series, np.ndarray],
                          volume: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate trend strength based on NVI momentum
        
        Args:
            close: Closing prices
            volume: Volume
            
        Returns:
            Tuple[pd.Series, pd.Series]: (nvi, trend_strength)
        """
        nvi = self.calculate(close, volume)
        
        # Calculate rate of change in NVI
        nvi_roc = nvi.pct_change(periods=20) * 100  # 20-period rate of change
        
        # Normalize trend strength
        trend_strength = pd.Series(0, index=nvi.index, dtype=float)
        trend_strength[nvi_roc > 2] = 1    # Strong uptrend
        trend_strength[(nvi_roc > 0) & (nvi_roc <= 2)] = 0.5  # Weak uptrend
        trend_strength[(nvi_roc < 0) & (nvi_roc >= -2)] = -0.5  # Weak downtrend
        trend_strength[nvi_roc < -2] = -1   # Strong downtrend
        
        return nvi, trend_strength
    
    def generate_signal(self, data):
        """
        Generate trading signals based on NVI.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            dict: Signal information
        """
        try:
            if 'close' not in data.columns or 'volume' not in data.columns:
                return {'signal': 'HOLD', 'strength': 0.0, 'nvi': None}
                
            close = data['close']
            volume = data['volume']
            
            nvi = self.calculate(close, volume)
            if nvi.empty or len(nvi) < 2:
                return {'signal': 'HOLD', 'strength': 0.0, 'nvi': None}
            
            current_nvi = nvi.iloc[-1]
            prev_nvi = nvi.iloc[-2]
            
            # Simple trend-based signal
            if current_nvi > prev_nvi:
                signal = 'BUY'
                strength = min(abs(current_nvi - prev_nvi) / prev_nvi * 100, 1.0)
            elif current_nvi < prev_nvi:
                signal = 'SELL'
                strength = min(abs(prev_nvi - current_nvi) / prev_nvi * 100, 1.0)
            else:
                signal = 'HOLD'
                strength = 0.0
            
            return {
                'signal': signal,
                'strength': strength,
                'nvi': current_nvi,
                'trend': 'UP' if current_nvi > prev_nvi else 'DOWN' if current_nvi < prev_nvi else 'FLAT'
            }
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating NVI signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'nvi': None}
    
    def generate_signal(self, data: MarketData, period_start=None, period_end=None) -> List[IndicatorSignal]:
        """
        Generate trading signals based on Negative Volume Index.
        
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
        # Calculate Negative Volume Index
        nvi = self.calculate(data=data.df)
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_nvi = nvi.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_nvi = nvi
        
        # Calculate moving average for signal generation (common practice with NVI)
        ema_period = 255  # Approximately 1 year of trading days, standard for NVI
        if len(working_nvi) >= ema_period:
            nvi_ema = working_nvi.ewm(span=ema_period, min_periods=ema_period).mean()
            
            # Generate signals only where we have sufficient data
            for i in range(ema_period + 5, len(working_nvi)):
                current_idx = working_df.index[i]
                current_price = working_df['close'].iloc[i]
                current_nvi = working_nvi.iloc[i]
                current_nvi_ema = nvi_ema.iloc[i]
                prev_nvi = working_nvi.iloc[i-1]
                prev_nvi_ema = nvi_ema.iloc[i-1]
                
                # Buy signal: NVI crosses above its long-term EMA
                if prev_nvi <= prev_nvi_ema and current_nvi > current_nvi_ema:
                    signals.append(
                        IndicatorSignal(
                            timestamp=current_idx,
                            signal_type=SignalType.BUY,
                            indicator_value=current_nvi,
                            price=current_price,
                            strength=0.7,  # NVI is considered a strong signal for informed buying
                            indicator_name=self.name
                        )
                    )
                
                # Sell signal: NVI crosses below its long-term EMA
                elif prev_nvi >= prev_nvi_ema and current_nvi < current_nvi_ema:
                    signals.append(
                        IndicatorSignal(
                            timestamp=current_idx,
                            signal_type=SignalType.SELL,
                            indicator_value=current_nvi,
                            price=current_price,
                            strength=0.6,  # Slightly less reliable for selling
                            indicator_name=self.name
                        )
                    )
                
                # Extreme reading: NVI substantially above/below its EMA
                # This indicates significant informed trading activity
                if abs(current_nvi - current_nvi_ema) / current_nvi_ema > 0.10:  # 10% deviation
                    signal_type = SignalType.BUY if current_nvi > current_nvi_ema else SignalType.SELL
                    strength = min(abs(current_nvi - current_nvi_ema) / current_nvi_ema, 1.0)
                    signals.append(
                        IndicatorSignal(
                            timestamp=current_idx,
                            signal_type=signal_type,
                            indicator_value=current_nvi,
                            price=current_price,
                            strength=strength,
                            indicator_name=f"{self.name} Extreme"
                        )
                    )
        
        return signals
    
    @staticmethod
    def validate_data(close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate input data
        
        Args:
            close: Closing prices
            volume: Volume
            
        Returns:
            bool: True if data is valid
        """
        if len(close) != len(volume):
            return False
        if len(close) < 2:
            return False
        return True
    
    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on Negative Volume Index for compatibility with indicator registry
        
        Parameters:
        -----------
        data : MarketData
            Market data containing OHLCV information
            
        Returns:
        --------
        IndicatorSignal
            Signal object with buy/sell/neutral recommendation
        """
        if not all(key in data.data for key in ["close", "volume"]):
            return IndicatorSignal(self.name, SignalType.NEUTRAL, None, None)
            
        close = data.data["close"]
        volume = data.data["volume"]
        
        # Calculate NVI and a 255-day EMA for signal generation
        nvi = self.calculate(close=close, volume=volume)
        nvi_ema = nvi.ewm(span=255, adjust=False).mean()  # Common lookback period for NVI
        
        # Determine signal based on NVI vs its long-term EMA
        last_nvi = nvi.iloc[-1]
        last_nvi_ema = nvi_ema.iloc[-1]
        
        if last_nvi > last_nvi_ema:
            signal_type = SignalType.BUY
            message = f"NVI ({last_nvi:.2f}) above its 255-day EMA ({last_nvi_ema:.2f}), suggesting bullish smart money activity"
        elif last_nvi < last_nvi_ema:
            signal_type = SignalType.SELL
            message = f"NVI ({last_nvi:.2f}) below its 255-day EMA ({last_nvi_ema:.2f}), suggesting bearish smart money activity"
        else:
            signal_type = SignalType.NEUTRAL
            message = f"NVI ({last_nvi:.2f}) at its 255-day EMA ({last_nvi_ema:.2f}), no clear signal"
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_type=signal_type,
            message=message,
            plot_data={
                "nvi": nvi.tolist(),
                "nvi_ema": nvi_ema.tolist()
            }
        )
    
def negative_volume_index(close: Union[pd.Series, np.ndarray],
                         volume: Union[pd.Series, np.ndarray],
                         base_value: float = 1000.0) -> pd.Series:
    """
    Calculate Negative Volume Index (functional interface)
    
    Args:
        close: Closing prices
        volume: Volume
        base_value: Starting value for the NVI (default: 1000.0)
        
    Returns:
        pd.Series: NVI values
    """
    indicator = NegativeVolumeIndex(base_value=base_value)
    return indicator.calculate(close, volume)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')  # Longer period for MA
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0, 0.015, 300)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    close_prices = pd.Series(prices, index=dates)
    
    # Generate volume with varying patterns
    base_volume = 1000000
    volume_changes = np.random.normal(0, 0.3, 300)
    volumes = [base_volume]
    for change in volume_changes[1:]:
        volumes.append(max(volumes[-1] * (1 + change), 100000))  # Minimum volume
    
    volumes = pd.Series(volumes, index=dates)
    
    # Test the indicator
    nvi = NegativeVolumeIndex(base_value=1000.0)
    
    print("Testing Negative Volume Index")
    print("=" * 40)
    
    # Basic calculation
    nvi_values = nvi.calculate(close_prices, volumes)
    print(f"Last 10 NVI values:")
    print(nvi_values.tail(10).round(4))
    
    # Statistical summary
    print(f"\nNVI Statistics:")
    print(f"Start Value: {nvi_values.iloc[0]:.4f}")
    print(f"End Value: {nvi_values.iloc[-1]:.4f}")
    print(f"Total Return: {((nvi_values.iloc[-1] / nvi_values.iloc[0]) - 1) * 100:.2f}%")
    print(f"Max: {nvi_values.max():.4f}")
    print(f"Min: {nvi_values.min():.4f}")
    
    # With moving average
    nvi_values, nvi_ma = nvi.calculate_with_ma(close_prices, volumes, ma_period=50)  # Shorter MA for testing
    print(f"\nNVI vs MA (last 5 values):")
    for i in range(-5, 0):
        print(f"NVI: {nvi_values.iloc[i]:.4f}, MA: {nvi_ma.iloc[i]:.4f}")
    
    # Signals
    nvi_values, nvi_ma, signals = nvi.calculate_with_signals(close_prices, volumes, ma_period=50)
    print(f"\nSignals summary:")
    print(f"Bullish periods: {(signals == 1).sum()}")
    print(f"Bearish periods: {(signals == -1).sum()}")
    print(f"Neutral periods: {(signals == 0).sum()}")
    
    # Bear market probability
    nvi_values, nvi_ma, bear_prob = nvi.get_bear_market_probability(close_prices, volumes, ma_period=50)
    current_bear_prob = bear_prob.iloc[-1]
    print(f"\nCurrent bear market probability: {current_bear_prob*100:.1f}%")
    
    # Trend strength
    nvi_values, trend_strength = nvi.get_trend_strength(close_prices, volumes)
    current_trend = trend_strength.iloc[-1]
    trend_desc = {1: "Strong Uptrend", 0.5: "Weak Uptrend", 0: "Neutral", 
                  -0.5: "Weak Downtrend", -1: "Strong Downtrend"}
    print(f"Current trend strength: {trend_desc.get(current_trend, 'Unknown')}")
    
    # Divergence analysis
    nvi_values, bullish_div, bearish_div = nvi.calculate_divergence(close_prices, volumes, lookback=30)
    print(f"\nDivergence signals:")
    print(f"Bullish divergences: {bullish_div.sum()}")
    print(f"Bearish divergences: {bearish_div.sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"Base Value: {nvi.base_value}")
    print(f"Indicator Name: {nvi.name}")
