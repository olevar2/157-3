"""
Positive Volume Index (PVI)
A cumulative indicator that uses the change in volume to decide when to update cumulative values.
Developed by Paul Dysart and refined by Norman Fosback.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType


class PositiveVolumeIndex(TechnicalIndicator):
    """
    Positive Volume Index (PVI)
    
    The PVI assumes that informed investors are active on days when volume 
    increases, while uninformed investors are active when volume decreases. 
    The PVI only changes on days when volume increases from the previous day.
    
    Formula:
    If Volume[today] > Volume[yesterday]:
        PVI[today] = PVI[yesterday] + ((Close[today] - Close[yesterday]) / Close[yesterday]) * PVI[yesterday]
    Else:
        PVI[today] = PVI[yesterday]
    """
    
    def __init__(self, base_value: float = 1000.0, config=None):
        """
        Initialize Positive Volume Index
        
        Args:
            base_value: Starting value for the PVI (default: 1000.0)
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
            
        self.name = "Positive Volume Index"
        
    def calculate(self, data=None, high=None, low=None, close=None, volume=None) -> pd.Series:
        """
        Calculate Positive Volume Index
        
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
            PVI values
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
            
        # Initialize PVI series
        pvi = pd.Series(index=close.index, dtype=float)
        pvi.iloc[0] = self.base_value
        
        # Calculate PVI
        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i-1]:
                # Volume increased - update PVI
                price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change_pct)
            else:
                # Volume decreased or stayed same - keep previous PVI
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi
    
    def calculate_with_ma(self, 
                         close: Union[pd.Series, np.ndarray],
                         volume: Union[pd.Series, np.ndarray],
                         ma_period: int = 255) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate PVI with its moving average (typically 255 days for yearly average)
        
        Args:
            close: Closing prices
            volume: Volume
            ma_period: Moving average period (default: 255)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (pvi, pvi_ma)
        """
        pvi = self.calculate(close, volume)
        pvi_ma = pvi.rolling(window=ma_period).mean()
        
        return pvi, pvi_ma
    
    def calculate_with_signals(self, 
                              close: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              ma_period: int = 255) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate PVI with buy/sell signals based on moving average crossover
        
        Args:
            close: Closing prices
            volume: Volume
            ma_period: Moving average period for signal generation (default: 255)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (pvi, pvi_ma, signals)
                signals: 1 for bullish (PVI > MA), -1 for bearish (PVI < MA), 0 for neutral
        """
        pvi, pvi_ma = self.calculate_with_ma(close, volume, ma_period)
        
        # Generate signals based on PVI vs its moving average
        signals = pd.Series(0, index=pvi.index, dtype=int)
        signals[pvi > pvi_ma] = 1   # Bullish
        signals[pvi < pvi_ma] = -1  # Bearish
        
        return pvi, pvi_ma, signals
    
    def calculate_divergence(self, 
                           close: Union[pd.Series, np.ndarray],
                           volume: Union[pd.Series, np.ndarray],
                           lookback: int = 50) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Identify bullish and bearish divergences between PVI and price
        
        Args:
            close: Closing prices
            volume: Volume
            lookback: Lookback period for divergence detection
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (pvi, bullish_divergence, bearish_divergence)
        """
        pvi = self.calculate(close, volume)
        
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        bullish_div = pd.Series(False, index=pvi.index)
        bearish_div = pd.Series(False, index=pvi.index)
        
        # Look for divergences over lookback period
        for i in range(lookback, len(close)):
            # Get segments for analysis
            price_segment = close.iloc[i-lookback:i+1]
            pvi_segment = pvi.iloc[i-lookback:i+1]
            
            # Find recent extremes
            price_low_idx = price_segment.idxmin()
            price_high_idx = price_segment.idxmax()
            pvi_low_idx = pvi_segment.idxmin()
            pvi_high_idx = pvi_segment.idxmax()
            
            current_price = close.iloc[i]
            current_pvi = pvi.iloc[i]
            
            # Bullish divergence: price makes lower low, PVI makes higher low
            if (current_price <= price_segment.loc[price_low_idx] and 
                current_pvi > pvi_segment.loc[pvi_low_idx]):
                bullish_div.iloc[i] = True
                
            # Bearish divergence: price makes higher high, PVI makes lower high
            if (current_price >= price_segment.loc[price_high_idx] and 
                current_pvi < pvi_segment.loc[pvi_high_idx]):
                bearish_div.iloc[i] = True
        
        return pvi, bullish_div, bearish_div
    
    def get_bull_market_probability(self, 
                                   close: Union[pd.Series, np.ndarray],
                                   volume: Union[pd.Series, np.ndarray],
                                   ma_period: int = 255) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate bull market probability based on PVI position relative to MA
        According to Fosback's research, when PVI is above its MA, there's a 79% chance of a bull market
        
        Args:
            close: Closing prices
            volume: Volume
            ma_period: Moving average period (default: 255)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (pvi, pvi_ma, bull_market_probability)
        """
        pvi, pvi_ma = self.calculate_with_ma(close, volume, ma_period)
        
        # Calculate bull market probability
        bull_market_prob = pd.Series(0.21, index=pvi.index)  # 21% when below MA
        bull_market_prob[pvi > pvi_ma] = 0.79  # 79% when above MA
        
        return pvi, pvi_ma, bull_market_prob
    
    def get_trend_strength(self, 
                          close: Union[pd.Series, np.ndarray],
                          volume: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate trend strength based on PVI momentum
        
        Args:
            close: Closing prices
            volume: Volume
            
        Returns:
            Tuple[pd.Series, pd.Series]: (pvi, trend_strength)
        """
        pvi = self.calculate(close, volume)
        
        # Calculate rate of change in PVI
        pvi_roc = pvi.pct_change(periods=20) * 100  # 20-period rate of change
        
        # Normalize trend strength
        trend_strength = pd.Series(0, index=pvi.index, dtype=float)
        trend_strength[pvi_roc > 2] = 1    # Strong uptrend
        trend_strength[(pvi_roc > 0) & (pvi_roc <= 2)] = 0.5  # Weak uptrend
        trend_strength[(pvi_roc < 0) & (pvi_roc >= -2)] = -0.5  # Weak downtrend
        trend_strength[pvi_roc < -2] = -1   # Strong downtrend
        
        return pvi, trend_strength
    
    def calculate_with_nvi(self, 
                          close: Union[pd.Series, np.ndarray],
                          volume: Union[pd.Series, np.ndarray],
                          nvi: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate PVI along with comparison to NVI
        
        Args:
            close: Closing prices
            volume: Volume
            nvi: Negative Volume Index values
            
        Returns:
            Tuple[pd.Series, pd.Series]: (pvi, pvi_nvi_ratio)
        """
        pvi = self.calculate(close, volume)        
        # Calculate PVI/NVI ratio for comparative analysis
        pvi_nvi_ratio = pvi / nvi
        
        return pvi, pvi_nvi_ratio
    
    def generate_signal(self, data):
        """
        Generate trading signals based on PVI.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            dict: Signal information
        """
        try:
            if 'close' not in data.columns or 'volume' not in data.columns:
                return {'signal': 'HOLD', 'strength': 0.0, 'pvi': None}
                
            close = data['close']
            volume = data['volume']
            
            pvi = self.calculate(close, volume)
            if pvi.empty or len(pvi) < 2:
                return {'signal': 'HOLD', 'strength': 0.0, 'pvi': None}
            
            current_pvi = pvi.iloc[-1]
            prev_pvi = pvi.iloc[-2]
            
            # Simple trend-based signal
            if current_pvi > prev_pvi:
                signal = 'BUY'
                strength = min(abs(current_pvi - prev_pvi) / prev_pvi * 100, 1.0)
            elif current_pvi < prev_pvi:
                signal = 'SELL'
                strength = min(abs(prev_pvi - current_pvi) / prev_pvi * 100, 1.0)
            else:
                signal = 'HOLD'
                strength = 0.0
            
            return {
                'signal': signal,
                'strength': strength,
                'pvi': current_pvi,
                'trend': 'UP' if current_pvi > prev_pvi else 'DOWN' if current_pvi < prev_pvi else 'FLAT'
            }
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating PVI signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'pvi': None}
    
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
    
    def generate_signal(self, data: MarketData, period_start=None, period_end=None) -> List[IndicatorSignal]:
        """
        Generate trading signals based on Positive Volume Index.
        
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
        # Calculate Positive Volume Index
        pvi = self.calculate(data=data.df)
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_pvi = pvi.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_pvi = pvi
        
        # Calculate moving average for signal generation (common practice with PVI)
        ema_period = 255  # Approximately 1 year of trading days, standard for PVI
        if len(working_pvi) >= ema_period:
            pvi_ema = working_pvi.ewm(span=ema_period, min_periods=ema_period).mean()
            
            # Generate signals only where we have sufficient data
            for i in range(ema_period + 5, len(working_pvi)):
                current_idx = working_df.index[i]
                current_price = working_df['close'].iloc[i]
                current_pvi = working_pvi.iloc[i]
                current_pvi_ema = pvi_ema.iloc[i]
                prev_pvi = working_pvi.iloc[i-1]
                prev_pvi_ema = pvi_ema.iloc[i-1]
                
                # Buy signal: PVI crosses above its long-term EMA
                if prev_pvi <= prev_pvi_ema and current_pvi > current_pvi_ema:
                    signals.append(
                        IndicatorSignal(
                            timestamp=current_idx,
                            signal_type=SignalType.BUY,
                            indicator_value=current_pvi,
                            price=current_price,
                            strength=0.6,  # PVI considered somewhat less reliable than NVI
                            indicator_name=self.name
                        )
                    )
                
                # Sell signal: PVI crosses below its long-term EMA
                elif prev_pvi >= prev_pvi_ema and current_pvi < current_pvi_ema:
                    signals.append(
                        IndicatorSignal(
                            timestamp=current_idx,
                            signal_type=SignalType.SELL,
                            indicator_value=current_pvi,
                            price=current_price,
                            strength=0.5,
                            indicator_name=self.name
                        )
                    )
                
                # Extreme reading: PVI substantially above/below its EMA (more than 10%)
                if abs(current_pvi - current_pvi_ema) / current_pvi_ema > 0.10:
                    signal_type = SignalType.BUY if current_pvi > current_pvi_ema else SignalType.SELL
                    strength = min(abs(current_pvi - current_pvi_ema) / current_pvi_ema, 1.0)
                    signals.append(
                        IndicatorSignal(
                            timestamp=current_idx,
                            signal_type=signal_type,
                            indicator_value=current_pvi,
                            price=current_price,
                            strength=strength,
                            indicator_name=f"{self.name} Extreme"
                        )
                    )
        
        return signals
    
    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on Positive Volume Index for compatibility with indicator registry
        
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
        
        # Calculate PVI and a 255-day EMA for signal generation
        pvi = self.calculate(close=close, volume=volume)
        pvi_ema = pvi.ewm(span=255, adjust=False).mean()  # Common lookback period for PVI
        
        # Determine signal based on PVI vs its long-term EMA
        last_pvi = pvi.iloc[-1]
        last_pvi_ema = pvi_ema.iloc[-1]
        
        if last_pvi > last_pvi_ema:
            signal_type = SignalType.BUY
            message = f"PVI ({last_pvi:.2f}) above its 255-day EMA ({last_pvi_ema:.2f}), suggesting bullish institutional activity"
        elif last_pvi < last_pvi_ema:
            signal_type = SignalType.SELL
            message = f"PVI ({last_pvi:.2f}) below its 255-day EMA ({last_pvi_ema:.2f}), suggesting bearish institutional activity"
        else:
            signal_type = SignalType.NEUTRAL
            message = f"PVI ({last_pvi:.2f}) at its 255-day EMA ({last_pvi_ema:.2f}), no clear signal"
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_type=signal_type,
            message=message,
            plot_data={
                "pvi": pvi.tolist(),
                "pvi_ema": pvi_ema.tolist()
            }
        )
    

def positive_volume_index(close: Union[pd.Series, np.ndarray],
                         volume: Union[pd.Series, np.ndarray],
                         base_value: float = 1000.0) -> pd.Series:
    """
    Calculate Positive Volume Index (functional interface)
    
    Args:
        close: Closing prices
        volume: Volume
        base_value: Starting value for the PVI (default: 1000.0)
        
    Returns:
        pd.Series: PVI values
    """
    indicator = PositiveVolumeIndex(base_value=base_value)
    return indicator.calculate(close, volume)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')  # Longer period for MA
    
    # Generate realistic price data with trend
    base_price = 100
    trend = np.linspace(0, 0.5, 300)  # Gradual uptrend
    noise = np.random.normal(0, 0.015, 300)
    returns = trend/300 + noise
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    close_prices = pd.Series(prices, index=dates)
    
    # Generate volume with positive correlation to price increases
    base_volume = 1000000
    volume_changes = np.random.normal(0, 0.3, 300)
    # Increase volume more on positive price days
    price_changes = close_prices.pct_change()
    for i in range(1, len(volume_changes)):
        if price_changes.iloc[i] > 0:
            volume_changes[i] += 0.1  # Higher volume on up days
    
    volumes = [base_volume]
    for change in volume_changes[1:]:
        volumes.append(max(volumes[-1] * (1 + change), 100000))  # Minimum volume
    
    volumes = pd.Series(volumes, index=dates)
    
    # Test the indicator
    pvi = PositiveVolumeIndex(base_value=1000.0)
    
    print("Testing Positive Volume Index")
    print("=" * 40)
    
    # Basic calculation
    pvi_values = pvi.calculate(close_prices, volumes)
    print(f"Last 10 PVI values:")
    print(pvi_values.tail(10).round(4))
    
    # Statistical summary
    print(f"\nPVI Statistics:")
    print(f"Start Value: {pvi_values.iloc[0]:.4f}")
    print(f"End Value: {pvi_values.iloc[-1]:.4f}")
    print(f"Total Return: {((pvi_values.iloc[-1] / pvi_values.iloc[0]) - 1) * 100:.2f}%")
    print(f"Max: {pvi_values.max():.4f}")
    print(f"Min: {pvi_values.min():.4f}")
    
    # Compare with price performance
    price_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100
    print(f"Price Total Return: {price_return:.2f}%")
    
    # With moving average
    pvi_values, pvi_ma = pvi.calculate_with_ma(close_prices, volumes, ma_period=50)  # Shorter MA for testing
    print(f"\nPVI vs MA (last 5 values):")
    for i in range(-5, 0):
        print(f"PVI: {pvi_values.iloc[i]:.4f}, MA: {pvi_ma.iloc[i]:.4f}")
    
    # Signals
    pvi_values, pvi_ma, signals = pvi.calculate_with_signals(close_prices, volumes, ma_period=50)
    print(f"\nSignals summary:")
    print(f"Bullish periods: {(signals == 1).sum()}")
    print(f"Bearish periods: {(signals == -1).sum()}")
    print(f"Neutral periods: {(signals == 0).sum()}")
    
    # Bull market probability
    pvi_values, pvi_ma, bull_prob = pvi.get_bull_market_probability(close_prices, volumes, ma_period=50)
    current_bull_prob = bull_prob.iloc[-1]
    print(f"\nCurrent bull market probability: {current_bull_prob*100:.1f}%")
    
    # Trend strength
    pvi_values, trend_strength = pvi.get_trend_strength(close_prices, volumes)
    current_trend = trend_strength.iloc[-1]
    trend_desc = {1: "Strong Uptrend", 0.5: "Weak Uptrend", 0: "Neutral", 
                  -0.5: "Weak Downtrend", -1: "Strong Downtrend"}
    print(f"Current trend strength: {trend_desc.get(current_trend, 'Unknown')}")
    
    # Divergence analysis
    pvi_values, bullish_div, bearish_div = pvi.calculate_divergence(close_prices, volumes, lookback=30)
    print(f"\nDivergence signals:")
    print(f"Bullish divergences: {bullish_div.sum()}")
    print(f"Bearish divergences: {bearish_div.sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"Base Value: {pvi.base_value}")
    print(f"Indicator Name: {pvi.name}")
