"""
Price Volume Rank (PVR)
Ranks price and volume performance to identify significant movements.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType


class PriceVolumeRank(TechnicalIndicator):
    """
    Price Volume Rank (PVR)
    
    The Price Volume Rank assigns ranks to price and volume movements 
    to identify periods when both price and volume are performing 
    significantly compared to their historical values. This helps 
    identify strong momentum periods.
    
    The indicator calculates percentile ranks for both price changes 
    and volume levels over a specified lookback period.
    """
    
    def __init__(self, period: int = 20, config=None):
        """
        Initialize Price Volume Rank
        
        Args:
            period: Period for ranking calculation (default: 20)
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
            
        self.name = "Price Volume Rank"
        
    def calculate(self, data=None, high=None, low=None, close=None, volume=None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Price Volume Rank
        
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
        Tuple[pd.Series, pd.Series, pd.Series]
            price_rank, volume_rank, combined_rank
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
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
            
        # Calculate price changes
        price_change = close.pct_change() * 100
        
        # Calculate rolling percentile ranks
        price_rank = price_change.rolling(window=self.period).rank(pct=True) * 100
        volume_rank = volume.rolling(window=self.period).rank(pct=True) * 100
        
        # Calculate combined rank (average of price and volume ranks)
        combined_rank = (price_rank + volume_rank) / 2
        
        return price_rank, volume_rank, combined_rank
    
    def calculate_with_signals(self, 
                              close: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              high_threshold: float = 80.0,
                              low_threshold: float = 20.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate PVR with buy/sell signals
        
        Args:
            close: Closing prices
            volume: Volume data
            high_threshold: Threshold for high rank signals (default: 80.0)
            low_threshold: Threshold for low rank signals (default: 20.0)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank, signals)
                signals: 1 for strong momentum, -1 for weak momentum, 0 for neutral
        """
        price_rank, volume_rank, combined_rank = self.calculate(close, volume)
        
        # Generate signals based on combined rank
        signals = pd.Series(0, index=combined_rank.index, dtype=int)
        signals[combined_rank > high_threshold] = 1   # Strong momentum
        signals[combined_rank < low_threshold] = -1   # Weak momentum
        
        return price_rank, volume_rank, combined_rank, signals
    
    def get_momentum_strength(self, 
                             close: Union[pd.Series, np.ndarray],
                             volume: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Classify momentum strength based on price and volume ranks
        
        Args:
            close: Closing prices
            volume: Volume data
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank, momentum_strength)
                momentum_strength: 2 for very strong, 1 for strong, 0 for neutral, -1 for weak, -2 for very weak
        """
        price_rank, volume_rank, combined_rank = self.calculate(close, volume)
        
        # Classify momentum strength
        momentum_strength = pd.Series(0, index=combined_rank.index, dtype=int)
        momentum_strength[combined_rank >= 90] = 2   # Very strong
        momentum_strength[(combined_rank >= 70) & (combined_rank < 90)] = 1   # Strong
        momentum_strength[(combined_rank <= 30) & (combined_rank > 10)] = -1  # Weak
        momentum_strength[combined_rank <= 10] = -2  # Very weak
        
        return price_rank, volume_rank, combined_rank, momentum_strength
    
    def identify_breakouts(self, 
                          close: Union[pd.Series, np.ndarray],
                          volume: Union[pd.Series, np.ndarray],
                          price_threshold: float = 85.0,
                          volume_threshold: float = 85.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Identify potential breakouts based on high price and volume ranks
        
        Args:
            close: Closing prices
            volume: Volume data
            price_threshold: Minimum price rank for breakout (default: 85.0)
            volume_threshold: Minimum volume rank for breakout (default: 85.0)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank, breakouts)
        """
        price_rank, volume_rank, combined_rank = self.calculate(close, volume)
        
        # Identify breakouts where both price and volume are highly ranked
        breakouts = (price_rank > price_threshold) & (volume_rank > volume_threshold)
        
        return price_rank, volume_rank, combined_rank, breakouts
    
    def get_divergence_signals(self, 
                              close: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              divergence_threshold: float = 30.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Identify divergences between price and volume ranks
        
        Args:
            close: Closing prices
            volume: Volume data
            divergence_threshold: Minimum difference for divergence detection (default: 30.0)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank, divergences)
                divergences: 1 for bullish divergence (high volume, low price), -1 for bearish, 0 for none
        """
        price_rank, volume_rank, combined_rank = self.calculate(close, volume)
        
        # Calculate rank difference
        rank_diff = volume_rank - price_rank
        
        # Identify divergences
        divergences = pd.Series(0, index=combined_rank.index, dtype=int)
        
        # Bullish divergence: High volume rank, low price rank
        bullish_div = (volume_rank > 70) & (price_rank < 30) & (rank_diff > divergence_threshold)
        divergences[bullish_div] = 1
        
        # Bearish divergence: Low volume rank, high price rank
        bearish_div = (volume_rank < 30) & (price_rank > 70) & (rank_diff < -divergence_threshold)
        divergences[bearish_div] = -1
        
        return price_rank, volume_rank, combined_rank, divergences
    
    def calculate_trend_confirmation(self, 
                                   close: Union[pd.Series, np.ndarray],
                                   volume: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate trend confirmation based on consistent high ranks
        
        Args:
            close: Closing prices
            volume: Volume data
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank, trend_confirmation)
        """
        price_rank, volume_rank, combined_rank = self.calculate(close, volume)
        
        # Calculate moving average of combined rank to smooth signals
        rank_ma = combined_rank.rolling(window=5).mean()
        
        # Trend confirmation based on sustained high/low ranks
        trend_confirmation = pd.Series(0, index=combined_rank.index, dtype=int)
        trend_confirmation[rank_ma > 75] = 1   # Strong uptrend confirmation
        trend_confirmation[rank_ma < 25] = -1  # Strong downtrend confirmation
        
        return price_rank, volume_rank, combined_rank, trend_confirmation
    
    def get_regime_classification(self, 
                                close: Union[pd.Series, np.ndarray],
                                volume: Union[pd.Series, np.ndarray]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Classify market regimes based on price-volume relationships
        
        Args:
            close: Closing prices
            volume: Volume data
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank, regime)
                regime: 1 for trending, 0 for ranging, -1 for volatile
        """
        price_rank, volume_rank, combined_rank = self.calculate(close, volume)
        
        # Calculate volatility of ranks
        rank_volatility = combined_rank.rolling(window=10).std()
        
        # Classify regimes
        regime = pd.Series(0, index=combined_rank.index, dtype=int)
        
        # Trending: Consistent high or low ranks with low volatility
        trending = ((combined_rank > 70) | (combined_rank < 30)) & (rank_volatility < 20)
        regime[trending] = 1
        
        # Volatile: High rank volatility
        volatile = rank_volatility > 35
        regime[volatile] = -1        
        return price_rank, volume_rank, combined_rank, regime
    
    def generate_signal(self, data):
        """
        Generate trading signals based on Price Volume Rank.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            dict: Signal information
        """
        try:
            if 'close' not in data.columns or 'volume' not in data.columns:
                return {'signal': 'HOLD', 'strength': 0.0, 'rank': None}
                
            close = data['close']
            volume = data['volume']
            
            price_rank, volume_rank, combined_rank = self.calculate(close, volume)
            if combined_rank.empty or len(combined_rank) < 2:
                return {'signal': 'HOLD', 'strength': 0.0, 'rank': None}
            
            current_rank = combined_rank.iloc[-1]
            
            # Signal based on rank thresholds
            if current_rank >= 80:  # Top quintile
                signal = 'BUY'
                strength = (current_rank - 50) / 50  # Scale from 50-100 to 0-1
            elif current_rank <= 20:  # Bottom quintile
                signal = 'SELL'
                strength = (50 - current_rank) / 50  # Scale from 50-0 to 0-1
            else:
                signal = 'HOLD'
                strength = 0.0
            
            return {
                'signal': signal,
                'strength': min(strength, 1.0),
                'rank': current_rank,
                'price_rank': price_rank.iloc[-1] if not price_rank.empty else None,
                'volume_rank': volume_rank.iloc[-1] if not volume_rank.empty else None
            }
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error generating PVR signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'rank': None}
    
    @staticmethod
    def validate_data(close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray]) -> bool:
        """
        Validate input data
        
        Args:
            close: Closing prices
            volume: Volume data
            
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
        Generate trading signals based on Price Volume Rank.
        
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
        # Calculate Price Volume Rank
        price_rank, volume_rank, combined_rank = self.calculate(data=data.df)
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_price_rank = price_rank.iloc[start_idx:end_idx+1]
            working_volume_rank = volume_rank.iloc[start_idx:end_idx+1]
            working_combined_rank = combined_rank.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_price_rank = price_rank
            working_volume_rank = volume_rank
            working_combined_rank = combined_rank
        
        # Generate signals only where we have sufficient data
        for i in range(self.period + 1, len(working_combined_rank)):
            current_idx = working_df.index[i]
            current_price = working_df['close'].iloc[i]
            current_combined_rank = working_combined_rank.iloc[i]
            prev_combined_rank = working_combined_rank.iloc[i-1]
            curr_price_rank = working_price_rank.iloc[i]
            curr_volume_rank = working_volume_rank.iloc[i]
            
            # Strong buy signal: High price rank and high volume rank (both above 80%)
            if curr_price_rank > 0.8 and curr_volume_rank > 0.8:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_combined_rank,
                        price=current_price,
                        strength=min((curr_price_rank + curr_volume_rank) / 2, 1.0),
                        indicator_name=f"{self.name} Strong"
                    )
                )
            
            # Strong sell signal: Low price rank and high volume rank
            # Low prices with high volume often signal distribution (bearish)
            elif curr_price_rank < 0.2 and curr_volume_rank > 0.8:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.SELL,
                        indicator_value=current_combined_rank,
                        price=current_price,
                        strength=min((1 - curr_price_rank + curr_volume_rank) / 2, 1.0),
                        indicator_name=f"{self.name} Strong"
                    )
                )
            
            # Emerging trend: Combined rank crosses above/below threshold
            if prev_combined_rank < 0.7 and current_combined_rank >= 0.7:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_combined_rank,
                        price=current_price,
                        strength=current_combined_rank,
                        indicator_name=self.name
                    )
                )
            elif prev_combined_rank > 0.3 and current_combined_rank <= 0.3:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.SELL,
                        indicator_value=current_combined_rank,
                        price=current_price,
                        strength=1 - current_combined_rank,
                        indicator_name=self.name
                    )
                )
        
        return signals
    
    def generate_signal(self, data: MarketData) -> IndicatorSignal:
        """
        Generate signals based on Price Volume Rank for compatibility with indicator registry
        
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
        
        # Calculate Price Volume Ranks
        price_rank, volume_rank, pvr = self.calculate(close=close, volume=volume)
        
        # Get the latest values
        last_pvr = pvr.iloc[-1]
        last_price_rank = price_rank.iloc[-1]
        last_volume_rank = volume_rank.iloc[-1]
        
        # Define thresholds
        high_threshold = 0.8  # 80th percentile
        low_threshold = 0.2   # 20th percentile
        
        if last_pvr > high_threshold:
            signal_type = SignalType.BUY
            message = f"Strong price-volume activity (PVR: {last_pvr:.2f}) with price rank: {last_price_rank:.2f} and volume rank: {last_volume_rank:.2f}"
        elif last_pvr < low_threshold:
            signal_type = SignalType.SELL
            message = f"Weak price-volume activity (PVR: {last_pvr:.2f}) with price rank: {last_price_rank:.2f} and volume rank: {last_volume_rank:.2f}"
        else:
            signal_type = SignalType.NEUTRAL
            message = f"Neutral price-volume activity (PVR: {last_pvr:.2f}) with price rank: {last_price_rank:.2f} and volume rank: {last_volume_rank:.2f}"
        
        return IndicatorSignal(
            indicator_name=self.name,
            signal_type=signal_type,
            message=message,
            plot_data={
                "price_rank": price_rank.tolist(),
                "volume_rank": volume_rank.tolist(),
                "pvr": pvr.tolist()
            }
        )
def price_volume_rank(close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray],
                     period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Price Volume Rank (functional interface)
    
    Args:
        close: Closing prices
        volume: Volume data
        period: Period for ranking calculation (default: 20)
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (price_rank, volume_rank, combined_rank)
    """
    indicator = PriceVolumeRank(period=period)
    return indicator.calculate(close, volume)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data with varying momentum
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    
    # Add some momentum periods
    returns[20:30] = np.random.normal(0.03, 0.015, 10)  # Strong uptrend
    returns[50:60] = np.random.normal(-0.025, 0.01, 10)  # Strong downtrend
    returns[80:90] = np.random.normal(0.01, 0.005, 10)   # Weak uptrend
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    close_prices = pd.Series(prices, index=dates)
    
    # Generate volume with correlation to price movements
    base_volume = 1000000
    volumes = []
    for i, ret in enumerate(returns):
        # Higher volume during strong moves
        vol_multiplier = 1 + abs(ret) * 3
        if 20 <= i < 30 or 50 <= i < 60:  # Extra volume during momentum periods
            vol_multiplier *= 1.5
        volume = base_volume * vol_multiplier * (1 + np.random.normal(0, 0.3))
        volumes.append(max(volume, 100000))
    
    volumes = pd.Series(volumes, index=dates)
    
    # Test the indicator
    pvr = PriceVolumeRank(period=20)
    
    print("Testing Price Volume Rank")
    print("=" * 40)
    
    # Basic calculation
    price_rank, volume_rank, combined_rank = pvr.calculate(close_prices, volumes)
    print(f"Last 10 values:")
    print("Price Rank | Volume Rank | Combined Rank")
    for i in range(-10, 0):
        print(f"{price_rank.iloc[i]:8.1f} | {volume_rank.iloc[i]:9.1f} | {combined_rank.iloc[i]:11.1f}")
    
    # Statistical summary
    print(f"\nCombined Rank Statistics:")
    print(f"Mean: {combined_rank.mean():.2f}")
    print(f"Max: {combined_rank.max():.2f}")
    print(f"Min: {combined_rank.min():.2f}")
    print(f"Std: {combined_rank.std():.2f}")
    
    # Signals
    price_rank, volume_rank, combined_rank, signals = pvr.calculate_with_signals(close_prices, volumes)
    print(f"\nSignals summary:")
    print(f"Strong momentum periods: {(signals == 1).sum()}")
    print(f"Weak momentum periods: {(signals == -1).sum()}")
    print(f"Neutral periods: {(signals == 0).sum()}")
    
    # Momentum strength
    price_rank, volume_rank, combined_rank, momentum = pvr.get_momentum_strength(close_prices, volumes)
    print(f"\nMomentum strength distribution:")
    print(f"Very strong: {(momentum == 2).sum()}")
    print(f"Strong: {(momentum == 1).sum()}")
    print(f"Neutral: {(momentum == 0).sum()}")
    print(f"Weak: {(momentum == -1).sum()}")
    print(f"Very weak: {(momentum == -2).sum()}")
    
    # Breakouts
    price_rank, volume_rank, combined_rank, breakouts = pvr.identify_breakouts(close_prices, volumes)
    print(f"\nBreakout signals: {breakouts.sum()}")
    
    # Divergences
    price_rank, volume_rank, combined_rank, divergences = pvr.get_divergence_signals(close_prices, volumes)
    print(f"\nDivergence signals:")
    print(f"Bullish divergences: {(divergences == 1).sum()}")
    print(f"Bearish divergences: {(divergences == -1).sum()}")
    
    # Trend confirmation
    price_rank, volume_rank, combined_rank, trend_conf = pvr.calculate_trend_confirmation(close_prices, volumes)
    print(f"\nTrend confirmation:")
    print(f"Uptrend confirmed: {(trend_conf == 1).sum()}")
    print(f"Downtrend confirmed: {(trend_conf == -1).sum()}")
    print(f"No confirmation: {(trend_conf == 0).sum()}")
    
    # Regime classification
    price_rank, volume_rank, combined_rank, regime = pvr.get_regime_classification(close_prices, volumes)
    print(f"\nMarket regime classification:")
    print(f"Trending periods: {(regime == 1).sum()}")
    print(f"Ranging periods: {(regime == 0).sum()}")
    print(f"Volatile periods: {(regime == -1).sum()}")
    
    print(f"\nIndicator parameters:")
    print(f"Period: {pvr.period}")
    print(f"Indicator Name: {pvr.name}")
