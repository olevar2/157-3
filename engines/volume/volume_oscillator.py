"""
Volume Oscillator Indicator

The Volume Oscillator measures the relationship between two moving averages of volume,
typically calculated as the difference between a short-period and long-period volume MA,
expressed as a percentage.

Formula:
Volume Oscillator = ((Short MA Volume - Long MA Volume) / Long MA Volume) * 100

Author: Platform3 Trading System
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from engines.indicator_base import TechnicalIndicator, MarketData, IndicatorSignal, SignalType


class VolumeOscillator(TechnicalIndicator):
    """
    Volume Oscillator Implementation
    
    The Volume Oscillator is a technical indicator that shows the relationship between    two moving averages of volume. It oscillates above and below zero, with positive
    values indicating volume strength and negative values indicating volume weakness.
    """
    
    def __init__(self, short_period: int = 14, long_period: int = 28, config=None):
        """
        Initialize Volume Oscillator
        
        Parameters:
        -----------
        short_period : int, default 14
            Period for short-term volume moving average
        long_period : int, default 28
            Period for long-term volume moving average
        config : dict, optional
            Configuration dictionary containing parameters
        """
        # Handle config parameter
        if config is not None:
            if isinstance(config, dict):
                self.short_period = config.get('short_period', short_period)
                self.long_period = config.get('long_period', long_period)
            else:
                self.short_period = short_period
                self.long_period = long_period
        else:
            self.short_period = short_period
            self.long_period = long_period
            
        self.name = f"Volume Oscillator ({self.short_period}/{self.long_period})"
        
        if self.short_period >= self.long_period:
            raise ValueError("Short period must be less than long period")
    
    def calculate(self, volume: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate Volume Oscillator
        
        Parameters:
        -----------
        volume : pd.Series or np.ndarray
            Volume data
            
        Returns:
        --------
        pd.Series
            Volume Oscillator values
        """
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        
        # Calculate moving averages
        short_ma = volume.rolling(window=self.short_period).mean()
        long_ma = volume.rolling(window=self.long_period).mean()
        
        # Calculate Volume Oscillator
        volume_oscillator = ((short_ma - long_ma) / long_ma) * 100
        
        return volume_oscillator
    
    def get_signals(self, volume: Union[pd.Series, np.ndarray], 
                   threshold: float = 10.0) -> pd.DataFrame:
        """
        Generate trading signals based on Volume Oscillator
        
        Parameters:
        -----------
        volume : pd.Series or np.ndarray
            Volume data
        threshold : float, default 10.0
            Threshold for strong volume signals
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with Volume Oscillator and signals
        """
        vol_osc = self.calculate(volume)
        
        signals = pd.DataFrame(index=vol_osc.index)
        signals['volume_oscillator'] = vol_osc
        
        # Generate signals
        signals['signal'] = 0
        signals['signal'][vol_osc > threshold] = 1  # High volume strength
        signals['signal'][vol_osc < -threshold] = -1  # Low volume strength
        
        # Zero line crossovers
        signals['crossover_up'] = (vol_osc > 0) & (vol_osc.shift(1) <= 0)
        signals['crossover_down'] = (vol_osc < 0) & (vol_osc.shift(1) >= 0)
        
        return signals
    
    def get_interpretation(self, volume: Union[pd.Series, np.ndarray]) -> dict:
        """
        Get interpretation of current Volume Oscillator reading
        
        Parameters:
        -----------
        volume : pd.Series or np.ndarray
            Volume data
            
        Returns:
        --------
        dict
            Interpretation of Volume Oscillator
        """
        vol_osc = self.calculate(volume)
        current_value = vol_osc.iloc[-1] if not pd.isna(vol_osc.iloc[-1]) else 0
        
        if current_value > 20:
            interpretation = "Very High Volume Activity"
            signal_strength = "Strong"
        elif current_value > 10:
            interpretation = "High Volume Activity"
            signal_strength = "Moderate"
        elif current_value > 0:
            interpretation = "Above Average Volume"
            signal_strength = "Weak"
        elif current_value > -10:
            interpretation = "Below Average Volume"
            signal_strength = "Weak"
        elif current_value > -20:
            interpretation = "Low Volume Activity"
            signal_strength = "Moderate"
        else:
            interpretation = "Very Low Volume Activity"
            signal_strength = "Strong"
        
        return {
            'current_value': current_value,
            'interpretation': interpretation,
            'signal_strength': signal_strength,
            'short_period': self.short_period,
            'long_period': self.long_period
        }
    
    def generate_signal(self, data: List[MarketData]) -> Optional[IndicatorSignal]:
        """
        Generate trading signal based on Volume Oscillator values
        
        Args:
            data: List of MarketData objects
            
        Returns:
            Optional[IndicatorSignal]: Trading signal or None
        """
        if len(data) < self.long_period + 1:
            return None
            
        # Extract volume data
        volumes = pd.Series([d.volume for d in data])
        
        # Calculate Volume Oscillator
        vo_values = self.calculate(volumes)
        
        if vo_values.isna().iloc[-1]:
            return None
            
        latest_vo = vo_values.iloc[-1]
        prev_vo = vo_values.iloc[-2] if len(vo_values) > 1 else 0
        
        # Generate signal based on Volume Oscillator crossover and levels
        signal_type = SignalType.HOLD
        strength = 0.0
        confidence = 0.5
        
        # Zero line crossover signals
        if latest_vo > 0 and prev_vo <= 0:
            signal_type = SignalType.BUY
            strength = min(abs(latest_vo) / 100.0, 1.0)  # VO is in percentage
            confidence = 0.6
        elif latest_vo < 0 and prev_vo >= 0:
            signal_type = SignalType.SELL
            strength = min(abs(latest_vo) / 100.0, 1.0)
            confidence = 0.6
        
        # Strong volume extremes
        if abs(latest_vo) > 20:  # Strong volume signals
            confidence = min(confidence + 0.2, 1.0)
        
        return IndicatorSignal(
            timestamp=data[-1].timestamp,
            indicator_name="Volume Oscillator",
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            metadata={'volume_oscillator': latest_vo}
        )
    
    def generate_signal(self, data: MarketData, period_start=None, period_end=None) -> List[IndicatorSignal]:
        """
        Generate trading signals based on Volume Oscillator.
        
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
        # Calculate Volume Oscillator
        vo = self.calculate(data.df['volume'])
        
        signals = []
        
        # Filter data by period if specified
        if period_start is not None or period_end is not None:
            start_idx = 0 if period_start is None else data.df.index.get_indexer([period_start], method='nearest')[0]
            end_idx = len(data.df) if period_end is None else data.df.index.get_indexer([period_end], method='nearest')[0]
            working_df = data.df.iloc[start_idx:end_idx+1]
            working_vo = vo.iloc[start_idx:end_idx+1]
        else:
            working_df = data.df
            working_vo = vo
        
        # Generate signals only where we have sufficient data
        for i in range(self.long_period + 1, len(working_vo)):
            current_idx = working_df.index[i]
            current_price = working_df['close'].iloc[i]
            current_vo = working_vo.iloc[i]
            prev_vo = working_vo.iloc[i-1]
            
            # Buy signal: Volume Oscillator crosses above zero (positive volume momentum)
            if prev_vo <= 0 and current_vo > 0:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_vo,
                        price=current_price,
                        strength=min(abs(current_vo) / 20, 1.0),  # Normalize strength
                        indicator_name=self.name
                    )
                )
            
            # Sell signal: Volume Oscillator crosses below zero (negative volume momentum)
            elif prev_vo >= 0 and current_vo < 0:
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.SELL,
                        indicator_value=current_vo,
                        price=current_price,
                        strength=min(abs(current_vo) / 20, 1.0),  # Normalize strength
                        indicator_name=self.name
                    )
                )
            
            # Additional trend confirmation signal (Volume expansion in existing trend)
            elif (current_price > working_df['close'].iloc[i-1] and 
                  current_vo > prev_vo and current_vo > 0 and
                  abs(current_vo - prev_vo) / abs(prev_vo + 0.0001) > 0.1):  # 10% volume expansion
                signals.append(
                    IndicatorSignal(
                        timestamp=current_idx,
                        signal_type=SignalType.BUY,
                        indicator_value=current_vo,
                        price=current_price,
                        strength=min(abs(current_vo) / 30, 1.0),  # Normalize strength
                        indicator_name=f"{self.name} (Volume Expansion)"
                    )
                )
        
        return signals


def volume_oscillator(volume: Union[pd.Series, np.ndarray], 
                     short_period: int = 14, 
                     long_period: int = 28) -> pd.Series:
    """
    Convenience function to calculate Volume Oscillator
    
    Parameters:
    -----------
    volume : pd.Series or np.ndarray
        Volume data
    short_period : int, default 14
        Period for short-term volume moving average
    long_period : int, default 28
        Period for long-term volume moving average
        
    Returns:
    --------
    pd.Series
        Volume Oscillator values
    """
    vo = VolumeOscillator(short_period=short_period, long_period=long_period)
    return vo.calculate(volume)


# Example usage and testing
if __name__ == "__main__":
    # Generate sample volume data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    volume_data = pd.Series(
        np.random.lognormal(mean=10, sigma=0.5, size=100) * 1000000,
        index=dates
    )
    
    # Calculate Volume Oscillator
    vo = VolumeOscillator(short_period=14, long_period=28)
    
    print("Volume Oscillator Test Results:")
    print("=" * 50)
    
    # Test calculation
    result = vo.calculate(volume_data)
    print(f"Latest Volume Oscillator values:")
    print(result.tail())
    
    # Test signals
    signals = vo.get_signals(volume_data)
    print(f"\nSignals (last 10 rows):")
    print(signals.tail(10))
    
    # Test interpretation
    interpretation = vo.get_interpretation(volume_data)
    print(f"\nCurrent Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
    
    print(f"\nVolume Oscillator calculation completed successfully!")
