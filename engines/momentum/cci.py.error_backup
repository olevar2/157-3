"""
Commodity Channel Index (CCI) Momentum Indicator
Measures the current price level relative to an average price level over a given period.
Part of Platform3's 67-indicator humanitarian trading system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional
from datetime import datetime
from ..indicator_base import (
    MomentumIndicator, IndicatorResult, IndicatorSignal, MarketData, 
    SignalType, TimeFrame, typical_price, sma
)

class CCI(MomentumIndicator):
    """
    Commodity Channel Index (CCI) momentum oscillator.
    
    CCI measures the current price level relative to an average price level
    over a given period of time. It oscillates around zero, with values
    above +100 indicating overbought conditions and values below -100
    indicating oversold conditions.
    
    Formula: CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)
    """
    
    def __init__(self, timeframe: TimeFrame, lookback_periods: int = 20):
        """
        Initialize CCI indicator.
        
        Args:
            timeframe: Timeframe for analysis
            lookback_periods: Period for calculation (default 20)
        """
        super().__init__("CCI", timeframe, lookback_periods=lookback_periods)
        self.period = lookback_periods
        self.constant = 0.015  # Lambert's constant
        self.overbought_level = 100
        self.oversold_level = -100
        self.extreme_overbought = 200
        self.extreme_oversold = -200
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """
        Calculate CCI value.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            IndicatorResult with CCI calculation
        """
        if len(data) < self.period:
            raise ValueError(f"Insufficient data for CCI calculation. Need {self.period} periods, got {len(data)}")
        
        start_time = datetime.now()
        
        try:
            # Calculate typical prices for the period
            calc_data = data[-self.period:]
            typical_prices = [typical_price(candle) for candle in calc_data]
            
            # Calculate Simple Moving Average of typical prices
            sma_tp = sma(typical_prices, self.period)
            
            # Calculate mean deviation
            deviations = [abs(tp - sma_tp) for tp in typical_prices]
            mean_deviation = sum(deviations) / len(deviations)
            
            # Calculate CCI
            current_tp = typical_prices[-1]
            if mean_deviation == 0:
                cci = 0  # Avoid division by zero
            else:
                cci = (current_tp - sma_tp) / (self.constant * mean_deviation)
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=cci,
                raw_data={
                    'cci': cci,
                    'typical_price': current_tp,
                    'sma_typical_price': sma_tp,
                    'mean_deviation': mean_deviation,
                    'period': self.period
                },
                calculation_time_ms=calculation_time
            )
            
            # Generate signal
            signal = self.generate_signal(result, [])
            if signal:
                result.signal = signal
                
            self.update_status(self.status)
            return result
            
        except Exception as e:
            self.update_status(self.status, str(e))
            raise ValueError(f"CCI calculation failed: {e}")
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on CCI levels and crossovers.
        
        Args:
            current_result: Current CCI calculation
            historical_results: Previous results for trend analysis
            
        Returns:
            IndicatorSignal if conditions are met
        """
        cci = current_result.value
        
        # Strong signals for extreme levels
        if cci >= self.extreme_overbought:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.STRONG_SELL,
                strength=min(1.0, abs(cci) / 300),
                confidence=0.8,
                metadata={
                    'cci': cci,
                    'level': 'extreme_overbought',
                    'threshold': self.extreme_overbought
                }
            )
        elif cci <= self.extreme_oversold:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.STRONG_BUY,
                strength=min(1.0, abs(cci) / 300),
                confidence=0.8,
                metadata={
                    'cci': cci,
                    'level': 'extreme_oversold',
                    'threshold': self.extreme_oversold
                }
            )
        
        # Regular overbought/oversold signals
        elif cci >= self.overbought_level:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.SELL,
                strength=min(1.0, cci / 200),
                confidence=0.6,
                metadata={
                    'cci': cci,
                    'level': 'overbought',
                    'threshold': self.overbought_level
                }
            )
        elif cci <= self.oversold_level:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.BUY,
                strength=min(1.0, abs(cci) / 200),
                confidence=0.6,
                metadata={
                    'cci': cci,
                    'level': 'oversold',
                    'threshold': self.oversold_level
                }
            )
        
        # Zero line crossover signals
        if len(historical_results) >= 1:
            prev_result = historical_results[-1]
            if hasattr(prev_result, 'value'):
                prev_cci = prev_result.value
                
                # Bullish zero line crossover
                if prev_cci < 0 and cci > 0:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.BUY,
                        strength=0.7,
                        confidence=0.7,
                        metadata={
                            'cci': cci,
                            'previous_cci': prev_cci,
                            'signal': 'bullish_zero_crossover'
                        }
                    )
                # Bearish zero line crossover
                elif prev_cci > 0 and cci < 0:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.SELL,
                        strength=0.7,
                        confidence=0.7,
                        metadata={
                            'cci': cci,
                            'previous_cci': prev_cci,
                            'signal': 'bearish_zero_crossover'
                        }
                    )
        
        return None

# Test function for validation
def test_cci():
    """Test CCI calculation with sample data."""
    from datetime import datetime, timedelta
    
    # Create sample market data with varying prices
    base_time = datetime.now()
    test_data = []
    
    # Generate test data with known pattern (trending up then down)
    prices = [
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,  # Uptrend
        59, 58, 57, 56, 55, 54, 53, 52, 51, 50,      # Downtrend
        51, 52, 53, 54, 55                           # Recovery
    ]
    
    for i, price in enumerate(prices):
        high = price + (i % 3) * 0.5  # Add some volatility
        low = price - (i % 2) * 0.3
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            open=price,
            high=high,
            low=low,
            close=price + 0.1,
            volume=1000 + i * 10,
            timeframe=TimeFrame.M1
        ))
    
    # Test CCI calculation
    cci = CCI(TimeFrame.M1, lookback_periods=20)
    result = cci.calculate(test_data)
    
    print(f"CCI Test Results:")
    print(f"Value: {result.value:.2f}")
    print(f"Calculation time: {result.calculation_time_ms:.2f}ms")
    print(f"Raw data: {result.raw_data}")
    
    if result.signal:
        print(f"Signal: {result.signal.signal_type.value} (strength: {result.signal.strength:.2f})")
        print(f"Confidence: {result.signal.confidence:.2f}")
        print(f"Metadata: {result.signal.metadata}")
    else:
        print("No signal generated")
    
    return result

if __name__ == "__main__":
    test_cci()
