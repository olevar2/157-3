"""
Money Flow Index (MFI) Momentum Indicator
A volume-weighted momentum oscillator that measures buying and selling pressure.
Part of Platform3's 67-indicator humanitarian trading system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional
from datetime import datetime
from indicator_base import (
    MomentumIndicator, IndicatorResult, IndicatorSignal, MarketData, 
    SignalType, TimeFrame, typical_price
)

class MFI(MomentumIndicator):
    """
    Money Flow Index (MFI) momentum indicator.
    
    MFI is a momentum indicator that uses both price and volume to measure
    buying and selling pressure. It oscillates between 0 and 100, where
    values above 80 indicate overbought conditions and values below 20
    indicate oversold conditions.
    
    Formula involves calculating money flow based on typical price and volume.
    """
    
    def __init__(self, timeframe: TimeFrame, lookback_periods: int = 14):
        """
        Initialize MFI indicator.
        
        Args:
            timeframe: Timeframe for analysis
            lookback_periods: Period for calculation (default 14)
        """
        super().__init__("MFI", timeframe, lookback_periods=lookback_periods + 1)
        self.period = lookback_periods
        self.overbought_level = 80
        self.oversold_level = 20
        self.extreme_overbought = 90
        self.extreme_oversold = 10
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """
        Calculate MFI value.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            IndicatorResult with MFI calculation
        """
        if len(data) < self.period + 1:
            raise ValueError(f"Insufficient data for MFI calculation. Need {self.period + 1} periods, got {len(data)}")
        
        start_time = datetime.now()
        
        try:
            # Calculate money flows
            money_flows = []
            
            for i in range(1, len(data)):
                current = data[i]
                previous = data[i-1]
                
                current_tp = typical_price(current)
                previous_tp = typical_price(previous)
                
                # Raw Money Flow = Typical Price * Volume
                raw_money_flow = current_tp * current.volume
                
                # Positive or Negative Money Flow based on typical price comparison
                if current_tp > previous_tp:
                    positive_flow = raw_money_flow
                    negative_flow = 0
                elif current_tp < previous_tp:
                    positive_flow = 0
                    negative_flow = raw_money_flow
                else:
                    positive_flow = 0
                    negative_flow = 0
                
                money_flows.append({
                    'positive': positive_flow,
                    'negative': negative_flow,
                    'raw': raw_money_flow
                })
            
            # Calculate MFI using the last 'period' money flows
            calc_flows = money_flows[-self.period:]
            
            positive_money_flow = sum(flow['positive'] for flow in calc_flows)
            negative_money_flow = sum(flow['negative'] for flow in calc_flows)
            
            # Money Flow Ratio = Positive Money Flow / Negative Money Flow
            if negative_money_flow == 0:
                if positive_money_flow > 0:
                    mfi = 100  # All positive flow
                else:
                    mfi = 50   # No flow
            else:
                money_flow_ratio = positive_money_flow / negative_money_flow
                # MFI = 100 - (100 / (1 + Money Flow Ratio))
                mfi = 100 - (100 / (1 + money_flow_ratio))
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=mfi,
                raw_data={
                    'mfi': mfi,
                    'positive_money_flow': positive_money_flow,
                    'negative_money_flow': negative_money_flow,
                    'money_flow_ratio': positive_money_flow / max(negative_money_flow, 1),
                    'period': self.period
                },
                calculation_time_ms=calculation_time
            )
            
            # Generate signal
            signal = self.generate_signal(result, [])
            if signal:
                result.signal = signal
                
            return result
            
        except Exception as e:
            raise ValueError(f"MFI calculation failed: {e}")
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on MFI levels and divergences.
        
        Args:
            current_result: Current MFI calculation
            historical_results: Previous results for trend analysis
            
        Returns:
            IndicatorSignal if conditions are met
        """
        mfi = current_result.value
        
        # Extreme level signals
        if mfi >= self.extreme_overbought:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.STRONG_SELL,
                strength=min(1.0, (mfi - self.extreme_overbought) / 10),
                confidence=0.85,
                metadata={
                    'mfi': mfi,
                    'level': 'extreme_overbought',
                    'threshold': self.extreme_overbought
                }
            )
        elif mfi <= self.extreme_oversold:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.STRONG_BUY,
                strength=min(1.0, (self.extreme_oversold - mfi) / 10),
                confidence=0.85,
                metadata={
                    'mfi': mfi,
                    'level': 'extreme_oversold',
                    'threshold': self.extreme_oversold
                }
            )
        
        # Regular overbought/oversold signals
        elif mfi >= self.overbought_level:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.SELL,
                strength=min(1.0, (mfi - self.overbought_level) / 20),
                confidence=0.7,
                metadata={
                    'mfi': mfi,
                    'level': 'overbought',
                    'threshold': self.overbought_level
                }
            )
        elif mfi <= self.oversold_level:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.BUY,
                strength=min(1.0, (self.oversold_level - mfi) / 20),
                confidence=0.7,
                metadata={
                    'mfi': mfi,
                    'level': 'oversold',
                    'threshold': self.oversold_level
                }
            )
        
        return None

def test_mfi():
    """Test MFI calculation with sample data."""
    from datetime import datetime, timedelta
    
    # Create sample market data with volume variations
    base_time = datetime.now()
    test_data = []
    
    # Generate test data with price and volume patterns
    base_price = 100
    base_volume = 1000
    
    for i in range(20):
        price_factor = 1 + (i % 8) * 0.01  # Price oscillation
        volume_factor = 1 + (i % 5) * 0.5   # Volume variations
        
        price = base_price * price_factor
        volume = base_volume * volume_factor
        
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=volume,
            timeframe=TimeFrame.M1
        ))
    
    # Test MFI calculation
    mfi = MFI(TimeFrame.M1)
    result = mfi.calculate(test_data)
    
    print(f"MFI Test Results:")
    print(f"Value: {result.value:.2f}")
    print(f"Calculation time: {result.calculation_time_ms:.2f}ms")
    print(f"Raw data: {result.raw_data}")
    
    if result.signal:
        print(f"Signal: {result.signal.signal_type.value} (strength: {result.signal.strength:.2f})")
    else:
        print("No signal generated")
    
    return result

if __name__ == "__main__":
    test_mfi()
