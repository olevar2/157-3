"""
On-Balance Volume (OBV) Volume Indicator
Measures cumulative buying and selling pressure by adding volume on up days and subtracting on down days.
Part of Platform3's 67-indicator humanitarian trading system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional
from datetime import datetime
from indicator_base import (
    VolumeIndicator, IndicatorResult, IndicatorSignal, MarketData, 
    SignalType, TimeFrame
)

class OBV(VolumeIndicator):
    """
    On-Balance Volume (OBV) volume indicator.
    
    OBV measures cumulative buying and selling pressure by adding the period's
    volume when the closing price is up and subtracting the period's volume when
    the closing price is down. It's used to predict price movements based on
    volume flows.
    """
    
    def __init__(self, timeframe: TimeFrame, lookback_periods: int = 20):
        """
        Initialize OBV indicator.
        
        Args:
            timeframe: Timeframe for analysis
            lookback_periods: Minimum periods needed for trend analysis
        """
        super().__init__("OBV", timeframe, lookback_periods=lookback_periods)
        self.period = lookback_periods
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """
        Calculate OBV value.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            IndicatorResult with OBV calculation
        """
        if len(data) < 2:
            raise ValueError(f"Insufficient data for OBV calculation. Need at least 2 periods, got {len(data)}")
        
        start_time = datetime.now()
        
        try:
            # Calculate cumulative OBV
            obv = 0
            obv_values = []
            
            # Start with first period
            obv_values.append(0)
            
            for i in range(1, len(data)):
                current = data[i]
                previous = data[i-1]
                
                if current.close > previous.close:
                    # Price up: add volume
                    obv += current.volume
                elif current.close < previous.close:
                    # Price down: subtract volume
                    obv -= current.volume
                # Price unchanged: no change to OBV
                
                obv_values.append(obv)
            
            current_obv = obv_values[-1]
            
            # Calculate OBV trend over specified period
            if len(obv_values) >= self.period:
                period_start_obv = obv_values[-self.period]
                obv_change = current_obv - period_start_obv
                obv_trend = obv_change / max(abs(period_start_obv), 1)  # Normalized trend
            else:
                obv_change = current_obv
                obv_trend = 0
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=current_obv,
                raw_data={
                    'obv': current_obv,
                    'obv_change': obv_change,
                    'obv_trend': obv_trend,
                    'obv_values': obv_values[-10:],  # Last 10 values for context
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
            raise ValueError(f"OBV calculation failed: {e}")
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on OBV trend and divergences.
        
        Args:
            current_result: Current OBV calculation
            historical_results: Previous results for trend analysis
            
        Returns:
            IndicatorSignal if conditions are met
        """
        obv_trend = current_result.raw_data.get('obv_trend', 0)
        obv_change = current_result.raw_data.get('obv_change', 0)
        
        # Strong volume trend signals
        if obv_trend > 0.1:  # Significant positive trend
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.BUY,
                strength=min(1.0, obv_trend * 2),
                confidence=0.7,
                metadata={
                    'obv': current_result.value,
                    'obv_trend': obv_trend,
                    'signal': 'strong_volume_accumulation'
                }
            )
        elif obv_trend < -0.1:  # Significant negative trend
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.SELL,
                strength=min(1.0, abs(obv_trend) * 2),
                confidence=0.7,
                metadata={
                    'obv': current_result.value,
                    'obv_trend': obv_trend,
                    'signal': 'strong_volume_distribution'
                }
            )
        
        # Volume momentum signals
        if len(historical_results) >= 2:
            prev_result = historical_results[-1]
            if hasattr(prev_result, 'value'):
                prev_obv = prev_result.value
                current_obv = current_result.value
                
                # Volume momentum acceleration
                if current_obv > prev_obv and obv_change > 0:
                    momentum_strength = min(1.0, abs(obv_change) / 100000)  # Normalize based on typical volume
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.BUY,
                        strength=momentum_strength,
                        confidence=0.6,
                        metadata={
                            'obv': current_obv,
                            'previous_obv': prev_obv,
                            'signal': 'volume_momentum_bullish'
                        }
                    )
                elif current_obv < prev_obv and obv_change < 0:
                    momentum_strength = min(1.0, abs(obv_change) / 100000)
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.SELL,
                        strength=momentum_strength,
                        confidence=0.6,
                        metadata={
                            'obv': current_obv,
                            'previous_obv': prev_obv,
                            'signal': 'volume_momentum_bearish'
                        }
                    )
        
        return None

def test_obv():
    """Test OBV calculation with sample data."""
    from datetime import datetime, timedelta
    
    # Create sample market data with volume and price patterns
    base_time = datetime.now()
    test_data = []
    
    # Generate test data: uptrend with increasing volume, then downtrend
    base_price = 100
    base_volume = 1000
    
    # Uptrend phase
    for i in range(15):
        price = base_price + i * 0.5  # Gradual price increase
        volume = base_volume + i * 100  # Increasing volume
        
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            open=price - 0.1,
            high=price + 0.2,
            low=price - 0.2,
            close=price,
            volume=volume,
            timeframe=TimeFrame.M1
        ))
    
    # Downtrend phase
    for i in range(10):
        price = base_price + 14 * 0.5 - i * 0.3  # Price decline
        volume = base_volume + 1400 + i * 50      # Still elevated volume
        
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=15 + i),
            open=price + 0.1,
            high=price + 0.1,
            low=price - 0.3,
            close=price,
            volume=volume,
            timeframe=TimeFrame.M1
        ))
    
    # Test OBV calculation
    obv = OBV(TimeFrame.M1)
    result = obv.calculate(test_data)
    
    print(f"OBV Test Results:")
    print(f"Value: {result.value:.2f}")
    print(f"Calculation time: {result.calculation_time_ms:.2f}ms")
    print(f"Raw data: {result.raw_data}")
    
    if result.signal:
        print(f"Signal: {result.signal.signal_type.value} (strength: {result.signal.strength:.2f})")
    else:
        print("No signal generated")
    
    return result

if __name__ == "__main__":
    test_obv()
