"""
Parabolic SAR Trend Indicator
A trend-following indicator that provides potential reversal points in price.
Part of Platform3's 67-indicator humanitarian trading system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional
from datetime import datetime
from indicator_base import (
    TrendIndicator, IndicatorResult, IndicatorSignal, MarketData, 
    SignalType, TimeFrame
)

class ParabolicSAR(TrendIndicator):
    """
    Parabolic SAR (Stop and Reverse) trend indicator.
    
    The Parabolic SAR is used to determine the direction of an asset's momentum
    and the point in time when this momentum has a higher-than-normal probability
    of switching directions.
    """
    
    def __init__(self, timeframe: TimeFrame, 
                 initial_af: float = 0.02, 
                 max_af: float = 0.20, 
                 af_increment: float = 0.02):
        """
        Initialize Parabolic SAR indicator.
        
        Args:
            timeframe: Timeframe for analysis
            initial_af: Initial acceleration factor (default 0.02)
            max_af: Maximum acceleration factor (default 0.20)
            af_increment: Acceleration factor increment (default 0.02)
        """
        super().__init__("Parabolic SAR", timeframe, lookback_periods=2)
        self.initial_af = initial_af
        self.max_af = max_af
        self.af_increment = af_increment
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """
        Calculate Parabolic SAR value.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            IndicatorResult with Parabolic SAR calculation
        """
        if len(data) < 2:
            raise ValueError(f"Insufficient data for Parabolic SAR calculation. Need at least 2 periods, got {len(data)}")
        
        start_time = datetime.now()
        
        try:
            # Initialize variables
            sar_values = []
            
            # Determine initial trend direction
            if data[1].close > data[0].close:
                trend = 1  # Uptrend
                sar = data[0].low
                ep = data[1].high  # Extreme point
            else:
                trend = -1  # Downtrend
                sar = data[0].high
                ep = data[1].low
            
            af = self.initial_af
            sar_values.append(sar)
            
            # Calculate SAR for each subsequent period
            for i in range(1, len(data)):
                current = data[i]
                
                # Calculate new SAR
                sar = sar + af * (ep - sar)
                
                if trend == 1:  # Uptrend
                    # Check for trend reversal
                    if current.low <= sar:
                        # Trend reversal to downtrend
                        trend = -1
                        sar = ep
                        ep = current.low
                        af = self.initial_af
                    else:
                        # Continue uptrend
                        if current.high > ep:
                            ep = current.high
                            af = min(af + self.af_increment, self.max_af)
                        
                        # Ensure SAR doesn't exceed recent lows
                        if i >= 1:
                            sar = min(sar, data[i-1].low)
                        if i >= 2:
                            sar = min(sar, data[i-2].low)
                
                else:  # Downtrend
                    # Check for trend reversal
                    if current.high >= sar:
                        # Trend reversal to uptrend
                        trend = 1
                        sar = ep
                        ep = current.high
                        af = self.initial_af
                    else:
                        # Continue downtrend
                        if current.low < ep:
                            ep = current.low
                            af = min(af + self.af_increment, self.max_af)
                        
                        # Ensure SAR doesn't fall below recent highs
                        if i >= 1:
                            sar = max(sar, data[i-1].high)
                        if i >= 2:
                            sar = max(sar, data[i-2].high)
                
                sar_values.append(sar)
            
            current_sar = sar_values[-1]
            current_price = data[-1].close
            
            # Determine position relative to SAR
            is_above_sar = current_price > current_sar
            distance_from_sar = abs(current_price - current_sar) / current_price * 100
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=current_sar,
                raw_data={
                    'sar': current_sar,
                    'current_price': current_price,
                    'trend': trend,
                    'acceleration_factor': af,
                    'extreme_point': ep,
                    'is_above_sar': is_above_sar,
                    'distance_from_sar': distance_from_sar,
                    'sar_values': sar_values[-5:]  # Last 5 values for context
                },
                calculation_time_ms=calculation_time
            )
            
            # Generate signal
            signal = self.generate_signal(result, [])
            if signal:
                result.signal = signal
                
            return result
            
        except Exception as e:
            raise ValueError(f"Parabolic SAR calculation failed: {e}")
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on Parabolic SAR.
        
        Args:
            current_result: Current Parabolic SAR calculation
            historical_results: Previous results for trend analysis
            
        Returns:
            IndicatorSignal if conditions are met
        """
        sar_data = current_result.raw_data
        current_price = sar_data['current_price']
        sar = sar_data['sar']
        trend = sar_data['trend']
        distance = sar_data['distance_from_sar']
        
        # Trend continuation signals
        if trend == 1 and current_price > sar:  # Strong uptrend
            strength = min(1.0, distance / 5)  # Normalize based on distance
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.BUY,
                strength=strength,
                confidence=0.75,
                stop_loss=sar,  # SAR acts as stop loss
                metadata={
                    'sar': sar,
                    'current_price': current_price,
                    'trend': 'uptrend',
                    'signal': 'trend_continuation_bullish',
                    'stop_loss': sar
                }
            )
        elif trend == -1 and current_price < sar:  # Strong downtrend
            strength = min(1.0, distance / 5)
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.SELL,
                strength=strength,
                confidence=0.75,
                stop_loss=sar,  # SAR acts as stop loss
                metadata={
                    'sar': sar,
                    'current_price': current_price,
                    'trend': 'downtrend',
                    'signal': 'trend_continuation_bearish',
                    'stop_loss': sar
                }
            )
        
        # Trend reversal signals (if we have historical data)
        if len(historical_results) >= 1:
            prev_result = historical_results[-1]
            if hasattr(prev_result, 'raw_data'):
                prev_trend = prev_result.raw_data.get('trend', 0)
                
                # Bullish reversal
                if prev_trend == -1 and trend == 1:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.STRONG_BUY,
                        strength=0.9,
                        confidence=0.8,
                        stop_loss=sar,
                        metadata={
                            'sar': sar,
                            'current_price': current_price,
                            'signal': 'bullish_trend_reversal',
                            'previous_trend': 'downtrend',
                            'new_trend': 'uptrend'
                        }
                    )
                # Bearish reversal
                elif prev_trend == 1 and trend == -1:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.STRONG_SELL,
                        strength=0.9,
                        confidence=0.8,
                        stop_loss=sar,
                        metadata={
                            'sar': sar,
                            'current_price': current_price,
                            'signal': 'bearish_trend_reversal',
                            'previous_trend': 'uptrend',
                            'new_trend': 'downtrend'
                        }
                    )
        
        return None

def test_parabolic_sar():
    """Test Parabolic SAR calculation with sample data."""
    from datetime import datetime, timedelta
    
    # Create sample market data with clear trend
    base_time = datetime.now()
    test_data = []
    
    # Generate test data: uptrend followed by downtrend
    base_price = 100
    
    # Uptrend phase
    for i in range(15):
        price = base_price + i * 0.5
        high = price + 0.3
        low = price - 0.2
        
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            open=price - 0.1,
            high=high,
            low=low,
            close=price,
            volume=1000,
            timeframe=TimeFrame.M1
        ))
    
    # Downtrend phase
    for i in range(15):
        price = base_price + 14 * 0.5 - i * 0.4
        high = price + 0.2
        low = price - 0.3
        
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=15 + i),
            open=price + 0.1,
            high=high,
            low=low,
            close=price,
            volume=1000,
            timeframe=TimeFrame.M1
        ))
    
    # Test Parabolic SAR calculation
    psar = ParabolicSAR(TimeFrame.M1)
    result = psar.calculate(test_data)
    
    print(f"Parabolic SAR Test Results:")
    print(f"SAR: {result.value:.4f}")
    print(f"Current Price: {result.raw_data['current_price']:.4f}")
    print(f"Trend: {result.raw_data['trend']} ({'Uptrend' if result.raw_data['trend'] == 1 else 'Downtrend'})")
    print(f"Acceleration Factor: {result.raw_data['acceleration_factor']:.4f}")
    print(f"Distance from SAR: {result.raw_data['distance_from_sar']:.2f}%")
    print(f"Calculation time: {result.calculation_time_ms:.2f}ms")
    
    if result.signal:
        print(f"Signal: {result.signal.signal_type.value} (strength: {result.signal.strength:.2f})")
        print(f"Stop Loss: {result.signal.stop_loss:.4f}")
        print(f"Metadata: {result.signal.metadata}")
    else:
        print("No signal generated")
    
    return result

if __name__ == "__main__":
    test_parabolic_sar()
