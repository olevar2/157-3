"""
Volume Weighted Average Price (VWAP) Volume Indicator
Calculates the average price weighted by volume, providing insight into the fair value.
Part of Platform3's 67-indicator humanitarian trading system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional
from datetime import datetime
from indicator_base import (
    VolumeIndicator, IndicatorResult, IndicatorSignal, MarketData, 
    SignalType, TimeFrame, typical_price
)

class VWAP(VolumeIndicator):
    """
    Volume Weighted Average Price (VWAP) volume indicator.
    
    VWAP provides the average price a security has traded at throughout the day,
    based on both volume and price. It's used as a trading benchmark and to
    identify value areas.
    """
    
    def __init__(self, timeframe: TimeFrame, lookback_periods: int = 20):
        """
        Initialize VWAP indicator.
        
        Args:
            timeframe: Timeframe for analysis
            lookback_periods: Period for VWAP calculation
        """
        super().__init__("VWAP", timeframe, lookback_periods=lookback_periods)
        self.period = lookback_periods
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """
        Calculate VWAP value.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            IndicatorResult with VWAP calculation
        """
        if len(data) < self.period:
            raise ValueError(f"Insufficient data for VWAP calculation. Need {self.period} periods, got {len(data)}")
        
        start_time = datetime.now()
        
        try:
            # Calculate VWAP over the specified period
            calc_data = data[-self.period:]
            
            total_pv = 0  # Price * Volume
            total_volume = 0
            
            for candle in calc_data:
                tp = typical_price(candle)
                pv = tp * candle.volume
                total_pv += pv
                total_volume += candle.volume
            
            if total_volume == 0:
                vwap = data[-1].close  # Fallback to current price
            else:
                vwap = total_pv / total_volume
            
            # Current price for comparison
            current_price = data[-1].close
            
            # Calculate deviation from VWAP
            vwap_deviation = (current_price - vwap) / vwap * 100
            
            # Calculate VWAP bands (standard deviation-based)
            price_deviations = []
            for candle in calc_data:
                tp = typical_price(candle)
                deviation = (tp - vwap) ** 2 * candle.volume
                price_deviations.append(deviation)
            
            if total_volume > 0:
                variance = sum(price_deviations) / total_volume
                std_dev = variance ** 0.5
            else:
                std_dev = 0
            
            upper_band = vwap + std_dev
            lower_band = vwap - std_dev
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=vwap,
                raw_data={
                    'vwap': vwap,
                    'current_price': current_price,
                    'vwap_deviation': vwap_deviation,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'standard_deviation': std_dev,
                    'total_volume': total_volume,
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
            raise ValueError(f"VWAP calculation failed: {e}")
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on VWAP position and bands.
        
        Args:
            current_result: Current VWAP calculation
            historical_results: Previous results for trend analysis
            
        Returns:
            IndicatorSignal if conditions are met
        """
        vwap = current_result.value
        current_price = current_result.raw_data['current_price']
        vwap_deviation = current_result.raw_data['vwap_deviation']
        upper_band = current_result.raw_data['upper_band']
        lower_band = current_result.raw_data['lower_band']
        
        # Band breakout signals
        if current_price >= upper_band:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.BUY,
                strength=min(1.0, abs(vwap_deviation) / 5),
                confidence=0.7,
                metadata={
                    'vwap': vwap,
                    'current_price': current_price,
                    'deviation': vwap_deviation,
                    'signal': 'upper_band_breakout'
                }
            )
        elif current_price <= lower_band:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.SELL,
                strength=min(1.0, abs(vwap_deviation) / 5),
                confidence=0.7,
                metadata={
                    'vwap': vwap,
                    'current_price': current_price,
                    'deviation': vwap_deviation,
                    'signal': 'lower_band_breakdown'
                }
            )
        
        # VWAP position signals
        if abs(vwap_deviation) > 2:  # Significant deviation from VWAP
            if vwap_deviation > 2:  # Price well above VWAP
                return IndicatorSignal(
                    timestamp=current_result.timestamp,
                    indicator_name=self.name,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, vwap_deviation / 10),
                    confidence=0.6,
                    metadata={
                        'vwap': vwap,
                        'current_price': current_price,
                        'deviation': vwap_deviation,
                        'signal': 'overvalued_vs_vwap'
                    }
                )
            elif vwap_deviation < -2:  # Price well below VWAP
                return IndicatorSignal(
                    timestamp=current_result.timestamp,
                    indicator_name=self.name,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, abs(vwap_deviation) / 10),
                    confidence=0.6,
                    metadata={
                        'vwap': vwap,
                        'current_price': current_price,
                        'deviation': vwap_deviation,
                        'signal': 'undervalued_vs_vwap'
                    }
                )
        
        # VWAP crossover signals
        if len(historical_results) >= 1:
            prev_result = historical_results[-1]
            if hasattr(prev_result, 'raw_data'):
                prev_price = prev_result.raw_data.get('current_price', 0)
                prev_vwap = prev_result.value
                
                # Bullish VWAP crossover
                if prev_price <= prev_vwap and current_price > vwap:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.BUY,
                        strength=0.8,
                        confidence=0.75,
                        metadata={
                            'vwap': vwap,
                            'current_price': current_price,
                            'signal': 'bullish_vwap_crossover'
                        }
                    )
                # Bearish VWAP crossover
                elif prev_price >= prev_vwap and current_price < vwap:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.SELL,
                        strength=0.8,
                        confidence=0.75,
                        metadata={
                            'vwap': vwap,
                            'current_price': current_price,
                            'signal': 'bearish_vwap_crossover'
                        }
                    )
        
        return None

def test_vwap():
    """Test VWAP calculation with sample data."""
    from datetime import datetime, timedelta
    
    # Create sample market data with realistic volume patterns
    base_time = datetime.now()
    test_data = []
    
    # Generate test data with varying prices and volumes
    base_price = 100
    
    for i in range(25):
        # Create price movement
        trend = i * 0.1
        volatility = (i % 5) * 0.2
        price = base_price + trend + volatility
        
        # Volume patterns (higher volume on moves)
        volume_base = 1000
        volume_multiplier = 1 + abs(volatility) * 2
        volume = volume_base * volume_multiplier
        
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            open=price - 0.05,
            high=price + 0.1,
            low=price - 0.1,
            close=price,
            volume=volume,
            timeframe=TimeFrame.M1
        ))
    
    # Test VWAP calculation
    vwap = VWAP(TimeFrame.M1)
    result = vwap.calculate(test_data)
    
    print(f"VWAP Test Results:")
    print(f"VWAP: {result.value:.4f}")
    print(f"Current Price: {result.raw_data['current_price']:.4f}")
    print(f"Deviation: {result.raw_data['vwap_deviation']:.2f}%")
    print(f"Upper Band: {result.raw_data['upper_band']:.4f}")
    print(f"Lower Band: {result.raw_data['lower_band']:.4f}")
    print(f"Calculation time: {result.calculation_time_ms:.2f}ms")
    
    if result.signal:
        print(f"Signal: {result.signal.signal_type.value} (strength: {result.signal.strength:.2f})")
    else:
        print("No signal generated")
    
    return result

if __name__ == "__main__":
    test_vwap()
