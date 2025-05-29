"""
Rate of Change (ROC) Momentum Indicator
Measures the percentage change in price over a specified period.
Part of Platform3's 67-indicator humanitarian trading system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional
from datetime import datetime
from indicator_base import (
    MomentumIndicator, IndicatorResult, IndicatorSignal, MarketData, 
    SignalType, TimeFrame
)

class ROC(MomentumIndicator):
    """
    Rate of Change (ROC) momentum indicator.
    
    ROC measures the percentage change in price from one period to the next.
    It oscillates around zero, with positive values indicating upward momentum
    and negative values indicating downward momentum.
    
    Formula: ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    """
    
    def __init__(self, timeframe: TimeFrame, lookback_periods: int = 12):
        """
        Initialize ROC indicator.
        
        Args:
            timeframe: Timeframe for analysis
            lookback_periods: Period for rate of change calculation (default 12)
        """
        super().__init__("ROC", timeframe, lookback_periods=lookback_periods + 1)  # +1 for comparison
        self.period = lookback_periods
        self.overbought_threshold = 15  # Positive threshold for overbought
        self.oversold_threshold = -15   # Negative threshold for oversold        self.strong_threshold = 25      # Strong momentum threshold
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """
        Calculate ROC value.
        
        Args:
            data: List of MarketData objects
            
        Returns:
            IndicatorResult with ROC calculation
        """
        if len(data) < (self.period + 1):
            raise ValueError(f"Insufficient data for ROC calculation. Need {self.period + 1} periods, got {len(data)}")
        
        start_time = datetime.now()
        
        try:
            # Get current price and price n periods ago
            current_price = data[-1].close
            past_price = data[-(self.period + 1)].close
            
            # Calculate Rate of Change
            if past_price == 0:
                roc = 0  # Avoid division by zero
            else:
                roc = ((current_price - past_price) / past_price) * 100
            
            calculation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=roc,
                raw_data={
                    'roc': roc,
                    'current_price': current_price,
                    'past_price': past_price,
                    'period': self.period,
                    'price_change': current_price - past_price,
                    'percentage_change': roc
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
            raise ValueError(f"ROC calculation failed: {e}")
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """
        Generate trading signals based on ROC momentum and crossovers.
        
        Args:
            current_result: Current ROC calculation
            historical_results: Previous results for trend analysis
            
        Returns:
            IndicatorSignal if conditions are met
        """
        roc = current_result.value
        
        # Strong momentum signals
        if roc >= self.strong_threshold:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.STRONG_BUY,
                strength=min(1.0, roc / 50),
                confidence=0.8,
                metadata={
                    'roc': roc,
                    'level': 'strong_bullish_momentum',
                    'threshold': self.strong_threshold
                }
            )
        elif roc <= -self.strong_threshold:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.STRONG_SELL,
                strength=min(1.0, abs(roc) / 50),
                confidence=0.8,
                metadata={
                    'roc': roc,
                    'level': 'strong_bearish_momentum',
                    'threshold': -self.strong_threshold
                }
            )
        
        # Regular momentum signals
        elif roc >= self.overbought_threshold:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.BUY,
                strength=min(1.0, roc / 30),
                confidence=0.6,
                metadata={
                    'roc': roc,
                    'level': 'bullish_momentum',
                    'threshold': self.overbought_threshold
                }
            )
        elif roc <= self.oversold_threshold:
            return IndicatorSignal(
                timestamp=current_result.timestamp,
                indicator_name=self.name,
                signal_type=SignalType.SELL,
                strength=min(1.0, abs(roc) / 30),
                confidence=0.6,
                metadata={
                    'roc': roc,
                    'level': 'bearish_momentum',
                    'threshold': self.oversold_threshold
                }
            )
        
        # Zero line crossover signals
        if len(historical_results) >= 1:
            prev_result = historical_results[-1]
            if hasattr(prev_result, 'value'):
                prev_roc = prev_result.value
                
                # Bullish zero line crossover
                if prev_roc < 0 and roc > 0:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.BUY,
                        strength=0.7,
                        confidence=0.7,
                        metadata={
                            'roc': roc,
                            'previous_roc': prev_roc,
                            'signal': 'bullish_zero_crossover'
                        }
                    )
                # Bearish zero line crossover
                elif prev_roc > 0 and roc < 0:
                    return IndicatorSignal(
                        timestamp=current_result.timestamp,
                        indicator_name=self.name,
                        signal_type=SignalType.SELL,
                        strength=0.7,
                        confidence=0.7,
                        metadata={
                            'roc': roc,
                            'previous_roc': prev_roc,
                            'signal': 'bearish_zero_crossover'
                        }
                    )
        
        # Momentum divergence signals (requires more historical data)
        if len(historical_results) >= 3:
            # Check for momentum divergence patterns
            recent_rocs = [hr.value for hr in historical_results[-3:]] + [roc]
            
            # Bullish divergence: ROC making higher lows while price might be making lower lows
            if (recent_rocs[-1] > recent_rocs[-2] and 
                recent_rocs[-2] > recent_rocs[-3] and 
                all(r < 0 for r in recent_rocs[-3:])):
                return IndicatorSignal(
                    timestamp=current_result.timestamp,
                    indicator_name=self.name,
                    signal_type=SignalType.BUY,
                    strength=0.6,
                    confidence=0.5,
                    metadata={
                        'roc': roc,
                        'signal': 'bullish_momentum_divergence',
                        'recent_values': recent_rocs
                    }
                )
            
            # Bearish divergence: ROC making lower highs while price might be making higher highs
            elif (recent_rocs[-1] < recent_rocs[-2] and 
                  recent_rocs[-2] < recent_rocs[-3] and 
                  all(r > 0 for r in recent_rocs[-3:])):
                return IndicatorSignal(
                    timestamp=current_result.timestamp,
                    indicator_name=self.name,
                    signal_type=SignalType.SELL,
                    strength=0.6,
                    confidence=0.5,
                    metadata={
                        'roc': roc,
                        'signal': 'bearish_momentum_divergence',
                        'recent_values': recent_rocs
                    }
                )
        
        return None

# Test function for validation
def test_roc():
    """Test ROC calculation with sample data."""
    from datetime import datetime, timedelta
    
    # Create sample market data with clear trend
    base_time = datetime.now()
    test_data = []
    
    # Generate test data with accelerating uptrend then reversal
    base_price = 100
    prices = []
    
    # Building momentum phase
    for i in range(15):
        momentum = 1 + (i * 0.02)  # Accelerating growth
        price = base_price * momentum
        prices.append(price)
    
    # Reversal phase
    for i in range(10):
        decline = 0.98 ** (i + 1)  # Decelerating decline
        price = prices[-1] * decline
        prices.append(price)
    
    for i, price in enumerate(prices):
        test_data.append(MarketData(
            timestamp=base_time + timedelta(minutes=i),
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=1000,
            timeframe=TimeFrame.M1
        ))
    
    # Test ROC calculation
    roc = ROC(TimeFrame.M1, lookback_periods=12)
    result = roc.calculate(test_data)
    
    print(f"ROC Test Results:")
    print(f"Value: {result.value:.2f}%")
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
    test_roc()
