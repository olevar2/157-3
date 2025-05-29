"""
Keltner Channels - Advanced Volatility-Based Trading Bands
Platform3 - Humanitarian Trading System
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from ..indicator_base import BaseIndicator, IndicatorResult, IndicatorSignal, SignalType, IndicatorType, MarketData, TimeFrame

@dataclass
class KeltnerChannelsConfig:
    """Configuration for Keltner Channels"""
    ema_period: int = 20
    atr_period: int = 14
    multiplier: float = 2.0

class KeltnerChannels(BaseIndicator):
    """
    Keltner Channels Implementation
    
    Features:
    - EMA-based center line with ATR volatility bands
    - Breakout and squeeze detection
    - Trend direction analysis
    - Enhanced signal generation with squeeze analysis
    - Band width monitoring for volatility analysis
    """
    
    def __init__(self, config: Optional[KeltnerChannelsConfig] = None):
        super().__init__(
            name="Keltner Channels",
            indicator_type=IndicatorType.VOLATILITY,
            timeframe=TimeFrame.H1,
            lookback_periods=25
        )
        self.config = config or KeltnerChannelsConfig()
        # Historical tracking for enhanced analysis
        self.band_widths: List[float] = []
        self.price_positions: List[str] = []  # 'above', 'below', 'within'
        self.max_history = 100  # Limit memory usage
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """Calculate Keltner Channels values"""
        try:
            if len(data) < self.config.atr_period + 5:
                return IndicatorResult(
                    timestamp=data[-1].timestamp if data else datetime.now(),
                    indicator_name=self.name,
                    indicator_type=self.indicator_type,
                    timeframe=self.timeframe,
                    value=0.0,
                    raw_data={'error': f"Insufficient data"}
                )
              # Calculate EMA (center line)
            closes = [candle.close for candle in data]
            ema_values = self._calculate_ema(closes, self.config.ema_period)
            
            # Calculate ATR
            atr_values = self._calculate_atr(data, self.config.atr_period)
              # Calculate bands
            current_ema = ema_values[-1] if ema_values else 0
            current_atr = atr_values[-1] if atr_values else 0
            
            upper_band = current_ema + (current_atr * self.config.multiplier)
            lower_band = current_ema - (current_atr * self.config.multiplier)
            
            # Calculate band width and store for squeeze analysis
            band_width = self._calculate_band_width(upper_band, lower_band, current_ema)
            self.band_widths.append(band_width)
            
            # Limit memory usage
            if len(self.band_widths) > self.max_history:
                self.band_widths = self.band_widths[-self.max_history:]
            
            # Determine price position
            current_price = data[-1].close
            if current_price > upper_band:
                position = 'above'
            elif current_price < lower_band:
                position = 'below'
            else:
                position = 'within'
            self.price_positions.append(position)
            
            # Limit memory usage
            if len(self.price_positions) > self.max_history:
                self.price_positions = self.price_positions[-self.max_history:]
            
            # Generate enhanced signal
            signal = self._generate_signal(current_price, upper_band, current_ema, lower_band, 
                                         data[-1].timestamp, data)
            
            # Add squeeze analysis to result
            squeeze_info = self._detect_squeeze(self.band_widths) if len(self.band_widths) >= 20 else {}
            
            return IndicatorResult(
                timestamp=data[-1].timestamp,
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value={
                    'upper_band': upper_band,
                    'center_line': current_ema,
                    'lower_band': lower_band,
                    'atr': current_atr,
                    'channel_width': upper_band - lower_band,
                    'band_width_pct': band_width,
                    'price_position': position,
                    'squeeze_analysis': squeeze_info
                },
                signal=signal
            )
            
        except Exception as e:
            return IndicatorResult(
                timestamp=data[-1].timestamp if data else datetime.now(),
                indicator_name=self.name,
                indicator_type=self.indicator_type,
                timeframe=self.timeframe,
                value=0.0,
                raw_data={'error': f"Calculation failed: {str(e)}"}
            )
    
    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(values) < period:
            return []
        
        alpha = 2.0 / (period + 1)
        ema_values = []
        
        # Start with SMA
        ema = sum(values[:period]) / period
        ema_values.append(ema)
        
        # Calculate EMA
        for i in range(period, len(values)):
            ema = (values[i] * alpha) + (ema * (1 - alpha))
            ema_values.append(ema)
        
        return ema_values
    
    def _calculate_atr(self, data: List[MarketData], period: int) -> List[float]:
        """Calculate Average True Range"""
        if len(data) < period + 1:
            return []
        
        true_ranges = []
        for i in range(1, len(data)):
            high_low = data[i].high - data[i].low
            high_close = abs(data[i].high - data[i-1].close)
            low_close = abs(data[i].low - data[i-1].close)
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
          # Calculate ATR
        atr_values = []
        for i in range(period - 1, len(true_ranges)):
            atr = sum(true_ranges[i - period + 1:i + 1]) / period
            atr_values.append(atr)
        
        return atr_values
    
    def _calculate_band_width(self, upper: float, lower: float, center: float) -> float:
        """Calculate normalized band width for squeeze detection"""
        if center == 0:
            return 0
        return ((upper - lower) / center) * 100
    
    def _detect_squeeze(self, band_widths: List[float], lookback: int = 20) -> dict:
        """Detect volatility squeeze conditions"""
        if len(band_widths) < lookback:
            return {'is_squeeze': False, 'squeeze_level': 50, 'squeeze_duration': 0}
        
        current_width = band_widths[-1]
        recent_widths = band_widths[-lookback:]
        avg_width = sum(recent_widths) / len(recent_widths)
        min_width = min(recent_widths)
        
        # Calculate squeeze metrics
        squeeze_percentile = sum(1 for w in recent_widths if w > current_width) / len(recent_widths) * 100
        is_squeeze = squeeze_percentile > 80  # Current width in bottom 20%
        
        # Calculate squeeze duration
        squeeze_duration = 0
        for i in range(len(band_widths) - 1, -1, -1):
            if band_widths[i] <= min_width * 1.1:  # Within 10% of minimum
                squeeze_duration += 1
            else:
                break
        
        return {
            'is_squeeze': is_squeeze,
            'squeeze_level': squeeze_percentile,
            'squeeze_duration': squeeze_duration,
            'current_vs_avg': current_width / avg_width if avg_width > 0 else 1
        }
    
    def _generate_signal(self, price: float, upper: float, center: float, 
                        lower: float, timestamp: datetime, data: List = None) -> IndicatorSignal:
        """Generate enhanced Keltner Channels signal with squeeze and breakout analysis"""
        
        # Calculate band width for current period
        band_width = self._calculate_band_width(upper, lower, center)
        
        # Basic position analysis
        if price > upper:
            distance_pct = ((price - upper) / (upper - center)) * 100
            if distance_pct > 10:  # Strong breakout
                signal_type = SignalType.BUY  # Bullish breakout                strength = min(0.9, 0.6 + (distance_pct / 100))
                pattern = "strong_bullish_breakout"
            else:
                signal_type = SignalType.SELL  # Potential resistance
                strength = 0.4
                pattern = "upper_resistance"
        
        elif price < lower:
            distance_pct = ((lower - price) / (center - lower)) * 100
            if distance_pct > 10:  # Strong breakout
                signal_type = SignalType.SELL  # Bearish breakout
                strength = min(0.9, 0.6 + (distance_pct / 100))
                pattern = "strong_bearish_breakout"
            else:
                signal_type = SignalType.BUY  # Potential support
                strength = 0.4
                pattern = "lower_support"
        
        else:
            # Price within bands - analyze position
            if upper - center == 0:  # Avoid division by zero
                center_distance = 0
            else:
                center_distance = abs(price - center) / (upper - center)
                
            if center_distance < 0.3:  # Near center line
                signal_type = SignalType.HOLD
                strength = 0.2
                pattern = "near_center_line"
            else:
                signal_type = SignalType.HOLD
                strength = 0.3
                pattern = "within_bands"
        
        # Enhanced metadata with squeeze analysis
        metadata = {
            'pattern': pattern,
            'upper': upper,
            'center': center,
            'lower': lower,
            'band_width': band_width,
            'price_position': 'above' if price > center else 'below',
            'distance_from_center_pct': ((price - center) / (upper - center)) * 100 if (upper - center) != 0 else 0
        }
        
        # Add squeeze analysis if we have historical data
        if hasattr(self, 'band_widths') and len(self.band_widths) > 0:
            squeeze_info = self._detect_squeeze(self.band_widths + [band_width])
            metadata.update(squeeze_info)
            
            # Adjust signal based on squeeze conditions
            if squeeze_info['is_squeeze'] and squeeze_info['squeeze_duration'] > 5:
                pattern += "_squeeze_setup"
                if signal_type in [SignalType.BUY, SignalType.SELL]:
                    strength = min(0.95, strength * 1.3)  # Boost signal strength
        
        return IndicatorSignal(
            timestamp=timestamp,
            indicator_name=self.name,
            signal_type=signal_type,
            strength=strength,
            confidence=0.8,
            metadata=metadata
        )
    
    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """Generate trading signal"""
        return current_result.signal


def test_keltner_channels():
    """Test Keltner Channels implementation"""
    # Create test data
    np.random.seed(42)
    test_data = []
    base_price = 100.0
    
    for i in range(50):
        trend = 0.002 if i < 25 else -0.001
        volatility = np.random.normal(0, 0.01)
        close_price = base_price * (1 + trend + volatility)
        
        open_price = close_price * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = 1000 + np.random.randint(0, 500)
        
        test_data.append(MarketData(
            timestamp=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timeframe=TimeFrame.H1
        ))
        
        base_price = close_price
    
    # Test Keltner Channels
    keltner = KeltnerChannels()
    result = keltner.calculate(test_data)
    
    print("=== KELTNER CHANNELS TEST ===")
    success = result.raw_data is None or 'error' not in result.raw_data
    print(f"Success: {success}")
    
    if success:
        values = result.value
        print(f"Upper Band: {values['upper_band']:.4f}")
        print(f"Center Line: {values['center_line']:.4f}")
        print(f"Lower Band: {values['lower_band']:.4f}")
        print(f"ATR: {values['atr']:.4f}")
        
        if result.signal:
            print(f"Signal: {result.signal.signal_type.value} - Strength: {result.signal.strength:.2f}")
    else:
        print(f"Error: {result.raw_data.get('error', 'Unknown error')}")
    
    return success


if __name__ == "__main__":
    test_keltner_channels()
