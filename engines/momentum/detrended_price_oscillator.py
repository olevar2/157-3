"""
Detrended Price Oscillator (DPO) - Price Cycle Analysis Indicator
Removes trend from price to identify underlying cycles and overbought/oversold conditions.
Used for cycle analysis and identifying price patterns without trend bias.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..indicator_base import BaseIndicator, IndicatorResult, IndicatorSignal, SignalType, IndicatorType, MarketData, TimeFrame


@dataclass
class DPOConfig:
    """Configuration for Detrended Price Oscillator"""
    period: int = 21
    price_type: str = 'close'  # 'close', 'high', 'low', 'hl2', 'hlc3', 'ohlc4'
    signal_threshold: float = 0.5  # Percentage threshold for signals
    overbought_threshold: float = 1.5  # Standard deviations for overbought
    oversold_threshold: float = -1.5  # Standard deviations for oversold


class DetrendedPriceOscillator(BaseIndicator):
    """
    Detrended Price Oscillator Implementation
    
    DPO = Price[X] - SMA[n][(n/2) + 1] periods ago
    Where:
    - X = current period
    - n = period length
    - SMA[n] = Simple Moving Average over n periods
    
    The oscillator removes trend by comparing current price to a displaced moving average,    revealing cyclical price movements and potential reversal points.
    """
    
    def __init__(self, config: Optional[DPOConfig] = None):
        super().__init__(
            name="Detrended Price Oscillator",
            indicator_type=IndicatorType.MOMENTUM,
            timeframe=TimeFrame.H1,  # Default timeframe
            lookback_periods=25  # period + 5 for displacement
        )
        self.config = config or DPOConfig()
        self.values: List[float] = []
        self.prices: List[float] = []
        self.sma_values: List[float] = []
        
    def calculate(self, data: List[MarketData]) -> IndicatorResult:
        """Calculate Detrended Price Oscillator values"""
        try:
            if len(data) < self.config.period + 5:
                return IndicatorResult(
                    success=False,
                    error=f"Insufficient data: need {self.config.period + 5}, got {len(data)}"
                )
            
            # Extract prices based on price type
            prices = self._extract_prices(data, self.config.price_type)
            
            # Calculate Simple Moving Average
            sma_values = self._calculate_sma(prices, self.config.period)
            
            # Calculate displacement (periods to look back for SMA)
            displacement = (self.config.period // 2) + 1
            
            # Calculate DPO values
            dpo_values = []
            for i in range(len(prices)):
                if i >= self.config.period - 1 + displacement:
                    # Current price minus SMA from displacement periods ago
                    sma_index = i - displacement
                    if sma_index < len(sma_values):
                        dpo = prices[i] - sma_values[sma_index]
                        dpo_values.append(dpo)
                    else:
                        dpo_values.append(0.0)
            
            # Store values for signal generation
            self.values = dpo_values
            self.prices = prices
            self.sma_values = sma_values
            
            # Generate signals
            signals = self._generate_signals(data, dpo_values, prices)
            
            # Calculate statistics
            stats = self._calculate_statistics(dpo_values)
            
            return IndicatorResult(
                success=True,
                values={
                    'dpo': dpo_values,
                    'prices': prices,
                    'sma': sma_values,
                    'normalized_dpo': self._normalize_dpo(dpo_values)
                },
                signals=signals,
                metadata={
                    'period': self.config.period,
                    'displacement': displacement,
                    'price_type': self.config.price_type,
                    'current_dpo': dpo_values[-1] if dpo_values else 0,
                    'dpo_position': self._get_dpo_position(dpo_values[-1] if dpo_values else 0, stats),
                    'cycle_phase': self._identify_cycle_phase(dpo_values),
                    **stats
                }
            )
            
        except Exception as e:
            return IndicatorResult(
                success=False,
                error=f"DPO calculation failed: {str(e)}"
            )
    
    def _extract_prices(self, data: List[MarketData], price_type: str) -> List[float]:
        """Extract prices based on specified type"""
        if price_type == 'close':
            return [candle.close for candle in data]
        elif price_type == 'high':
            return [candle.high for candle in data]
        elif price_type == 'low':
            return [candle.low for candle in data]
        elif price_type == 'hl2':
            return [(candle.high + candle.low) / 2 for candle in data]
        elif price_type == 'hlc3':
            return [(candle.high + candle.low + candle.close) / 3 for candle in data]
        elif price_type == 'ohlc4':
            return [(candle.open + candle.high + candle.low + candle.close) / 4 for candle in data]
        else:
            return [candle.close for candle in data]  # Default to close
    
    def _calculate_sma(self, values: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma_values = []
        for i in range(len(values)):
            if i >= period - 1:
                sma = sum(values[i - period + 1:i + 1]) / period
                sma_values.append(sma)
        return sma_values
    
    def _calculate_statistics(self, dpo_values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for DPO"""
        if not dpo_values:
            return {}
        
        mean_dpo = np.mean(dpo_values)
        std_dpo = np.std(dpo_values)
        max_dpo = max(dpo_values)
        min_dpo = min(dpo_values)
        
        return {
            'dpo_mean': mean_dpo,
            'dpo_std': std_dpo,
            'dpo_max': max_dpo,
            'dpo_min': min_dpo,
            'dpo_range': max_dpo - min_dpo,
            'overbought_level': mean_dpo + (std_dpo * self.config.overbought_threshold),
            'oversold_level': mean_dpo + (std_dpo * self.config.oversold_threshold)
        }
    
    def _normalize_dpo(self, dpo_values: List[float]) -> List[float]:
        """Normalize DPO values to standard deviations from mean"""
        if not dpo_values:
            return []
        
        mean_dpo = np.mean(dpo_values)
        std_dpo = np.std(dpo_values)
        
        if std_dpo == 0:
            return [0.0] * len(dpo_values)
        
        return [(value - mean_dpo) / std_dpo for value in dpo_values]
    
    def _get_dpo_position(self, current_dpo: float, stats: Dict[str, float]) -> str:
        """Determine current DPO position relative to statistical levels"""
        if not stats:
            return 'neutral'
        
        overbought = stats.get('overbought_level', 0)
        oversold = stats.get('oversold_level', 0)
        mean = stats.get('dpo_mean', 0)
        
        if current_dpo >= overbought:
            return 'overbought'
        elif current_dpo <= oversold:
            return 'oversold'
        elif current_dpo > mean:
            return 'above_mean'
        elif current_dpo < mean:
            return 'below_mean'
        else:
            return 'neutral'
    
    def _identify_cycle_phase(self, dpo_values: List[float]) -> str:
        """Identify current cycle phase based on DPO pattern"""
        if len(dpo_values) < 5:
            return 'unknown'
        
        recent = dpo_values[-5:]
        
        # Analyze trend direction
        if all(recent[i] >= recent[i-1] for i in range(1, len(recent))):
            return 'rising'
        elif all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
            return 'falling'
        elif recent[-1] > recent[0]:
            return 'uptrend'
        elif recent[-1] < recent[0]:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _generate_signals(self, data: List[MarketData], dpo_values: List[float], prices: List[float]) -> List[IndicatorSignal]:
        """Generate trading signals based on DPO analysis"""
        signals = []
        
        if len(dpo_values) < 3:
            return signals
        
        # Calculate statistics for thresholds
        stats = self._calculate_statistics(dpo_values)
        
        # Overbought/Oversold signals
        self._detect_extreme_levels(dpo_values, stats, signals)
        
        # Zero line crossover signals
        self._detect_zero_crossover(dpo_values, signals)
        
        # Divergence signals
        self._detect_divergence(dpo_values, prices, signals)
        
        # Cycle turning points
        self._detect_cycle_extremes(dpo_values, signals)
        
        return signals
    
    def _detect_extreme_levels(self, dpo_values: List[float], stats: Dict[str, float], signals: List[IndicatorSignal]):
        """Detect overbought/oversold levels"""
        if not stats or len(dpo_values) < 2:
            return
        
        current = dpo_values[-1]
        overbought = stats.get('overbought_level', 0)
        oversold = stats.get('oversold_level', 0)
        
        # Overbought condition (potential sell signal)
        if current >= overbought:
            signals.append(IndicatorSignal(
                signal_type=SignalType.SELL,
                strength=0.7,
                confidence=0.75,
                metadata={
                    'pattern': 'dpo_overbought',
                    'dpo_value': current,
                    'overbought_level': overbought,
                    'excess': current - overbought
                }
            ))
        
        # Oversold condition (potential buy signal)
        elif current <= oversold:
            signals.append(IndicatorSignal(
                signal_type=SignalType.BUY,
                strength=0.7,
                confidence=0.75,
                metadata={
                    'pattern': 'dpo_oversold',
                    'dpo_value': current,
                    'oversold_level': oversold,
                    'excess': oversold - current
                }
            ))
    
    def _detect_zero_crossover(self, dpo_values: List[float], signals: List[IndicatorSignal]):
        """Detect zero line crossover signals"""
        if len(dpo_values) < 2:
            return
        
        current = dpo_values[-1]
        previous = dpo_values[-2]
        
        # Bullish crossover (from negative to positive)
        if previous <= 0 < current:
            signals.append(IndicatorSignal(
                signal_type=SignalType.BUY,
                strength=0.6,
                confidence=0.7,
                metadata={
                    'pattern': 'dpo_zero_crossover_bullish',
                    'dpo_value': current,
                    'crossover_strength': current
                }
            ))
        
        # Bearish crossover (from positive to negative)
        elif previous >= 0 > current:
            signals.append(IndicatorSignal(
                signal_type=SignalType.SELL,
                strength=0.6,
                confidence=0.7,
                metadata={
                    'pattern': 'dpo_zero_crossover_bearish',
                    'dpo_value': current,
                    'crossover_strength': abs(current)
                }
            ))
    
    def _detect_divergence(self, dpo_values: List[float], prices: List[float], signals: List[IndicatorSignal]):
        """Detect price-DPO divergences"""
        if len(dpo_values) < 10 or len(prices) < 10:
            return
        
        # Look for divergence in recent data
        lookback = min(10, len(dpo_values))
        recent_dpo = dpo_values[-lookback:]
        recent_prices = prices[-len(recent_dpo):]
        
        # Find peaks and troughs
        dpo_peaks = self._find_peaks(recent_dpo, True)
        dpo_troughs = self._find_peaks(recent_dpo, False)
        price_peaks = self._find_peaks(recent_prices, True)
        price_troughs = self._find_peaks(recent_prices, False)
        
        # Bullish divergence (price makes lower low, DPO makes higher low)
        if len(price_troughs) >= 2 and len(dpo_troughs) >= 2:
            if (price_troughs[-1] < price_troughs[-2] and 
                dpo_troughs[-1] > dpo_troughs[-2]):
                signals.append(IndicatorSignal(
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    confidence=0.7,
                    metadata={
                        'pattern': 'dpo_bullish_divergence',
                        'price_trend': 'lower_low',
                        'dpo_trend': 'higher_low'
                    }
                ))
        
        # Bearish divergence (price makes higher high, DPO makes lower high)
        if len(price_peaks) >= 2 and len(dpo_peaks) >= 2:
            if (price_peaks[-1] > price_peaks[-2] and 
                dpo_peaks[-1] < dpo_peaks[-2]):
                signals.append(IndicatorSignal(
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    confidence=0.7,
                    metadata={
                        'pattern': 'dpo_bearish_divergence',
                        'price_trend': 'higher_high',
                        'dpo_trend': 'lower_high'
                    }
                ))
    
    def _detect_cycle_extremes(self, dpo_values: List[float], signals: List[IndicatorSignal]):
        """Detect cycle turning points"""
        if len(dpo_values) < 5:
            return
        
        # Check for cycle peak or trough
        recent = dpo_values[-5:]
        middle_index = len(recent) // 2
        middle_value = recent[middle_index]
        
        # Local maximum (cycle peak)
        is_peak = all(middle_value >= recent[i] for i in range(len(recent)) if i != middle_index)
        
        # Local minimum (cycle trough)
        is_trough = all(middle_value <= recent[i] for i in range(len(recent)) if i != middle_index)
        
        if is_peak and middle_value > 0:
            signals.append(IndicatorSignal(
                signal_type=SignalType.SELL,
                strength=0.5,
                confidence=0.6,
                metadata={
                    'pattern': 'dpo_cycle_peak',
                    'peak_value': middle_value,
                    'cycle_position': 'top'
                }
            ))
        
        elif is_trough and middle_value < 0:
            signals.append(IndicatorSignal(
                signal_type=SignalType.BUY,
                strength=0.5,
                confidence=0.6,
                metadata={
                    'pattern': 'dpo_cycle_trough',
                    'trough_value': middle_value,
                    'cycle_position': 'bottom'
                }
            ))
    
    def _find_peaks(self, values: List[float], find_max: bool = True) -> List[float]:
        """Find local peaks (max or min) in the values"""
        peaks = []
        for i in range(1, len(values) - 1):
            if find_max:
                if values[i] > values[i-1] and values[i] > values[i+1]:
                    peaks.append(values[i])
            else:
                if values[i] < values[i-1] and values[i] < values[i+1]:
                    peaks.append(values[i])
        return peaks

    def generate_signal(self, current_result: IndicatorResult, 
                       historical_results: List[IndicatorResult]) -> Optional[IndicatorSignal]:
        """Generate a single signal based on current DPO state"""
        if not current_result.success or not current_result.values:
            return None
            
        dpo_values = current_result.values.get('dpo', [])
        if not dpo_values or len(dpo_values) < 3:
            return None
        
        current_dpo = dpo_values[-1]
        metadata = current_result.metadata
        
        # Primary signal based on extreme levels
        if current_dpo >= metadata.get('overbought_level', 0):
            return IndicatorSignal(
                signal_type=SignalType.SELL,
                strength=0.7,
                confidence=0.75,
                metadata={
                    'pattern': 'dpo_overbought',
                    'dpo_value': current_dpo,
                    'position': metadata.get('dpo_position', 'unknown')
                }
            )
        elif current_dpo <= metadata.get('oversold_level', 0):
            return IndicatorSignal(
                signal_type=SignalType.BUY,
                strength=0.7,
                confidence=0.75,
                metadata={
                    'pattern': 'dpo_oversold',
                    'dpo_value': current_dpo,
                    'position': metadata.get('dpo_position', 'unknown')
                }
            )
        
        # Zero line crossover signals
        if len(dpo_values) >= 2:
            previous_dpo = dpo_values[-2]
            if previous_dpo <= 0 < current_dpo:
                return IndicatorSignal(
                    signal_type=SignalType.BUY,
                    strength=0.6,
                    confidence=0.7,
                    metadata={
                        'pattern': 'dpo_zero_crossover_bullish',
                        'dpo_value': current_dpo
                    }
                )
            elif previous_dpo >= 0 > current_dpo:
                return IndicatorSignal(
                    signal_type=SignalType.SELL,
                    strength=0.6,
                    confidence=0.7,
                    metadata={
                        'pattern': 'dpo_zero_crossover_bearish',
                        'dpo_value': current_dpo
                    }
                )
        
        # Default to None (no signal)
        return None


def test_detrended_price_oscillator():
    """Test the Detrended Price Oscillator implementation"""
    # Create test data with cycles
    np.random.seed(42)
    test_data = []
    base_price = 100.0
    
    for i in range(60):
        # Generate cyclical data with trend
        cycle = 5 * np.sin(i * 0.3)  # Main cycle
        trend = 0.05 * i  # Small upward trend
        noise = np.random.normal(0, 0.5)
        
        close_price = base_price + cycle + trend + noise
        open_price = close_price + np.random.normal(0, 0.2)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
        volume = 1000 + np.random.randint(0, 500)
        
        test_data.append(MarketData(
            timestamp=i,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timeframe='1H'
        ))
    
    # Test DPO
    dpo = DetrendedPriceOscillator()
    result = dpo.calculate(test_data)
    
    print("=== DETRENDED PRICE OSCILLATOR TEST ===")
    print(f"Success: {result.success}")
    if result.success:
        dpo_values = result.values['dpo']
        print(f"DPO Values (last 5): {[round(v, 4) for v in dpo_values[-5:]]}")
        print(f"Current DPO: {round(dpo_values[-1], 4)}")
        print(f"DPO Position: {result.metadata['dpo_position']}")
        print(f"Cycle Phase: {result.metadata['cycle_phase']}")
        print(f"Overbought Level: {round(result.metadata['overbought_level'], 4)}")
        print(f"Oversold Level: {round(result.metadata['oversold_level'], 4)}")
        print(f"Number of signals: {len(result.signals)}")
        
        for i, signal in enumerate(result.signals[-3:]):  # Show last 3 signals
            print(f"Signal {i+1}: {signal.signal_type.value} - Strength: {signal.strength:.2f} - {signal.metadata.get('pattern', 'N/A')}")
    else:
        print(f"Error: {result.error}")
    
    return result.success


if __name__ == "__main__":
    test_detrended_price_oscillator()
