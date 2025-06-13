"""
Vortex Indicator - Platform3 Financial Indicator

The Vortex Indicator (VI) was developed by Etienne Botes and Douglas Siepman to identify
the start of a new trend or the continuation of an existing trend in financial markets.
It consists of two oscillating lines, VI+ and VI-, that capture positive and negative
vortex movements.

Platform3 compliant implementation with CCI proven patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import logging
import sys
import os

# Add the base path for imports when running as script
if __name__ == "__main__":
    
try:
    from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
except ImportError:
    # For direct script execution
        from base_indicator import StandardIndicatorInterface


class VortexIndicator(StandardIndicatorInterface):
    """
    Vortex Indicator - Platform3 Implementation
    
    The Vortex Indicator measures the relationship between closing prices and the
    trading range (high and low) to determine the direction of the trend.
    
    Formula:
    VI+ = Sum(|High[i] - Low[i-1]|, period) / Sum(True Range, period)
    VI- = Sum(|Low[i] - High[i-1]|, period) / Sum(True Range, period)
    
    Where True Range = max(High-Low, |High-Close[i-1]|, |Low-Close[i-1]|)
    
    Signals:
    - VI+ > VI-: Bullish trend
    - VI- > VI+: Bearish trend
    - Crossovers indicate trend changes
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Performance Optimization  
    - Robust Error Handling
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize Vortex Indicator.
        
        Args:
            period (int): Lookback period for calculation (default: 14)
        """
        self.period = period
        self.name = "VortexIndicator"
        self.version = "1.0.0"
        self.logger = logging.getLogger(f"Platform3.{self.name}")
        
        # Initialize base class without calling super().__init__()
        self.metadata = None
        self._is_initialized = False
        self._last_calculation = None
        self._performance_stats = {}
        
        self.logger.info(f"{self.name} initialized with period={period}")

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return {"period": self.period}

    def validate_parameters(self) -> bool:
        """Validate parameters."""
        return isinstance(self.period, int) and self.period > 0

    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate True Range."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value
        
        range1 = high - low
        range2 = np.abs(high - prev_close)
        range3 = np.abs(low - prev_close)
        
        return np.maximum(range1, np.maximum(range2, range3))

    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate Vortex Indicator.
        
        Args:
            data: Market data (DataFrame with OHLC columns or 4D array [open, high, low, close])
            
        Returns:
            Dict containing Vortex Indicator values and analysis
        """
        try:
            # Extract OHLC prices
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    open_prices = data['open'].values
                    high_prices = data['high'].values
                    low_prices = data['low'].values
                    close_prices = data['close'].values
                elif all(col in data.columns for col in ['high', 'low', 'close']):
                    high_prices = data['high'].values
                    low_prices = data['low'].values
                    close_prices = data['close'].values
                    open_prices = close_prices  # Use close as open if not available
                else:
                    return {"error": "Insufficient OHLC data columns"}
            else:
                if data.ndim == 2 and data.shape[1] >= 3:
                    if data.shape[1] == 4:
                        open_prices = data[:, 0]
                        high_prices = data[:, 1]
                        low_prices = data[:, 2]
                        close_prices = data[:, 3]
                    else:  # Assume HLC
                        high_prices = data[:, 0]
                        low_prices = data[:, 1]
                        close_prices = data[:, 2]
                        open_prices = close_prices
                else:
                    return {"error": "Data must have at least 3 columns (H, L, C)"}
            
            if len(high_prices) < self.minimum_periods:
                return {"error": f"Insufficient data. Need at least {self.minimum_periods} periods"}
            
            # Calculate previous values
            prev_high = np.roll(high_prices, 1)
            prev_low = np.roll(low_prices, 1)
            prev_close = np.roll(close_prices, 1)
            
            # Handle first values
            prev_high[0] = high_prices[0]
            prev_low[0] = low_prices[0]
            prev_close[0] = close_prices[0]
            
            # Calculate vortex movements
            vm_plus = np.abs(high_prices - prev_low)
            vm_minus = np.abs(low_prices - prev_high)
            
            # Calculate true range
            true_range = self._calculate_true_range(high_prices, low_prices, close_prices)
            
            # Calculate Vortex Indicators
            vi_plus = np.full(len(high_prices), np.nan)
            vi_minus = np.full(len(high_prices), np.nan)
            
            for i in range(self.period - 1, len(high_prices)):
                sum_vm_plus = np.sum(vm_plus[i - self.period + 1:i + 1])
                sum_vm_minus = np.sum(vm_minus[i - self.period + 1:i + 1])
                sum_tr = np.sum(true_range[i - self.period + 1:i + 1])
                
                if sum_tr != 0:
                    vi_plus[i] = sum_vm_plus / sum_tr
                    vi_minus[i] = sum_vm_minus / sum_tr
            
            # Generate signals
            signals = self._generate_signals(vi_plus, vi_minus)
            
            # Quality assessment
            quality_score = self._assess_quality(vi_plus, vi_minus, true_range)
            
            return {
                "vi_plus": vi_plus,
                "vi_minus": vi_minus,
                "vi_plus_values": vi_plus[~np.isnan(vi_plus)].tolist(),
                "vi_minus_values": vi_minus[~np.isnan(vi_minus)].tolist(),
                "current_vi_plus": float(vi_plus[-1]) if not np.isnan(vi_plus[-1]) else None,
                "current_vi_minus": float(vi_minus[-1]) if not np.isnan(vi_minus[-1]) else None,
                "vm_plus": vm_plus,
                "vm_minus": vm_minus,
                "true_range": true_range,
                "signals": signals,
                "quality_score": quality_score,
                "signal_strength": self._calculate_signal_strength(vi_plus, vi_minus),
                "trend_state": self._determine_trend_state(vi_plus, vi_minus)
            }
            
        except Exception as e:
            self.logger.error(f"Vortex Indicator calculation error: {e}")
            return {"error": str(e)}

    def _generate_signals(self, vi_plus: np.ndarray, vi_minus: np.ndarray) -> Dict[str, List[int]]:
        """Generate trading signals based on Vortex Indicator crossovers."""
        signals = {
            "bullish_crossover": [],
            "bearish_crossover": [],
            "strong_bullish": [],
            "strong_bearish": []
        }
        
        for i in range(1, len(vi_plus)):
            if np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(vi_plus[i-1]) or np.isnan(vi_minus[i-1]):
                continue
            
            # Bullish crossover: VI+ crosses above VI-
            if vi_plus[i-1] <= vi_minus[i-1] and vi_plus[i] > vi_minus[i]:
                signals["bullish_crossover"].append(i)
            
            # Bearish crossover: VI- crosses above VI+
            if vi_minus[i-1] <= vi_plus[i-1] and vi_minus[i] > vi_plus[i]:
                signals["bearish_crossover"].append(i)
            
            # Strong signals (significant separation)
            if vi_plus[i] > vi_minus[i] + 0.1:
                signals["strong_bullish"].append(i)
            elif vi_minus[i] > vi_plus[i] + 0.1:
                signals["strong_bearish"].append(i)
        
        return signals

    def _assess_quality(self, vi_plus: np.ndarray, vi_minus: np.ndarray, true_range: np.ndarray) -> float:
        """Assess the quality of Vortex Indicator calculation."""
        try:
            valid_plus = vi_plus[~np.isnan(vi_plus)]
            valid_minus = vi_minus[~np.isnan(vi_minus)]
            
            if len(valid_plus) == 0 or len(valid_minus) == 0:
                return 0.0
            
            # Check for reasonable values (typically between 0.5 and 2.0)
            reasonable_plus = np.logical_and(valid_plus > 0.3, valid_plus < 3.0)
            reasonable_minus = np.logical_and(valid_minus > 0.3, valid_minus < 3.0)
            range_quality = (np.mean(reasonable_plus) + np.mean(reasonable_minus)) / 2
            
            # Check volatility consistency
            volatility = np.std(true_range[-min(50, len(true_range)):])
            volatility_quality = min(1.0, volatility / np.mean(true_range[-min(50, len(true_range)):]))
            
            # Data completeness
            completeness = (len(valid_plus) + len(valid_minus)) / (2 * len(vi_plus))
            
            # Oscillation quality (good indicators should cross each other regularly)
            crossovers = 0
            for i in range(1, min(len(valid_plus), len(valid_minus))):
                if ((valid_plus[i-1] < valid_minus[i-1] and valid_plus[i] > valid_minus[i]) or
                    (valid_plus[i-1] > valid_minus[i-1] and valid_plus[i] < valid_minus[i])):
                    crossovers += 1
            oscillation_quality = min(1.0, crossovers / max(1, len(valid_plus) / 20))
            
            return float(np.mean([range_quality, volatility_quality, completeness, oscillation_quality]))
            
        except Exception:
            return 0.5

    def _calculate_signal_strength(self, vi_plus: np.ndarray, vi_minus: np.ndarray) -> float:
        """Calculate signal strength based on VI separation and trend consistency."""
        try:
            valid_indices = ~(np.isnan(vi_plus) | np.isnan(vi_minus))
            if not np.any(valid_indices):
                return 0.0
            
            recent_plus = vi_plus[valid_indices][-min(10, np.sum(valid_indices)):]
            recent_minus = vi_minus[valid_indices][-min(10, np.sum(valid_indices)):]
            
            if len(recent_plus) == 0 or len(recent_minus) == 0:
                return 0.0
            
            # Signal strength based on separation and consistency
            separation = np.abs(recent_plus[-1] - recent_minus[-1])
            consistency = 1.0 - np.std(recent_plus - recent_minus) / max(0.1, np.mean(np.abs(recent_plus - recent_minus)))
            
            # Normalize signal strength
            separation_factor = min(1.0, separation / 0.5)  # 0.5 is considered strong separation
            consistency_factor = min(1.0, max(0.0, consistency))
            
            return float((separation_factor + consistency_factor) / 2)
            
        except Exception:
            return 0.5

    def _determine_trend_state(self, vi_plus: np.ndarray, vi_minus: np.ndarray) -> str:
        """Determine current trend state based on Vortex Indicator."""
        try:
            if np.isnan(vi_plus[-1]) or np.isnan(vi_minus[-1]):
                return "undefined"
            
            diff = vi_plus[-1] - vi_minus[-1]
            
            if diff > 0.1:
                return "strong_bullish"
            elif diff > 0.05:
                return "bullish"
            elif diff > -0.05:
                return "neutral"
            elif diff > -0.1:
                return "bearish"
            else:
                return "strong_bearish"
                
        except Exception:
            return "undefined"

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required for calculation."""
        return self.period

    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "category": "trend",
            "subcategory": "trend_following",
            "parameters": self.parameters,
            "output_keys": [
                "vi_plus", "vi_minus", "vi_plus_values", "vi_minus_values",
                "current_vi_plus", "current_vi_minus", "vm_plus", "vm_minus",
                "true_range", "signals", "quality_score", "signal_strength", "trend_state"
            ],
            "minimum_periods": self.minimum_periods,
            "platform3_compliant": True,
            "description": "Vortex Indicator for trend identification and momentum analysis"
        }


def export_indicator():
    """Export the indicator for registry discovery."""
    return VortexIndicator


# Test code
if __name__ == "__main__":
    print("Testing Vortex Indicator...")
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create realistic OHLC data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate realistic OHLC from close prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    # Test the indicator
    vortex = VortexIndicator(period=14)
    result = vortex.calculate(test_data)
    
    if "error" not in result:
        print("[OK] Vortex Indicator calculation successful")
        print(f"[OK] Current VI+: {result.get('current_vi_plus', 'N/A'):.4f}")
        print(f"[OK] Current VI-: {result.get('current_vi_minus', 'N/A'):.4f}")
        print(f"[OK] Trend State: {result['trend_state']}")
        print(f"[OK] Quality Score: {result['quality_score']:.4f}")
        print(f"[OK] Signal Strength: {result['signal_strength']:.4f}")
        print(f"[OK] Bullish Crossovers: {len(result['signals']['bullish_crossover'])}")
        print(f"[OK] Bearish Crossovers: {len(result['signals']['bearish_crossover'])}")
        
        # Show some recent values
        if result['vi_plus_values']:
            recent_plus = result['vi_plus_values'][-5:]
            recent_minus = result['vi_minus_values'][-5:]
            print(f"[OK] Recent VI+ values: {[f'{v:.4f}' for v in recent_plus]}")
            print(f"[OK] Recent VI- values: {[f'{v:.4f}' for v in recent_minus]}")
    else:
        print(f"[ERROR] Error: {result['error']}")
    
    # Test metadata
    metadata = vortex.get_metadata()
    print(f"[OK] Metadata: {metadata['name']} v{metadata['version']}")
    print(f"[OK] Minimum periods: {metadata['minimum_periods']}")
    
    print("\nVortex Indicator test completed!")