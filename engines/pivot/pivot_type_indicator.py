"""
Pivot Type Indicator

Advanced pivot point analysis with multiple calculation methods and 
real-time support/resistance level detection.

Features:
- Multiple pivot calculation methods (Standard, Fibonacci, Camarilla, Woodie)
- Dynamic support and resistance levels
- Breakout detection and confirmation
- Pivot strength scoring
- Multi-timeframe analysis

Author: Platform3 AI Framework
Created: 2025-06-10
"""

import os
import sys
from typing import Any, Dict, List, Union, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from engines.ai_enhancement.indicators.base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PivotMethod(Enum):
    """Pivot calculation methods"""
    STANDARD = "standard"
    FIBONACCI = "fibonacci"
    CAMARILLA = "camarilla"
    WOODIE = "woodie"
    DEMARK = "demark"


class PivotTypeIndicator(StandardIndicatorInterface):
    """
    Pivot Type Indicator
    
    Calculates pivot points using various methods and provides support/resistance
    levels with breakout detection and strength analysis.
    """

    # Class-level metadata (REQUIRED)
    CATEGORY: str = "pivot"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"

    def __init__(
        self,
        period: int = 20,
        method: str = "standard",
        strength_threshold: float = 0.7,
        breakout_threshold: float = 0.5,
        max_levels: int = 5,
        **kwargs,
    ):
        """
        Initialize Pivot Type indicator

        Args:
            period: Period for pivot calculation and analysis (default: 20)
            method: Pivot calculation method (default: "standard")
            strength_threshold: Minimum strength for significant pivots (default: 0.7)
            breakout_threshold: Threshold for breakout confirmation (default: 0.5)
            max_levels: Maximum number of S/R levels to track (default: 5)
        """
        super().__init__(
            period=period,
            method=method,
            strength_threshold=strength_threshold,
            breakout_threshold=breakout_threshold,
            max_levels=max_levels,
            **kwargs,
        )
        
        # Initialize pivot analysis components
        self._pivot_levels = {}
        self._support_levels = []
        self._resistance_levels = []
        self._pivot_history = []
        self._breakout_signals = []

    def calculate(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Calculate pivot type indicator values

        Args:
            data: DataFrame with OHLC data
                 Required columns: 'open', 'high', 'low', 'close'
                 Optional: 'volume' for strength confirmation

        Returns:
            pd.Series: Pivot indicator values showing current pivot relative position
        """
        # Validate input data
        self.validate_input_data(data)
        
        if not isinstance(data, pd.DataFrame):
            raise IndicatorValidationError("Pivot Type requires DataFrame with OHLC data")
            
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise IndicatorValidationError(f"Missing required columns: {missing_columns}")
        
        period = self.parameters.get("period", 20)
        method = self.parameters.get("method", "standard")
        strength_threshold = self.parameters.get("strength_threshold", 0.7)
        breakout_threshold = self.parameters.get("breakout_threshold", 0.5)
        max_levels = self.parameters.get("max_levels", 5)
        
        # Initialize result series
        result_index = data.index if hasattr(data, 'index') else range(len(data))
        pivot_values = np.zeros(len(data))
        
        try:
            # Calculate pivot points for each period
            for i in range(period, len(data)):
                # Get data window for pivot calculation
                window_data = data.iloc[i-period+1:i+1]
                
                # Calculate pivot levels using specified method
                pivot_levels = self._calculate_pivot_levels(window_data, method)
                
                # Calculate current position relative to pivot levels
                current_price = data['close'].iloc[i]
                relative_position = self._calculate_relative_position(current_price, pivot_levels)
                
                # Calculate pivot strength
                pivot_strength = self._calculate_pivot_strength(window_data, pivot_levels)
                
                # Check for breakouts
                breakout_signal = self._detect_breakout(data, i, pivot_levels, breakout_threshold)
                
                # Combine factors for final indicator value
                pivot_values[i] = self._combine_pivot_factors(
                    relative_position, pivot_strength, breakout_signal, strength_threshold
                )
                
                # Store pivot data for analysis
                self._pivot_levels[i] = pivot_levels
                self._pivot_history.append({
                    'index': i,
                    'levels': pivot_levels,
                    'strength': pivot_strength,
                    'breakout': breakout_signal,
                    'relative_position': relative_position
                })
                
                # Update support/resistance tracking
                self._update_support_resistance_levels(pivot_levels, max_levels)
                
            # Store calculation details for debugging
            self._last_calculation = {
                "pivot_values": pivot_values,
                "method": method,
                "period": period,
                "total_pivots": len(self._pivot_history),
                "support_levels": len(self._support_levels),
                "resistance_levels": len(self._resistance_levels),
                "strength_threshold": strength_threshold,
                "breakout_threshold": breakout_threshold,
            }
            
            return pd.Series(
                pivot_values, 
                index=result_index, 
                name="PivotType"
            )
            
        except Exception as e:
            logger.warning(f"Error in pivot type calculation: {e}")
            # Return neutral values on error
            return pd.Series(
                np.zeros(len(data)), 
                index=result_index, 
                name="PivotType"
            )

    def _calculate_pivot_levels(self, data: pd.DataFrame, method: str) -> Dict[str, float]:
        """Calculate pivot levels using specified method"""
        high = data['high'].max()
        low = data['low'].min()
        close = data['close'].iloc[-1]
        
        # Get open for some calculations
        open_price = data['open'].iloc[0] if 'open' in data.columns else close
        
        if method == PivotMethod.STANDARD.value:
            return self._calculate_standard_pivots(high, low, close)
        elif method == PivotMethod.FIBONACCI.value:
            return self._calculate_fibonacci_pivots(high, low, close)
        elif method == PivotMethod.CAMARILLA.value:
            return self._calculate_camarilla_pivots(high, low, close)
        elif method == PivotMethod.WOODIE.value:
            return self._calculate_woodie_pivots(high, low, close, open_price)
        elif method == PivotMethod.DEMARK.value:
            return self._calculate_demark_pivots(high, low, close, open_price)
        else:
            # Default to standard method
            return self._calculate_standard_pivots(high, low, close)

    def _calculate_standard_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate standard pivot points"""
        pivot = (high + low + close) / 3
        
        levels = {
            'PP': pivot,  # Pivot Point
            'R1': 2 * pivot - low,  # Resistance 1
            'R2': pivot + (high - low),  # Resistance 2
            'R3': high + 2 * (pivot - low),  # Resistance 3
            'S1': 2 * pivot - high,  # Support 1
            'S2': pivot - (high - low),  # Support 2
            'S3': low - 2 * (high - pivot),  # Support 3
        }
        
        return levels

    def _calculate_fibonacci_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Fibonacci pivot points"""
        pivot = (high + low + close) / 3
        range_hl = high - low
        
        levels = {
            'PP': pivot,
            'R1': pivot + 0.382 * range_hl,
            'R2': pivot + 0.618 * range_hl,
            'R3': pivot + 1.000 * range_hl,
            'S1': pivot - 0.382 * range_hl,
            'S2': pivot - 0.618 * range_hl,
            'S3': pivot - 1.000 * range_hl,
        }
        
        return levels

    def _calculate_camarilla_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Camarilla pivot points"""
        range_hl = high - low
        
        levels = {
            'PP': close,  # Camarilla uses close as pivot
            'R1': close + range_hl * 1.1 / 12,
            'R2': close + range_hl * 1.1 / 6,
            'R3': close + range_hl * 1.1 / 4,
            'R4': close + range_hl * 1.1 / 2,
            'S1': close - range_hl * 1.1 / 12,
            'S2': close - range_hl * 1.1 / 6,
            'S3': close - range_hl * 1.1 / 4,
            'S4': close - range_hl * 1.1 / 2,
        }
        
        return levels

    def _calculate_woodie_pivots(self, high: float, low: float, close: float, open_price: float) -> Dict[str, float]:
        """Calculate Woodie pivot points"""
        pivot = (high + low + 2 * close) / 4
        
        levels = {
            'PP': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + high - low,
            'S1': 2 * pivot - high,
            'S2': pivot - high + low,
        }
        
        return levels

    def _calculate_demark_pivots(self, high: float, low: float, close: float, open_price: float) -> Dict[str, float]:
        """Calculate DeMark pivot points"""
        # DeMark calculation depends on close vs open relationship
        if close < open_price:
            x = high + 2 * low + close
        elif close > open_price:
            x = 2 * high + low + close
        else:  # close == open
            x = high + low + 2 * close
            
        levels = {
            'PP': x / 4,
            'R1': x / 2 - low,
            'S1': x / 2 - high,
        }
        
        return levels

    def _calculate_relative_position(self, current_price: float, pivot_levels: Dict[str, float]) -> float:
        """Calculate current price position relative to pivot levels"""
        pivot = pivot_levels.get('PP', current_price)
        
        # Find the closest support and resistance levels
        resistance_levels = [v for k, v in pivot_levels.items() if k.startswith('R') and v > current_price]
        support_levels = [v for k, v in pivot_levels.items() if k.startswith('S') and v < current_price]
        
        if resistance_levels and support_levels:
            nearest_resistance = min(resistance_levels)
            nearest_support = max(support_levels)
            
            # Calculate relative position between support and resistance
            if nearest_resistance != nearest_support:
                relative_pos = (current_price - nearest_support) / (nearest_resistance - nearest_support)
                return (relative_pos - 0.5) * 2  # Scale to [-1, 1]
        
        # Fall back to position relative to pivot point
        if pivot != 0:
            # Normalize by typical range (could be improved with ATR)
            range_estimate = abs(max(pivot_levels.values()) - min(pivot_levels.values()))
            if range_estimate > 0:
                return np.tanh((current_price - pivot) / (range_estimate * 0.5))
        
        return 0.0

    def _calculate_pivot_strength(self, data: pd.DataFrame, pivot_levels: Dict[str, float]) -> float:
        """Calculate strength/reliability of pivot levels"""
        # Factors that contribute to pivot strength:
        # 1. Volume confirmation (if available)
        # 2. Price action around pivot levels
        # 3. Number of times levels were tested
        # 4. Range of the pivot calculation period
        
        strength_factors = []
        
        # Volume factor (if available)
        if 'volume' in data.columns and len(data['volume']) > 1:
            avg_volume = data['volume'].mean()
            recent_volume = data['volume'].iloc[-1]
            volume_factor = min(2.0, recent_volume / avg_volume) if avg_volume > 0 else 1.0
            strength_factors.append(volume_factor)
        
        # Range factor - larger ranges often indicate stronger pivots
        price_range = data['high'].max() - data['low'].min()
        avg_range = (data['high'] - data['low']).mean()
        range_factor = min(2.0, price_range / avg_range) if avg_range > 0 else 1.0
        strength_factors.append(range_factor)
        
        # Price action factor - how much price respects pivot levels
        current_price = data['close'].iloc[-1]
        pivot = pivot_levels.get('PP', current_price)
        
        # Check if price is near a significant level
        tolerance = price_range * 0.01  # 1% of range
        near_level = False
        for level_value in pivot_levels.values():
            if abs(current_price - level_value) <= tolerance:
                near_level = True
                break
        
        level_factor = 1.5 if near_level else 1.0
        strength_factors.append(level_factor)
        
        # Calculate overall strength
        if strength_factors:
            strength = np.mean(strength_factors)
            return min(1.0, strength / 2.0)  # Normalize to [0, 1]
        
        return 0.5  # Default moderate strength

    def _detect_breakout(self, data: pd.DataFrame, index: int, pivot_levels: Dict[str, float], threshold: float) -> float:
        """Detect breakout from pivot levels"""
        if index < 3:  # Need some history
            return 0.0
            
        current_price = data['close'].iloc[index]
        previous_prices = data['close'].iloc[max(0, index-3):index]
        
        breakout_signal = 0.0
        
        # Check for breakouts above resistance levels
        resistance_levels = [v for k, v in pivot_levels.items() if k.startswith('R')]
        for resistance in resistance_levels:
            if current_price > resistance and all(p <= resistance for p in previous_prices):
                # Confirmed breakout above resistance
                strength = (current_price - resistance) / resistance if resistance > 0 else 0
                breakout_signal = max(breakout_signal, min(1.0, strength * 100))  # Scale appropriately
                
        # Check for breakouts below support levels  
        support_levels = [v for k, v in pivot_levels.items() if k.startswith('S')]
        for support in support_levels:
            if current_price < support and all(p >= support for p in previous_prices):
                # Confirmed breakout below support
                strength = (support - current_price) / support if support > 0 else 0
                breakout_signal = min(breakout_signal, -min(1.0, strength * 100))  # Negative for bearish
                
        return breakout_signal

    def _combine_pivot_factors(self, relative_position: float, strength: float, breakout: float, strength_threshold: float) -> float:
        """Combine various pivot factors into final indicator value"""
        # Base signal from relative position
        base_signal = relative_position
        
        # Amplify signal based on pivot strength
        if strength >= strength_threshold:
            strength_multiplier = 1.0 + (strength - strength_threshold) / (1.0 - strength_threshold)
            base_signal *= strength_multiplier
        else:
            # Reduce signal for weak pivots
            base_signal *= strength / strength_threshold
            
        # Add breakout component
        breakout_component = breakout * 0.3  # Breakouts get 30% weight
        
        # Combine signals
        final_signal = base_signal + breakout_component
        
        # Bound to [-1, 1]
        return np.tanh(final_signal)

    def _update_support_resistance_levels(self, pivot_levels: Dict[str, float], max_levels: int):
        """Update tracked support and resistance levels"""
        # Extract and sort levels
        resistance_values = [v for k, v in pivot_levels.items() if k.startswith('R')]
        support_values = [v for k, v in pivot_levels.items() if k.startswith('S')]
        
        # Update resistance levels
        for resistance in resistance_values:
            if resistance not in self._resistance_levels:
                self._resistance_levels.append(resistance)
                
        # Update support levels
        for support in support_values:
            if support not in self._support_levels:
                self._support_levels.append(support)
                
        # Keep only the most relevant levels
        self._resistance_levels = sorted(self._resistance_levels)[-max_levels:]
        self._support_levels = sorted(self._support_levels, reverse=True)[:max_levels]

    def validate_parameters(self) -> bool:
        """Validate Pivot Type parameters"""
        period = self.parameters.get("period", 20)
        method = self.parameters.get("method", "standard")
        strength_threshold = self.parameters.get("strength_threshold", 0.7)
        breakout_threshold = self.parameters.get("breakout_threshold", 0.5)
        max_levels = self.parameters.get("max_levels", 5)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        valid_methods = [method.value for method in PivotMethod]
        if method not in valid_methods:
            raise IndicatorValidationError(
                f"method must be one of {valid_methods}, got {method}"
            )

        if not isinstance(strength_threshold, (int, float)) or not (0 <= strength_threshold <= 1):
            raise IndicatorValidationError(
                f"strength_threshold must be between 0 and 1, got {strength_threshold}"
            )

        if not isinstance(breakout_threshold, (int, float)) or not (0 <= breakout_threshold <= 1):
            raise IndicatorValidationError(
                f"breakout_threshold must be between 0 and 1, got {breakout_threshold}"
            )

        if not isinstance(max_levels, int) or max_levels < 1:
            raise IndicatorValidationError(
                f"max_levels must be positive integer, got {max_levels}"
            )

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return Pivot Type metadata as dictionary for compatibility"""
        return {
            "name": "PivotType",
            "category": self.CATEGORY,
            "description": "Pivot Type - Advanced pivot point analysis with multiple methods",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """Pivot Type requires OHLC data"""
        return ["high", "low", "close"]  # open and volume are optional but beneficial

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for pivot calculation"""
        return self.parameters.get("period", 20)

    def _setup_defaults(self):
        """Setup default parameter values"""
        if "period" not in self.parameters:
            self.parameters["period"] = 20
        if "method" not in self.parameters:
            self.parameters["method"] = "standard"
        if "strength_threshold" not in self.parameters:
            self.parameters["strength_threshold"] = 0.7
        if "breakout_threshold" not in self.parameters:
            self.parameters["breakout_threshold"] = 0.5
        if "max_levels" not in self.parameters:
            self.parameters["max_levels"] = 5

    # Property accessors for backward compatibility
    @property
    def period(self) -> int:
        """Period for backward compatibility"""
        return self.parameters.get("period", 20)

    @property
    def method(self) -> str:
        """Method for backward compatibility"""
        return self.parameters.get("method", "standard")

    @property
    def strength_threshold(self) -> float:
        """Strength threshold for backward compatibility"""
        return self.parameters.get("strength_threshold", 0.7)

    def get_current_pivot_levels(self) -> Dict[str, float]:
        """Get the most recent pivot levels"""
        if self._pivot_history:
            return self._pivot_history[-1]['levels'].copy()
        return {}

    def get_support_resistance_levels(self) -> Tuple[List[float], List[float]]:
        """Get current support and resistance levels"""
        return self._support_levels.copy(), self._resistance_levels.copy()

    def get_pivot_signal(self, pivot_value: float) -> str:
        """
        Get trading signal based on pivot indicator value

        Args:
            pivot_value: Current pivot indicator value

        Returns:
            str: Trading signal description
        """
        if pivot_value > 0.7:
            return "strong_resistance_breakout"
        elif pivot_value > 0.3:
            return "approaching_resistance"
        elif pivot_value > 0.1:
            return "above_pivot"
        elif pivot_value > -0.1:
            return "at_pivot"
        elif pivot_value > -0.3:
            return "below_pivot"
        elif pivot_value > -0.7:
            return "approaching_support"
        else:
            return "strong_support_breakout"


# Export for dynamic discovery
def get_indicator_class():
    """Return the indicator class for dynamic registration"""
    return PivotTypeIndicator


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt

    # Generate sample OHLC data
    np.random.seed(42)
    n_points = 200
    
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    
    # Generate realistic OHLC data
    base_price = 100
    prices = [base_price]
    opens = [base_price]
    highs = [base_price]
    lows = [base_price]
    closes = [base_price]
    volumes = [10000]
    
    for i in range(1, n_points):
        # Price evolution with some trend and noise
        change = np.random.randn() * 0.5 + np.sin(i / 20) * 0.2  # Add some cyclical behavior
        new_close = closes[-1] + change
        
        # Generate OHLC from close
        volatility = abs(np.random.randn() * 0.3)
        high = new_close + volatility
        low = new_close - volatility
        open_price = closes[-1] + np.random.randn() * 0.1  # Small gap
        
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        closes.append(new_close)
        volumes.append(np.random.randint(5000, 20000))
    
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    # Calculate pivot indicator with different methods
    methods = ['standard', 'fibonacci', 'camarilla']
    results = {}
    
    for method in methods:
        pivot_indicator = PivotTypeIndicator(period=20, method=method)
        results[method] = pivot_indicator.calculate(data)

    # Plot results
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(12, 12))

    # Price chart with pivot levels
    axes[0].plot(dates, closes, label="Close Price", color="blue", linewidth=1)
    
    # Add current pivot levels from standard method
    standard_indicator = PivotTypeIndicator(period=20, method='standard')
    standard_result = standard_indicator.calculate(data)
    current_levels = standard_indicator.get_current_pivot_levels()
    
    if current_levels:
        for level_name, level_value in current_levels.items():
            color = 'red' if level_name.startswith('R') else 'green' if level_name.startswith('S') else 'blue'
            axes[0].axhline(y=level_value, color=color, linestyle='--', alpha=0.7, 
                          label=f'{level_name}: {level_value:.2f}')
    
    axes[0].set_title("Price Data with Pivot Levels (Standard Method)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True)

    # Plot each method's results
    for i, (method, result) in enumerate(results.items(), 1):
        axes[i].plot(dates, result.values, label=f"Pivot Type ({method})", linewidth=2)
        axes[i].axhline(y=0.7, color="red", linestyle="--", alpha=0.7, label="Strong Resistance (+0.7)")
        axes[i].axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Approaching Resistance (+0.3)")
        axes[i].axhline(y=-0.3, color="orange", linestyle="--", alpha=0.7, label="Approaching Support (-0.3)")
        axes[i].axhline(y=-0.7, color="green", linestyle="--", alpha=0.7, label="Strong Support (-0.7)")
        axes[i].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[i].set_title(f"Pivot Type Indicator - {method.title()} Method")
        axes[i].set_ylabel("Pivot Value")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    print("Pivot Type calculation completed successfully!")
    print(f"Data points: {len(data)}")
    
    # Show results for each method
    for method, result in results.items():
        pivot_indicator = PivotTypeIndicator(period=20, method=method)
        pivot_indicator.calculate(data)  # Recalculate to get internal state
        
        print(f"\n{method.title()} Method:")
        print(f"  Current value: {result.iloc[-1]:.3f}")
        print(f"  Signal: {pivot_indicator.get_pivot_signal(result.iloc[-1])}")
        
        current_levels = pivot_indicator.get_current_pivot_levels()
        if current_levels:
            print(f"  Current pivot levels:")
            for level_name, level_value in sorted(current_levels.items()):
                print(f"    {level_name}: {level_value:.2f}")
        
        support_levels, resistance_levels = pivot_indicator.get_support_resistance_levels()
        print(f"  Tracked support levels: {len(support_levels)}")
        print(f"  Tracked resistance levels: {len(resistance_levels)}")

    # Statistics for standard method
    standard_result = results['standard']
    valid_pivot = standard_result.dropna()
    print(f"\nStandard Method Statistics:")
    print(f"Min: {valid_pivot.min():.3f}")
    print(f"Max: {valid_pivot.max():.3f}")
    print(f"Mean: {valid_pivot.mean():.3f}")
    print(f"Std: {valid_pivot.std():.3f}")
    print(f"Above pivot: {(valid_pivot > 0).sum()}")
    print(f"Below pivot: {(valid_pivot < 0).sum()}")
    print(f"Strong signals: {(abs(valid_pivot) > 0.7).sum()}")