"""
Fisher Transform - Platform3 Financial Indicator

The Fisher Transform was developed by John F. Ehlers to transform prices into a 
Gaussian normal distribution. The indicator oscillates between -3 and +3, with
values beyond ±1.75 considered significant. It's particularly useful for identifying
turning points in the market.

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
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

try:
    from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
except ImportError:
    # For direct script execution
    sys.path.append(os.path.dirname(__file__))
    from base_indicator import StandardIndicatorInterface


class FisherTransform(StandardIndicatorInterface):
    """
    Fisher Transform - Platform3 Implementation
    
    The Fisher Transform converts prices into a near-normal distribution to better
    identify extreme price movements and potential reversal points.
    
    Formula:
    1. HL2 = (High + Low) / 2
    2. MinLow = Lowest(Low, period)
    3. MaxHigh = Highest(High, period)
    4. Value1 = 0.33 * 2 * ((HL2 - MinLow) / (MaxHigh - MinLow) - 0.5) + 0.67 * Previous Value1
    5. Value2 = If Value1 > 0.99 then 0.999, If Value1 < -0.99 then -0.999, else Value1
    6. Fisher = 0.5 * Log((1 + Value2) / (1 - Value2)) + 0.5 * Previous Fisher
    7. Trigger = Previous Fisher
    
    Signals:
    - Fisher > Trigger: Bullish
    - Fisher < Trigger: Bearish  
    - Fisher crossing above +1.75: Strong bearish reversal signal
    - Fisher crossing below -1.75: Strong bullish reversal signal
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Performance Optimization  
    - Robust Error Handling
    """
    
    def __init__(self, period: int = 10):
        """
        Initialize Fisher Transform.
        
        Args:
            period (int): Lookback period for min/max calculation (default: 10)
        """
        self.period = period
        self.name = "FisherTransform"
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

    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate Fisher Transform.
        
        Args:
            data: Market data (DataFrame with 'high', 'low' columns or 2D array [high, low])
            
        Returns:
            Dict containing Fisher Transform values and analysis
        """
        try:
            # Extract high and low prices
            if isinstance(data, pd.DataFrame):
                if 'high' in data.columns and 'low' in data.columns:
                    high_prices = data['high'].values
                    low_prices = data['low'].values
                elif len(data.columns) >= 2:
                    high_prices = data.iloc[:, 1].values  # Assuming [low, high] or [open, high, low, close]
                    low_prices = data.iloc[:, 0].values if len(data.columns) == 2 else data.iloc[:, 2].values
                else:
                    return {"error": "Insufficient price data columns"}
            else:
                if data.ndim == 2 and data.shape[1] >= 2:
                    high_prices = data[:, 1] if data.shape[1] == 2 else data[:, 1]
                    low_prices = data[:, 0] if data.shape[1] == 2 else data[:, 2]
                else:
                    return {"error": "Data must have high and low prices"}
            
            if len(high_prices) < self.minimum_periods:
                return {"error": f"Insufficient data. Need at least {self.minimum_periods} periods"}
            
            # Calculate median price (HL2)
            hl2 = (high_prices + low_prices) / 2.0
            
            # Calculate Fisher Transform
            fisher = np.full(len(high_prices), np.nan)
            trigger = np.full(len(high_prices), np.nan)
            value1 = 0.0
            prev_fisher = 0.0
            
            for i in range(self.period - 1, len(high_prices)):
                # Calculate min/max over the period
                start_idx = max(0, i - self.period + 1)
                min_low = np.min(low_prices[start_idx:i + 1])
                max_high = np.max(high_prices[start_idx:i + 1])
                
                # Avoid division by zero
                if max_high == min_low:
                    normalized = 0.0
                else:
                    normalized = (hl2[i] - min_low) / (max_high - min_low) - 0.5
                
                # Smooth the normalized value
                value1 = 0.33 * 2 * normalized + 0.67 * value1
                
                # Constrain value to prevent log errors
                if value1 > 0.99:
                    value2 = 0.999
                elif value1 < -0.99:
                    value2 = -0.999
                else:
                    value2 = value1
                
                # Calculate Fisher Transform
                try:
                    fisher_raw = 0.5 * np.log((1 + value2) / (1 - value2))
                    current_fisher = 0.5 * fisher_raw + 0.5 * prev_fisher
                except:
                    current_fisher = prev_fisher
                
                fisher[i] = current_fisher
                trigger[i] = prev_fisher
                prev_fisher = current_fisher
            
            # Generate signals
            signals = self._generate_signals(fisher, trigger)
            
            # Quality assessment
            quality_score = self._assess_quality(fisher, hl2)
            
            return {
                "fisher": fisher,
                "trigger": trigger,
                "fisher_values": fisher[~np.isnan(fisher)].tolist(),
                "trigger_values": trigger[~np.isnan(trigger)].tolist(),
                "current_fisher": float(fisher[-1]) if not np.isnan(fisher[-1]) else None,
                "current_trigger": float(trigger[-1]) if not np.isnan(trigger[-1]) else None,
                "hl2": hl2,
                "signals": signals,
                "quality_score": quality_score,
                "signal_strength": self._calculate_signal_strength(fisher, trigger),
                "trend_state": self._determine_trend_state(fisher, trigger),
                "extremes": self._identify_extremes(fisher)
            }
            
        except Exception as e:
            self.logger.error(f"Fisher Transform calculation error: {e}")
            return {"error": str(e)}

    def _generate_signals(self, fisher: np.ndarray, trigger: np.ndarray) -> Dict[str, List[int]]:
        """Generate trading signals based on Fisher Transform crossovers and extremes."""
        signals = {
            "bullish_crossover": [],
            "bearish_crossover": [],
            "extreme_high": [],
            "extreme_low": [],
            "reversal_zones": []
        }
        
        for i in range(1, len(fisher)):
            if np.isnan(fisher[i]) or np.isnan(trigger[i]) or np.isnan(fisher[i-1]) or np.isnan(trigger[i-1]):
                continue
            
            # Bullish crossover: Fisher crosses above trigger
            if fisher[i-1] <= trigger[i-1] and fisher[i] > trigger[i]:
                signals["bullish_crossover"].append(i)
            
            # Bearish crossover: Fisher crosses below trigger
            if fisher[i-1] >= trigger[i-1] and fisher[i] < trigger[i]:
                signals["bearish_crossover"].append(i)
            
            # Extreme readings
            if fisher[i] > 1.75:
                signals["extreme_high"].append(i)
            elif fisher[i] < -1.75:
                signals["extreme_low"].append(i)
            
            # Reversal zones (extreme readings with crossover)
            if ((fisher[i] > 1.75 and fisher[i-1] <= trigger[i-1] and fisher[i] < trigger[i]) or
                (fisher[i] < -1.75 and fisher[i-1] >= trigger[i-1] and fisher[i] > trigger[i])):
                signals["reversal_zones"].append(i)
        
        return signals

    def _assess_quality(self, fisher: np.ndarray, hl2: np.ndarray) -> float:
        """Assess the quality of Fisher Transform calculation."""
        try:
            valid_fisher = fisher[~np.isnan(fisher)]
            if len(valid_fisher) == 0:
                return 0.0
            
            # Check for reasonable oscillation range
            range_quality = 1.0 if np.max(valid_fisher) > 0.5 and np.min(valid_fisher) < -0.5 else 0.5
            
            # Check for data completeness
            completeness = len(valid_fisher) / len(fisher)
            
            # Check for price movement (Fisher works best with volatile markets)
            price_volatility = np.std(hl2[-min(50, len(hl2)):])
            volatility_quality = min(1.0, price_volatility / (np.mean(hl2[-min(50, len(hl2)):]) * 0.02))
            
            # Check for oscillations (good Fisher should cross zero regularly)
            zero_crossings = 0
            for i in range(1, len(valid_fisher)):
                if (valid_fisher[i-1] < 0 and valid_fisher[i] > 0) or (valid_fisher[i-1] > 0 and valid_fisher[i] < 0):
                    zero_crossings += 1
            oscillation_quality = min(1.0, zero_crossings / max(1, len(valid_fisher) / 20))
            
            return float(np.mean([range_quality, completeness, volatility_quality, oscillation_quality]))
            
        except Exception:
            return 0.5

    def _calculate_signal_strength(self, fisher: np.ndarray, trigger: np.ndarray) -> float:
        """Calculate signal strength based on Fisher Transform behavior."""
        try:
            if np.isnan(fisher[-1]) or np.isnan(trigger[-1]):
                return 0.0
            
            # Signal strength based on current separation and extreme levels
            separation = abs(fisher[-1] - trigger[-1])
            extreme_factor = min(1.0, abs(fisher[-1]) / 2.0)  # 2.0 is considered very extreme
            
            # Recent trend consistency
            valid_indices = ~(np.isnan(fisher) | np.isnan(trigger))
            if not np.any(valid_indices):
                return 0.0
            
            recent_fisher = fisher[valid_indices][-min(5, np.sum(valid_indices)):]
            recent_trigger = trigger[valid_indices][-min(5, np.sum(valid_indices)):]
            
            if len(recent_fisher) < 2:
                return 0.0
            
            # Trend consistency
            fisher_trend = np.mean(np.diff(recent_fisher))
            trend_strength = min(1.0, abs(fisher_trend) * 10)  # Normalize trend strength
            
            # Combine factors
            separation_factor = min(1.0, separation / 0.5)  # 0.5 is considered good separation
            
            return float((separation_factor + extreme_factor + trend_strength) / 3)
            
        except Exception:
            return 0.5

    def _determine_trend_state(self, fisher: np.ndarray, trigger: np.ndarray) -> str:
        """Determine current trend state based on Fisher Transform."""
        try:
            if np.isnan(fisher[-1]) or np.isnan(trigger[-1]):
                return "undefined"
            
            fisher_val = fisher[-1]
            trigger_val = trigger[-1]
            
            if fisher_val > trigger_val:
                if fisher_val > 1.75:
                    return "strong_bullish_extreme"
                elif fisher_val > 1.0:
                    return "strong_bullish"
                else:
                    return "bullish"
            else:
                if fisher_val < -1.75:
                    return "strong_bearish_extreme"
                elif fisher_val < -1.0:
                    return "strong_bearish"
                else:
                    return "bearish"
                
        except Exception:
            return "undefined"

    def _identify_extremes(self, fisher: np.ndarray) -> List[Dict[str, Any]]:
        """Identify extreme Fisher Transform readings."""
        extremes = []
        try:
            valid_indices = ~np.isnan(fisher)
            if not np.any(valid_indices):
                return extremes
            
            for i, val in enumerate(fisher):
                if np.isnan(val):
                    continue
                    
                if val > 1.75:
                    extremes.append({
                        "index": i,
                        "value": float(val),
                        "type": "extreme_high",
                        "strength": min(1.0, (val - 1.75) / 1.25)  # Normalize to 3.0 max
                    })
                elif val < -1.75:
                    extremes.append({
                        "index": i,
                        "value": float(val),
                        "type": "extreme_low",
                        "strength": min(1.0, abs(val + 1.75) / 1.25)  # Normalize to -3.0 min
                    })
            
            return extremes
            
        except Exception:
            return []

    @property
    def minimum_periods(self) -> int:
        """Minimum periods required for calculation."""
        return self.period

    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "category": "momentum",
            "subcategory": "oscillator",
            "parameters": self.parameters,
            "output_keys": [
                "fisher", "trigger", "fisher_values", "trigger_values",
                "current_fisher", "current_trigger", "hl2", "signals",
                "quality_score", "signal_strength", "trend_state", "extremes"
            ],
            "minimum_periods": self.minimum_periods,
            "platform3_compliant": True,
            "description": "Fisher Transform oscillator for identifying turning points and extremes"
        }


def export_indicator():
    """Export the indicator for registry discovery."""
    return FisherTransform


# Test code
if __name__ == "__main__":
    print("Testing Fisher Transform Indicator...")
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create realistic price data with some trends and volatility
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some volatility to high/low
    high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    
    test_data = pd.DataFrame({
        'date': dates,
        'high': high_prices,
        'low': low_prices,
        'close': prices
    })
    
    # Test the indicator
    fisher = FisherTransform(period=10)
    result = fisher.calculate(test_data)
    
    if "error" not in result:
        print("[OK] Fisher Transform calculation successful")
        print(f"[OK] Current Fisher: {result.get('current_fisher', 'N/A'):.4f}")
        print(f"[OK] Current Trigger: {result.get('current_trigger', 'N/A'):.4f}")
        print(f"[OK] Trend State: {result['trend_state']}")
        print(f"[OK] Quality Score: {result['quality_score']:.4f}")
        print(f"[OK] Signal Strength: {result['signal_strength']:.4f}")
        print(f"[OK] Bullish Crossovers: {len(result['signals']['bullish_crossover'])}")
        print(f"[OK] Bearish Crossovers: {len(result['signals']['bearish_crossover'])}")
        print(f"[OK] Extreme Highs: {len(result['signals']['extreme_high'])}")
        print(f"[OK] Extreme Lows: {len(result['signals']['extreme_low'])}")
        
        # Show some recent values
        if result['fisher_values']:
            recent_fisher = result['fisher_values'][-5:]
            recent_trigger = result['trigger_values'][-5:]
            print(f"[OK] Recent Fisher values: {[f'{v:.4f}' for v in recent_fisher]}")
            print(f"[OK] Recent Trigger values: {[f'{v:.4f}' for v in recent_trigger]}")
    else:
        print(f"[ERROR] Error: {result['error']}")
    
    # Test metadata
    metadata = fisher.get_metadata()
    print(f"[OK] Metadata: {metadata['name']} v{metadata['version']}")
    print(f"[OK] Minimum periods: {metadata['minimum_periods']}")
    
    print("\nFisher Transform test completed!")